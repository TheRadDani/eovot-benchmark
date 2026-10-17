"""Structured challenge scenarios for synthetic tracker evaluation.

Standard tracking benchmarks test accuracy on natural videos but do not
isolate *why* a tracker fails.  This module addresses that by generating
synthetic sequences in which exactly one well-defined challenge is active.
Each challenge scenario targets a specific tracker failure mode common in
edge deployments:

+------------------------+--------------------------------------------------+
| Challenge              | What it tests                                    |
+========================+==================================================+
| ``illumination``       | Sinusoidal brightness oscillation — tests color  |
|                        | histogram and appearance model stability.        |
+------------------------+--------------------------------------------------+
| ``size_change``        | Target area grows then shrinks — tests scale     |
|                        | adaptation in correlation-filter trackers.       |
+------------------------+--------------------------------------------------+
| ``partial_occlusion``  | A same-width bar sweeps across the target —      |
|                        | tests robustness to sudden area loss.            |
+------------------------+--------------------------------------------------+
| ``disappearance``      | Target exits frame, re-enters from the other     |
|                        | side — tests long-term recovery.                 |
+------------------------+--------------------------------------------------+
| ``distractors``        | Multiple same-colour decoys in background —      |
|                        | tests foreground/background discriminability.    |
+------------------------+--------------------------------------------------+

Each scenario uses the ``linear`` motion pattern from
:class:`~eovot.datasets.synthetic.SyntheticDataset` as the base, then
applies the challenge on top so that the challenge is the *only* variable.

The ground-truth boxes always represent the *true* target position and size,
regardless of whether the target is (partially) occluded or invisible.

Typical usage::

    from eovot.datasets.synthetic_challenge import ChallengeSyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.kcf import KCFTracker

    for challenge in ChallengeSyntheticDataset.ALL_CHALLENGES:
        dataset = ChallengeSyntheticDataset(
            challenge=challenge,
            num_sequences=5,
            num_frames=120,
            severity=0.8,
        )
        engine  = BenchmarkEngine(verbose=False)
        result  = engine.run(KCFTracker(), dataset, dataset_name=f"Challenge-{challenge}")
        print(f"{challenge:25s}  mIoU={result.mean_iou:.4f}  FPS={result.mean_fps:.1f}")
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np

from .base import BaseDataset, Sequence

ChallengeType = Literal[
    "illumination",
    "size_change",
    "partial_occlusion",
    "disappearance",
    "distractors",
]


class _ChallengeSequence(Sequence):
    """In-memory sequence whose frames are pre-rendered numpy arrays."""

    def __init__(
        self,
        name: str,
        frames: List[np.ndarray],
        ground_truth: np.ndarray,
    ) -> None:
        self._frames = frames
        super().__init__(
            name=name,
            frame_paths=["<memory>"] * len(frames),
            ground_truth=ground_truth,
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self._frames


class ChallengeSyntheticDataset(BaseDataset):
    """Synthetic dataset with one structured tracking challenge per sequence.

    Each sequence uses the same linear-motion base as
    :class:`~eovot.datasets.synthetic.SyntheticDataset` but layers exactly
    one challenge on top.  This lets you run all five challenges with a single
    tracker and attribute accuracy differences to specific failure modes.

    Args:
        challenge: Which challenge to activate.  One of:
            ``"illumination"``, ``"size_change"``, ``"partial_occlusion"``,
            ``"disappearance"``, ``"distractors"``.
        num_sequences: Number of sequences to generate.  Default: ``10``.
        num_frames: Frames per sequence.  Default: ``120``.
        frame_size: ``(width, height)`` of each frame in pixels.
            Default: ``(320, 240)``.
        bbox_size: ``(width, height)`` of the base target rectangle.
            Default: ``(40, 40)``.
        severity: Challenge strength in ``[0, 1]``.  ``0.0`` = near-invisible
            effect (close to the clean baseline), ``1.0`` = maximum challenge.
            Default: ``0.7``.
        seed: Base RNG seed; sequence *i* uses ``seed + i``.  Default: ``42``.

    Raises:
        ValueError: If *challenge* is not one of :attr:`ALL_CHALLENGES`, or if
            *severity* is outside ``[0, 1]``.

    Example::

        ds = ChallengeSyntheticDataset("size_change", num_sequences=3,
                                        severity=0.9)
        for seq in ds:
            print(seq.name, seq.ground_truth.shape)
        # challenge_size_change_000  (120, 4)
        # challenge_size_change_001  (120, 4)
        # challenge_size_change_002  (120, 4)
    """

    ALL_CHALLENGES: Tuple[ChallengeType, ...] = (
        "illumination",
        "size_change",
        "partial_occlusion",
        "disappearance",
        "distractors",
    )

    def __init__(
        self,
        challenge: ChallengeType,
        num_sequences: int = 10,
        num_frames: int = 120,
        frame_size: Tuple[int, int] = (320, 240),
        bbox_size: Tuple[int, int] = (40, 40),
        severity: float = 0.7,
        seed: int = 42,
    ) -> None:
        if challenge not in self.ALL_CHALLENGES:
            raise ValueError(
                f"Unknown challenge: {challenge!r}. "
                f"Choose from {self.ALL_CHALLENGES}."
            )
        if not 0.0 <= severity <= 1.0:
            raise ValueError(
                f"severity must be in [0, 1], got {severity}."
            )
        self.challenge = challenge
        self.num_sequences = num_sequences
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.bbox_size = bbox_size
        self.severity = severity
        self.seed = seed
        self._cache: List[Optional[_ChallengeSequence]] = [None] * num_sequences

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Sequence:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(
                f"Sequence index {idx} out of range [0, {self.num_sequences})"
            )
        if self._cache[idx] is None:
            self._cache[idx] = self._build_sequence(idx)
        return self._cache[idx]  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"ChallengeSyntheticDataset("
            f"challenge={self.challenge!r}, "
            f"sequences={self.num_sequences}, "
            f"frames={self.num_frames}, "
            f"severity={self.severity})"
        )

    # ------------------------------------------------------------------
    # Sequence generation dispatch
    # ------------------------------------------------------------------

    def _build_sequence(self, idx: int) -> _ChallengeSequence:
        rng = np.random.default_rng(self.seed + idx)
        W, H = self.frame_size
        bw, bh = self.bbox_size
        half_bw, half_bh = bw / 2.0, bh / 2.0

        # Shared linear motion base
        cx0 = float(rng.integers(bw, W - bw))
        cy0 = float(rng.integers(bh, H - bh))
        vx = float(rng.uniform(1.0, 3.0)) * float(rng.choice([-1, 1]))
        vy = float(rng.uniform(0.5, 2.0)) * float(rng.choice([-1, 1]))
        positions = self._linear_positions(cx0, cy0, vx, vy, W, H, half_bw, half_bh)

        background = rng.integers(40, 100, (H, W, 3), dtype=np.uint8)
        colour = tuple(int(c) for c in rng.integers(160, 256, 3))

        builders: Dict[str, object] = {
            "illumination": self._build_illumination,
            "size_change": self._build_size_change,
            "partial_occlusion": self._build_partial_occlusion,
            "disappearance": self._build_disappearance,
            "distractors": self._build_distractors,
        }
        builder = builders[self.challenge]
        name = f"challenge_{self.challenge}_{idx:03d}"
        return builder(  # type: ignore[operator]
            name, positions, background, colour, rng
        )

    # ------------------------------------------------------------------
    # Challenge implementations
    # ------------------------------------------------------------------

    def _build_illumination(
        self,
        name: str,
        positions: List[Tuple[float, float]],
        background: np.ndarray,
        colour: Tuple[int, ...],
        rng: np.random.Generator,
    ) -> _ChallengeSequence:
        """Sinusoidal brightness oscillation.

        Brightness multiplier cycles between (1 - severity) and 1.0.
        At severity=1.0 the darkest frames drop to zero brightness (black).
        """
        W, H = self.frame_size
        bw, bh = self.bbox_size
        T = self.num_frames
        # Brightness oscillates with ~3 full cycles.
        amp = 0.5 * self.severity
        base = 1.0 - amp

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []

        for i, (cx, cy) in enumerate(positions):
            brightness = base + amp * math.cos(2 * math.pi * 3 * i / max(T - 1, 1))
            frame = (background.astype(np.float32) * brightness).clip(0, 255).astype(np.uint8)
            x1, y1 = int(round(cx - bw / 2)), int(round(cy - bh / 2))
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x1 + bw), min(H, y1 + bh)
            scaled_colour = tuple(
                int(c * brightness) for c in colour
            )
            frame[y1c:y2c, x1c:x2c] = np.clip(scaled_colour, 0, 255)
            frames.append(frame)
            gt_boxes.append((float(x1), float(y1), float(bw), float(bh)))

        return _ChallengeSequence(
            name=name,
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    def _build_size_change(
        self,
        name: str,
        positions: List[Tuple[float, float]],
        background: np.ndarray,
        colour: Tuple[int, ...],
        rng: np.random.Generator,
    ) -> _ChallengeSequence:
        """Target area expands then contracts.

        At severity=1.0 the target reaches 3× its base area at peak.
        The base bbox_size is maintained at the start and end of the sequence.
        """
        W, H = self.frame_size
        bw, bh = self.bbox_size
        T = self.num_frames
        # Scale factor oscillates: 1.0 → (1 + 2*severity) → 1.0
        max_scale = 1.0 + 2.0 * self.severity

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []

        for i, (cx, cy) in enumerate(positions):
            t = i / max(T - 1, 1)
            scale = 1.0 + (max_scale - 1.0) * math.sin(math.pi * t)
            w_i = int(round(bw * scale))
            h_i = int(round(bh * scale))
            w_i = max(2, w_i)
            h_i = max(2, h_i)

            frame = background.copy()
            x1 = int(round(cx - w_i / 2))
            y1 = int(round(cy - h_i / 2))
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x1 + w_i), min(H, y1 + h_i)
            frame[y1c:y2c, x1c:x2c] = colour
            frames.append(frame)
            gt_boxes.append((float(x1), float(y1), float(w_i), float(h_i)))

        return _ChallengeSequence(
            name=name,
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    def _build_partial_occlusion(
        self,
        name: str,
        positions: List[Tuple[float, float]],
        background: np.ndarray,
        colour: Tuple[int, ...],
        rng: np.random.Generator,
    ) -> _ChallengeSequence:
        """A grey occluder bar sweeps vertically through the target zone.

        The bar enters from the left at frame T//3, fully covers the target
        by frame T//2, and exits by frame 2*T//3.  The occluder height is
        ``severity`` × target height.  GT boxes always show the true target
        position and size (not the visible portion).
        """
        W, H = self.frame_size
        bw, bh = self.bbox_size
        T = self.num_frames
        occluder_h = max(1, int(round(bh * (0.3 + 0.7 * self.severity))))
        occluder_colour = (128, 128, 128)

        # Occluder sweeps from x=-occluder_w → x=W across the middle third.
        enter_frame = T // 3
        exit_frame = 2 * T // 3
        occluder_w = bw + 20  # slightly wider than target

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []

        for i, (cx, cy) in enumerate(positions):
            frame = background.copy()
            x1 = int(round(cx - bw / 2))
            y1 = int(round(cy - bh / 2))
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x1 + bw), min(H, y1 + bh)
            frame[y1c:y2c, x1c:x2c] = colour

            if enter_frame <= i <= exit_frame:
                span = max(1, exit_frame - enter_frame)
                prog = (i - enter_frame) / span  # 0 → 1
                # Occluder's left edge sweeps from -occluder_w to W
                occ_x = int(round(-occluder_w + (W + occluder_w) * prog))
                occ_y = int(round(cy - occluder_h / 2))
                occ_x1c = max(0, occ_x)
                occ_y1c = max(0, occ_y)
                occ_x2c = min(W, occ_x + occluder_w)
                occ_y2c = min(H, occ_y + occluder_h)
                if occ_x2c > occ_x1c and occ_y2c > occ_y1c:
                    frame[occ_y1c:occ_y2c, occ_x1c:occ_x2c] = occluder_colour

            frames.append(frame)
            gt_boxes.append((float(x1), float(y1), float(bw), float(bh)))

        return _ChallengeSequence(
            name=name,
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    def _build_disappearance(
        self,
        name: str,
        positions: List[Tuple[float, float]],
        background: np.ndarray,
        colour: Tuple[int, ...],
        rng: np.random.Generator,
    ) -> _ChallengeSequence:
        """Target leaves the frame and re-enters from the opposite side.

        Absence duration = ``severity × 30 %`` of total frames, centered at
        the midpoint of the sequence.  During absence the target is outside
        the frame; the GT box still shows the (off-screen) true position so
        IoU is 0 during the absence window.  This tests long-term recovery.
        """
        W, H = self.frame_size
        bw, bh = self.bbox_size
        T = self.num_frames
        absence_len = max(1, int(round(T * 0.30 * self.severity)))
        mid = T // 2
        abs_start = max(0, mid - absence_len // 2)
        abs_end = min(T, abs_start + absence_len)

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []

        for i, (cx, cy) in enumerate(positions):
            frame = background.copy()

            if abs_start <= i < abs_end:
                # Target is off-screen: GT placed just outside frame boundary
                gt_x = float(-bw)
                gt_y = float(cy - bh / 2)
            else:
                x1 = int(round(cx - bw / 2))
                y1 = int(round(cy - bh / 2))
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(W, x1 + bw), min(H, y1 + bh)
                if x2c > x1c and y2c > y1c:
                    frame[y1c:y2c, x1c:x2c] = colour
                gt_x = float(x1)
                gt_y = float(y1)

            frames.append(frame)
            gt_boxes.append((gt_x, gt_y, float(bw), float(bh)))

        return _ChallengeSequence(
            name=name,
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    def _build_distractors(
        self,
        name: str,
        positions: List[Tuple[float, float]],
        background: np.ndarray,
        colour: Tuple[int, ...],
        rng: np.random.Generator,
    ) -> _ChallengeSequence:
        """Background distractors of the same colour as the target.

        Number of distractors scales with severity: 1 at 0.2, up to 5 at 1.0.
        Distractors move independently using random-walk motion.  They are the
        same colour as the target to maximise confusion for colour-histogram
        trackers (e.g. CamShift, ParticleFilter).
        """
        W, H = self.frame_size
        bw, bh = self.bbox_size
        n_distractors = max(1, int(round(self.severity * 5)))

        # Generate random-walk trajectories for each distractor.
        distractor_positions: List[List[Tuple[float, float]]] = []
        for _ in range(n_distractors):
            dx0 = float(rng.integers(bw, W - bw))
            dy0 = float(rng.integers(bh, H - bh))
            step = float(rng.uniform(2.0, 4.0))
            traj: List[Tuple[float, float]] = []
            dx, dy = dx0, dy0
            for _ in range(self.num_frames):
                traj.append((dx, dy))
                dx = float(
                    np.clip(dx + rng.uniform(-step, step), bw / 2, W - bw / 2)
                )
                dy = float(
                    np.clip(dy + rng.uniform(-step, step), bh / 2, H - bh / 2)
                )
            distractor_positions.append(traj)

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []

        for i, (cx, cy) in enumerate(positions):
            frame = background.copy()

            # Draw distractors first (underneath target)
            for traj in distractor_positions:
                dcx, dcy = traj[i]
                dx1 = int(round(dcx - bw / 2))
                dy1 = int(round(dcy - bh / 2))
                dx1c, dy1c = max(0, dx1), max(0, dy1)
                dx2c, dy2c = min(W, dx1 + bw), min(H, dy1 + bh)
                if dx2c > dx1c and dy2c > dy1c:
                    frame[dy1c:dy2c, dx1c:dx2c] = colour

            # Draw target on top
            x1 = int(round(cx - bw / 2))
            y1 = int(round(cy - bh / 2))
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x1 + bw), min(H, y1 + bh)
            if x2c > x1c and y2c > y1c:
                frame[y1c:y2c, x1c:x2c] = colour

            frames.append(frame)
            gt_boxes.append((float(x1), float(y1), float(bw), float(bh)))

        return _ChallengeSequence(
            name=name,
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Motion helpers
    # ------------------------------------------------------------------

    def _linear_positions(
        self,
        cx0: float,
        cy0: float,
        vx: float,
        vy: float,
        W: int,
        H: int,
        half_bw: float,
        half_bh: float,
    ) -> List[Tuple[float, float]]:
        """Generate linear motion with wall-bounce."""
        positions: List[Tuple[float, float]] = []
        cx, cy = cx0, cy0
        for _ in range(self.num_frames):
            positions.append((cx, cy))
            cx += vx
            cy += vy
            if cx < half_bw or cx > W - half_bw:
                vx = -vx
                cx = float(np.clip(cx, half_bw, W - half_bw))
            if cy < half_bh or cy > H - half_bh:
                vy = -vy
                cy = float(np.clip(cy, half_bh, H - half_bh))
        return positions


def run_challenge_suite(
    tracker,
    challenges: Optional[List[ChallengeType]] = None,
    num_sequences: int = 5,
    num_frames: int = 120,
    severity: float = 0.7,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run a tracker against all challenge scenarios and return a summary.

    Convenience function for quick tracker profiling across all challenge types.

    Args:
        tracker: Any :class:`~eovot.trackers.base.BaseTracker` instance.
        challenges: List of challenge names to run.  Defaults to all five.
        num_sequences: Sequences per challenge.  Default: ``5``.
        num_frames: Frames per sequence.  Default: ``120``.
        severity: Challenge strength in ``[0, 1]``.  Default: ``0.7``.
        seed: RNG seed.  Default: ``42``.
        verbose: Print per-challenge progress.  Default: ``True``.

    Returns:
        Dict mapping challenge name → ``{"mean_iou": float, "mean_fps": float}``.

    Example::

        from eovot.trackers.kcf import KCFTracker
        from eovot.datasets.synthetic_challenge import run_challenge_suite

        summary = run_challenge_suite(KCFTracker(), severity=0.8)
        for ch, metrics in summary.items():
            print(f"{ch:25s}  mIoU={metrics['mean_iou']:.4f}")
    """
    from ..benchmark.engine import BenchmarkEngine

    if challenges is None:
        challenges = list(ChallengeSyntheticDataset.ALL_CHALLENGES)

    engine = BenchmarkEngine(verbose=False)
    summary: Dict[str, object] = {}

    for challenge in challenges:
        dataset = ChallengeSyntheticDataset(
            challenge=challenge,
            num_sequences=num_sequences,
            num_frames=num_frames,
            severity=severity,
            seed=seed,
        )
        result = engine.run(tracker, dataset, dataset_name=f"Challenge-{challenge}")
        summary[challenge] = {
            "mean_iou": round(result.mean_iou, 4),
            "mean_fps": round(result.mean_fps, 1),
            "success_auc": round(result.mean_success_auc or 0.0, 4),
        }
        if verbose:
            print(
                f"  {challenge:25s}  "
                f"mIoU={result.mean_iou:.4f}  "
                f"FPS={result.mean_fps:.1f}"
            )

    return summary
