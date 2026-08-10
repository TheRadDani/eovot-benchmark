"""Challenge-aware synthetic dataset for controlled VOT evaluation.

Extends :class:`~eovot.datasets.synthetic.SyntheticDataset` with four
challenge modes that deliberately introduce VOT difficulty attributes.
This bridges the gap between the attribute analysis module
(:mod:`eovot.metrics.attributes`) and actually having synthetic sequences
that exercise those attributes, enabling rapid controlled ablation without
downloading OTB, LaSOT, or GOT-10k.

Challenge modes
---------------
``"occlusion"``
    A rectangular occluder passes over the target at regular intervals,
    causing single-frame area-drop events that trigger the
    ``partial_occlusion`` attribute.  The tracker must survive brief
    complete occlusions and re-acquire the target.

``"scale_change"``
    The target rectangle smoothly grows to 2.5× its initial size and
    shrinks back over the sequence, triggering the ``scale_variation``
    attribute (max/min area ratio > 4).  Tests a tracker's ability to
    handle significant target size changes.

``"fast_motion"``
    Periodic velocity bursts (5× normal speed for 3 frames every 20 frames)
    produce large inter-frame displacements that trigger ``fast_motion``
    (displacement > 20 % of mean box diagonal).  Tests whether a tracker
    can recover from sudden target jumps.

``"illumination"``
    Frame-wide brightness pulses simulate sudden illumination changes
    (a common real-world failure mode for appearance-based trackers).
    The target undergoes a random-walk motion so the dataset also provides
    basic position jitter for temporal consistency analysis.

All modes build on the in-memory rendering infrastructure of
:class:`~eovot.datasets.synthetic.SyntheticDataset`, requiring no file
I/O and no external libraries beyond NumPy and OpenCV.

Example::

    from eovot.datasets.challenge import ChallengeSyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.metrics.attributes import AttributeAnalyzer
    from eovot.trackers.registry import build_tracker

    dataset = ChallengeSyntheticDataset(
        challenge="occlusion",
        num_sequences=5,
        num_frames=120,
    )
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(build_tracker("MOSSE"), dataset, dataset_name="occlusion")

    analyzer = AttributeAnalyzer()
    table = analyzer.breakdown(result)
    print(table.to_markdown())
"""

from __future__ import annotations

import math
from typing import Iterator, List, Literal, Optional, Tuple

import numpy as np

from .base import BaseDataset, Sequence

ChallengeMode = Literal["occlusion", "scale_change", "fast_motion", "illumination"]

_VALID_CHALLENGES: Tuple[str, ...] = (
    "occlusion",
    "scale_change",
    "fast_motion",
    "illumination",
)


class _InMemorySequence(Sequence):
    """Sequence whose frames are held in memory (no file I/O)."""

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
    """Synthetic dataset with deliberate VOT challenge attributes.

    Each sequence renders a coloured filled rectangle against a textured
    noise background and then introduces the selected challenge pattern
    so that :class:`~eovot.metrics.attributes.AttributeDetector` reliably
    flags the corresponding attribute.

    Args:
        challenge: One of ``"occlusion"``, ``"scale_change"``,
            ``"fast_motion"``, or ``"illumination"``.
        num_sequences: Number of sequences to generate.  Default: ``5``.
        num_frames: Frames per sequence.  Default: ``120``.
        frame_size: ``(width, height)`` of each frame in pixels.
            Default: ``(320, 240)``.
        base_bbox_size: ``(width, height)`` of the target at its nominal
            (un-scaled) size.  For ``"scale_change"`` the target grows to
            ``max_scale_factor`` times this size.
            Default: ``(40, 40)``.
        max_scale_factor: Maximum scale multiplier applied in
            ``"scale_change"`` mode.  Must be ≥ 2.01 to trigger the
            ``scale_variation`` attribute (area ratio > 4).  Default: ``2.5``.
        occlusion_period: Frames between occlusion events in
            ``"occlusion"`` mode.  Default: ``20``.
        occlusion_duration: Frames each occlusion event lasts.  Default: ``5``.
        burst_period: Frames between velocity burst events in
            ``"fast_motion"`` mode.  Default: ``20``.
        burst_duration: Frames each velocity burst lasts.  Default: ``3``.
        burst_speed_factor: Speed multiplier during a burst.  Default: ``5.0``.
        illumination_period: Frames between brightness pulses in
            ``"illumination"`` mode.  Default: ``25``.
        illumination_duration: Frames each brightness pulse lasts.  Default: ``8``.
        illumination_delta: Maximum absolute brightness shift (0-255 scale).
            Default: ``80``.
        seed: Base RNG seed; sequence ``i`` uses ``seed + i``.
            Default: ``42``.

    Raises:
        ValueError: If *challenge* is not one of the four accepted strings.

    Example::

        ds = ChallengeSyntheticDataset("fast_motion", num_sequences=3)
        for seq in ds:
            print(seq.name, seq.ground_truth.shape)
    """

    def __init__(
        self,
        challenge: ChallengeMode,
        num_sequences: int = 5,
        num_frames: int = 120,
        frame_size: Tuple[int, int] = (320, 240),
        base_bbox_size: Tuple[int, int] = (40, 40),
        max_scale_factor: float = 2.5,
        occlusion_period: int = 20,
        occlusion_duration: int = 5,
        burst_period: int = 20,
        burst_duration: int = 3,
        burst_speed_factor: float = 5.0,
        illumination_period: int = 25,
        illumination_duration: int = 8,
        illumination_delta: int = 80,
        seed: int = 42,
    ) -> None:
        if challenge not in _VALID_CHALLENGES:
            raise ValueError(
                f"Unknown challenge {challenge!r}. "
                f"Choose from {_VALID_CHALLENGES}."
            )
        self.challenge = challenge
        self.num_sequences = num_sequences
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.base_bbox_size = base_bbox_size
        self.max_scale_factor = max_scale_factor
        self.occlusion_period = occlusion_period
        self.occlusion_duration = occlusion_duration
        self.burst_period = burst_period
        self.burst_duration = burst_duration
        self.burst_speed_factor = burst_speed_factor
        self.illumination_period = illumination_period
        self.illumination_duration = illumination_duration
        self.illumination_delta = illumination_delta
        self.seed = seed
        self._cache: List[Optional[_InMemorySequence]] = [None] * num_sequences

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
            f"frame_size={self.frame_size})"
        )

    # ------------------------------------------------------------------
    # Sequence routing
    # ------------------------------------------------------------------

    def _build_sequence(self, idx: int) -> _InMemorySequence:
        """Dispatch to the correct challenge-specific builder."""
        builders = {
            "occlusion":    self._build_occlusion,
            "scale_change": self._build_scale_change,
            "fast_motion":  self._build_fast_motion,
            "illumination": self._build_illumination,
        }
        return builders[self.challenge](idx)

    # ------------------------------------------------------------------
    # Shared rendering helper
    # ------------------------------------------------------------------

    def _render_frame(
        self,
        background: np.ndarray,
        cx: float,
        cy: float,
        bw: int,
        bh: int,
        colour: Tuple[int, int, int],
    ) -> np.ndarray:
        """Draw a filled rectangle on a copy of *background*."""
        W, H = self.frame_size
        frame = background.copy()
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        x1c, y1c = max(0, x1), max(0, y1)
        x2c = min(W, x1 + bw)
        y2c = min(H, y1 + bh)
        frame[y1c:y2c, x1c:x2c] = colour
        return frame

    # ------------------------------------------------------------------
    # occlusion — partial_occlusion attribute
    # ------------------------------------------------------------------

    def _build_occlusion(self, idx: int) -> _InMemorySequence:
        """Linear motion with periodic rectangular occluder events."""
        rng = np.random.default_rng(self.seed + idx)
        W, H = self.frame_size
        bw, bh = self.base_bbox_size

        cx0 = float(rng.integers(bw, W - bw))
        cy0 = float(rng.integers(bh, H - bh))
        vx = float(rng.uniform(1.0, 2.5)) * float(rng.choice([-1, 1]))
        vy = float(rng.uniform(0.5, 1.5)) * float(rng.choice([-1, 1]))

        half_bw, half_bh = bw / 2.0, bh / 2.0
        background = rng.integers(40, 100, (H, W, 3), dtype=np.uint8)
        colour = tuple(int(c) for c in rng.integers(160, 256, 3))
        occluder_colour = tuple(int(c) for c in rng.integers(100, 150, 3))

        # Occluder is slightly wider than the target to fully cover it.
        ow, oh = int(bw * 1.2), int(bh * 1.2)

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []
        cx, cy = cx0, cy0

        for i in range(self.num_frames):
            frame = self._render_frame(background, cx, cy, bw, bh, colour)  # type: ignore[arg-type]

            # Occluder passes over target for `occlusion_duration` frames.
            in_occlusion = (
                i % self.occlusion_period < self.occlusion_duration
                and i > 0  # leave first frame clean for init
            )
            if in_occlusion:
                # GT box shrinks to near-zero to trigger area-drop detection.
                occ_gt_w = max(1, int(bw * 0.05))
                occ_gt_h = max(1, int(bh * 0.05))
                x1 = int(round(cx - ow / 2))
                y1 = int(round(cy - oh / 2))
                x1c, y1c = max(0, x1), max(0, y1)
                x2c = min(W, x1 + ow)
                y2c = min(H, y1 + oh)
                frame[y1c:y2c, x1c:x2c] = occluder_colour
                gt_boxes.append((
                    float(cx - occ_gt_w / 2), float(cy - occ_gt_h / 2),
                    float(occ_gt_w), float(occ_gt_h),
                ))
            else:
                gt_boxes.append((
                    float(cx - half_bw), float(cy - half_bh),
                    float(bw), float(bh),
                ))

            frames.append(frame)

            # Update position (wall-bounce).
            cx += vx
            cy += vy
            if cx < half_bw or cx > W - half_bw:
                vx = -vx
                cx = float(np.clip(cx, half_bw, W - half_bw))
            if cy < half_bh or cy > H - half_bh:
                vy = -vy
                cy = float(np.clip(cy, half_bh, H - half_bh))

        return _InMemorySequence(
            name=f"challenge_occlusion_{idx:03d}",
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # scale_change — scale_variation attribute
    # ------------------------------------------------------------------

    def _build_scale_change(self, idx: int) -> _InMemorySequence:
        """Linear motion with smooth sinusoidal scale growth/shrink cycle.

        The target area ratio over the sequence is ``max_scale_factor ** 2``
        (>= 6.25 at default 2.5x), reliably triggering ``scale_variation``
        (area ratio > 4).
        """
        rng = np.random.default_rng(self.seed + idx)
        W, H = self.frame_size
        bw0, bh0 = self.base_bbox_size

        cx0 = float(rng.integers(bw0 * 2, W - bw0 * 2))
        cy0 = float(rng.integers(bh0 * 2, H - bh0 * 2))
        vx = float(rng.uniform(0.5, 1.5)) * float(rng.choice([-1, 1]))
        vy = float(rng.uniform(0.3, 1.0)) * float(rng.choice([-1, 1]))

        background = rng.integers(40, 100, (H, W, 3), dtype=np.uint8)
        colour = tuple(int(c) for c in rng.integers(160, 256, 3))

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []
        cx, cy = cx0, cy0

        for i in range(self.num_frames):
            # Smooth sinusoidal scale: 1 → max_scale_factor → 1 over sequence.
            t = i / max(self.num_frames - 1, 1)
            scale = 1.0 + (self.max_scale_factor - 1.0) * math.sin(math.pi * t)
            bw = max(4, int(round(bw0 * scale)))
            bh = max(4, int(round(bh0 * scale)))

            half_bw_max = bw0 * self.max_scale_factor / 2.0
            half_bh_max = bh0 * self.max_scale_factor / 2.0

            frame = self._render_frame(background, cx, cy, bw, bh, colour)  # type: ignore[arg-type]
            frames.append(frame)
            gt_boxes.append((
                float(cx - bw / 2), float(cy - bh / 2),
                float(bw), float(bh),
            ))

            cx += vx
            cy += vy
            if cx < half_bw_max or cx > W - half_bw_max:
                vx = -vx
                cx = float(np.clip(cx, half_bw_max, W - half_bw_max))
            if cy < half_bh_max or cy > H - half_bh_max:
                vy = -vy
                cy = float(np.clip(cy, half_bh_max, H - half_bh_max))

        return _InMemorySequence(
            name=f"challenge_scale_{idx:03d}",
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # fast_motion — fast_motion attribute
    # ------------------------------------------------------------------

    def _build_fast_motion(self, idx: int) -> _InMemorySequence:
        """Slow linear motion with periodic high-speed burst phases.

        During a burst, the target moves at ``burst_speed_factor`` times its
        normal speed, creating per-frame displacements well above the 20 %
        mean-diagonal threshold required to trigger ``fast_motion``.
        """
        rng = np.random.default_rng(self.seed + idx)
        W, H = self.frame_size
        bw, bh = self.base_bbox_size
        half_bw, half_bh = bw / 2.0, bh / 2.0

        cx0 = float(rng.integers(bw, W - bw))
        cy0 = float(rng.integers(bh, H - bh))
        # Base velocity (slow)
        vx = float(rng.uniform(0.5, 1.5)) * float(rng.choice([-1, 1]))
        vy = float(rng.uniform(0.3, 1.0)) * float(rng.choice([-1, 1]))

        background = rng.integers(40, 100, (H, W, 3), dtype=np.uint8)
        colour = tuple(int(c) for c in rng.integers(160, 256, 3))

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []
        cx, cy = cx0, cy0

        for i in range(self.num_frames):
            in_burst = (
                i > 0
                and (i % self.burst_period) < self.burst_duration
            )
            speed = self.burst_speed_factor if in_burst else 1.0

            frame = self._render_frame(background, cx, cy, bw, bh, colour)  # type: ignore[arg-type]
            frames.append(frame)
            gt_boxes.append((
                float(cx - half_bw), float(cy - half_bh),
                float(bw), float(bh),
            ))

            cx += vx * speed
            cy += vy * speed
            if cx < half_bw or cx > W - half_bw:
                vx = -vx
                cx = float(np.clip(cx, half_bw, W - half_bw))
            if cy < half_bh or cy > H - half_bh:
                vy = -vy
                cy = float(np.clip(cy, half_bh, H - half_bh))

        return _InMemorySequence(
            name=f"challenge_fastmotion_{idx:03d}",
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # illumination — illumination change (appearance challenge)
    # ------------------------------------------------------------------

    def _build_illumination(self, idx: int) -> _InMemorySequence:
        """Random-walk motion with periodic frame-wide brightness pulses.

        Each illumination event adds a uniform brightness offset to all
        pixels, simulating sudden lighting changes that confuse appearance-
        based correlation filters.  The target colour is also shifted by
        the same delta so the visual contrast ratio is preserved.
        """
        rng = np.random.default_rng(self.seed + idx)
        W, H = self.frame_size
        bw, bh = self.base_bbox_size
        half_bw, half_bh = bw / 2.0, bh / 2.0

        cx0 = float(rng.integers(bw, W - bw))
        cy0 = float(rng.integers(bh, H - bh))
        step = float(rng.uniform(2.0, 5.0))

        background = rng.integers(40, 100, (H, W, 3), dtype=np.uint8)
        base_colour = rng.integers(160, 220, 3)  # leave headroom for brightness shift

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []
        cx, cy = cx0, cy0

        for i in range(self.num_frames):
            # Determine brightness delta for this frame.
            in_pulse = (
                i > 0
                and (i % self.illumination_period) < self.illumination_duration
            )
            if in_pulse:
                phase = (i % self.illumination_period) / self.illumination_duration
                brightness_shift = int(
                    self.illumination_delta * math.sin(math.pi * phase)
                )
            else:
                brightness_shift = 0

            shifted_bg = np.clip(
                background.astype(np.int32) + brightness_shift, 0, 255
            ).astype(np.uint8)
            shifted_colour = tuple(
                int(np.clip(int(c) + brightness_shift, 0, 255))
                for c in base_colour
            )

            frame = self._render_frame(
                shifted_bg, cx, cy, bw, bh, shifted_colour  # type: ignore[arg-type]
            )
            frames.append(frame)
            gt_boxes.append((
                float(cx - half_bw), float(cy - half_bh),
                float(bw), float(bh),
            ))

            cx = float(np.clip(
                cx + rng.uniform(-step, step), half_bw, W - half_bw
            ))
            cy = float(np.clip(
                cy + rng.uniform(-step, step), half_bh, H - half_bh
            ))

        return _InMemorySequence(
            name=f"challenge_illumination_{idx:03d}",
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )
