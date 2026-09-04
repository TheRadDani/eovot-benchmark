"""Synthetic dataset with configurable occlusion events for EOVOT.

Extends the concept of :class:`~eovot.datasets.synthetic.SyntheticDataset`
to inject temporally-structured occluder rectangles that periodically mask
the target.  This enables **systematic, reproducible evaluation** of tracker
behavior under controlled occlusion — a failure mode ubiquitous in real
datasets (pedestrians behind cars, objects behind pillars) but difficult to
isolate because occlusion duration, frequency, and severity cannot be
controlled in natural footage.

Occlusion model
~~~~~~~~~~~~~~~
An occluder is a filled rectangle drawn **over** the target at the predicted
center for the duration of each occlusion window.  The occluder is sized
larger than the target (configurable) to fully hide it.  The target continues
moving normally behind the occluder; ground-truth boxes always reflect the
true position regardless of visibility.

Multiple non-overlapping occlusion windows are spaced ``occlusion_interval``
frames apart.  Each window lasts ``occlusion_duration`` frames.  The dataset
exposes a per-frame ``occlusion_mask`` boolean array on every sequence,
which is consumed by :class:`~eovot.metrics.occlusion.OcclusionRobustnessAnalyzer`
to compute pre/during/post-occlusion statistics.

Evaluation protocol
~~~~~~~~~~~~~~~~~~~
1. Run any ``BaseTracker`` on ``OcclusionSyntheticDataset`` using ``BenchmarkEngine``.
2. Extract ``seq_result.ious`` and ``seq.occlusion_mask`` for each sequence.
3. Pass both to ``OcclusionRobustnessAnalyzer.analyze()`` to obtain structured
   pre/during/post metrics and recovery AUC.

Usage::

    from eovot.datasets.occlusion import OcclusionSyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.metrics.occlusion import OcclusionRobustnessAnalyzer

    ds = OcclusionSyntheticDataset(
        num_sequences=5,
        num_frames=100,
        occlusion_interval=25,
        occlusion_duration=8,
    )

    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), ds, dataset_name="OcclusionSynth")

    analyzer = OcclusionRobustnessAnalyzer()
    for seq_result, seq in zip(result.sequence_results, ds):
        r = analyzer.analyze(seq_result.ious, seq.occlusion_mask,
                             tracker_name="MOSSE", sequence_name=seq.name)
        print(r)
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, Sequence

BBox = Tuple[float, float, float, float]


class OcclusionSequence(Sequence):
    """A :class:`~eovot.datasets.base.Sequence` with an attached occlusion mask.

    Attributes:
        occlusion_mask: Boolean array of shape ``(N,)`` — ``True`` on frames
            where the target is partially or fully hidden by an occluder.
    """

    def __init__(
        self,
        name: str,
        frames: List[np.ndarray],
        ground_truth: np.ndarray,
        occlusion_mask: np.ndarray,
    ) -> None:
        self._frames = frames
        self.occlusion_mask = occlusion_mask
        super().__init__(
            name=name,
            frame_paths=["<memory>"] * len(frames),
            ground_truth=ground_truth,
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self._frames

    def __repr__(self) -> str:
        n_occ = int(self.occlusion_mask.sum())
        return (
            f"OcclusionSequence(name={self.name!r}, "
            f"frames={len(self)}, occluded_frames={n_occ})"
        )


class OcclusionSyntheticDataset(BaseDataset):
    """Synthetic dataset with periodic occlusion events.

    Each sequence contains a moving target rectangle against a textured noise
    background.  Occluder rectangles are drawn over the target at regular
    intervals, completely hiding its appearance for a configurable number of
    frames while the target continues moving underneath.

    Args:
        num_sequences:      Number of sequences.  Default: ``5``.
        num_frames:         Frames per sequence.  Default: ``100``.
        frame_size:         ``(width, height)`` in pixels.  Default: ``(320, 240)``.
        target_size:        ``(width, height)`` of the tracking target.
                            Default: ``(40, 40)``.
        occluder_size:      ``(width, height)`` of each occluder.  Should be
                            ≥ ``target_size`` to fully hide the target.
                            Default: ``(60, 60)``.
        occlusion_interval: Frames between successive occlusion onsets.
                            The first occlusion starts at this frame index.
                            Must be > ``occlusion_duration``.  Default: ``25``.
        occlusion_duration: Frames each occlusion window lasts.  Default: ``8``.
        motion:             Target motion pattern — ``"linear"`` or ``"random"``.
                            Default: ``"linear"``.
        seed:               Base RNG seed; sequence ``i`` uses ``seed + i``.
                            Default: ``42``.

    Raises:
        ValueError: If ``occlusion_interval <= occlusion_duration`` or if
            ``motion`` is not a recognised pattern name.

    Example::

        ds = OcclusionSyntheticDataset(
            num_sequences=3,
            num_frames=80,
            occlusion_interval=20,
            occlusion_duration=6,
        )
        seq = ds[0]
        print(seq.occlusion_mask.sum(), "frames occluded")  # ~3 events × 6 frames
    """

    _VALID_MOTIONS = ("linear", "random")

    def __init__(
        self,
        num_sequences: int = 5,
        num_frames: int = 100,
        frame_size: Tuple[int, int] = (320, 240),
        target_size: Tuple[int, int] = (40, 40),
        occluder_size: Tuple[int, int] = (60, 60),
        occlusion_interval: int = 25,
        occlusion_duration: int = 8,
        motion: str = "linear",
        seed: int = 42,
    ) -> None:
        if motion not in self._VALID_MOTIONS:
            raise ValueError(
                f"Unknown motion pattern: {motion!r}. "
                f"Choose from {self._VALID_MOTIONS}."
            )
        if occlusion_interval <= occlusion_duration:
            raise ValueError(
                f"occlusion_interval ({occlusion_interval}) must be greater than "
                f"occlusion_duration ({occlusion_duration})."
            )
        if occlusion_duration < 1:
            raise ValueError("occlusion_duration must be >= 1.")

        self.num_sequences = num_sequences
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.target_size = target_size
        self.occluder_size = occluder_size
        self.occlusion_interval = occlusion_interval
        self.occlusion_duration = occlusion_duration
        self.motion = motion
        self.seed = seed
        self._cache: List[Optional[OcclusionSequence]] = [None] * num_sequences

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> OcclusionSequence:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(
                f"Sequence index {idx} out of range [0, {self.num_sequences})"
            )
        if self._cache[idx] is None:
            self._cache[idx] = self._build_sequence(idx)
        return self._cache[idx]  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"OcclusionSyntheticDataset("
            f"sequences={self.num_sequences}, "
            f"frames={self.num_frames}, "
            f"motion={self.motion!r}, "
            f"interval={self.occlusion_interval}, "
            f"duration={self.occlusion_duration})"
        )

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------

    def _build_sequence(self, idx: int) -> OcclusionSequence:
        """Render one sequence with injected occlusion windows."""
        rng = np.random.default_rng(self.seed + idx)
        W, H = self.frame_size
        bw, bh = self.target_size
        ow, oh = self.occluder_size

        # Initial target centre (fully inside frame bounds)
        cx0 = float(rng.integers(bw, W - bw))
        cy0 = float(rng.integers(bh, H - bh))
        target_positions = self._generate_positions(cx0, cy0, rng)

        # Textured background (fixed per sequence, varied per sequence via seed)
        background = rng.integers(30, 90, (H, W, 3), dtype=np.uint8)

        # Target and occluder colours: target is brighter than background;
        # occluder is a mid-range colour distinct from both
        target_color = tuple(int(c) for c in rng.integers(160, 220, 3))
        occluder_color = tuple(int(c) for c in rng.integers(90, 140, 3))

        # Build occlusion mask: True for frames in an occlusion window
        occlusion_mask = np.zeros(self.num_frames, dtype=bool)
        onset = self.occlusion_interval
        while onset + self.occlusion_duration <= self.num_frames:
            occlusion_mask[onset : onset + self.occlusion_duration] = True
            onset += self.occlusion_interval

        frames: List[np.ndarray] = []
        gt_boxes: List[BBox] = []

        for f_idx, (cx, cy) in enumerate(target_positions):
            frame = background.copy()

            # Draw target (always — ground truth is visibility-independent)
            tx1 = int(round(cx - bw / 2))
            ty1 = int(round(cy - bh / 2))
            tx1c, ty1c = max(0, tx1), max(0, ty1)
            tx2c, ty2c = min(W, tx1 + bw), min(H, ty1 + bh)
            frame[ty1c:ty2c, tx1c:tx2c] = target_color

            # Draw occluder centred on target during occlusion windows
            if occlusion_mask[f_idx]:
                ox1 = int(round(cx - ow / 2))
                oy1 = int(round(cy - oh / 2))
                ox1c, oy1c = max(0, ox1), max(0, oy1)
                ox2c, oy2c = min(W, ox1 + ow), min(H, oy1 + oh)
                frame[oy1c:oy2c, ox1c:ox2c] = occluder_color

            frames.append(frame)
            gt_boxes.append((float(tx1), float(ty1), float(bw), float(bh)))

        return OcclusionSequence(
            name=f"occ_{self.motion}_{idx:03d}",
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
            occlusion_mask=occlusion_mask,
        )

    def _generate_positions(
        self,
        cx0: float,
        cy0: float,
        rng: np.random.Generator,
    ) -> List[Tuple[float, float]]:
        """Generate target centre positions for all frames."""
        W, H = self.frame_size
        bw, bh = self.target_size
        half_bw, half_bh = bw / 2.0, bh / 2.0
        positions: List[Tuple[float, float]] = []

        if self.motion == "linear":
            vx = float(rng.uniform(1.0, 3.0)) * float(rng.choice([-1, 1]))
            vy = float(rng.uniform(0.5, 2.0)) * float(rng.choice([-1, 1]))
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
        else:  # random
            step = float(rng.uniform(2.0, 5.0))
            cx, cy = cx0, cy0
            for _ in range(self.num_frames):
                positions.append((cx, cy))
                cx = float(np.clip(cx + rng.uniform(-step, step), half_bw, W - half_bw))
                cy = float(np.clip(cy + rng.uniform(-step, step), half_bh, H - half_bh))

        return positions
