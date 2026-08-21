"""Synthetic sequence generator for EOVOT — offline benchmarking and CI.

Generates sequences of BGR frames containing a coloured filled rectangle
(the target) moving against a textured noise background.  No external data
download is required, making this module ideal for:

- **CI/CD integration tests** — the full benchmark pipeline can be exercised
  without downloading GOT-10k, LaSOT, or OTB.
- **Algorithm development** — quickly validate a new tracker before running
  expensive real-data experiments.
- **Attribute-based evaluation** — the ``scale`` and ``occlusion`` patterns
  generate sequences whose challenge attributes (scale variation and partial
  occlusion) can be detected by :class:`~eovot.metrics.attributes.AttributeDetector`
  without requiring external annotation.

Motion patterns
---------------
- ``"linear"``   — constant-velocity drift with wall-bounce.
- ``"circular"`` — target orbits the frame centre (3 full rotations).
- ``"random"``   — Gaussian random walk clamped to frame boundaries.
- ``"scale"``    — linear motion + target size oscillates 0.5× to 2.0×,
  simulating a target moving toward and away from the camera.
- ``"occlusion"`` — linear motion; target disappears for the middle 30 % of
  frames, simulating full occlusion.  GT bbox during occlusion holds the
  extrapolated (invisible) position.

Usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    dataset = SyntheticDataset(num_sequences=5, num_frames=100, motion="linear")
    engine  = BenchmarkEngine(verbose=False)
    result  = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic-Linear")
    print(result)
"""

from __future__ import annotations

import math
from typing import Iterator, List, Literal, Optional, Tuple

import numpy as np

from .base import BaseDataset, Sequence

MotionPattern = Literal["linear", "circular", "random", "scale", "occlusion"]


class _InMemorySequence(Sequence):
    """Sequence whose frames are held in memory rather than read from disk.

    Overrides the file-path-based ``__iter__`` of :class:`~.base.Sequence`
    to iterate directly over pre-rendered numpy arrays, with no I/O overhead.
    """

    def __init__(
        self,
        name: str,
        frames: List[np.ndarray],
        ground_truth: np.ndarray,
    ) -> None:
        self._frames = frames
        # Pass synthetic paths; __iter__ is overridden so they are never opened.
        super().__init__(
            name=name,
            frame_paths=["<memory>"] * len(frames),
            ground_truth=ground_truth,
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self._frames


class SyntheticDataset(BaseDataset):
    """Dataset of procedurally-generated single-object tracking sequences.

    Each sequence renders a coloured filled rectangle moving against a static
    noise background.  Sequences are generated lazily on first access and
    cached for the lifetime of the dataset object.

    Args:
        num_sequences: Number of sequences to generate.  Default: ``10``.
        num_frames: Frames per sequence.  Default: ``100``.
        frame_size: ``(width, height)`` of each frame in pixels.
            Default: ``(320, 240)``.
        bbox_size: ``(width, height)`` of the target rectangle in pixels
            (the base size for the ``scale`` pattern).
            Default: ``(40, 40)``.
        motion: Motion pattern — ``"linear"``, ``"circular"``, ``"random"``,
            ``"scale"``, or ``"occlusion"``.
            Default: ``"linear"``.
        seed: Base RNG seed; sequence ``i`` uses ``seed + i`` so every
            sequence is distinct yet reproducible.  Default: ``42``.

    Raises:
        ValueError: If ``motion`` is not one of the accepted pattern names.

    Example::

        ds = SyntheticDataset(num_sequences=3, num_frames=50, motion="scale")
        for seq in ds:
            print(seq.name, seq.ground_truth[:, 2].min(), seq.ground_truth[:, 2].max())
        # synth_scale_000  20.0  80.0   (width varies with scale)
    """

    _VALID_MOTIONS = ("linear", "circular", "random", "scale", "occlusion")

    def __init__(
        self,
        num_sequences: int = 10,
        num_frames: int = 100,
        frame_size: Tuple[int, int] = (320, 240),
        bbox_size: Tuple[int, int] = (40, 40),
        motion: MotionPattern = "linear",
        seed: int = 42,
    ) -> None:
        if motion not in self._VALID_MOTIONS:
            raise ValueError(
                f"Unknown motion pattern: {motion!r}. "
                f"Choose from {self._VALID_MOTIONS}."
            )
        self.num_sequences = num_sequences
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.bbox_size = bbox_size
        self.motion = motion
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
            f"SyntheticDataset(sequences={self.num_sequences}, "
            f"frames={self.num_frames}, motion={self.motion!r}, "
            f"frame_size={self.frame_size})"
        )

    # ------------------------------------------------------------------
    # Sequence generation — dispatcher
    # ------------------------------------------------------------------

    def _build_sequence(self, idx: int) -> _InMemorySequence:
        """Render one sequence; dispatches to the pattern-specific builder."""
        if self.motion == "scale":
            return self._build_scale_sequence(idx)
        if self.motion == "occlusion":
            return self._build_occlusion_sequence(idx)
        # linear / circular / random — original implementation
        rng = np.random.default_rng(self.seed + idx)
        W, H = self.frame_size
        bw, bh = self.bbox_size

        cx0 = float(rng.integers(bw, W - bw))
        cy0 = float(rng.integers(bh, H - bh))
        positions = self._generate_positions(cx0, cy0, rng)

        background = rng.integers(40, 100, (H, W, 3), dtype=np.uint8)
        colour = tuple(int(c) for c in rng.integers(160, 256, 3))

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []

        for cx, cy in positions:
            frame = background.copy()
            x1 = int(round(cx - bw / 2))
            y1 = int(round(cy - bh / 2))
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x1 + bw), min(H, y1 + bh)
            frame[y1c:y2c, x1c:x2c] = colour
            frames.append(frame)
            gt_boxes.append((float(x1), float(y1), float(bw), float(bh)))

        return _InMemorySequence(
            name=f"synth_{self.motion}_{idx:03d}",
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Scale challenge pattern
    # ------------------------------------------------------------------

    def _build_scale_sequence(self, idx: int) -> _InMemorySequence:
        """Linear motion with oscillating target size (0.5× → 2.0× → 0.5×).

        The scale factor follows a half-sine over the full sequence, producing
        one full cycle of size expansion and contraction.  This tests tracker
        robustness to scale variation — a key OTB/VOT challenge attribute.
        The GT bounding box at each frame reflects the actual target size.
        """
        rng = np.random.default_rng(self.seed + idx)
        W, H = self.frame_size
        bw, bh = self.bbox_size
        half_bw, half_bh = bw / 2.0, bh / 2.0

        cx0 = float(rng.integers(bw, W - bw))
        cy0 = float(rng.integers(bh, H - bh))
        vx = float(rng.uniform(1.0, 2.0)) * float(rng.choice([-1, 1]))
        vy = float(rng.uniform(0.5, 1.5)) * float(rng.choice([-1, 1]))

        background = rng.integers(40, 100, (H, W, 3), dtype=np.uint8)
        colour = tuple(int(c) for c in rng.integers(160, 256, 3))

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []
        cx, cy = cx0, cy0

        for i in range(self.num_frames):
            # Scale oscillates 0.5× → 2.0× over the sequence (half-sine)
            t = i / max(self.num_frames - 1, 1)
            scale = 0.5 + 1.5 * math.sin(math.pi * t)
            curr_bw = max(2, int(round(bw * scale)))
            curr_bh = max(2, int(round(bh * scale)))

            frame = background.copy()
            x1 = int(round(cx - curr_bw / 2))
            y1 = int(round(cy - curr_bh / 2))
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x1 + curr_bw), min(H, y1 + curr_bh)
            frame[y1c:y2c, x1c:x2c] = colour
            frames.append(frame)
            gt_boxes.append((float(x1), float(y1), float(curr_bw), float(curr_bh)))

            cx += vx
            cy += vy
            if cx < half_bw or cx > W - half_bw:
                vx = -vx
                cx = float(np.clip(cx, half_bw, W - half_bw))
            if cy < half_bh or cy > H - half_bh:
                vy = -vy
                cy = float(np.clip(cy, half_bh, H - half_bh))

        return _InMemorySequence(
            name=f"synth_scale_{idx:03d}",
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Occlusion challenge pattern
    # ------------------------------------------------------------------

    def _build_occlusion_sequence(self, idx: int) -> _InMemorySequence:
        """Linear motion with full occlusion in the middle 30 % of frames.

        During the occlusion window the target rectangle is NOT drawn — the
        frame shows background only.  The GT bbox continues to hold the
        extrapolated (invisible) target position, matching how real datasets
        (OTB, VOT) annotate fully occluded targets.

        This tests tracker ability to maintain state through complete
        occlusion and recover correctly when the target reappears.
        """
        rng = np.random.default_rng(self.seed + idx)
        W, H = self.frame_size
        bw, bh = self.bbox_size
        half_bw, half_bh = bw / 2.0, bh / 2.0

        cx0 = float(rng.integers(bw, W - bw))
        cy0 = float(rng.integers(bh, H - bh))
        vx = float(rng.uniform(1.0, 3.0)) * float(rng.choice([-1, 1]))
        vy = float(rng.uniform(0.5, 2.0)) * float(rng.choice([-1, 1]))

        # Occlusion covers frames in [35 %, 65 %) of the sequence
        occ_start = int(self.num_frames * 0.35)
        occ_end = int(self.num_frames * 0.65)

        background = rng.integers(40, 100, (H, W, 3), dtype=np.uint8)
        colour = tuple(int(c) for c in rng.integers(160, 256, 3))

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []
        cx, cy = cx0, cy0

        for i in range(self.num_frames):
            occluded = occ_start <= i < occ_end
            frame = background.copy()
            if not occluded:
                x1c = max(0, int(round(cx - bw / 2)))
                y1c = max(0, int(round(cy - bh / 2)))
                x2c = min(W, x1c + bw)
                y2c = min(H, y1c + bh)
                frame[y1c:y2c, x1c:x2c] = colour
            frames.append(frame)
            # GT holds extrapolated position even during occlusion
            gt_boxes.append((
                float(round(cx - bw / 2)),
                float(round(cy - bh / 2)),
                float(bw),
                float(bh),
            ))

            cx += vx
            cy += vy
            if cx < half_bw or cx > W - half_bw:
                vx = -vx
                cx = float(np.clip(cx, half_bw, W - half_bw))
            if cy < half_bh or cy > H - half_bh:
                vy = -vy
                cy = float(np.clip(cy, half_bh, H - half_bh))

        return _InMemorySequence(
            name=f"synth_occlusion_{idx:03d}",
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Position generation (linear / circular / random)
    # ------------------------------------------------------------------

    def _generate_positions(
        self,
        cx0: float,
        cy0: float,
        rng: np.random.Generator,
    ) -> List[Tuple[float, float]]:
        """Generate target centre positions for all frames."""
        W, H = self.frame_size
        bw, bh = self.bbox_size
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

        elif self.motion == "circular":
            # 3 full rotations over the sequence.
            radius = min(W, H) * 0.25
            cx_c, cy_c = W / 2.0, H / 2.0
            for i in range(self.num_frames):
                angle = 2 * math.pi * 3 * i / max(self.num_frames - 1, 1)
                positions.append((cx_c + radius * math.cos(angle),
                                  cy_c + radius * math.sin(angle)))

        else:  # random
            step = float(rng.uniform(2.0, 5.0))
            cx, cy = cx0, cy0
            for _ in range(self.num_frames):
                positions.append((cx, cy))
                cx = float(np.clip(cx + rng.uniform(-step, step), half_bw, W - half_bw))
                cy = float(np.clip(cy + rng.uniform(-step, step), half_bh, H - half_bh))

        return positions
