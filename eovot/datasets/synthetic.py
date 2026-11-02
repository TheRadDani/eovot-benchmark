"""Synthetic sequence generator for EOVOT — offline benchmarking and CI.

Generates sequences of BGR frames containing a coloured filled rectangle
(the target) moving against a textured noise background.  No external data
download is required, making this module ideal for:

- **CI/CD integration tests** — the full benchmark pipeline can be exercised
  without downloading GOT-10k, LaSOT, or OTB.
- **Algorithm development** — quickly validate a new tracker before running
  expensive real-data experiments.
- **Demos** — demonstrate the framework to new contributors without a dataset.

Motion patterns
---------------
- ``"linear"``       — constant-velocity drift with wall-bounce.
- ``"circular"``     — target orbits the frame centre (3 full rotations).
- ``"random"``       — Gaussian random walk clamped to frame boundaries.
- ``"scale_change"`` — linear translation with sinusoidally-varying target size;
                       scale factor oscillates between 0.25× and 1.75× over the
                       sequence.  Stresses trackers that assume constant target area.
- ``"occlusion"``    — random-walk translation with periodic gray occlusion
                       patches that cover the target for ``occ_length`` frames
                       every ``occ_period`` frames.  Ground-truth boxes are
                       always accurate; only the rendered frames are affected.
                       Stresses tracker re-detection and recovery.

Usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    dataset = SyntheticDataset(5, 100, motion="scale_change")
    engine  = BenchmarkEngine(verbose=False)
    result  = engine.run(MOSSETracker(), dataset, dataset_name="Synth-Scale")
    print(result)
"""

from __future__ import annotations

import math
from typing import Iterator, List, Literal, Optional, Tuple

import numpy as np

from .base import BaseDataset, Sequence

MotionPattern = Literal["linear", "circular", "random", "scale_change", "occlusion"]

# Internal frame descriptor: (centre_x, centre_y, scale_factor, is_occluded)
_FrameDesc = Tuple[float, float, float, bool]


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
        bbox_size: ``(width, height)`` of the target rectangle at scale 1.0.
            Default: ``(40, 40)``.
        motion: Motion pattern — ``"linear"``, ``"circular"``, ``"random"``,
            ``"scale_change"``, or ``"occlusion"``.  Default: ``"linear"``.
        seed: Base RNG seed; sequence ``i`` uses ``seed + i`` so every
            sequence is distinct yet reproducible.  Default: ``42``.

    Raises:
        ValueError: If ``motion`` is not one of the accepted pattern names.

    Example::

        ds = SyntheticDataset(num_sequences=3, num_frames=50, motion="circular")
        for seq in ds:
            print(seq.name, seq.ground_truth.shape)
        # synth_circular_000 (50, 4)
        # synth_circular_001 (50, 4)
        # synth_circular_002 (50, 4)

        # Scale-change and occlusion challenges:
        ds_scale = SyntheticDataset(5, 100, motion="scale_change")
        ds_occ   = SyntheticDataset(5, 100, motion="occlusion")
    """

    _VALID_MOTIONS = ("linear", "circular", "random", "scale_change", "occlusion")

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
    # Sequence generation
    # ------------------------------------------------------------------

    def _build_sequence(self, idx: int) -> _InMemorySequence:
        """Render one sequence with a moving coloured target rectangle."""
        rng = np.random.default_rng(self.seed + idx)
        W, H = self.frame_size
        bw, bh = self.bbox_size

        # Random initial centre, fully inside the frame.
        cx0 = float(rng.integers(bw, W - bw))
        cy0 = float(rng.integers(bh, H - bh))
        descs: List[_FrameDesc] = self._generate_positions(cx0, cy0, rng)

        # Static noise background — varied per sequence but fixed across frames.
        background = rng.integers(40, 100, (H, W, 3), dtype=np.uint8)
        # Bright target colour distinct from the dim background.
        colour = tuple(int(c) for c in rng.integers(160, 256, 3))
        # Neutral gray occlusion patch (drawn over the target region).
        occ_colour = (110, 110, 110)

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []

        for cx, cy, scale, occluded in descs:
            frame = background.copy()
            bw_s = max(4, int(round(bw * scale)))
            bh_s = max(4, int(round(bh * scale)))
            x1 = int(round(cx - bw_s / 2))
            y1 = int(round(cy - bh_s / 2))

            # Always draw the target first at its true location.
            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(W, x1 + bw_s); y2c = min(H, y1 + bh_s)
            frame[y1c:y2c, x1c:x2c] = colour

            if occluded:
                # Overlay a gray patch slightly larger than the target box.
                pad = max(4, bw_s // 5)
                ox1 = max(0, x1 - pad); oy1 = max(0, y1 - pad)
                ox2 = min(W, x1 + bw_s + pad); oy2 = min(H, y1 + bh_s + pad)
                frame[oy1:oy2, ox1:ox2] = occ_colour

            frames.append(frame)
            # Ground-truth is always the true (unoccluded) box.
            gt_boxes.append((float(x1), float(y1), float(bw_s), float(bh_s)))

        return _InMemorySequence(
            name=f"synth_{self.motion}_{idx:03d}",
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    def _generate_positions(
        self,
        cx0: float,
        cy0: float,
        rng: np.random.Generator,
    ) -> List[_FrameDesc]:
        """Generate per-frame descriptors: (cx, cy, scale, occluded).

        Returns:
            List of ``(centre_x, centre_y, scale_factor, is_occluded)`` tuples,
            one per frame.  ``scale_factor`` multiplies the base ``bbox_size``;
            ``is_occluded`` signals that an occlusion patch covers the target in
            the rendered frame (ground-truth is still accurate).
        """
        W, H = self.frame_size
        bw, bh = self.bbox_size
        half_bw, half_bh = bw / 2.0, bh / 2.0
        descs: List[_FrameDesc] = []

        if self.motion == "linear":
            vx = float(rng.uniform(1.0, 3.0)) * float(rng.choice([-1, 1]))
            vy = float(rng.uniform(0.5, 2.0)) * float(rng.choice([-1, 1]))
            cx, cy = cx0, cy0
            for _ in range(self.num_frames):
                descs.append((cx, cy, 1.0, False))
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
                descs.append((
                    cx_c + radius * math.cos(angle),
                    cy_c + radius * math.sin(angle),
                    1.0,
                    False,
                ))

        elif self.motion == "random":
            step = float(rng.uniform(2.0, 5.0))
            cx, cy = cx0, cy0
            for _ in range(self.num_frames):
                descs.append((cx, cy, 1.0, False))
                cx = float(np.clip(cx + rng.uniform(-step, step), half_bw, W - half_bw))
                cy = float(np.clip(cy + rng.uniform(-step, step), half_bh, H - half_bh))

        elif self.motion == "scale_change":
            # Linear translation with wall-bounce; target scale oscillates
            # sinusoidally between 0.25x and 1.75x the base bbox_size over 2
            # complete cycles.  This tests trackers that assume fixed target area.
            vx = float(rng.uniform(1.0, 3.0)) * float(rng.choice([-1, 1]))
            vy = float(rng.uniform(0.5, 2.0)) * float(rng.choice([-1, 1]))
            cx, cy = cx0, cy0
            for i in range(self.num_frames):
                # scale in [0.25, 1.75] — two full sinusoidal cycles
                scale = 1.0 + 0.75 * math.sin(
                    2 * math.pi * 2 * i / max(self.num_frames - 1, 1)
                )
                descs.append((cx, cy, scale, False))
                cx += vx
                cy += vy
                if cx < half_bw or cx > W - half_bw:
                    vx = -vx
                    cx = float(np.clip(cx, half_bw, W - half_bw))
                if cy < half_bh or cy > H - half_bh:
                    vy = -vy
                    cy = float(np.clip(cy, half_bh, H - half_bh))

        elif self.motion == "occlusion":
            # Random-walk translation; periodically covers the target with a
            # gray patch for occ_length frames every occ_period frames.
            # Ground-truth remains accurate even during occlusion.
            step = float(rng.uniform(2.0, 5.0))
            # Aim for ~4 occlusion bursts; each burst is ~1/5 of its period.
            occ_period = max(8, self.num_frames // 4)
            occ_length = max(2, occ_period // 5)
            cx, cy = cx0, cy0
            for i in range(self.num_frames):
                # Never occlude the first frame (needed for tracker init).
                frame_in_period = i % occ_period
                occluded = i > 0 and frame_in_period < occ_length
                descs.append((cx, cy, 1.0, occluded))
                cx = float(np.clip(
                    cx + rng.uniform(-step, step), half_bw, W - half_bw
                ))
                cy = float(np.clip(
                    cy + rng.uniform(-step, step), half_bh, H - half_bh
                ))

        return descs
