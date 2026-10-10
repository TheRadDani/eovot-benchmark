"""Synthetic sequence generator for EOVOT — offline benchmarking and CI.

Generates sequences of BGR frames containing a coloured filled rectangle
(the target) moving against a textured noise background.  No external data
download is required, making this module ideal for:

- **CI/CD integration tests** — the full benchmark pipeline can be exercised
  without downloading GOT-10k, LaSOT, or OTB.
- **Algorithm development** — quickly validate a new tracker before running
  expensive real-data experiments.
- **Demos** — demonstrate the framework to new contributors without a dataset.
- **Controlled challenge experiments** — generate sequences with specific VOT
  challenge attributes (occlusion, scale variation) to study tracker behaviour
  under precisely defined difficulty conditions.

Motion patterns
---------------
- ``"linear"``       — constant-velocity drift with wall-bounce.
- ``"circular"``     — target orbits the frame centre (3 full rotations).
- ``"random"``       — Gaussian random walk clamped to frame boundaries.
- ``"occlusion"``    — linear target motion; a rectangular occluder sweeps
  across the frame periodically, covering the target for several frames at a
  time.  Ground-truth boxes record the true target position even when occluded,
  matching real benchmark annotation conventions.  Triggers the
  ``partial_occlusion`` attribute in
  :class:`~eovot.metrics.attributes.AttributeDetector`.
- ``"scale_change"`` — linear target motion combined with sinusoidal
  scale variation: the target bbox grows to ``scale_max`` × and shrinks back
  to ``scale_min`` × its nominal size over ``scale_period`` frames.  Triggers
  the ``scale_variation`` attribute.

Usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    dataset = SyntheticDataset(num_sequences=5, num_frames=100, motion="linear")
    engine  = BenchmarkEngine(verbose=False)
    result  = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic-Linear")
    print(result)

    # Controlled occlusion experiment
    occ_ds = SyntheticDataset(num_sequences=5, num_frames=120, motion="occlusion")

    # Controlled scale-change experiment
    sc_ds  = SyntheticDataset(num_sequences=5, num_frames=120, motion="scale_change")
"""

from __future__ import annotations

import math
from typing import Iterator, List, Literal, Optional, Tuple

import numpy as np

from .base import BaseDataset, Sequence

MotionPattern = Literal["linear", "circular", "random", "occlusion", "scale_change"]


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
        bbox_size: ``(width, height)`` of the target rectangle in pixels.
            Default: ``(40, 40)``.
        motion: Motion pattern — ``"linear"``, ``"circular"``, or ``"random"``.
            Default: ``"linear"``.
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
    """

    _VALID_MOTIONS = ("linear", "circular", "random", "occlusion", "scale_change")

    def __init__(
        self,
        num_sequences: int = 10,
        num_frames: int = 100,
        frame_size: Tuple[int, int] = (320, 240),
        bbox_size: Tuple[int, int] = (40, 40),
        motion: MotionPattern = "linear",
        seed: int = 42,
        occlusion_period: int = 30,
        occlusion_frames: int = 10,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        scale_period: int = 40,
    ) -> None:
        """
        Args:
            num_sequences:    Number of sequences to generate.
            num_frames:       Frames per sequence.
            frame_size:       ``(width, height)`` in pixels.
            bbox_size:        Nominal ``(width, height)`` of the target.
            motion:           Motion pattern — one of
                              ``"linear"``, ``"circular"``, ``"random"``,
                              ``"occlusion"``, ``"scale_change"``.
            seed:             Base RNG seed; sequence ``i`` uses ``seed + i``.
            occlusion_period: Frames between successive occlusion events
                              (``"occlusion"`` pattern only).  Default: ``30``.
            occlusion_frames: Number of consecutive frames the occluder covers
                              the target per event.  Default: ``10``.
            scale_min:        Minimum bbox scale factor relative to ``bbox_size``
                              (``"scale_change"`` pattern only).  Default: ``0.5``.
            scale_max:        Maximum bbox scale factor.  Default: ``2.0``.
            scale_period:     Frames for one full scale oscillation.  Default: ``40``.
        """
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
        self.occlusion_period = max(1, occlusion_period)
        self.occlusion_frames = max(1, occlusion_frames)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.scale_period = max(1, scale_period)
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
        positions = self._generate_positions(cx0, cy0, rng)
        sizes = self._generate_sizes(len(positions))

        # Static noise background — varied per sequence but fixed across frames.
        background = rng.integers(40, 100, (H, W, 3), dtype=np.uint8)
        # Bright target colour distinct from the dim background.
        colour = tuple(int(c) for c in rng.integers(160, 256, 3))
        # Occluder colour: neutral grey, clearly distinct from target.
        occ_colour = (128, 128, 128)

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []

        for i, ((cx, cy), (bw_i, bh_i)) in enumerate(zip(positions, sizes)):
            frame = background.copy()
            x1 = int(round(cx - bw_i / 2))
            y1 = int(round(cy - bh_i / 2))
            # Clip rectangle to frame boundaries before drawing.
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x1 + bw_i), min(H, y1 + bh_i)
            frame[y1c:y2c, x1c:x2c] = colour

            if self.motion == "occlusion":
                self._draw_occluder(frame, i, cx, cy, bw_i, bh_i, occ_colour)

            frames.append(frame)
            # Ground-truth uses the un-clipped box (may extend outside frame).
            gt_boxes.append((float(x1), float(y1), float(bw_i), float(bh_i)))

        return _InMemorySequence(
            name=f"synth_{self.motion}_{idx:03d}",
            frames=frames,
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    def _generate_sizes(
        self, num_frames: int
    ) -> List[Tuple[int, int]]:
        """Return per-frame ``(bw, bh)`` tuples.

        For all patterns except ``"scale_change"`` the size is constant and
        equal to ``self.bbox_size``.  For ``"scale_change"`` the size
        oscillates sinusoidally between ``scale_min × bbox_size`` and
        ``scale_max × bbox_size`` over ``scale_period`` frames.

        Args:
            num_frames: Length of the sequence.

        Returns:
            List of ``num_frames`` ``(int, int)`` size tuples.
        """
        bw, bh = self.bbox_size
        if self.motion != "scale_change":
            return [(bw, bh)] * num_frames

        sizes: List[Tuple[int, int]] = []
        mid = (self.scale_min + self.scale_max) / 2.0
        amp = (self.scale_max - self.scale_min) / 2.0
        for i in range(num_frames):
            scale = mid + amp * math.sin(2 * math.pi * i / self.scale_period)
            bw_i = max(4, int(round(bw * scale)))
            bh_i = max(4, int(round(bh * scale)))
            sizes.append((bw_i, bh_i))
        return sizes

    def _draw_occluder(
        self,
        frame: np.ndarray,
        frame_idx: int,
        cx: float,
        cy: float,
        bw: int,
        bh: int,
        colour: Tuple[int, int, int],
    ) -> None:
        """Paint a rectangular occluder on top of the target when active.

        The occluder is active for ``occlusion_frames`` consecutive frames out
        of every ``occlusion_period`` frames, starting at the first full period.
        When active it is painted centred on the target so it occludes it
        completely, simulating a foreground object passing over the target.

        The ground-truth box is **not** modified — it always records the true
        target position, matching real benchmark annotation conventions.

        Args:
            frame:     BGR frame to paint in-place.
            frame_idx: 0-indexed frame number within the sequence.
            cx, cy:    Target centre coordinates.
            bw, bh:    Target bbox dimensions.
            colour:    BGR colour tuple for the occluder.
        """
        H, W = frame.shape[:2]
        phase = frame_idx % self.occlusion_period
        if phase >= self.occlusion_frames:
            return  # occluder not active on this frame

        # Occluder is 1.5× the target size so it fully covers it.
        ow = int(bw * 1.5)
        oh = int(bh * 1.5)
        ox1 = max(0, int(round(cx - ow / 2)))
        oy1 = max(0, int(round(cy - oh / 2)))
        ox2 = min(W, ox1 + ow)
        oy2 = min(H, oy1 + oh)
        frame[oy1:oy2, ox1:ox2] = colour

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

        if self.motion in ("linear", "occlusion", "scale_change"):
            # All three use constant-velocity motion with wall-bounce.
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
