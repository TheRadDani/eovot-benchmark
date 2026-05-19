"""Synthetic dataset generator for offline benchmarking and CI testing.

Produces procedurally-generated tracking sequences entirely in memory —
no dataset download required.  Sequences consist of a filled coloured
rectangle (the "target") moving against a randomised noise background.

Supported motion patterns
~~~~~~~~~~~~~~~~~~~~~~~~~
* **linear** — constant-velocity drift that bounces off frame edges.
* **circular** — target orbits the frame centre at a fixed radius.
* **random** — zero-mean Gaussian random walk.

All three patterns keep the target fully inside the frame at all times,
so every frame has a valid, non-degenerate ground-truth box.

Typical usage::

    from eovot.datasets.synthetic import SyntheticDataset

    # 10 sequences, 100 frames each, linear motion
    dataset = SyntheticDataset(
        num_sequences=10,
        num_frames=100,
        frame_size=(320, 240),
        bbox_size=(40, 40),
        motion="linear",
        seed=42,
    )

    for seq in dataset:
        for frame in seq:
            ...  # process frame (H×W×3 uint8 BGR)
        print(seq.ground_truth.shape)  # (100, 4) float64

The dataset is safe to use in parallel / multi-process settings because
every call to ``__iter__`` regenerates frames from a deterministic RNG
seeded per sequence.
"""

from __future__ import annotations

import math
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from .base import BaseDataset, Sequence


class SyntheticSequence(Sequence):
    """A single procedurally-generated tracking sequence.

    Args:
        name:        Sequence identifier string.
        num_frames:  Number of frames to generate.
        frame_size:  ``(width, height)`` in pixels.
        bbox_size:   ``(width, height)`` of the target bounding box in pixels.
        motion:      One of ``"linear"``, ``"circular"``, ``"random"``.
        seed:        Integer RNG seed for reproducibility.
    """

    def __init__(
        self,
        name: str,
        num_frames: int,
        frame_size: Tuple[int, int],
        bbox_size: Tuple[int, int],
        motion: str,
        seed: int,
    ) -> None:
        self._num_frames = num_frames
        self._frame_size = frame_size  # (W, H)
        self._bbox_size = bbox_size    # (bw, bh)
        self._motion = motion
        self._seed = seed

        gt = self._generate_ground_truth()
        dummy_paths = [f"{name}_frame_{i:05d}.jpg" for i in range(num_frames)]
        super().__init__(name=name, frame_paths=dummy_paths, ground_truth=gt)

    # ------------------------------------------------------------------
    # BaseDataset / Sequence interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield BGR frames generated in-memory (no disk I/O)."""
        rng = np.random.default_rng(self._seed)
        W, H = self._frame_size
        bw, bh = self._bbox_size
        gt = self.ground_truth  # (N, 4) float64

        # Assign a random BGR colour to this target
        target_color = tuple(int(c) for c in rng.integers(80, 220, size=3))

        for i in range(self._num_frames):
            # Noise background
            frame = rng.integers(30, 80, size=(H, W, 3), dtype=np.uint8)
            x, y = int(gt[i, 0]), int(gt[i, 1])
            x2, y2 = min(x + bw, W), min(y + bh, H)
            # Draw target rectangle
            frame[y:y2, x:x2] = target_color
            # Thin dark border so the target has a distinct edge
            cv2.rectangle(frame, (x, y), (x2 - 1, y2 - 1), (0, 0, 0), 1)
            yield frame

    # ------------------------------------------------------------------
    # Ground-truth trajectory generation
    # ------------------------------------------------------------------

    def _generate_ground_truth(self) -> np.ndarray:
        """Compute the ``(N, 4)`` float64 ground-truth array for this sequence."""
        rng = np.random.default_rng(self._seed)
        W, H = self._frame_size
        bw, bh = self._bbox_size
        N = self._num_frames

        if self._motion == "linear":
            gt = self._linear_motion(rng, N, W, H, bw, bh)
        elif self._motion == "circular":
            gt = self._circular_motion(N, W, H, bw, bh)
        elif self._motion == "random":
            gt = self._random_walk(rng, N, W, H, bw, bh)
        else:
            raise ValueError(
                f"Unknown motion type '{self._motion}'. "
                "Choose 'linear', 'circular', or 'random'."
            )
        return gt.astype(np.float64)

    @staticmethod
    def _linear_motion(
        rng: np.random.Generator,
        N: int, W: int, H: int, bw: int, bh: int,
    ) -> np.ndarray:
        """Constant-velocity motion that bounces off frame edges."""
        gt = np.empty((N, 4), dtype=np.float64)
        # Random initial position
        x = float(rng.integers(0, max(1, W - bw)))
        y = float(rng.integers(0, max(1, H - bh)))
        # Random velocity in [-4, 4] px/frame, avoid zero
        vx = float(rng.uniform(-4, 4))
        vy = float(rng.uniform(-4, 4))
        vx = vx if abs(vx) > 0.5 else 1.5
        vy = vy if abs(vy) > 0.5 else 1.5

        for i in range(N):
            gt[i] = [x, y, float(bw), float(bh)]
            x += vx
            y += vy
            if x < 0 or x + bw > W:
                vx = -vx
                x = max(0.0, min(float(W - bw), x))
            if y < 0 or y + bh > H:
                vy = -vy
                y = max(0.0, min(float(H - bh), y))
        return gt

    @staticmethod
    def _circular_motion(
        N: int, W: int, H: int, bw: int, bh: int,
    ) -> np.ndarray:
        """Target orbits the frame centre at a fixed radius."""
        gt = np.empty((N, 4), dtype=np.float64)
        cx, cy = W / 2.0, H / 2.0
        radius = min(cx, cy) * 0.5
        for i in range(N):
            angle = 2 * math.pi * i / N
            x = cx + radius * math.cos(angle) - bw / 2
            y = cy + radius * math.sin(angle) - bh / 2
            x = max(0.0, min(float(W - bw), x))
            y = max(0.0, min(float(H - bh), y))
            gt[i] = [x, y, float(bw), float(bh)]
        return gt

    @staticmethod
    def _random_walk(
        rng: np.random.Generator,
        N: int, W: int, H: int, bw: int, bh: int,
    ) -> np.ndarray:
        """Zero-mean Gaussian random walk, clipped to frame boundaries."""
        gt = np.empty((N, 4), dtype=np.float64)
        x = float(rng.integers(0, max(1, W - bw)))
        y = float(rng.integers(0, max(1, H - bh)))
        for i in range(N):
            gt[i] = [x, y, float(bw), float(bh)]
            x = float(np.clip(x + rng.normal(0, 3), 0, W - bw))
            y = float(np.clip(y + rng.normal(0, 3), 0, H - bh))
        return gt


class SyntheticDataset(BaseDataset):
    """A collection of in-memory synthetic tracking sequences.

    All sequences are generated at construction time and kept as lightweight
    ``SyntheticSequence`` objects (frames are generated lazily on iteration).

    Args:
        num_sequences: Number of sequences to generate.  Default: ``10``.
        num_frames:    Frames per sequence.  Default: ``100``.
        frame_size:    ``(width, height)`` in pixels.  Default: ``(320, 240)``.
        bbox_size:     Target box size in pixels.  Default: ``(40, 40)``.
        motion:        Motion pattern: ``"linear"`` | ``"circular"`` | ``"random"``.
                       Default: ``"linear"``.
        seed:          Base RNG seed; each sequence gets ``seed + sequence_index``
                       for deterministic but distinct trajectories.  Default: ``42``.

    Example::

        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=5, motion="circular")
        print(len(dataset))  # 5
        seq = dataset[0]
        for frame in seq:
            ...  # 100 in-memory BGR frames
    """

    def __init__(
        self,
        num_sequences: int = 10,
        num_frames: int = 100,
        frame_size: Tuple[int, int] = (320, 240),
        bbox_size: Tuple[int, int] = (40, 40),
        motion: str = "linear",
        seed: int = 42,
    ) -> None:
        W, H = frame_size
        bw, bh = bbox_size
        if bw >= W or bh >= H:
            raise ValueError(
                f"bbox_size {bbox_size} must be smaller than frame_size {frame_size}."
            )

        self._sequences = [
            SyntheticSequence(
                name=f"synthetic_{motion}_{i:03d}",
                num_frames=num_frames,
                frame_size=frame_size,
                bbox_size=bbox_size,
                motion=motion,
                seed=seed + i,
            )
            for i in range(num_sequences)
        ]

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> SyntheticSequence:
        return self._sequences[idx]
