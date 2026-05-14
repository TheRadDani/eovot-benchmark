"""Synthetic sequence generator for EOVOT pipeline testing.

Creates fully in-memory tracking sequences with a moving rectangular target
on a uniform background.  No dataset download required — ideal for CI/CD
pipelines, quick smoke tests, and algorithm development.

Supported motion patterns:

- ``"linear"``     — constant-velocity straight-line trajectory
- ``"sinusoidal"`` — sinusoidal horizontal oscillation with linear vertical drift
- ``"random_walk"``— Gaussian-noise displacement clamped to image bounds

All ground-truth boxes are exact by construction, so a perfect tracker will
achieve mIoU = 1.0 on every synthetic sequence.

Usage::

    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(
        num_sequences=5,
        sequence_length=100,
        frame_size=(240, 320),   # (height, width)
        target_size=(40, 40),
        motion="sinusoidal",
        seed=42,
    )

    for seq in dataset:
        print(seq.name, len(seq), seq.init_bbox)

    # Plug directly into BenchmarkEngine
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")
    print(result)
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, Sequence

# (height, width) of the generated frame
FrameSize = Tuple[int, int]


class SyntheticSequence(Sequence):
    """An in-memory tracking sequence with exact ground-truth boxes.

    Frames are rendered on demand by :meth:`__iter__` rather than stored in
    memory all at once, keeping peak RAM usage proportional to one frame.

    Args:
        name:           Sequence identifier.
        ground_truth:   ``(N, 4)`` array of ``(x, y, w, h)`` boxes.
        frame_size:     ``(height, width)`` of generated frames in pixels.
        target_color:   BGR tuple for the target rectangle.
        background_color: BGR tuple for the background.
    """

    def __init__(
        self,
        name: str,
        ground_truth: np.ndarray,
        frame_size: FrameSize,
        target_color: Tuple[int, int, int] = (0, 0, 255),
        background_color: Tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        if ground_truth.ndim != 2 or ground_truth.shape[1] != 4:
            raise ValueError(
                f"ground_truth must be (N, 4), got {ground_truth.shape}"
            )
        self.name = name
        self._frame_paths: List[str] = ["<synthetic>"] * len(ground_truth)
        self.ground_truth = ground_truth
        self._frame_size = frame_size
        self._target_color = target_color
        self._background_color = background_color

    def __len__(self) -> int:
        return len(self.ground_truth)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield BGR frames rendered from the stored ground-truth boxes."""
        h, w = self._frame_size
        bg = self._background_color
        tc = self._target_color
        for bbox in self.ground_truth:
            frame = np.full((h, w, 3), bg, dtype=np.uint8)
            x, y, bw, bh = (int(round(v)) for v in bbox)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            if x2 > x1 and y2 > y1:
                frame[y1:y2, x1:x2] = tc
            yield frame


class SyntheticDataset(BaseDataset):
    """Generate a collection of synthetic tracking sequences.

    All sequences share the same structural parameters but use independent
    random trajectories when ``seed`` is set (each sequence gets a
    deterministic sub-seed derived from the global seed).

    Args:
        num_sequences:    Number of sequences to generate.  Default: ``5``.
        sequence_length:  Frames per sequence.  Default: ``100``.
        frame_size:       ``(height, width)`` of each frame in pixels.
                          Default: ``(240, 320)``.
        target_size:      ``(height, width)`` of the moving target in pixels.
                          Default: ``(40, 40)``.
        motion:           One of ``"linear"``, ``"sinusoidal"``,
                          ``"random_walk"``.  Default: ``"linear"``.
        speed:            Pixels per frame for linear/sinusoidal motion.
                          Default: ``2.0``.
        seed:             Master random seed for reproducibility.
                          ``None`` uses system entropy.  Default: ``0``.
        target_color:     BGR colour of the target rectangle.
        background_color: BGR colour of the background.

    Example::

        dataset = SyntheticDataset(num_sequences=3, motion="random_walk", seed=7)
        for seq in dataset:
            print(seq.name, seq.init_bbox)
    """

    def __init__(
        self,
        num_sequences: int = 5,
        sequence_length: int = 100,
        frame_size: FrameSize = (240, 320),
        target_size: Tuple[int, int] = (40, 40),
        motion: str = "linear",
        speed: float = 2.0,
        seed: int = 0,
        target_color: Tuple[int, int, int] = (0, 0, 255),
        background_color: Tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        _valid = {"linear", "sinusoidal", "random_walk"}
        if motion not in _valid:
            raise ValueError(f"motion must be one of {_valid}, got {motion!r}")
        if num_sequences < 1:
            raise ValueError("num_sequences must be >= 1")
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")

        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.target_size = target_size
        self.motion = motion
        self.speed = float(speed)
        self.seed = seed
        self.target_color = target_color
        self.background_color = background_color

        self._sequences: List[SyntheticSequence] = [
            self._make_sequence(i) for i in range(num_sequences)
        ]

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> SyntheticSequence:
        return self._sequences[idx]

    def __repr__(self) -> str:
        return (
            f"SyntheticDataset(sequences={self.num_sequences}, "
            f"length={self.sequence_length}, motion={self.motion!r})"
        )

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------

    def _make_sequence(self, idx: int) -> SyntheticSequence:
        """Build one synthetic sequence with a deterministic trajectory."""
        rng = np.random.default_rng(self.seed * 1000 + idx)
        gt = self._generate_trajectory(rng)
        return SyntheticSequence(
            name=f"synthetic_{self.motion}_{idx:03d}",
            ground_truth=gt,
            frame_size=self.frame_size,
            target_color=self.target_color,
            background_color=self.background_color,
        )

    def _generate_trajectory(self, rng: np.random.Generator) -> np.ndarray:
        """Return ``(N, 4)`` ground-truth box array for one sequence."""
        fh, fw = self.frame_size
        th, tw = self.target_size
        n = self.sequence_length

        # Random starting position (keep target fully inside frame)
        x0 = float(rng.integers(0, max(1, fw - tw)))
        y0 = float(rng.integers(0, max(1, fh - th)))

        xs = np.empty(n, dtype=np.float64)
        ys = np.empty(n, dtype=np.float64)

        if self.motion == "linear":
            # Random direction unit vector scaled by speed
            angle = rng.uniform(0, 2 * np.pi)
            vx = self.speed * np.cos(angle)
            vy = self.speed * np.sin(angle)
            for t in range(n):
                xs[t] = np.clip(x0 + vx * t, 0, fw - tw)
                ys[t] = np.clip(y0 + vy * t, 0, fh - th)

        elif self.motion == "sinusoidal":
            # Horizontal sinusoidal oscillation + slow vertical drift
            amplitude = min(fw / 4.0, 60.0)
            period = max(n / 2.0, 10.0)
            vy = self.speed * 0.3
            for t in range(n):
                xs[t] = np.clip(
                    x0 + amplitude * np.sin(2 * np.pi * t / period), 0, fw - tw
                )
                ys[t] = np.clip(y0 + vy * t, 0, fh - th)

        else:  # random_walk
            sigma = self.speed
            xs[0], ys[0] = x0, y0
            for t in range(1, n):
                xs[t] = np.clip(xs[t - 1] + rng.normal(0, sigma), 0, fw - tw)
                ys[t] = np.clip(ys[t - 1] + rng.normal(0, sigma), 0, fh - th)

        gt = np.column_stack([xs, ys,
                               np.full(n, tw, dtype=np.float64),
                               np.full(n, th, dtype=np.float64)])
        return gt
