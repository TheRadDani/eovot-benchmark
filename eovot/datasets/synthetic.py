"""Synthetic sequence generator for EOVOT.

Provides :class:`SyntheticDataset` — a fully in-memory dataset that renders
BGR frames with a moving target rectangle.  No real video or annotation files
are required, making it ideal for:

- CI pipelines that need a runnable benchmark without downloading datasets.
- Ablation studies where motion complexity is the controlled variable.
- Smoke-testing new trackers before evaluating on real benchmarks.

Supported motion patterns
-------------------------
* ``linear``     — constant velocity in a random direction.
* ``circular``   — uniform circular motion around the frame centre.
* ``sinusoidal`` — sinusoidal oscillation along the x-axis.
* ``random``     — independent Gaussian displacement each frame (random walk).

All patterns clamp the target box to stay fully within the frame so that
out-of-bounds frames are never generated.

Example::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker
    from eovot.benchmark.engine import BenchmarkEngine

    dataset = SyntheticDataset(n_sequences=5, n_frames=60, motion="circular")
    engine  = BenchmarkEngine(verbose=True)
    result  = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic-circular")
    print(result.summary())
"""

from __future__ import annotations

import math
from typing import Iterator, List, Literal, Optional, Tuple

import numpy as np

from .base import BaseDataset, BBox, Sequence

# Motion patterns supported by SyntheticDataset.
MotionPattern = Literal["linear", "circular", "sinusoidal", "random"]

_DEFAULT_FRAME_SIZE = (320, 240)   # (width, height) in pixels
_DEFAULT_TARGET_SIZE = (40, 30)    # (width, height) in pixels
_TARGET_COLOR = (0, 200, 80)       # BGR — vivid green rectangle
_BG_COLOR = (30, 30, 30)          # near-black background


class SyntheticSequence(Sequence):
    """An in-memory sequence that renders BGR frames with a visible moving target.

    Frames are generated lazily on iteration so memory consumption is O(1)
    regardless of sequence length.

    Args:
        name:        Unique sequence identifier.
        n_frames:    Number of frames in the sequence.
        gt_boxes:    Ground-truth boxes ``(x, y, w, h)`` per frame, shape ``(N, 4)``.
        frame_size:  ``(width, height)`` of rendered frames in pixels.
        target_size: ``(width, height)`` of the target rectangle in pixels.
        rng_seed:    Seed for reproducible texture noise. None = random.
    """

    def __init__(
        self,
        name: str,
        n_frames: int,
        gt_boxes: np.ndarray,
        frame_size: Tuple[int, int] = _DEFAULT_FRAME_SIZE,
        target_size: Tuple[int, int] = _DEFAULT_TARGET_SIZE,
        rng_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            name=name,
            frame_paths=[f"synthetic_frame_{i:05d}" for i in range(n_frames)],
            ground_truth=gt_boxes,
        )
        self._n_frames = n_frames
        self._frame_size = frame_size   # (W, H)
        self._target_size = target_size  # (tw, th)
        self._rng = np.random.default_rng(rng_seed)

    def __iter__(self) -> Iterator[np.ndarray]:  # type: ignore[override]
        """Yield BGR uint8 frames with the target rendered at each GT position."""
        fw, fh = self._frame_size
        tw, th = self._target_size
        gt = self.ground_truth  # (N, 4)

        # Pre-generate subtle background noise for realism.
        rng = np.random.default_rng(int(self._rng.integers(0, 2**31)))
        noise_base = rng.integers(0, 20, (fh, fw, 3), dtype=np.uint8)

        for i in range(self._n_frames):
            frame = np.full((fh, fw, 3), _BG_COLOR, dtype=np.uint8)
            frame = np.clip(frame.astype(np.int16) + noise_base, 0, 255).astype(np.uint8)

            x, y, w, h = gt[i]
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + w)), int(round(y + h))
            x1 = max(0, min(x1, fw - 1))
            y1 = max(0, min(y1, fh - 1))
            x2 = max(x1 + 1, min(x2, fw))
            y2 = max(y1 + 1, min(y2, fh))

            frame[y1:y2, x1:x2] = _TARGET_COLOR
            yield frame


class SyntheticDataset(BaseDataset):
    """In-memory benchmark dataset with configurable motion patterns.

    Each sequence starts with the target centred in the frame and then moves
    according to the chosen ``motion`` pattern.

    Args:
        n_sequences:    Number of independent sequences to generate.
        n_frames:       Frames per sequence.
        motion:         One of ``"linear"``, ``"circular"``, ``"sinusoidal"``,
                        ``"random"``.
        frame_size:     ``(width, height)`` of rendered frames. Default: ``(320, 240)``.
        target_size:    ``(width, height)`` of the target box. Default: ``(40, 30)``.
        speed:          Pixels moved per frame (interpretation varies by motion
                        pattern). Default: ``3.0``.
        seed:           Global RNG seed for reproducibility. Default: ``42``.

    Example::

        dataset = SyntheticDataset(n_sequences=10, n_frames=50, motion="circular")
        for seq in dataset:
            for frame in seq:
                ...
    """

    MOTIONS: Tuple[str, ...] = ("linear", "circular", "sinusoidal", "random")

    def __init__(
        self,
        n_sequences: int = 5,
        n_frames: int = 50,
        motion: MotionPattern = "linear",
        frame_size: Tuple[int, int] = _DEFAULT_FRAME_SIZE,
        target_size: Tuple[int, int] = _DEFAULT_TARGET_SIZE,
        speed: float = 3.0,
        seed: int = 42,
    ) -> None:
        if motion not in self.MOTIONS:
            raise ValueError(f"motion must be one of {self.MOTIONS}, got {motion!r}")
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")

        self._n_sequences = n_sequences
        self._n_frames = n_frames
        self._motion = motion
        self._frame_size = frame_size
        self._target_size = target_size
        self._speed = speed
        self._rng = np.random.default_rng(seed)
        self._sequences: List[SyntheticSequence] = self._generate_all()

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Sequence:
        if idx < 0 or idx >= len(self._sequences):
            raise IndexError(f"Index {idx} out of range [0, {len(self._sequences)})")
        return self._sequences[idx]

    def __repr__(self) -> str:
        return (
            f"SyntheticDataset(n_sequences={self._n_sequences}, "
            f"n_frames={self._n_frames}, motion={self._motion!r})"
        )

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _generate_all(self) -> List[SyntheticSequence]:
        sequences = []
        for i in range(self._n_sequences):
            seq_seed = int(self._rng.integers(0, 2**31))
            gt_boxes = self._generate_trajectory(seq_seed)
            seq = SyntheticSequence(
                name=f"synthetic_{self._motion}_{i:03d}",
                n_frames=self._n_frames,
                gt_boxes=gt_boxes,
                frame_size=self._frame_size,
                target_size=self._target_size,
                rng_seed=seq_seed,
            )
            sequences.append(seq)
        return sequences

    def _generate_trajectory(self, seq_seed: int) -> np.ndarray:
        """Generate per-frame GT boxes for one sequence using the chosen motion pattern."""
        rng = np.random.default_rng(seq_seed)
        fw, fh = self._frame_size
        tw, th = self._target_size

        # Start at centre of frame.
        cx0 = fw / 2.0
        cy0 = fh / 2.0

        boxes = np.empty((self._n_frames, 4), dtype=np.float64)

        if self._motion == "linear":
            angle = rng.uniform(0.0, 2 * math.pi)
            vx = self._speed * math.cos(angle)
            vy = self._speed * math.sin(angle)
            cx, cy = cx0, cy0
            for t in range(self._n_frames):
                cx, cy = self._reflect_centre(cx + vx, cy + vy, vx, vy, fw, fh, tw, th)
                boxes[t] = [cx - tw / 2, cy - th / 2, float(tw), float(th)]

        elif self._motion == "circular":
            # Radius chosen so the full circle stays inside the frame.
            max_r = min(fw / 2 - tw / 2, fh / 2 - th / 2) * 0.7
            radius = max(self._speed * 5, min(self._speed * 20, max_r))
            phase0 = rng.uniform(0.0, 2 * math.pi)
            angular_speed = self._speed / max(radius, 1.0)
            for t in range(self._n_frames):
                angle = phase0 + t * angular_speed
                cx = cx0 + radius * math.cos(angle)
                cy = cy0 + radius * math.sin(angle)
                cx = float(np.clip(cx, tw / 2, fw - tw / 2))
                cy = float(np.clip(cy, th / 2, fh - th / 2))
                boxes[t] = [cx - tw / 2, cy - th / 2, float(tw), float(th)]

        elif self._motion == "sinusoidal":
            amplitude = min(fw / 2 - tw / 2, self._speed * 15)
            freq = 2 * math.pi / max(self._n_frames / 2, 1)
            phase0 = rng.uniform(0.0, 2 * math.pi)
            vy = self._speed * rng.choice([-1.0, 1.0])
            cy = cy0
            for t in range(self._n_frames):
                cx = cx0 + amplitude * math.sin(freq * t + phase0)
                cy = cy + vy
                if cy - th / 2 < 0 or cy + th / 2 > fh:
                    vy = -vy
                    cy = float(np.clip(cy, th / 2, fh - th / 2))
                cx = float(np.clip(cx, tw / 2, fw - tw / 2))
                boxes[t] = [cx - tw / 2, cy - th / 2, float(tw), float(th)]

        elif self._motion == "random":
            cx, cy = cx0, cy0
            for t in range(self._n_frames):
                dx = rng.normal(0.0, self._speed)
                dy = rng.normal(0.0, self._speed)
                cx = float(np.clip(cx + dx, tw / 2, fw - tw / 2))
                cy = float(np.clip(cy + dy, th / 2, fh - th / 2))
                boxes[t] = [cx - tw / 2, cy - th / 2, float(tw), float(th)]

        return boxes

    @staticmethod
    def _reflect_centre(
        cx: float, cy: float,
        vx: float, vy: float,
        fw: int, fh: int,
        tw: int, th: int,
    ) -> Tuple[float, float]:
        """Reflect centre coordinates off frame walls to keep target in-bounds."""
        half_w, half_h = tw / 2.0, th / 2.0
        cx = float(np.clip(cx, half_w, fw - half_w))
        cy = float(np.clip(cy, half_h, fh - half_h))
        return cx, cy
