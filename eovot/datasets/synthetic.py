"""Synthetic tracking dataset for zero-dependency benchmarking and CI.

Generates fully synthetic video sequences with random-walk target motion and
simple visual appearance (solid-colour rectangles on gradient backgrounds).
No external data download is required — frames and annotations are produced
on-the-fly in memory using NumPy alone.

Intended use cases:

- **CI / unit tests** — deterministic (fixed seed), fast (<1 ms per frame),
  and zero-download.  Lets the full benchmark pipeline run in any environment.
- **Demos and tutorials** — shows the complete API without needing a dataset
  licence or large storage footprint.
- **Controlled ablations** — appearance variation is minimal, isolating the
  effect of motion complexity on tracker performance.

Motion model::

    v(t+1) = v(t) * damping + N(0, noise_std²)
    pos(t+1) = pos(t) + v(t+1)

The damping factor keeps the target drifting slowly rather than flying out
of frame.  When the centre would leave a margin band, the velocity is
reflected back toward the frame centre.

Visual model:

* Background: a smoothly varying RGB gradient — different each sequence.
* Target: a filled rectangle drawn over the background in a distinct colour.
  Colour is chosen per-sequence to ensure it contrasts with the background.

Example::

    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, seq_len=50, seed=0)
    for seq in dataset:
        print(seq.name, len(seq), seq.init_bbox)
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, BBox, Sequence

# ---------------------------------------------------------------------------
# Internal frame generator
# ---------------------------------------------------------------------------


def _make_frame(
    frame_h: int,
    frame_w: int,
    bg_color_tl: np.ndarray,
    bg_color_br: np.ndarray,
    bbox: BBox,
    target_color: np.ndarray,
) -> np.ndarray:
    """Render a single synthetic BGR frame.

    Args:
        frame_h: Frame height in pixels.
        frame_w: Frame width in pixels.
        bg_color_tl: BGR colour of the top-left corner background.
        bg_color_br: BGR colour of the bottom-right corner background.
        bbox: Target bounding box ``(x, y, w, h)`` in pixel coordinates.
        target_color: BGR colour of the target rectangle.

    Returns:
        ``(H, W, 3)`` uint8 BGR array.
    """
    # Build a bilinear-interpolated gradient background.
    ys = np.linspace(0.0, 1.0, frame_h)[:, np.newaxis]  # (H, 1)
    xs = np.linspace(0.0, 1.0, frame_w)[np.newaxis, :]  # (1, W)

    # Interpolate top and bottom rows, then blend vertically.
    top = bg_color_tl[np.newaxis, :] * (1 - xs[0, :, np.newaxis]) + bg_color_br[np.newaxis, :] * xs[0, :, np.newaxis]
    bot = bg_color_br[np.newaxis, :] * (1 - xs[0, :, np.newaxis]) + bg_color_tl[np.newaxis, :] * xs[0, :, np.newaxis]

    frame = (
        top[np.newaxis, :, :] * (1 - ys[:, :, np.newaxis])
        + bot[np.newaxis, :, :] * ys[:, :, np.newaxis]
    ).clip(0, 255).astype(np.uint8)

    # Draw the target rectangle.
    x, y, w, h = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(frame_w, int(round(x + w)))
    y2 = min(frame_h, int(round(y + h)))
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = target_color.astype(np.uint8)

    return frame


# ---------------------------------------------------------------------------
# Per-sequence generator
# ---------------------------------------------------------------------------


class _SyntheticSequence:
    """Generate frames and ground-truth boxes for one synthetic sequence.

    This class does all the work; :class:`SyntheticDataset` wraps multiple
    instances and exposes them as :class:`~eovot.datasets.base.Sequence`
    objects.

    Args:
        seq_id: Integer ID used to seed the per-sequence RNG.
        global_seed: Global seed combined with seq_id for full reproducibility.
        seq_len: Number of frames in the sequence.
        frame_size: ``(width, height)`` in pixels.
        target_size: ``(width, height)`` of the target rectangle in pixels.
        damping: Velocity decay factor in ``(0, 1)``.  Lower = smoother motion.
        noise_std: Standard deviation of the velocity noise (pixels/frame).
        margin: Minimum distance from the frame edge to the target centre
            (pixels).  Prevents the target from leaving the frame.
    """

    def __init__(
        self,
        seq_id: int,
        global_seed: int,
        seq_len: int,
        frame_size: Tuple[int, int],
        target_size: Tuple[int, int],
        damping: float,
        noise_std: float,
        margin: int,
    ) -> None:
        self._rng = np.random.default_rng(global_seed + seq_id * 1_000_003)
        self.seq_len = seq_len
        self.frame_w, self.frame_h = frame_size
        self.target_w, self.target_h = target_size
        self.damping = damping
        self.noise_std = noise_std
        self.margin = margin

        # Pre-generate the full trajectory so it is consistent across multiple
        # calls to __iter__ (important for reproducibility in the benchmark loop).
        self._bboxes: List[BBox] = self._generate_trajectory()

        # Per-sequence colours.
        self._bg_tl = self._rng.integers(30, 180, size=3).astype(np.float32)
        self._bg_br = self._rng.integers(30, 180, size=3).astype(np.float32)
        # Target colour: high contrast — flip all channels and offset.
        self._target_color = np.clip(255 - self._bg_tl + 40, 0, 255)

    def _generate_trajectory(self) -> List[BBox]:
        """Simulate the random-walk motion model for all frames."""
        min_cx = self.margin + self.target_w // 2
        max_cx = self.frame_w - self.margin - self.target_w // 2
        min_cy = self.margin + self.target_h // 2
        max_cy = self.frame_h - self.margin - self.target_h // 2

        # Clamp to valid range (guards against very large targets/small frames).
        min_cx = min(min_cx, max_cx - 1) if min_cx < max_cx else 0
        min_cy = min(min_cy, max_cy - 1) if min_cy < max_cy else 0

        cx = float(self._rng.uniform(min_cx, max_cx))
        cy = float(self._rng.uniform(min_cy, max_cy))
        vx = float(self._rng.normal(0.0, self.noise_std))
        vy = float(self._rng.normal(0.0, self.noise_std))

        bboxes: List[BBox] = []
        for _ in range(self.seq_len):
            x = cx - self.target_w / 2.0
            y = cy - self.target_h / 2.0
            bboxes.append((x, y, float(self.target_w), float(self.target_h)))

            # Update velocity with damping and random noise.
            vx = vx * self.damping + float(self._rng.normal(0.0, self.noise_std))
            vy = vy * self.damping + float(self._rng.normal(0.0, self.noise_std))

            # Reflect velocity at boundaries to keep the target inside.
            new_cx = cx + vx
            new_cy = cy + vy
            if new_cx < min_cx or new_cx > max_cx:
                vx = -vx * 0.8
                new_cx = np.clip(new_cx, min_cx, max_cx)
            if new_cy < min_cy or new_cy > max_cy:
                vy = -vy * 0.8
                new_cy = np.clip(new_cy, min_cy, max_cy)
            cx, cy = new_cx, new_cy

        return bboxes

    def to_sequence(self, name: str) -> "SyntheticInMemorySequence":
        """Convert to a :class:`~eovot.datasets.base.Sequence`-compatible object."""
        return SyntheticInMemorySequence(
            name=name,
            bboxes=self._bboxes,
            frame_h=self.frame_h,
            frame_w=self.frame_w,
            bg_tl=self._bg_tl,
            bg_br=self._bg_br,
            target_color=self._target_color,
        )


# ---------------------------------------------------------------------------
# Sequence subclass — renders frames lazily
# ---------------------------------------------------------------------------


class SyntheticInMemorySequence(Sequence):
    """A :class:`~eovot.datasets.base.Sequence` backed by synthetic frames.

    Frames are rendered on-the-fly from pre-computed bounding boxes, so no
    disk I/O is involved.  Each call to :meth:`__iter__` re-renders the
    identical frames for reproducibility.

    This class is not intended to be instantiated directly; use
    :class:`SyntheticDataset` instead.
    """

    def __init__(
        self,
        name: str,
        bboxes: List[BBox],
        frame_h: int,
        frame_w: int,
        bg_tl: np.ndarray,
        bg_br: np.ndarray,
        target_color: np.ndarray,
    ) -> None:
        gt = np.array(bboxes, dtype=np.float64)
        # Pass an empty frame-paths list — we override __iter__ below.
        super().__init__(name=name, frame_paths=[], ground_truth=gt)
        self._bboxes = bboxes
        self._frame_h = frame_h
        self._frame_w = frame_w
        self._bg_tl = bg_tl
        self._bg_br = bg_br
        self._target_color = target_color

    def __len__(self) -> int:
        return len(self._bboxes)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield BGR frames rendered from the pre-computed trajectory."""
        for bbox in self._bboxes:
            yield _make_frame(
                frame_h=self._frame_h,
                frame_w=self._frame_w,
                bg_color_tl=self._bg_tl,
                bg_color_br=self._bg_br,
                bbox=bbox,
                target_color=self._target_color,
            )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SyntheticDataset(BaseDataset):
    """Synthetic tracking dataset with random-walk motion and rendered frames.

    All sequences are fully reproducible given the same ``seed``.  No
    external files or downloads are required.

    Args:
        num_sequences: Number of sequences in the dataset. Default: ``10``.
        seq_len: Number of frames per sequence. Default: ``100``.
        frame_size: ``(width, height)`` of each frame in pixels.
            Default: ``(320, 240)``.
        target_size: ``(width, height)`` of the target rectangle in pixels.
            Default: ``(40, 30)``.
        damping: Velocity decay factor controlling motion smoothness.
            ``1.0`` = constant velocity; ``0.0`` = Gaussian noise each frame.
            Default: ``0.85``.
        noise_std: Standard deviation of per-frame velocity noise (pixels).
            Default: ``2.0``.
        margin: Minimum distance from the frame edge to the target centre
            (pixels).  Prevents the target from going out of frame.
            Default: ``20``.
        seed: Global random seed for full reproducibility. Default: ``42``.

    Example::

        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker
        from eovot.benchmark.engine import BenchmarkEngine

        dataset = SyntheticDataset(num_sequences=5, seq_len=60, seed=0)
        tracker = MOSSETracker()
        engine  = BenchmarkEngine(verbose=True)
        result  = engine.run(tracker, dataset, dataset_name="Synthetic")
        print(result)
    """

    def __init__(
        self,
        num_sequences: int = 10,
        seq_len: int = 100,
        frame_size: Tuple[int, int] = (320, 240),
        target_size: Tuple[int, int] = (40, 30),
        damping: float = 0.85,
        noise_std: float = 2.0,
        margin: int = 20,
        seed: int = 42,
    ) -> None:
        if num_sequences <= 0:
            raise ValueError(f"num_sequences must be > 0, got {num_sequences}")
        if seq_len <= 1:
            raise ValueError(f"seq_len must be > 1 (need at least init + 1 update), got {seq_len}")
        if frame_size[0] <= 0 or frame_size[1] <= 0:
            raise ValueError(f"frame_size dimensions must be positive, got {frame_size}")
        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(f"target_size dimensions must be positive, got {target_size}")

        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.target_size = target_size
        self.damping = damping
        self.noise_std = noise_std
        self.margin = margin
        self.seed = seed

        # Pre-build all sequences so __getitem__ is O(1).
        self._sequences: List[SyntheticInMemorySequence] = [
            _SyntheticSequence(
                seq_id=i,
                global_seed=seed,
                seq_len=seq_len,
                frame_size=frame_size,
                target_size=target_size,
                damping=damping,
                noise_std=noise_std,
                margin=margin,
            ).to_sequence(name=f"synthetic_{i:04d}")
            for i in range(num_sequences)
        ]

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> SyntheticInMemorySequence:
        if idx < 0 or idx >= len(self._sequences):
            raise IndexError(f"Sequence index {idx} out of range [0, {len(self._sequences)})")
        return self._sequences[idx]

    def __repr__(self) -> str:
        fw, fh = self.frame_size
        tw, th = self.target_size
        return (
            f"SyntheticDataset("
            f"sequences={self.num_sequences}, "
            f"len={self.seq_len}, "
            f"frame={fw}×{fh}, "
            f"target={tw}×{th}, "
            f"seed={self.seed})"
        )
