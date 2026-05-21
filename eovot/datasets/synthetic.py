"""Procedural synthetic dataset for offline benchmarking and CI.

:class:`SyntheticDataset` generates tracking sequences entirely in memory —
no filesystem access, no dataset download required.  Each sequence places a
coloured rectangle on a textured background and moves it according to one of
three motion models (``linear``, ``circular``, ``random``).

This is invaluable for:

* **CI / unit tests** — fast, deterministic, zero I/O.
* **Ablation sweeps** — run many configs without downloading OTB / GOT-10k.
* **Framework smoke tests** — verify the full pipeline end-to-end.
* **Tracker debugging** — controlled motion makes failure analysis easy.

Usage::

    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(
        num_sequences=10,
        num_frames=100,
        frame_size=(320, 240),
        bbox_size=(40, 40),
        motion="circular",
        seed=42,
    )
    print(f"{len(dataset)} sequences")
    seq = dataset[0]
    for frame in seq:
        h, w = frame.shape[:2]  # (240, 320, 3) uint8 BGR
    gt = seq.ground_truth       # shape (N, 4) — (x, y, w, h)
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, Sequence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_background(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random low-frequency texture background (BGR uint8)."""
    noise = rng.integers(30, 200, size=(height // 8 + 1, width // 8 + 1, 3), dtype=np.uint8)
    # Upscale to full resolution by repeating blocks (avoids cv2 dependency)
    bg = np.repeat(np.repeat(noise, 8, axis=0), 8, axis=1)
    return bg[:height, :width]


def _draw_rect(
    frame: np.ndarray,
    x: float,
    y: float,
    w: int,
    h: int,
    color: Tuple[int, int, int],
) -> np.ndarray:
    """Stamp a filled rectangle onto *frame* in place."""
    x1 = int(round(max(0, x)))
    y1 = int(round(max(0, y)))
    x2 = min(frame.shape[1], x1 + w)
    y2 = min(frame.shape[0], y1 + h)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = color
    return frame


def _linear_trajectory(
    num_frames: int,
    frame_w: int,
    frame_h: int,
    bbox_w: int,
    bbox_h: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Straight-line trajectory with bounce off frame edges."""
    cx = float(rng.integers(bbox_w, frame_w - bbox_w))
    cy = float(rng.integers(bbox_h, frame_h - bbox_h))
    speed = float(rng.uniform(1.5, 4.0))
    angle = float(rng.uniform(0, 2 * np.pi))
    vx, vy = speed * np.cos(angle), speed * np.sin(angle)

    xs, ys = [cx], [cy]
    for _ in range(num_frames - 1):
        cx += vx
        cy += vy
        half_w, half_h = bbox_w / 2.0, bbox_h / 2.0
        if cx - half_w < 0 or cx + half_w > frame_w:
            vx = -vx
            cx = max(half_w, min(frame_w - half_w, cx))
        if cy - half_h < 0 or cy + half_h > frame_h:
            vy = -vy
            cy = max(half_h, min(frame_h - half_h, cy))
        xs.append(cx)
        ys.append(cy)

    return np.stack([xs, ys], axis=1)


def _circular_trajectory(
    num_frames: int,
    frame_w: int,
    frame_h: int,
    bbox_w: int,
    bbox_h: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Elliptical orbit centred in the frame."""
    cx0, cy0 = frame_w / 2.0, frame_h / 2.0
    rx = float(rng.uniform(frame_w * 0.15, frame_w * 0.35))
    ry = float(rng.uniform(frame_h * 0.15, frame_h * 0.35))
    omega = float(rng.uniform(0.03, 0.08))
    phase = float(rng.uniform(0, 2 * np.pi))
    t = np.arange(num_frames, dtype=np.float64)
    xs = cx0 + rx * np.cos(omega * t + phase)
    ys = cy0 + ry * np.sin(omega * t + phase)
    return np.stack([xs, ys], axis=1)


def _random_trajectory(
    num_frames: int,
    frame_w: int,
    frame_h: int,
    bbox_w: int,
    bbox_h: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random walk with inertia (auto-regressive velocity)."""
    cx = float(rng.uniform(bbox_w, frame_w - bbox_w))
    cy = float(rng.uniform(bbox_h, frame_h - bbox_h))
    vx = float(rng.uniform(-2, 2))
    vy = float(rng.uniform(-2, 2))
    alpha = 0.85  # velocity retention

    xs, ys = [cx], [cy]
    for _ in range(num_frames - 1):
        vx = alpha * vx + (1 - alpha) * float(rng.uniform(-3, 3))
        vy = alpha * vy + (1 - alpha) * float(rng.uniform(-3, 3))
        cx = float(np.clip(cx + vx, bbox_w / 2.0, frame_w - bbox_w / 2.0))
        cy = float(np.clip(cy + vy, bbox_h / 2.0, frame_h - bbox_h / 2.0))
        xs.append(cx)
        ys.append(cy)

    return np.stack([xs, ys], axis=1)


# ---------------------------------------------------------------------------
# In-memory Sequence
# ---------------------------------------------------------------------------


class _SyntheticSequence(Sequence):
    """A single synthetic tracking sequence backed by NumPy arrays.

    Frames are materialised lazily on first access and cached in memory.
    """

    def __init__(
        self,
        name: str,
        frames: List[np.ndarray],
        gt: np.ndarray,
    ) -> None:
        self._name = name
        self._frames = frames
        self._gt = gt

    @property
    def name(self) -> str:
        return self._name

    @property
    def init_bbox(self):
        return tuple(float(v) for v in self._gt[0])

    @property
    def ground_truth(self) -> np.ndarray:
        return self._gt

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._frames)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._frames[idx]


# ---------------------------------------------------------------------------
# SyntheticDataset
# ---------------------------------------------------------------------------

#: Available motion models for :class:`SyntheticDataset`.
MOTION_MODELS = ("linear", "circular", "random")


class SyntheticDataset(BaseDataset):
    """In-memory procedural dataset for offline benchmarking and CI/CD.

    All sequences are generated deterministically from ``seed``.  No files are
    read from disk — every frame is a NumPy array created at construction time.

    Args:
        num_sequences: Number of independent tracking sequences.  Default: ``10``.
        num_frames: Frames per sequence.  Default: ``100``.
        frame_size: ``(width, height)`` of each frame in pixels.  Default: ``(320, 240)``.
        bbox_size: ``(width, height)`` of the tracked target rectangle.  Default: ``(40, 40)``.
        motion: Motion model for the target.  One of ``"linear"``, ``"circular"``,
            ``"random"``.  Default: ``"linear"``.
        seed: Random seed for reproducible generation.  Default: ``0``.

    Raises:
        ValueError: If *motion* is not one of :data:`MOTION_MODELS`.

    Example::

        dataset = SyntheticDataset(num_sequences=5, num_frames=50, motion="circular", seed=7)
        print(f"{len(dataset)} sequences, {len(dataset[0])} frames each")

        seq = dataset[0]
        for frame in seq:
            pass  # frame is (H, W, 3) BGR uint8 ndarray
        gt = seq.ground_truth   # shape (50, 4) — (x, y, w, h) in pixels
    """

    def __init__(
        self,
        num_sequences: int = 10,
        num_frames: int = 100,
        frame_size: Tuple[int, int] = (320, 240),
        bbox_size: Tuple[int, int] = (40, 40),
        motion: str = "linear",
        seed: int = 0,
    ) -> None:
        if motion not in MOTION_MODELS:
            raise ValueError(
                f"Unknown motion model '{motion}'. Choose from {MOTION_MODELS}."
            )
        self._num_sequences = num_sequences
        self._num_frames = num_frames
        self._frame_w, self._frame_h = frame_size
        self._bbox_w, self._bbox_h = bbox_size
        self._motion = motion
        self._seed = seed
        self._sequences: Optional[List[_SyntheticSequence]] = None

    # ------------------------------------------------------------------
    # Lazy generation
    # ------------------------------------------------------------------

    def _ensure_generated(self) -> None:
        if self._sequences is not None:
            return
        rng = np.random.default_rng(self._seed)
        self._sequences = [
            self._generate_sequence(i, rng) for i in range(self._num_sequences)
        ]

    def _generate_sequence(
        self, idx: int, rng: np.random.Generator
    ) -> _SyntheticSequence:
        name = f"synthetic_{self._motion}_{idx:04d}"
        bg_seed = rng.integers(0, 2**31)
        bg_rng = np.random.default_rng(int(bg_seed))
        bg = _make_background(self._frame_h, self._frame_w, bg_rng)
        color = tuple(int(c) for c in rng.integers(80, 255, size=3))

        traj_fn = {
            "linear": _linear_trajectory,
            "circular": _circular_trajectory,
            "random": _random_trajectory,
        }[self._motion]
        centres = traj_fn(
            self._num_frames,
            self._frame_w,
            self._frame_h,
            self._bbox_w,
            self._bbox_h,
            rng,
        )

        frames: List[np.ndarray] = []
        gt_boxes: List[Tuple[float, float, float, float]] = []

        for cx, cy in centres:
            x = cx - self._bbox_w / 2.0
            y = cy - self._bbox_h / 2.0
            frame = bg.copy()
            _draw_rect(frame, x, y, self._bbox_w, self._bbox_h, color)
            frames.append(frame)
            gt_boxes.append((float(x), float(y), float(self._bbox_w), float(self._bbox_h)))

        gt = np.array(gt_boxes, dtype=np.float64)
        return _SyntheticSequence(name=name, frames=frames, gt=gt)

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, idx: int) -> _SyntheticSequence:
        self._ensure_generated()
        assert self._sequences is not None
        if idx < 0 or idx >= self._num_sequences:
            raise IndexError(
                f"Sequence index {idx} out of range [0, {self._num_sequences})."
            )
        return self._sequences[idx]
