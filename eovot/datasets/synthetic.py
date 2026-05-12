"""Procedurally generated synthetic tracking dataset for EOVOT.

Produces fully deterministic, in-memory tracking sequences without requiring
any downloaded data.  Primary use cases:

- **CI / smoke tests** — run the full benchmark pipeline in seconds.
- **Algorithm prototyping** — validate tracker behaviour on controlled motion
  before committing to long real-data runs.
- **Ablation studies** — isolate the effect of motion complexity, scale change,
  or noise level by toggling config parameters.
- **Edge-case generation** — fast motion, target scale change, noisy frames.

Supported motion patterns
-------------------------
``linear``
    Constant-velocity target with elastic boundary reflection.

``circular``
    Uniform circular motion around the frame centre.

``sinusoidal``
    Independent sine-wave trajectories along x and y axes.

``random`` / ``random_walk``
    Velocity perturbed by Gaussian noise each frame (random walk).

Constructor API
---------------
Flat (backward-compatible with the original version)::

    dataset = SyntheticDataset(n_sequences=5, n_frames=60, motion="circular")

Config-based (preferred for experiment YAML configs)::

    from eovot.datasets.synthetic import SyntheticConfig, SyntheticDataset

    cfg = SyntheticConfig(num_sequences=10, sequence_length=100, motion="sinusoidal",
                          scale_change=1.5, add_noise=True)
    dataset = SyntheticDataset(config=cfg)

Engine integration::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    result = BenchmarkEngine(verbose=False).run(
        MOSSETracker(), dataset, dataset_name="Synthetic"
    )
    print(result.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Literal, Optional, Tuple

import cv2
import numpy as np

from .base import BaseDataset, BBox, Sequence

# Supported motion patterns (kept as a Literal for static analysis).
MotionPattern = Literal["linear", "circular", "sinusoidal", "random", "random_walk"]

_DEFAULT_FRAME_SIZE = (320, 240)   # (width, height) in pixels
_DEFAULT_TARGET_SIZE = (40, 30)    # (width, height) in pixels
_TARGET_COLOR = (0, 200, 80)       # BGR — vivid green rectangle
_BG_COLOR = (30, 30, 30)          # near-black background


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    """All parameters that fully specify a :class:`SyntheticDataset`.

    Every sequence generated from the same config and seed is identical,
    guaranteeing reproducibility across runs, machines, and Python versions.

    Args:
        num_sequences:    Number of distinct sequences.
        sequence_length:  Frames per sequence.
        frame_size:       ``(width, height)`` of every frame in pixels.
        target_size:      ``(width, height)`` of the target bounding box in pixels.
        motion:           Motion model — one of ``"linear"``, ``"circular"``,
                          ``"sinusoidal"``, ``"random"``, ``"random_walk"``.
        speed:            Pixels moved per frame (base speed for all motions).
        seed:             Master RNG seed.  Each sequence uses a derived seed.
        background_type:  ``"gradient"`` or ``"solid"`` background.
        add_noise:        Add zero-mean Gaussian pixel noise (sigma=8) to frames.
        scale_change:     ``> 1.0`` enables smooth target scale oscillation.
                          The target smoothly grows to ``target_size * scale_change``
                          and back over the sequence.  ``1.0`` disables variation.
    """

    num_sequences: int = 5
    sequence_length: int = 50
    frame_size: Tuple[int, int] = _DEFAULT_FRAME_SIZE   # (W, H)
    target_size: Tuple[int, int] = _DEFAULT_TARGET_SIZE  # (W, H)
    motion: str = "linear"
    speed: float = 3.0
    seed: int = 42
    background_type: str = "solid"
    add_noise: bool = False
    scale_change: float = 1.0


# ---------------------------------------------------------------------------
# In-memory sequence
# ---------------------------------------------------------------------------

class SyntheticSequence(Sequence):
    """An in-memory sequence that renders BGR frames with a visible moving target.

    Subclasses :class:`~eovot.datasets.base.Sequence` and overrides
    ``__iter__`` to generate frames procedurally instead of reading from disk.
    Frames are yielded as independent copies so that trackers can modify them
    in-place without corrupting the cached sequence.

    Args:
        name:             Unique sequence identifier.
        n_frames:         Number of frames in the sequence.
        gt_boxes:         Ground-truth boxes ``(x, y, w, h)`` per frame,
                          shape ``(N, 4)``.
        frame_size:       ``(width, height)`` of rendered frames in pixels.
        target_size:      ``(width, height)`` of the target rectangle in pixels.
        rng_seed:         Seed for reproducible texture noise.
        background_type:  ``"gradient"`` or ``"solid"``.
        add_noise:        Add Gaussian noise if True.
        scale_change:     Scale oscillation factor (1.0 = fixed size).
    """

    def __init__(
        self,
        name: str,
        n_frames: int,
        gt_boxes: np.ndarray,
        frame_size: Tuple[int, int] = _DEFAULT_FRAME_SIZE,
        target_size: Tuple[int, int] = _DEFAULT_TARGET_SIZE,
        rng_seed: Optional[int] = None,
        background_type: str = "solid",
        add_noise: bool = False,
        scale_change: float = 1.0,
    ) -> None:
        super().__init__(
            name=name,
            frame_paths=[f"synthetic_frame_{i:05d}" for i in range(n_frames)],
            ground_truth=gt_boxes,
        )
        self._n_frames = n_frames
        self._frame_size = frame_size    # (W, H)
        self._target_size = target_size  # (tw, th)
        self._rng_seed = rng_seed
        self._background_type = background_type
        self._add_noise = add_noise
        self._scale_change = scale_change

    def __iter__(self) -> Iterator[np.ndarray]:  # type: ignore[override]
        """Yield independent BGR uint8 frame copies with the target rendered.

        Copies are returned so trackers cannot corrupt the cached sequence.
        """
        fw, fh = self._frame_size
        tw, th = self._target_size
        gt = self.ground_truth   # (N, 4)
        n = self._n_frames

        rng = np.random.default_rng(self._rng_seed)

        # Build background canvas once and reuse it.
        if self._background_type == "gradient":
            xs = np.linspace(60, 160, fw, dtype=np.float32)
            ys = np.linspace(60, 160, fh, dtype=np.float32)
            bg_gray = ((xs[np.newaxis, :] + ys[:, np.newaxis]) / 2.0).astype(np.uint8)
            bg_bgr = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
        else:
            bg_bgr = np.full((fh, fw, 3), _BG_COLOR, dtype=np.uint8)
            noise_base = rng.integers(0, 20, (fh, fw, 3), dtype=np.uint8)
            bg_bgr = np.clip(bg_bgr.astype(np.int16) + noise_base, 0, 255).astype(np.uint8)

        for i in range(n):
            frame = bg_bgr.copy()
            x, y, w, h = gt[i]

            # Apply smooth scale oscillation when configured.
            if self._scale_change > 1.0:
                phase = math.pi * i / max(n - 1, 1)
                sf = 1.0 + (self._scale_change - 1.0) * 0.5 * (1.0 - math.cos(phase))
                w = max(4, int(round(w * sf)))
                h = max(4, int(round(h * sf)))

            x1 = max(0, min(int(round(x)), fw - int(w)))
            y1 = max(0, min(int(round(y)), fh - int(h)))
            x2 = min(x1 + int(w), fw)
            y2 = min(y1 + int(h), fh)

            cv2.rectangle(frame, (x1, y1), (x2, y2), _TARGET_COLOR, thickness=-1)
            border = tuple(max(0, c - 60) for c in _TARGET_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, thickness=2)

            if self._add_noise:
                noise = rng.standard_normal(frame.shape).astype(np.float32) * 8.0
                frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            yield frame


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticDataset(BaseDataset):
    """In-memory benchmark dataset with configurable motion patterns.

    Accepts either a :class:`SyntheticConfig` object or flat keyword arguments
    matching the original flat API, so it is backward-compatible.

    Args:
        config:           :class:`SyntheticConfig` with all parameters.  When
                          provided, all flat keyword arguments are ignored.
        n_sequences:      Number of independent sequences (flat API).
        n_frames:         Frames per sequence (flat API).
        motion:           Motion pattern (flat API).
        frame_size:       ``(width, height)`` of rendered frames (flat API).
        target_size:      ``(width, height)`` of the target box (flat API).
        speed:            Pixels moved per frame (flat API).
        seed:             RNG seed (flat API).
        background_type:  ``"gradient"`` or ``"solid"`` (flat API).
        add_noise:        Add Gaussian pixel noise (flat API).
        scale_change:     Scale oscillation factor (flat API).

    Example::

        # Config-based (preferred):
        ds = SyntheticDataset(SyntheticConfig(num_sequences=10, motion="circular"))

        # Flat API (backward-compatible with original version):
        ds = SyntheticDataset(n_sequences=10, motion="circular")
    """

    MOTIONS: Tuple[str, ...] = ("linear", "circular", "sinusoidal", "random", "random_walk")

    def __init__(
        self,
        config: Optional[SyntheticConfig] = None,
        *,
        n_sequences: int = 5,
        n_frames: int = 50,
        motion: str = "linear",
        frame_size: Tuple[int, int] = _DEFAULT_FRAME_SIZE,
        target_size: Tuple[int, int] = _DEFAULT_TARGET_SIZE,
        speed: float = 3.0,
        seed: int = 42,
        background_type: str = "solid",
        add_noise: bool = False,
        scale_change: float = 1.0,
    ) -> None:
        if config is not None:
            n_sequences = config.num_sequences
            n_frames = config.sequence_length
            motion = config.motion
            frame_size = config.frame_size
            target_size = config.target_size
            speed = config.speed
            seed = config.seed
            background_type = config.background_type
            add_noise = config.add_noise
            scale_change = config.scale_change

        # Normalise "random_walk" to "random" for internal use.
        _motion = "random" if motion == "random_walk" else motion
        if _motion not in self.MOTIONS:
            raise ValueError(
                f"motion must be one of {self.MOTIONS!r}, got {motion!r}"
            )
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")

        self._n_sequences = n_sequences
        self._n_frames = n_frames
        self._motion = _motion
        self._frame_size = frame_size
        self._target_size = target_size
        self._speed = speed
        self._seed = seed
        self._background_type = background_type
        self._add_noise = add_noise
        self._scale_change = scale_change

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
    # Generation internals
    # ------------------------------------------------------------------

    def _generate_all(self) -> List[SyntheticSequence]:
        sequences: List[SyntheticSequence] = []
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
                background_type=self._background_type,
                add_noise=self._add_noise,
                scale_change=self._scale_change,
            )
            sequences.append(seq)
        return sequences

    def _generate_trajectory(self, seq_seed: int) -> np.ndarray:
        """Generate per-frame GT boxes for one sequence using the chosen motion."""
        rng = np.random.default_rng(seq_seed)
        fw, fh = self._frame_size
        tw, th = self._target_size
        cx0, cy0 = fw / 2.0, fh / 2.0
        boxes = np.empty((self._n_frames, 4), dtype=np.float64)

        if self._motion == "linear":
            angle = rng.uniform(0.0, 2 * math.pi)
            vx = self._speed * math.cos(angle)
            vy = self._speed * math.sin(angle)
            cx, cy = cx0, cy0
            for t in range(self._n_frames):
                cx += vx
                cy += vy
                if cx - tw / 2 < 0 or cx + tw / 2 > fw:
                    vx = -vx
                    cx = float(np.clip(cx, tw / 2, fw - tw / 2))
                if cy - th / 2 < 0 or cy + th / 2 > fh:
                    vy = -vy
                    cy = float(np.clip(cy, th / 2, fh - th / 2))
                boxes[t] = [cx - tw / 2, cy - th / 2, float(tw), float(th)]

        elif self._motion == "circular":
            max_r = min(fw / 2 - tw / 2, fh / 2 - th / 2) * 0.7
            radius = max(self._speed * 5, min(self._speed * 20, max_r))
            phase0 = rng.uniform(0.0, 2 * math.pi)
            angular_speed = self._speed / max(radius, 1.0)
            for t in range(self._n_frames):
                angle = phase0 + t * angular_speed
                cx = float(np.clip(cx0 + radius * math.cos(angle), tw / 2, fw - tw / 2))
                cy = float(np.clip(cy0 + radius * math.sin(angle), th / 2, fh - th / 2))
                boxes[t] = [cx - tw / 2, cy - th / 2, float(tw), float(th)]

        elif self._motion == "sinusoidal":
            amplitude = min(fw / 2 - tw / 2, self._speed * 15)
            freq = 2 * math.pi / max(self._n_frames / 2, 1)
            phase_x = rng.uniform(0.0, 2 * math.pi)
            phase_y = rng.uniform(0.0, 2 * math.pi)
            amp_y = min(fh / 2 - th / 2, self._speed * 10)
            for t in range(self._n_frames):
                cx = float(np.clip(
                    cx0 + amplitude * math.sin(freq * t + phase_x), tw / 2, fw - tw / 2
                ))
                cy = float(np.clip(
                    cy0 + amp_y * math.sin(freq * t + phase_y), th / 2, fh - th / 2
                ))
                boxes[t] = [cx - tw / 2, cy - th / 2, float(tw), float(th)]

        elif self._motion == "random":
            cx, cy = cx0, cy0
            vx = float(rng.uniform(-self._speed, self._speed))
            vy = float(rng.uniform(-self._speed, self._speed))
            noise_scale = self._speed * 0.2
            for t in range(self._n_frames):
                vx += float(rng.normal(0.0, noise_scale))
                vy += float(rng.normal(0.0, noise_scale))
                cur_speed = math.sqrt(vx ** 2 + vy ** 2) + 1e-9
                max_speed = self._speed * 2.0
                if cur_speed > max_speed:
                    vx *= max_speed / cur_speed
                    vy *= max_speed / cur_speed
                cx = float(np.clip(cx + vx, tw / 2, fw - tw / 2))
                cy = float(np.clip(cy + vy, th / 2, fh - th / 2))
                boxes[t] = [cx - tw / 2, cy - th / 2, float(tw), float(th)]

        return boxes
