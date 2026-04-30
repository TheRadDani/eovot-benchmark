"""Synthetic dataset generator for EOVOT benchmarking.

Generates tracking sequences entirely in memory — no dataset downloads required.
Each sequence renders a moving target (solid rectangle, checkerboard, or Gaussian
blob) onto a configurable background with a programmable motion model.

Supported motion models:
- **LinearMotion** — constant velocity, useful as a basic drift baseline
- **SinusoidalMotion** — oscillating trajectory for repeatability testing
- **RandomWalkMotion** — Brownian motion for robustness stress testing

All generated sequences are compatible with :class:`~eovot.datasets.base.Sequence`
and :class:`~eovot.benchmark.engine.BenchmarkEngine`, so synthetic data slots
directly into any existing experiment config or benchmark loop.

Typical usage::

    from eovot.datasets.synthetic import SyntheticDataset, SyntheticSequenceConfig

    # Quick 5-sequence benchmark with default linear motion
    dataset = SyntheticDataset.quick(n_sequences=5)
    print(len(dataset))  # → 5

    # Custom single sequence with sinusoidal motion
    cfg = SyntheticSequenceConfig(
        name="sin_test",
        n_frames=100,
        frame_size=(480, 640),
        init_bbox=(100, 100, 80, 60),
        motion="sinusoidal",
        amplitude=(60, 40),
        frequency=0.05,
    )
    dataset = SyntheticDataset([cfg])
    seq = dataset[0]
    for frame in seq:
        pass  # process frame
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, List, Literal, Optional, Tuple

import cv2
import numpy as np

from .base import BaseDataset, Sequence

BBox = Tuple[float, float, float, float]

# Allowed motion model names
MotionType = Literal["linear", "sinusoidal", "random_walk"]

# Allowed target appearance styles
AppearanceType = Literal["solid", "checkerboard", "gradient"]


# ---------------------------------------------------------------------------
# Sequence configuration
# ---------------------------------------------------------------------------

@dataclass
class SyntheticSequenceConfig:
    """Configuration for a single synthetic tracking sequence.

    Args:
        name:         Sequence identifier used in benchmark reports.
        n_frames:     Number of frames to generate. Default: ``60``.
        frame_size:   ``(height, width)`` in pixels. Default: ``(240, 320)``.
        init_bbox:    Initial bounding box ``(x, y, w, h)`` in pixel coords.
                      Default: target centred in the frame at ~25% of frame size.
        motion:       Motion model — ``"linear"``, ``"sinusoidal"``, or
                      ``"random_walk"``. Default: ``"linear"``.
        velocity:     ``(vx, vy)`` pixels/frame for ``"linear"`` motion.
                      Default: ``(2.0, 1.5)``.
        amplitude:    ``(ax, ay)`` pixel amplitude for ``"sinusoidal"`` motion.
                      Default: ``(60, 40)``.
        frequency:    Oscillation frequency (cycles/frame) for ``"sinusoidal"``
                      motion. Default: ``0.02``.
        random_walk_std: Standard deviation (pixels/frame) for ``"random_walk"``
                      Gaussian step. Default: ``3.0``.
        scale_factor: Per-frame multiplicative scale change applied to target
                      size.  ``1.0`` = no scale change; ``1.005`` = 0.5%
                      growth/frame. Default: ``1.0``.
        appearance:   Target appearance — ``"solid"``, ``"checkerboard"``, or
                      ``"gradient"``. Default: ``"solid"``.
        target_color: BGR tuple for ``"solid"`` appearance. Default: ``(0, 0, 200)``
                      (red).
        bg_color:     BGR tuple for the background. Default: ``(180, 180, 180)``
                      (light grey).
        seed:         RNG seed for reproducible ``"random_walk"`` sequences.
                      Default: ``None`` (non-deterministic).
        add_noise:    If ``True``, adds Gaussian pixel noise (σ=5) to simulate
                      sensor noise. Default: ``False``.
        noise_sigma:  Standard deviation of additive Gaussian noise when
                      ``add_noise=True``. Default: ``5.0``.
    """

    name: str = "synthetic_seq"
    n_frames: int = 60
    frame_size: Tuple[int, int] = (240, 320)        # (H, W)
    init_bbox: Optional[BBox] = None                 # defaults to centred target
    motion: MotionType = "linear"
    velocity: Tuple[float, float] = (2.0, 1.5)      # for linear motion
    amplitude: Tuple[float, float] = (60.0, 40.0)   # for sinusoidal motion
    frequency: float = 0.02                          # for sinusoidal motion
    random_walk_std: float = 3.0                     # for random_walk motion
    scale_factor: float = 1.0                        # per-frame scale multiplier
    appearance: AppearanceType = "solid"
    target_color: Tuple[int, int, int] = (0, 0, 200)
    bg_color: Tuple[int, int, int] = (180, 180, 180)
    seed: Optional[int] = None
    add_noise: bool = False
    noise_sigma: float = 5.0

    def resolve_init_bbox(self) -> BBox:
        """Return the initial bounding box, computing a centred default if needed."""
        if self.init_bbox is not None:
            return self.init_bbox
        H, W = self.frame_size
        w = max(1, int(W * 0.25))
        h = max(1, int(H * 0.25))
        x = (W - w) // 2
        y = (H - h) // 2
        return (float(x), float(y), float(w), float(h))


# ---------------------------------------------------------------------------
# In-memory sequence
# ---------------------------------------------------------------------------

class _SyntheticSequence(Sequence):
    """A :class:`~eovot.datasets.base.Sequence` whose frames are generated in memory.

    Overrides ``__iter__`` to render frames on the fly without any disk I/O.
    """

    def __init__(self, config: SyntheticSequenceConfig) -> None:
        self._cfg = config
        gt = _compute_ground_truth(config)
        super().__init__(
            name=config.name,
            frame_paths=[f"synthetic://{config.name}/frame_{i:06d}" for i in range(config.n_frames)],
            ground_truth=gt,
        )

    def __iter__(self) -> Iterator[np.ndarray]:  # type: ignore[override]
        """Yield rendered BGR frames without touching the filesystem."""
        cfg = self._cfg
        rng = np.random.default_rng(cfg.seed)
        for i in range(cfg.n_frames):
            bbox = tuple(self.ground_truth[i])  # type: ignore[arg-type]
            frame = _render_frame(cfg, bbox, rng if cfg.add_noise else None)
            yield frame


# ---------------------------------------------------------------------------
# Ground-truth trajectory computation
# ---------------------------------------------------------------------------

def _compute_ground_truth(cfg: SyntheticSequenceConfig) -> np.ndarray:
    """Generate the (N, 4) ground-truth array for a SyntheticSequenceConfig."""
    H, W = cfg.frame_size
    x0, y0, w0, h0 = cfg.resolve_init_bbox()
    cx0, cy0 = x0 + w0 / 2, y0 + h0 / 2

    rng = np.random.default_rng(cfg.seed)
    boxes = np.empty((cfg.n_frames, 4), dtype=np.float64)

    cx, cy = cx0, cy0
    w, h = float(w0), float(h0)

    for i in range(cfg.n_frames):
        if cfg.motion == "linear":
            cx = cx0 + cfg.velocity[0] * i
            cy = cy0 + cfg.velocity[1] * i
        elif cfg.motion == "sinusoidal":
            cx = cx0 + cfg.amplitude[0] * math.sin(2 * math.pi * cfg.frequency * i)
            cy = cy0 + cfg.amplitude[1] * math.sin(2 * math.pi * cfg.frequency * i + math.pi / 4)
        elif cfg.motion == "random_walk":
            if i > 0:
                cx += float(rng.normal(0.0, cfg.random_walk_std))
                cy += float(rng.normal(0.0, cfg.random_walk_std))
        else:
            raise ValueError(f"Unknown motion model: {cfg.motion!r}")

        # Apply per-frame scale change
        scale = cfg.scale_factor ** i
        wi = w0 * scale
        hi = h0 * scale

        # Clamp centre so the target stays inside the frame
        wi = max(4.0, min(wi, W - 2))
        hi = max(4.0, min(hi, H - 2))
        cx = max(wi / 2, min(cx, W - wi / 2))
        cy = max(hi / 2, min(cy, H - hi / 2))

        boxes[i] = [cx - wi / 2, cy - hi / 2, wi, hi]

    return boxes


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _render_frame(
    cfg: SyntheticSequenceConfig,
    bbox: BBox,
    noise_rng: Optional[np.random.Generator],
) -> np.ndarray:
    """Render a single BGR frame with the target at *bbox*."""
    H, W = cfg.frame_size
    frame = np.full((H, W, 3), cfg.bg_color, dtype=np.uint8)

    x, y, w, h = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(W, int(round(x + w)))
    y2 = min(H, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return frame

    if cfg.appearance == "solid":
        frame[y1:y2, x1:x2] = cfg.target_color

    elif cfg.appearance == "checkerboard":
        patch = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
        cell = max(4, min((x2 - x1) // 4, (y2 - y1) // 4))
        for r in range(y2 - y1):
            for c in range(x2 - x1):
                if ((r // cell) + (c // cell)) % 2 == 0:
                    patch[r, c] = cfg.target_color
                else:
                    patch[r, c] = (255 - cfg.target_color[0],
                                   255 - cfg.target_color[1],
                                   255 - cfg.target_color[2])
        frame[y1:y2, x1:x2] = patch

    elif cfg.appearance == "gradient":
        ph, pw = y2 - y1, x2 - x1
        xs = np.linspace(0, 1, pw, dtype=np.float32)
        ys = np.linspace(0, 1, ph, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        mask = ((gx + gy) / 2)[..., np.newaxis]
        c0 = np.array(cfg.target_color, dtype=np.float32)
        c1 = np.array([255 - v for v in cfg.target_color], dtype=np.float32)
        patch = (c0 * (1 - mask) + c1 * mask).astype(np.uint8)
        frame[y1:y2, x1:x2] = patch

    if noise_rng is not None:
        noise = noise_rng.normal(0.0, cfg.noise_sigma, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return frame


# ---------------------------------------------------------------------------
# Public dataset class
# ---------------------------------------------------------------------------

class SyntheticDataset(BaseDataset):
    """A dataset of fully synthetic tracking sequences.

    Can be constructed from a list of :class:`SyntheticSequenceConfig` objects
    or via the convenience factory methods :meth:`quick`, :meth:`stress_test`,
    and :meth:`scale_challenge`.

    Args:
        configs: One :class:`SyntheticSequenceConfig` per sequence.

    Example — minimal usage::

        dataset = SyntheticDataset.quick(n_sequences=5, n_frames=40)
        engine = BenchmarkEngine(verbose=True)
        result = engine.run(tracker, dataset, dataset_name="Synthetic-Quick")

    Example — custom sequences::

        configs = [
            SyntheticSequenceConfig(name="linear_fast", motion="linear", velocity=(5, 3)),
            SyntheticSequenceConfig(name="sine_wave",   motion="sinusoidal", amplitude=(80, 50)),
            SyntheticSequenceConfig(name="random_walk", motion="random_walk", seed=7),
        ]
        dataset = SyntheticDataset(configs)
    """

    def __init__(self, configs: List[SyntheticSequenceConfig]) -> None:
        self._sequences: List[_SyntheticSequence] = [
            _SyntheticSequence(cfg) for cfg in configs
        ]

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Sequence:
        return self._sequences[idx]

    def __repr__(self) -> str:
        return f"SyntheticDataset(sequences={len(self)})"

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def quick(
        cls,
        n_sequences: int = 5,
        n_frames: int = 60,
        frame_size: Tuple[int, int] = (240, 320),
        motion: MotionType = "linear",
        seed: Optional[int] = 0,
    ) -> "SyntheticDataset":
        """Create a small dataset for quick sanity checks.

        Each sequence has the same motion model but different initial positions
        and velocities to provide variety.

        Args:
            n_sequences: Number of sequences to generate.
            n_frames:    Frames per sequence.
            frame_size:  ``(H, W)`` in pixels.
            motion:      Motion model for all sequences.
            seed:        Base RNG seed; each sequence gets ``seed + i``.

        Returns:
            :class:`SyntheticDataset` ready to pass to :class:`~eovot.benchmark.engine.BenchmarkEngine`.
        """
        H, W = frame_size
        rng = np.random.default_rng(seed)
        configs = []
        for i in range(n_sequences):
            w = int(rng.integers(W // 8, W // 4))
            h = int(rng.integers(H // 8, H // 4))
            x = int(rng.integers(0, max(1, W - w)))
            y = int(rng.integers(0, max(1, H - h)))
            vx = float(rng.uniform(-3.0, 3.0))
            vy = float(rng.uniform(-2.0, 2.0))
            configs.append(SyntheticSequenceConfig(
                name=f"quick_{i:02d}",
                n_frames=n_frames,
                frame_size=frame_size,
                init_bbox=(float(x), float(y), float(w), float(h)),
                motion=motion,
                velocity=(vx, vy),
                seed=None if seed is None else seed + i,
            ))
        return cls(configs)

    @classmethod
    def stress_test(
        cls,
        n_sequences: int = 10,
        n_frames: int = 100,
        frame_size: Tuple[int, int] = (480, 640),
        seed: int = 42,
    ) -> "SyntheticDataset":
        """Create a mixed-motion dataset for robustness stress-testing.

        Generates sequences with a mix of linear, sinusoidal, and random-walk
        motion plus varying target sizes and appearances.

        Args:
            n_sequences: Number of sequences to generate.
            n_frames:    Frames per sequence.
            frame_size:  ``(H, W)`` in pixels.
            seed:        Base RNG seed for reproducibility.

        Returns:
            :class:`SyntheticDataset` with diverse motion patterns.
        """
        rng = np.random.default_rng(seed)
        motions: List[MotionType] = ["linear", "sinusoidal", "random_walk"]
        appearances: List[AppearanceType] = ["solid", "checkerboard", "gradient"]
        H, W = frame_size

        configs = []
        for i in range(n_sequences):
            motion = motions[i % len(motions)]
            appearance = appearances[i % len(appearances)]
            w = int(rng.integers(W // 10, W // 4))
            h = int(rng.integers(H // 10, H // 4))
            x = int(rng.integers(0, max(1, W - w)))
            y = int(rng.integers(0, max(1, H - h)))
            color = (
                int(rng.integers(50, 200)),
                int(rng.integers(50, 200)),
                int(rng.integers(50, 200)),
            )
            configs.append(SyntheticSequenceConfig(
                name=f"stress_{i:02d}_{motion}",
                n_frames=n_frames,
                frame_size=frame_size,
                init_bbox=(float(x), float(y), float(w), float(h)),
                motion=motion,
                velocity=(float(rng.uniform(-4.0, 4.0)), float(rng.uniform(-3.0, 3.0))),
                amplitude=(float(rng.uniform(30.0, 80.0)), float(rng.uniform(20.0, 60.0))),
                frequency=float(rng.uniform(0.01, 0.05)),
                random_walk_std=float(rng.uniform(2.0, 6.0)),
                appearance=appearance,
                target_color=color,
                seed=seed + i,
                add_noise=(i % 3 == 2),
            ))
        return cls(configs)

    @classmethod
    def scale_challenge(
        cls,
        n_sequences: int = 6,
        n_frames: int = 80,
        frame_size: Tuple[int, int] = (360, 480),
        seed: int = 99,
    ) -> "SyntheticDataset":
        """Create sequences with progressive target scale change.

        Alternates between growing and shrinking targets to stress-test trackers
        that assume fixed object size (e.g. MOSSE and KCF without scale estimation).

        Args:
            n_sequences: Number of sequences to generate.
            n_frames:    Frames per sequence.
            frame_size:  ``(H, W)`` in pixels.
            seed:        RNG seed.

        Returns:
            :class:`SyntheticDataset` with scale-varying sequences.
        """
        rng = np.random.default_rng(seed)
        H, W = frame_size
        configs = []
        for i in range(n_sequences):
            # Alternate growing / shrinking
            scale_factor = 1.005 if i % 2 == 0 else 0.996
            w = int(rng.integers(W // 8, W // 5))
            h = int(rng.integers(H // 8, H // 5))
            x = (W - w) // 2
            y = (H - h) // 2
            configs.append(SyntheticSequenceConfig(
                name=f"scale_{i:02d}_{'grow' if i % 2 == 0 else 'shrink'}",
                n_frames=n_frames,
                frame_size=frame_size,
                init_bbox=(float(x), float(y), float(w), float(h)),
                motion="sinusoidal",
                amplitude=(30.0, 20.0),
                frequency=0.03,
                scale_factor=scale_factor,
                appearance="checkerboard",
                seed=seed + i,
            ))
        return cls(configs)
