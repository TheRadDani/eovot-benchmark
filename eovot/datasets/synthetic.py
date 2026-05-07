"""Synthetic dataset generator for EOVOT benchmarking.

Generates tracking sequences entirely in memory — no dataset downloads required.
Each sequence renders a moving target onto a configurable background with a
programmable motion model, making this module suitable for:

- CI/CD pipeline tests of the full benchmark stack without any data downloads.
- Debugging new tracker implementations with controlled, known ground truth.
- Stress-testing the profiling and metrics engines at arbitrary sequence length.

Supported motion models
-----------------------
- ``"linear"``       — constant-velocity drift; optionally bounces at frame edges.
- ``"sinusoidal"``   — Lissajous-style oscillation, deterministic and repeatable.
- ``"random_walk"``  — bounded Gaussian step noise, seeded for reproducibility.

Supported target appearances
-----------------------------
- ``"solid"``        — filled rectangle in a single BGR colour.
- ``"checkerboard"`` — alternating black/white tiles, useful for tracking
                       algorithms that rely on texture features.
- ``"gradient"``     — horizontal luminance gradient, mimicking a partially
                       illuminated surface.

Factory methods on :class:`SyntheticDataset`
--------------------------------------------
- :meth:`~SyntheticDataset.quick`           — minimal same-motion dataset.
- :meth:`~SyntheticDataset.stress_test`     — mixed motions and appearances.
- :meth:`~SyntheticDataset.scale_challenge` — progressively growing/shrinking targets.

Usage::

    from eovot.datasets.synthetic import SyntheticDataset, SyntheticSequenceConfig

    # Quick 5-sequence sinusoidal benchmark
    dataset = SyntheticDataset.quick(n_sequences=5, motion="sinusoidal", seed=0)

    # Fully custom sequence
    cfg = SyntheticSequenceConfig(
        name="sin_test",
        n_frames=100,
        frame_size=(240, 320),
        init_bbox=(60.0, 50.0, 80.0, 60.0),
        motion="sinusoidal",
        amplitude=(60.0, 40.0),
        frequency=0.03,
        appearance="checkerboard",
        seed=42,
    )
    dataset = SyntheticDataset([cfg])
    for frame in dataset[0]:
        pass  # iterate without any disk I/O
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, List, Literal, Optional, Tuple

import cv2
import numpy as np

from .base import BaseDataset, Sequence

BBox = Tuple[float, float, float, float]

MotionType = Literal["linear", "sinusoidal", "random_walk"]
AppearanceType = Literal["solid", "checkerboard", "gradient"]


# ---------------------------------------------------------------------------
# Sequence configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SyntheticSequenceConfig:
    """All parameters that define a single synthetic tracking sequence.

    Args:
        name:             Identifier used in benchmark reports.
        n_frames:         Total frames (including the init frame). Default: ``60``.
        frame_size:       ``(H, W)`` in pixels. Default: ``(240, 320)``.
        init_bbox:        Initial bounding box ``(x, y, w, h)``.  When ``None``
                          the target is centred in the frame at ~25% of frame
                          dimensions.
        motion:           Motion model. Default: ``"linear"``.
        velocity:         ``(vx, vy)`` pixels/frame for ``"linear"`` motion.
        amplitude:        ``(ax, ay)`` pixel oscillation amplitude for
                          ``"sinusoidal"`` motion.
        frequency:        Oscillation cycles/frame for ``"sinusoidal"`` motion.
        random_walk_std:  Gaussian step σ (px/frame) for ``"random_walk"`` motion.
        scale_factor:     Per-frame multiplicative size change.  ``1.0`` = fixed
                          size; ``1.005`` grows by 0.5%/frame.
        appearance:       Visual style of the rendered target.
        target_color:     BGR color for ``"solid"`` appearance.
        bg_color:         BGR background color.
        seed:             RNG seed for ``"random_walk"`` motion and noise.
        add_noise:        If ``True``, adds Gaussian pixel noise to each frame.
        noise_sigma:      Noise σ when ``add_noise=True``. Default: ``5.0``.
    """

    name: str = "synthetic_seq"
    n_frames: int = 60
    frame_size: Tuple[int, int] = (240, 320)          # (H, W)
    init_bbox: Optional[BBox] = None                   # auto-centres if None
    motion: MotionType = "linear"
    velocity: Tuple[float, float] = (2.0, 1.5)         # for linear
    amplitude: Tuple[float, float] = (60.0, 40.0)      # for sinusoidal
    frequency: float = 0.02                             # for sinusoidal
    random_walk_std: float = 3.0                        # for random_walk
    scale_factor: float = 1.0
    appearance: AppearanceType = "solid"
    target_color: Tuple[int, int, int] = (0, 0, 200)   # BGR red
    bg_color: Tuple[int, int, int] = (180, 180, 180)   # BGR grey
    seed: Optional[int] = None
    add_noise: bool = False
    noise_sigma: float = 5.0

    def resolve_init_bbox(self) -> BBox:
        """Return the initial bounding box, computing a centred default if None."""
        if self.init_bbox is not None:
            return self.init_bbox
        H, W = self.frame_size
        w = max(1, int(W * 0.25))
        h = max(1, int(H * 0.25))
        x = (W - w) // 2
        y = (H - h) // 2
        return (float(x), float(y), float(w), float(h))


# ---------------------------------------------------------------------------
# Ground-truth computation (pure function — no rendering)
# ---------------------------------------------------------------------------

def _compute_ground_truth(cfg: SyntheticSequenceConfig) -> np.ndarray:
    """Compute per-frame ground-truth bounding boxes without rendering frames.

    Args:
        cfg: Sequence configuration.

    Returns:
        ``(n_frames, 4)`` float64 array in ``(x, y, w, h)`` format, clamped
        to the frame boundaries.

    Raises:
        ValueError: If ``cfg.motion`` is not a recognised motion model.
    """
    H, W = cfg.frame_size
    x0, y0, w0, h0 = cfg.resolve_init_bbox()
    cx0, cy0 = x0 + w0 / 2.0, y0 + h0 / 2.0

    rng = np.random.default_rng(cfg.seed)
    gt = np.zeros((cfg.n_frames, 4), dtype=np.float64)

    for i in range(cfg.n_frames):
        # --- Target size (may scale over time) ---
        scale = cfg.scale_factor ** i
        tw = max(1.0, w0 * scale)
        th = max(1.0, h0 * scale)

        # --- Target centre position ---
        if cfg.motion == "linear":
            cx = cx0 + cfg.velocity[0] * i
            cy = cy0 + cfg.velocity[1] * i
            # Bounce at frame edges so the target stays visible.
            cx = _bounce(cx, tw / 2, W - tw / 2)
            cy = _bounce(cy, th / 2, H - th / 2)

        elif cfg.motion == "sinusoidal":
            ax, ay = cfg.amplitude
            cx = cx0 + ax * math.sin(2 * math.pi * cfg.frequency * i)
            cy = cy0 + ay * math.sin(4 * math.pi * cfg.frequency * i)

        elif cfg.motion == "random_walk":
            if i == 0:
                cx, cy = cx0, cy0
            else:
                prev_x = gt[i - 1, 0] + gt[i - 1, 2] / 2
                prev_y = gt[i - 1, 1] + gt[i - 1, 3] / 2
                dx = float(rng.normal(0.0, cfg.random_walk_std))
                dy = float(rng.normal(0.0, cfg.random_walk_std))
                cx = prev_x + dx
                cy = prev_y + dy

        else:
            raise ValueError(
                f"Unknown motion model '{cfg.motion}'. "
                "Expected 'linear', 'sinusoidal', or 'random_walk'."
            )

        # Clamp to frame
        cx = float(np.clip(cx, tw / 2, W - tw / 2))
        cy = float(np.clip(cy, th / 2, H - th / 2))
        gt[i] = [cx - tw / 2, cy - th / 2, tw, th]

    return gt


def _bounce(pos: float, lo: float, hi: float) -> float:
    """Reflect ``pos`` at ``lo`` and ``hi`` boundaries (triangle-wave)."""
    if hi <= lo:
        return (lo + hi) / 2.0
    span = hi - lo
    t = (pos - lo) % (2 * span)
    if t > span:
        t = 2 * span - t
    return lo + t


# ---------------------------------------------------------------------------
# Frame renderer (pure function)
# ---------------------------------------------------------------------------

def _render_frame(
    cfg: SyntheticSequenceConfig,
    bbox: BBox,
    rng: Optional[np.random.Generator],
) -> np.ndarray:
    """Render a single BGR frame with the target at ``bbox``.

    Args:
        cfg:  Sequence config (appearance, colors, noise settings).
        bbox: Bounding box ``(x, y, w, h)`` of the target in this frame.
        rng:  RNG for pixel noise; pass ``None`` to skip noise even if
              ``cfg.add_noise`` is True.

    Returns:
        ``(H, W, 3)`` uint8 BGR image.
    """
    H, W = cfg.frame_size
    frame = np.full((H, W, 3), cfg.bg_color, dtype=np.uint8)

    x, y, w, h = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x), max(0, y)
    x2 = min(W, x + max(1, w))
    y2 = min(H, y + max(1, h))

    if x2 > x1 and y2 > y1:
        if cfg.appearance == "solid":
            frame[y1:y2, x1:x2] = cfg.target_color

        elif cfg.appearance == "checkerboard":
            tile = 8
            for row in range(y1, y2):
                for col in range(x1, x2):
                    if ((row - y1) // tile + (col - x1) // tile) % 2 == 0:
                        frame[row, col] = (0, 0, 0)
                    else:
                        frame[row, col] = (255, 255, 255)

        elif cfg.appearance == "gradient":
            grad = np.linspace(50, 230, x2 - x1, dtype=np.uint8)
            for row in range(y1, y2):
                frame[row, x1:x2, 0] = grad
                frame[row, x1:x2, 1] = grad
                frame[row, x1:x2, 2] = grad

    if cfg.add_noise and rng is not None:
        noise = rng.normal(0, cfg.noise_sigma, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return frame


# ---------------------------------------------------------------------------
# In-memory Sequence implementation
# ---------------------------------------------------------------------------

class _SyntheticSequence(Sequence):
    """A :class:`~eovot.datasets.base.Sequence` generated entirely in memory.

    Overrides ``__iter__`` to render frames lazily without any disk I/O.
    The ground-truth array is pre-computed at construction time so it is
    immediately available via ``.ground_truth`` and ``.init_bbox``.
    """

    def __init__(self, cfg: SyntheticSequenceConfig) -> None:
        self._cfg = cfg
        gt = _compute_ground_truth(cfg)
        super().__init__(
            name=cfg.name,
            frame_paths=[f"{cfg.name}_frame_{i:05d}" for i in range(cfg.n_frames)],
            ground_truth=gt,
        )

    def __iter__(self) -> Iterator[np.ndarray]:  # type: ignore[override]
        """Yield rendered BGR frames without touching the filesystem."""
        rng = np.random.default_rng(self._cfg.seed) if self._cfg.add_noise else None
        for i in range(self._cfg.n_frames):
            bbox = tuple(self.ground_truth[i])  # type: ignore[arg-type]
            yield _render_frame(self._cfg, bbox, rng)

    def __repr__(self) -> str:
        return (
            f"SyntheticSequence(name={self.name!r}, "
            f"frames={len(self)}, motion={self._cfg.motion!r})"
        )


# ---------------------------------------------------------------------------
# Public dataset class
# ---------------------------------------------------------------------------

class SyntheticDataset(BaseDataset):
    """A dataset of fully synthetic tracking sequences requiring no data files.

    Accepts a list of :class:`SyntheticSequenceConfig` objects, one per
    sequence, or can be constructed via the convenience factory methods
    :meth:`quick`, :meth:`stress_test`, and :meth:`scale_challenge`.

    Args:
        configs: One :class:`SyntheticSequenceConfig` per sequence.

    Example — factory method::

        dataset = SyntheticDataset.quick(n_sequences=5, n_frames=40)
        engine  = BenchmarkEngine(verbose=False)
        result  = engine.run(KCFTracker(), dataset, dataset_name="Synthetic")

    Example — custom sequences::

        configs = [
            SyntheticSequenceConfig(name="linear_fast", motion="linear", velocity=(5.0, 3.0)),
            SyntheticSequenceConfig(name="sine_wave",   motion="sinusoidal", amplitude=(80.0, 50.0)),
            SyntheticSequenceConfig(name="rw_seq",      motion="random_walk", seed=7),
        ]
        dataset = SyntheticDataset(configs)
    """

    def __init__(self, configs: List[SyntheticSequenceConfig]) -> None:
        self._sequences: List[_SyntheticSequence] = [
            _SyntheticSequence(cfg) for cfg in configs
        ]

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
        """Create a small dataset for quick sanity checks or CI smoke tests.

        Each sequence shares the same motion model but draws a random initial
        bbox and velocity from ``seed`` so all sequences differ.

        Args:
            n_sequences: Number of sequences. Default: ``5``.
            n_frames:    Frames per sequence. Default: ``60``.
            frame_size:  ``(H, W)`` in pixels. Default: ``(240, 320)``.
            motion:      Motion model for all sequences.
            seed:        Base RNG seed; sequence ``i`` uses ``seed + i``.

        Returns:
            Ready-to-use :class:`SyntheticDataset`.
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
        """Create a mixed-motion, mixed-appearance dataset for robustness testing.

        Cycles through all three motion models and all target appearances, with
        randomised sizes, velocities, amplitudes, and frequencies.  Every third
        sequence adds pixel noise to further stress-test tracker robustness.

        Args:
            n_sequences: Number of sequences. Default: ``10``.
            n_frames:    Frames per sequence. Default: ``100``.
            frame_size:  ``(H, W)`` in pixels. Default: ``(480, 640)``.
            seed:        Base RNG seed.

        Returns:
            :class:`SyntheticDataset` with diverse motion and appearance patterns.
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

        Alternates between growing and shrinking targets to test trackers that
        assume fixed object size (e.g. MOSSE and KCF without scale estimation).

        Args:
            n_sequences: Number of sequences. Default: ``6``.
            n_frames:    Frames per sequence. Default: ``80``.
            frame_size:  ``(H, W)`` in pixels. Default: ``(360, 480)``.
            seed:        RNG seed.

        Returns:
            :class:`SyntheticDataset` with scale-varying sinusoidal sequences.
        """
        rng = np.random.default_rng(seed)
        H, W = frame_size
        configs = []
        for i in range(n_sequences):
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
