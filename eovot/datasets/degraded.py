"""Degraded dataset wrapper for edge-robustness evaluation.

Edge-deployed trackers operate under conditions that differ substantially from
the clean sequences in standard VOT benchmarks: cheap CMOS sensors introduce
Gaussian noise, fast-moving platforms cause motion blur, poor outdoor
lighting compresses dynamic range, and partial occlusions are commonplace.

This module provides :class:`DegradedDataset`, a lightweight composable wrapper
that applies parametric degradations to any :class:`~.base.BaseDataset` at
frame-read time — without writing modified frames to disk.  The same tracker
code path is exercised as in a normal benchmark run; only the pixel values change.

Design goals:

* **Zero copies of raw data** — degradations are applied on-the-fly inside
  :class:`DegradedSequence.__iter__`, so RAM consumption matches the un-wrapped
  dataset.
* **Reproducibility** — every degradation that uses randomness (noise,
  occlusion patch position) seeds its RNG from a deterministic combination of
  the sequence index, frame index, and a user-supplied global seed.
* **Composability** — multiple degradations stack in order; they are applied
  as a pipeline so the interaction effects are explicit and controllable.
* **Drop-in replacement** — :class:`DegradedDataset` is a :class:`BaseDataset`,
  so it passes directly to :class:`~eovot.benchmark.engine.BenchmarkEngine`
  without any change to the benchmark code.

Available degradations
----------------------
+---------------------+---------------------------------+---------------------+
| Degradation         | ``DegradationConfig`` field     | Default (off)       |
+=====================+=================================+=====================+
| Gaussian noise      | ``noise_std``                   | 0.0 (disabled)      |
+---------------------+---------------------------------+---------------------+
| Motion blur         | ``blur_kernel_size``            | 0 (disabled)        |
+---------------------+---------------------------------+---------------------+
| Brightness/contrast | ``brightness_alpha``,           | 1.0, 0 (identity)   |
|                     | ``brightness_beta``             |                     |
+---------------------+---------------------------------+---------------------+
| Partial occlusion   | ``occlusion_fraction``          | 0.0 (disabled)      |
+---------------------+---------------------------------+---------------------+

Usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.datasets.degraded import DegradedDataset, DegradationConfig
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.adaptive_kcf import AdaptiveKCFTracker

    cfg = DegradationConfig(noise_std=15.0, blur_kernel_size=5)
    ds  = DegradedDataset(SyntheticDataset(num_sequences=5, num_frames=100), cfg)
    result = BenchmarkEngine(verbose=False).run(
        AdaptiveKCFTracker(psr_threshold=7.0), ds, dataset_name="Synthetic-Degraded"
    )
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import cv2
import numpy as np

from .base import BaseDataset, Sequence


@dataclass
class DegradationConfig:
    """Parameters controlling which degradations are applied and how strongly.

    All degradations default to their identity / disabled state, so an
    un-modified config is a no-op wrapper.

    Args:
        noise_std:         Standard deviation of additive zero-mean Gaussian
                           noise in the range ``[0, 255]``.  ``0.0`` disables.
        blur_kernel_size:  Side length (odd integer ≥ 3) of the motion-blur
                           kernel applied horizontally.  ``0`` disables.
        brightness_alpha:  Contrast scaling factor applied via
                           ``alpha * pixel + beta``.  ``1.0`` = identity.
        brightness_beta:   Additive brightness offset (−255 … 255).  ``0`` = identity.
        occlusion_fraction: Fraction of the frame area covered by a single
                            grey occlusion patch, in ``[0.0, 1.0)``.
                            The patch is randomly placed on each frame.
                            ``0.0`` disables.
        seed:              Base seed for all random degradations.  Combined
                           with the sequence and frame index to ensure
                           frame-level reproducibility.
    """

    noise_std: float = 0.0
    blur_kernel_size: int = 0
    brightness_alpha: float = 1.0
    brightness_beta: float = 0.0
    occlusion_fraction: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.noise_std < 0.0:
            raise ValueError(f"noise_std must be ≥ 0, got {self.noise_std}")
        if self.blur_kernel_size != 0:
            if self.blur_kernel_size < 3 or self.blur_kernel_size % 2 == 0:
                raise ValueError(
                    f"blur_kernel_size must be an odd integer ≥ 3 or 0, "
                    f"got {self.blur_kernel_size}"
                )
        if not 0.0 <= self.occlusion_fraction < 1.0:
            raise ValueError(
                f"occlusion_fraction must be in [0, 1), got {self.occlusion_fraction}"
            )
        if self.brightness_alpha <= 0.0:
            raise ValueError(
                f"brightness_alpha must be positive, got {self.brightness_alpha}"
            )

    @property
    def is_identity(self) -> bool:
        """True if this config applies no degradation."""
        return (
            self.noise_std == 0.0
            and self.blur_kernel_size == 0
            and self.brightness_alpha == 1.0
            and self.brightness_beta == 0.0
            and self.occlusion_fraction == 0.0
        )


class DegradedSequence(Sequence):
    """A :class:`~.base.Sequence` that applies degradations on frame iteration.

    Wraps a source sequence and applies the configured degradations to each
    frame as it is yielded.  Ground-truth boxes are passed through unchanged
    (degradations are purely visual; the target position is unaffected).

    Args:
        source:    Original sequence to wrap.
        config:    Degradation configuration.
        seq_index: Index of this sequence within the dataset (used to seed
                   per-frame RNGs deterministically).
    """

    def __init__(
        self,
        source: Sequence,
        config: DegradationConfig,
        seq_index: int = 0,
    ) -> None:
        super().__init__(
            name=source.name,
            frame_paths=["<degraded>"] * len(source),
            ground_truth=source.ground_truth.copy(),
        )
        self._source = source
        self._config = config
        self._seq_index = seq_index

    def __len__(self) -> int:
        return len(self._source)

    def __iter__(self) -> Iterator[np.ndarray]:
        cfg = self._config
        for frame_idx, frame in enumerate(self._source):
            yield self._degrade(frame, frame_idx, cfg)

    def _degrade(
        self, frame: np.ndarray, frame_idx: int, cfg: DegradationConfig
    ) -> np.ndarray:
        """Apply the full degradation pipeline to one frame."""
        if cfg.is_identity:
            return frame

        out = frame.astype(np.float32)

        if cfg.blur_kernel_size > 0:
            k = cfg.blur_kernel_size
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            out = cv2.filter2D(out, -1, kernel)

        if cfg.brightness_alpha != 1.0 or cfg.brightness_beta != 0.0:
            out = cfg.brightness_alpha * out + cfg.brightness_beta

        if cfg.noise_std > 0.0:
            rng = np.random.default_rng(
                cfg.seed * 100_000 + self._seq_index * 10_000 + frame_idx
            )
            out = out + rng.normal(0.0, cfg.noise_std, out.shape).astype(np.float32)

        out = np.clip(out, 0.0, 255.0).astype(np.uint8)

        if cfg.occlusion_fraction > 0.0:
            out = self._add_occlusion(out, frame_idx, cfg)

        return out

    def _add_occlusion(
        self, frame: np.ndarray, frame_idx: int, cfg: DegradationConfig
    ) -> np.ndarray:
        """Overlay a randomly-placed grey rectangle on the frame."""
        h, w = frame.shape[:2]
        total_area = h * w
        patch_area = cfg.occlusion_fraction * total_area

        rng = np.random.default_rng(
            cfg.seed * 200_000 + self._seq_index * 10_000 + frame_idx + 1
        )
        aspect = float(rng.uniform(0.5, 2.0))
        ph = int(round(np.sqrt(patch_area / aspect)))
        pw = int(round(ph * aspect))
        ph = max(1, min(ph, h))
        pw = max(1, min(pw, w))

        py = int(rng.integers(0, max(1, h - ph + 1)))
        px = int(rng.integers(0, max(1, w - pw + 1)))

        out = frame.copy()
        grey = int(rng.integers(60, 130))
        out[py : py + ph, px : px + pw] = grey
        return out


class DegradedDataset(BaseDataset):
    """Drop-in :class:`~.base.BaseDataset` wrapper that degrades frames on the fly.

    Wraps any dataset and applies a shared :class:`DegradationConfig` to every
    sequence it returns.  The wrapper is transparent to
    :class:`~eovot.benchmark.engine.BenchmarkEngine` — pass it wherever you
    would pass the original dataset.

    Args:
        source: Any :class:`~.base.BaseDataset` instance.
        config: Degradation parameters.  Defaults to an identity no-op if omitted.

    Example::

        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.datasets.degraded import DegradedDataset, DegradationConfig

        base = SyntheticDataset(num_sequences=10, num_frames=100)
        noisy = DegradedDataset(base, DegradationConfig(noise_std=20.0))
        blurry = DegradedDataset(base, DegradationConfig(blur_kernel_size=7))
        dark = DegradedDataset(base, DegradationConfig(brightness_alpha=0.4, brightness_beta=-30))
    """

    def __init__(
        self,
        source: BaseDataset,
        config: Optional[DegradationConfig] = None,
    ) -> None:
        self._source = source
        self._config = config if config is not None else DegradationConfig()

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, idx: int) -> DegradedSequence:
        seq = self._source[idx]
        return DegradedSequence(seq, self._config, seq_index=idx)

    def __repr__(self) -> str:
        return (
            f"DegradedDataset(source={self._source!r}, "
            f"noise_std={self._config.noise_std}, "
            f"blur_kernel_size={self._config.blur_kernel_size}, "
            f"occlusion_fraction={self._config.occlusion_fraction})"
        )
