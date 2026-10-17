"""Image degradation pipeline for robustness stress testing.

Wraps any :class:`~eovot.datasets.base.Sequence`-compatible object and
applies systematic visual degradations on-the-fly, enabling trackers to
be evaluated under edge-camera conditions without creating copies of
dataset files.

Supported degradation types
----------------------------
============== ===============================================================
Type           Severity->Parameter mapping (severity in [0, 1])
============== ===============================================================
gaussian_noise sigma: 0 -> 50 pixel intensity units
motion_blur    horizontal kernel length: 1 -> 31 pixels (always odd)
jpeg           JPEG quality: 95 -> 10 (lower quality = more artifact)
brightness     additive bias: -80 -> +80 intensity units (0.5 = neutral)
salt_pepper    corrupt fraction: 0 -> 20% of pixels
pixelation     block size: 1 -> 32 pixels
============== ===============================================================

Example::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.datasets.degradation import (
        DegradationConfig, DegradationType, ImageDegrader, DegradedSequence
    )

    ds = SyntheticDataset(num_sequences=1, num_frames=50)
    seq = ds[0]

    cfg = DegradationConfig(DegradationType.GAUSSIAN_NOISE, severity=0.5)
    degrader = ImageDegrader([cfg], rng_seed=42)
    degraded = DegradedSequence(seq, degrader)

    # Pass directly to BenchmarkEngine -- interface is duck-type compatible.
    for frame in degraded:
        pass  # frame is degraded uint8 BGR
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .base import Sequence


class DegradationType(str, Enum):
    """Supported image degradation types."""

    GAUSSIAN_NOISE = "gaussian_noise"
    MOTION_BLUR = "motion_blur"
    JPEG_COMPRESSION = "jpeg_compression"
    BRIGHTNESS = "brightness"
    SALT_PEPPER = "salt_pepper"
    PIXELATION = "pixelation"


@dataclass
class DegradationConfig:
    """Configuration for one degradation effect.

    Args:
        degradation_type: Which effect to apply.
        severity: Normalised intensity in ``[0, 1]``.
            0 = no effect; 1 = maximum effect for this type.
        enabled: When ``False`` this entry is silently skipped.

    Raises:
        ValueError: If ``severity`` is outside ``[0, 1]``.
    """

    degradation_type: DegradationType
    severity: float = 0.5
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(
                f"severity must be in [0, 1], got {self.severity!r}"
            )


class ImageDegrader:
    """Apply a chain of :class:`DegradationConfig` effects to BGR frames.

    All math is performed in ``float32`` to avoid uint8 overflow/wrap.
    The final result is clipped and cast back to ``uint8``.

    Severity-to-parameter mapping per type:

    ================== =====================================================
    Type               Range at severity 0 .. 1
    ================== =====================================================
    GAUSSIAN_NOISE     sigma 0 .. 50 intensity units
    MOTION_BLUR        kernel 1 .. 31 px (horizontal, always odd)
    JPEG_COMPRESSION   quality 95 .. 10
    BRIGHTNESS         offset -80 .. +80 intensity units (0.5 = neutral)
    SALT_PEPPER        corrupt fraction 0 .. 20% of pixels
    PIXELATION         block size 1 .. 32 px
    ================== =====================================================

    Args:
        configs: Effects to apply, in order.  Disabled configs are skipped.
        rng_seed: Seed for reproducible noise generation.
            ``None`` gives non-reproducible randomness.

    Example::

        degrader = ImageDegrader([
            DegradationConfig(DegradationType.GAUSSIAN_NOISE, severity=0.3),
            DegradationConfig(DegradationType.MOTION_BLUR, severity=0.4),
        ], rng_seed=0)
        out = degrader.apply(frame)
    """

    _SIGMA_MAX = 50.0
    _KERNEL_MAX_ODD = 31
    _JPEG_QUALITY_MIN = 10
    _JPEG_QUALITY_MAX = 95
    _BRIGHTNESS_AMP = 80.0
    _SP_FRAC_MAX = 0.20
    _PIXEL_MAX = 32

    def __init__(
        self,
        configs: List[DegradationConfig],
        rng_seed: Optional[int] = None,
    ) -> None:
        self._configs = [c for c in configs if c.enabled]
        self._rng = np.random.default_rng(rng_seed)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply all enabled degradations and return a ``uint8`` BGR image.

        Args:
            frame: Input BGR image, shape ``(H, W, 3)``, dtype uint8.

        Returns:
            Degraded BGR image, same shape and dtype.
        """
        out = frame.astype(np.float32)
        for cfg in self._configs:
            out = self._apply_one(out, cfg)
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def _apply_one(self, frame: np.ndarray, cfg: DegradationConfig) -> np.ndarray:
        s = cfg.severity
        t = cfg.degradation_type

        if t == DegradationType.GAUSSIAN_NOISE:
            sigma = s * self._SIGMA_MAX
            if sigma < 1e-4:
                return frame
            noise = self._rng.normal(0.0, sigma, frame.shape).astype(np.float32)
            return frame + noise

        if t == DegradationType.MOTION_BLUR:
            k = max(1, int(round(s * self._KERNEL_MAX_ODD)))
            if k % 2 == 0:
                k += 1
            if k <= 1:
                return frame
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            return cv2.filter2D(frame, -1, kernel)

        if t == DegradationType.JPEG_COMPRESSION:
            quality = int(
                self._JPEG_QUALITY_MAX
                - s * (self._JPEG_QUALITY_MAX - self._JPEG_QUALITY_MIN)
            )
            quality = max(self._JPEG_QUALITY_MIN, min(self._JPEG_QUALITY_MAX, quality))
            uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            ok, buf = cv2.imencode(".jpg", uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok or buf is None:
                return frame
            decoded = cv2.imdecode(
                np.frombuffer(buf.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if decoded is None:
                return frame
            return decoded.astype(np.float32)

        if t == DegradationType.BRIGHTNESS:
            offset = (s - 0.5) * 2.0 * self._BRIGHTNESS_AMP
            return frame + offset

        if t == DegradationType.SALT_PEPPER:
            frac = s * self._SP_FRAC_MAX
            out = frame.copy()
            h, w = frame.shape[:2]
            n = max(0, int(frac * h * w))
            if n > 0:
                sr = self._rng.integers(0, h, n)
                sc = self._rng.integers(0, w, n)
                out[sr, sc] = 255.0
                pr = self._rng.integers(0, h, n)
                pc = self._rng.integers(0, w, n)
                out[pr, pc] = 0.0
            return out

        if t == DegradationType.PIXELATION:
            block = max(1, int(round(s * self._PIXEL_MAX)))
            if block <= 1:
                return frame
            h, w = frame.shape[:2]
            small_w = max(1, w // block)
            small_h = max(1, h // block)
            uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            small = cv2.resize(uint8, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST).astype(
                np.float32
            )

        return frame  # unknown type -> no-op


class DegradedSequence:
    """Sequence wrapper that applies :class:`ImageDegrader` to each frame.

    Ground-truth boxes are unchanged -- degradations affect appearance only.
    The class is duck-type compatible with
    :class:`~eovot.datasets.base.Sequence` so it can be passed directly to
    :class:`~eovot.benchmark.engine.BenchmarkEngine`.

    Args:
        sequence: Any object with ``__iter__``, ``ground_truth``,
            ``init_bbox``, ``name``, and ``__len__`` attributes
            (a :class:`~eovot.datasets.base.Sequence` or derivative).
        degrader: :class:`ImageDegrader` applied to every frame on iteration.

    Example::

        degraded = DegradedSequence(seq, degrader)
        # Can be evaluated directly:
        result = engine._run_sequence(tracker, degraded)
    """

    def __init__(self, sequence: Sequence, degrader: ImageDegrader) -> None:
        self._sequence = sequence
        self._degrader = degrader

    @property
    def name(self) -> str:
        types = "+".join(c.degradation_type.value for c in self._degrader._configs)
        return f"{self._sequence.name}[{types}]" if types else self._sequence.name

    @property
    def ground_truth(self) -> np.ndarray:
        return self._sequence.ground_truth

    @property
    def init_bbox(self):
        return self._sequence.init_bbox

    def __len__(self) -> int:
        return len(self._sequence)

    def __iter__(self) -> Iterator[np.ndarray]:
        for frame in self._sequence:
            yield self._degrader.apply(frame)

    def __repr__(self) -> str:
        return f"DegradedSequence({self.name!r}, frames={len(self)})"


@dataclass
class SweepPoint:
    """One result from a :class:`DegradationSweep` run."""

    degradation_type: str
    severity: float
    mean_iou: float
    mean_fps: float

    def to_dict(self) -> Dict:
        return {
            "degradation_type": self.degradation_type,
            "severity": round(self.severity, 3),
            "mean_iou": round(self.mean_iou, 4),
            "mean_fps": round(self.mean_fps, 2),
        }


class DegradationSweep:
    """Sweep degradation severity levels and measure accuracy / speed impact.

    For each :class:`DegradationType` in scope, creates a
    :class:`DegradedSequence` at each severity level and calls a user-supplied
    evaluation function, collecting the resulting accuracy and speed.

    Args:
        severities: Severity levels to sweep.
            Default: ``[0.0, 0.25, 0.5, 0.75, 1.0]``.
        types: Degradation types to include.  Default: all six types.
        rng_seed: Seed for :class:`ImageDegrader` so noise is reproducible.

    Example::

        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker

        ds = SyntheticDataset(num_sequences=1, num_frames=50)
        seq = ds[0]
        engine = BenchmarkEngine(verbose=False)
        tracker = MOSSETracker()

        def eval_fn(degraded_seq):
            ious, fps = [], 0.0
            # Minimal inline eval without BenchmarkEngine wrapper:
            frames = list(degraded_seq)
            gt = degraded_seq.ground_truth
            tracker.initialize(frames[0], degraded_seq.init_bbox)
            for frame in frames[1:]:
                tracker.update(frame)
            return 0.8, 120.0  # placeholder

        sweep = DegradationSweep(severities=[0.0, 0.5, 1.0])
        pts = sweep.run(seq, eval_fn)
        print(sweep.to_markdown(pts))
    """

    def __init__(
        self,
        severities: Optional[List[float]] = None,
        types: Optional[List[DegradationType]] = None,
        rng_seed: Optional[int] = 42,
    ) -> None:
        self.severities = list(severities or [0.0, 0.25, 0.5, 0.75, 1.0])
        self.types = list(types or DegradationType)
        self.rng_seed = rng_seed

    def run(
        self,
        sequence: Sequence,
        eval_fn: Callable[[DegradedSequence], Tuple[float, float]],
    ) -> List[SweepPoint]:
        """Run the sweep over all (type, severity) combinations.

        Args:
            sequence: Clean source sequence.
            eval_fn: Callable that receives a :class:`DegradedSequence`
                and returns ``(mean_iou, mean_fps)``.

        Returns:
            Ordered list of :class:`SweepPoint` objects --
            one per (type, severity) pair.
        """
        results: List[SweepPoint] = []
        for deg_type in self.types:
            for severity in self.severities:
                cfg = DegradationConfig(deg_type, severity=severity)
                degrader = ImageDegrader([cfg], rng_seed=self.rng_seed)
                degraded_seq = DegradedSequence(sequence, degrader)
                mean_iou, mean_fps = eval_fn(degraded_seq)
                results.append(
                    SweepPoint(
                        degradation_type=deg_type.value,
                        severity=severity,
                        mean_iou=mean_iou,
                        mean_fps=mean_fps,
                    )
                )
        return results

    @staticmethod
    def to_markdown(results: List[SweepPoint]) -> str:
        """Format *results* as a Markdown table.

        Args:
            results: Output of :meth:`run`.

        Returns:
            Markdown table string ready for inclusion in reports.
        """
        header = "| Degradation        | Severity | mIoU   |     FPS |"
        sep    = "|--------------------|----------|--------|---------|"
        rows = [header, sep]
        for pt in results:
            rows.append(
                f"| {pt.degradation_type:<18} | {pt.severity:>8.2f} "
                f"| {pt.mean_iou:>6.4f} | {pt.mean_fps:>7.1f} |"
            )
        return "\n".join(rows)
