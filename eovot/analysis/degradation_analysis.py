"""Input degradation robustness analysis for edge-deployment evaluation.

Edge devices receive image inputs under conditions that differ significantly
from clean academic benchmarks: low-bandwidth streams introduce JPEG
compression, outdoor cameras suffer motion blur, low-light environments
reduce contrast, and cheap sensors add Gaussian noise.

This module quantifies how much each of these degradations reduces a tracker's
accuracy, producing a **degradation robustness curve** and a single scalar
**Degradation Tolerance Score (DTS)** that summarises resilience across
the full severity range.

Degradation types
-----------------
- ``GAUSSIAN_NOISE``       — additive zero-mean Gaussian noise (σ: 0 → 50 px std)
- ``GAUSSIAN_BLUR``        — Gaussian smoothing (kernel: 1 → 21 px, odd)
- ``JPEG_COMPRESSION``     — JPEG re-encode at reducing quality (100 → 10)
- ``BRIGHTNESS_REDUCTION`` — multiplicative darkening (1.0 → 0.1 ×)

Degradation Tolerance Score (DTS)
----------------------------------
For a tracker evaluated at severities [s_0=0, s_1, …, s_K]:

    DTS = AUC(severity → mean_IoU) / AUC_clean

where AUC is computed via the trapezoidal rule.  ``DTS ∈ [0, 1]``.
A tracker with DTS=1.0 maintains full accuracy at all severities.
A tracker with DTS=0.5 loses half its IoU integrated over the severity range.

Typical usage::

    from eovot.analysis.degradation_analysis import (
        DegradationRobustnessAnalyzer, DegradationType
    )
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker

    dataset  = SyntheticDataset(num_sequences=3, num_frames=60)
    tracker  = MOSSETracker()
    analyzer = DegradationRobustnessAnalyzer()

    reports = analyzer.analyze(tracker, dataset)
    for r in reports:
        print(r)

    print(analyzer.to_markdown_table(reports))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import MetricsEngine


# ---------------------------------------------------------------------------
# Degradation type enum
# ---------------------------------------------------------------------------

class DegradationType(str, Enum):
    """Supported image degradation types."""

    GAUSSIAN_NOISE = "gaussian_noise"
    GAUSSIAN_BLUR = "gaussian_blur"
    JPEG_COMPRESSION = "jpeg_compression"
    BRIGHTNESS_REDUCTION = "brightness_reduction"

    def display_name(self) -> str:
        return {
            DegradationType.GAUSSIAN_NOISE: "Gaussian Noise",
            DegradationType.GAUSSIAN_BLUR: "Gaussian Blur",
            DegradationType.JPEG_COMPRESSION: "JPEG Compression",
            DegradationType.BRIGHTNESS_REDUCTION: "Brightness Reduction",
        }[self]


# ---------------------------------------------------------------------------
# Pipeline — applies one degradation at one severity level
# ---------------------------------------------------------------------------

class ImageDegradationPipeline:
    """Apply controlled image degradations for robustness evaluation.

    Each degradation is parameterised by a ``severity`` value in ``[0, 1]``
    where ``0`` means no degradation (clean input) and ``1`` is the maximum
    degradation modelled.

    Args:
        rng_seed: Seed for the random-number generator used by
            ``GAUSSIAN_NOISE``.  Pass a fixed seed for reproducible results.
            Default: ``42``.
    """

    def __init__(self, rng_seed: int = 42) -> None:
        self._rng = np.random.default_rng(rng_seed)

    def apply(
        self,
        frame: np.ndarray,
        deg_type: DegradationType,
        severity: float,
    ) -> np.ndarray:
        """Return a degraded copy of *frame* at the given *severity*.

        Args:
            frame:    BGR uint8 image of shape ``(H, W, 3)``.
            deg_type: Which degradation to apply.
            severity: Degradation strength in ``[0, 1]``.  ``0`` returns an
                      unchanged copy; ``1`` applies the maximum modelled
                      degradation.

        Returns:
            Degraded BGR uint8 image, same shape as *frame*.
        """
        if severity <= 0.0:
            return frame.copy()
        severity = float(np.clip(severity, 0.0, 1.0))

        if deg_type == DegradationType.GAUSSIAN_NOISE:
            return self._gaussian_noise(frame, severity)
        if deg_type == DegradationType.GAUSSIAN_BLUR:
            return self._gaussian_blur(frame, severity)
        if deg_type == DegradationType.JPEG_COMPRESSION:
            return self._jpeg_compression(frame, severity)
        if deg_type == DegradationType.BRIGHTNESS_REDUCTION:
            return self._brightness_reduction(frame, severity)
        return frame.copy()

    # ------------------------------------------------------------------
    # Private degradation implementations
    # ------------------------------------------------------------------

    def _gaussian_noise(self, frame: np.ndarray, severity: float) -> np.ndarray:
        sigma = severity * 50.0  # 0 → 50 px std
        noise = self._rng.normal(0.0, sigma, frame.shape).astype(np.float32)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _gaussian_blur(self, frame: np.ndarray, severity: float) -> np.ndarray:
        # Map severity → odd kernel size in [1, 21]
        ksize = max(1, int(2 * round(severity * 10) + 1))
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.GaussianBlur(frame, (ksize, ksize), 0)

    def _jpeg_compression(self, frame: np.ndarray, severity: float) -> np.ndarray:
        quality = max(1, int(100 - severity * 90))  # 100 → 10
        _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _brightness_reduction(self, frame: np.ndarray, severity: float) -> np.ndarray:
        factor = 1.0 - severity * 0.9  # 1.0 → 0.1
        return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Results dataclass
# ---------------------------------------------------------------------------

@dataclass
class DegradationSweepResult:
    """Robustness curve for one tracker × one degradation type.

    Attributes:
        tracker_name:    Human-readable tracker identifier.
        degradation_type: Which degradation was applied.
        severities:      Severity levels evaluated (including 0 = clean).
        mean_ious:       Mean IoU at each severity, same length as *severities*.
        dts:             Degradation Tolerance Score — normalised AUC of the
                         severity → mean_IoU curve.  Range ``[0, 1]``; higher
                         is more robust.
        clean_iou:       Mean IoU at severity 0 (baseline with no degradation).
        half_life:       Severity at which mean_IoU first drops below
                         ``0.5 × clean_iou``.  ``None`` if IoU never halves.
    """

    tracker_name: str
    degradation_type: DegradationType
    severities: List[float]
    mean_ious: List[float]
    dts: float
    clean_iou: float
    half_life: Optional[float] = None

    def __str__(self) -> str:
        hl = f"{self.half_life:.2f}" if self.half_life is not None else "never"
        return (
            f"DegradationSweepResult[{self.tracker_name}  "
            f"{self.degradation_type.display_name()}]  "
            f"clean_IoU={self.clean_iou:.4f}  DTS={self.dts:.4f}  "
            f"half_life_severity={hl}"
        )


@dataclass
class DegradationReport:
    """Full degradation robustness report for one tracker across all degradation types.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        sweeps:       One :class:`DegradationSweepResult` per degradation type.
        mean_dts:     Macro-average DTS across all degradation types.
    """

    tracker_name: str
    sweeps: List[DegradationSweepResult] = field(default_factory=list)

    @property
    def mean_dts(self) -> float:
        if not self.sweeps:
            return 0.0
        return float(np.mean([s.dts for s in self.sweeps]))

    def __str__(self) -> str:
        lines = [f"DegradationReport[{self.tracker_name}]  mean_DTS={self.mean_dts:.4f}"]
        for s in self.sweeps:
            lines.append(f"  {s}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class DegradationRobustnessAnalyzer:
    """Run a tracker at multiple image degradation levels and report robustness.

    Args:
        severities: Severity levels to evaluate, including ``0.0`` (clean).
            Default: ``[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]``.
        degradation_types: Subset of :class:`DegradationType` values to test.
            Default: all four types.
        rng_seed: Seed for :class:`ImageDegradationPipeline`.  Fixed for
            reproducibility.  Default: ``42``.

    Example::

        analyzer = DegradationRobustnessAnalyzer(
            severities=[0.0, 0.25, 0.5, 0.75, 1.0],
            degradation_types=[DegradationType.GAUSSIAN_NOISE],
        )
        reports = analyzer.analyze(tracker, dataset)
    """

    _DEFAULT_SEVERITIES = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    def __init__(
        self,
        severities: Optional[List[float]] = None,
        degradation_types: Optional[List[DegradationType]] = None,
        rng_seed: int = 42,
    ) -> None:
        self.severities = severities if severities is not None else self._DEFAULT_SEVERITIES
        self.degradation_types = (
            degradation_types if degradation_types is not None else list(DegradationType)
        )
        self._pipeline = ImageDegradationPipeline(rng_seed=rng_seed)
        self._metrics = MetricsEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        tracker,
        dataset: BaseDataset,
        tracker_name: Optional[str] = None,
        verbose: bool = False,
    ) -> DegradationReport:
        """Run the full degradation sweep and return a :class:`DegradationReport`.

        For each degradation type × severity combination, the tracker is
        re-initialized from scratch on every sequence (clean initial frame)
        and evaluated on the degraded stream of subsequent frames.  This
        isolates the effect of degradation on tracking rather than
        initialization.

        Args:
            tracker:      Any :class:`~eovot.trackers.base.BaseTracker` instance.
            dataset:      Dataset providing sequences to evaluate on.
            tracker_name: Optional display name override.  Defaults to
                          ``tracker.name``.
            verbose:      Print progress to stdout.  Default: ``False``.

        Returns:
            :class:`DegradationReport` with one sweep per degradation type.
        """
        name = tracker_name or getattr(tracker, "name", repr(tracker))
        report = DegradationReport(tracker_name=name)

        for deg_type in self.degradation_types:
            if verbose:
                print(f"  [{name}] {deg_type.display_name()} …", flush=True)
            sweep = self._sweep_degradation(tracker, dataset, name, deg_type, verbose)
            report.sweeps.append(sweep)

        return report

    def compare(
        self,
        trackers,
        dataset: BaseDataset,
        verbose: bool = False,
    ) -> List[DegradationReport]:
        """Convenience wrapper: analyze multiple trackers and return sorted reports.

        Args:
            trackers: Iterable of tracker instances.
            dataset:  Shared evaluation dataset.
            verbose:  Print progress.

        Returns:
            List of :class:`DegradationReport`, sorted by ``mean_dts`` descending.
        """
        reports = [self.analyze(t, dataset, verbose=verbose) for t in trackers]
        reports.sort(key=lambda r: r.mean_dts, reverse=True)
        return reports

    def to_markdown_table(self, reports: List[DegradationReport]) -> str:
        """Format robustness results as a Markdown table.

        Columns: Tracker | <DegradationType> … | Mean DTS

        Args:
            reports: Output of :meth:`analyze` or :meth:`compare`.

        Returns:
            Multi-line Markdown string ready to embed in reports or READMEs.
        """
        type_names = [dt.display_name() for dt in self.degradation_types]
        header_cols = ["Tracker"] + type_names + ["Mean DTS"]
        sep_cols = ["-------"] + ["---:" for _ in type_names] + ["--------:"]
        lines = [
            "| " + " | ".join(header_cols) + " |",
            "| " + " | ".join(sep_cols) + " |",
        ]
        for report in reports:
            dts_by_type: Dict[str, str] = {}
            for sweep in report.sweeps:
                dts_by_type[sweep.degradation_type.display_name()] = f"{sweep.dts:.4f}"
            row = [report.tracker_name]
            for tn in type_names:
                row.append(dts_by_type.get(tn, "—"))
            row.append(f"{report.mean_dts:.4f}")
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sweep_degradation(
        self,
        tracker,
        dataset: BaseDataset,
        tracker_name: str,
        deg_type: DegradationType,
        verbose: bool,
    ) -> DegradationSweepResult:
        """Evaluate the tracker at every configured severity for one degradation type."""
        mean_ious: List[float] = []

        for severity in self.severities:
            iou = self._mean_iou_at_severity(tracker, dataset, deg_type, severity)
            mean_ious.append(iou)
            if verbose:
                print(f"      severity={severity:.2f}  mIoU={iou:.4f}", flush=True)

        clean_iou = mean_ious[0] if mean_ious else 0.0
        dts = self._compute_dts(self.severities, mean_ious, clean_iou)
        half_life = self._find_half_life(self.severities, mean_ious, clean_iou)

        return DegradationSweepResult(
            tracker_name=tracker_name,
            degradation_type=deg_type,
            severities=list(self.severities),
            mean_ious=mean_ious,
            dts=dts,
            clean_iou=clean_iou,
            half_life=half_life,
        )

    def _mean_iou_at_severity(
        self,
        tracker,
        dataset: BaseDataset,
        deg_type: DegradationType,
        severity: float,
    ) -> float:
        """Return mean IoU across the entire dataset at one severity level."""
        all_ious: List[np.ndarray] = []

        for seq in dataset:
            ious = self._run_sequence(tracker, seq, deg_type, severity)
            if len(ious) > 0:
                all_ious.append(ious)

        if not all_ious:
            return 0.0
        combined = np.concatenate(all_ious)
        return float(combined.mean())

    def _run_sequence(
        self,
        tracker,
        seq: Sequence,
        deg_type: DegradationType,
        severity: float,
    ) -> np.ndarray:
        """Run the tracker on a degraded version of *seq* and return per-frame IoUs."""
        frames = list(seq)
        gt = seq.ground_truth
        preds: List = []

        for i, frame in enumerate(frames):
            # Always initialize on the clean frame so degradation affects tracking
            # continuity, not just initialization quality.
            if i == 0:
                tracker.initialize(frame, seq.init_bbox)
                preds.append(seq.init_bbox)
            else:
                degraded = self._pipeline.apply(frame, deg_type, severity)
                bbox = tracker.update(degraded)
                preds.append(bbox)

        preds_arr = np.array(preds, dtype=np.float64)
        n = min(len(preds_arr), len(gt))
        return self._metrics.batch_iou(preds_arr[:n], gt[:n])

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dts(
        severities: List[float],
        mean_ious: List[float],
        clean_iou: float,
    ) -> float:
        """Compute the Degradation Tolerance Score via trapezoidal AUC.

        DTS = AUC(severity → mean_IoU) / max_possible_AUC

        where max_possible_AUC = clean_IoU × severity_range.
        A tracker with DTS=1.0 maintains full clean accuracy at all severities.
        """
        if len(severities) < 2 or clean_iou <= 0:
            return 0.0
        s = np.array(severities, dtype=np.float64)
        iou = np.array(mean_ious, dtype=np.float64)

        try:
            auc = float(np.trapezoid(iou, s))  # numpy >= 2.0
        except AttributeError:
            auc = float(np.trapz(iou, s))  # numpy < 2.0

        max_auc = clean_iou * (s[-1] - s[0])
        return float(np.clip(auc / max_auc, 0.0, 1.0)) if max_auc > 0 else 0.0

    @staticmethod
    def _find_half_life(
        severities: List[float],
        mean_ious: List[float],
        clean_iou: float,
    ) -> Optional[float]:
        """Return the severity at which IoU first drops below 50 % of clean_iou.

        Uses linear interpolation between the two surrounding severity points.
        Returns ``None`` if IoU never drops to half, or if clean_iou is 0.
        """
        if clean_iou <= 0:
            return None
        threshold = 0.5 * clean_iou
        for i in range(1, len(mean_ious)):
            if mean_ious[i] <= threshold:
                # Linear interpolation between severities[i-1] and severities[i]
                s0, s1 = severities[i - 1], severities[i]
                v0, v1 = mean_ious[i - 1], mean_ious[i]
                if abs(v1 - v0) < 1e-9:
                    return float(s0)
                t = (threshold - v0) / (v1 - v0)
                return float(s0 + t * (s1 - s0))
        return None
