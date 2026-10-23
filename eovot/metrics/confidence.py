"""Confidence calibration metrics for correlation-filter tracker evaluation.

Correlation-filter trackers (MOSSE, KCF) compute a response map on every
frame; the **Peak-to-Sidelobe Ratio (PSR)** of that map is an intrinsic
confidence estimate that requires no ground-truth labels at inference time.

A well-calibrated tracker is one where high PSR reliably predicts high IoU
and low PSR reliably predicts tracking failure — making PSR a practical
proxy for re-detection triggers in edge deployment pipelines.

This module measures how well PSR tracks actual accuracy:

- :class:`ConfidenceCalibrationResult` — per-sequence calibration summary.
- :class:`ConfidenceCalibrationAnalyzer` — computes Pearson / Spearman
  correlation, per-bin IoU breakdown, and the optimal PSR failure threshold.

Typical usage::

    from eovot.metrics.confidence import ConfidenceCalibrationAnalyzer

    analyzer = ConfidenceCalibrationAnalyzer()
    result = analyzer.analyze(confidences, ious, tracker_name="MOSSE", sequence_name="car1")
    print(result)
    # ConfidenceCalibration[MOSSE on car1]
    #   Pearson r=0.72  Spearman ρ=0.68
    #   high-conf mIoU=0.74  low-conf mIoU=0.23
    #   optimal PSR threshold=5.2

    threshold = analyzer.optimal_failure_threshold(confidences, ious)
    print(f"Re-detect when PSR < {threshold:.1f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ConfidenceCalibrationResult:
    """Per-sequence calibration summary between PSR confidence and IoU.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        sequence_name: Sequence identifier.
        pearson_r: Pearson correlation between PSR and IoU (linear association).
        spearman_rho: Spearman rank correlation (monotonic association,
            robust to outliers and non-linear relationships).
        high_conf_mean_iou: Mean IoU on frames where PSR ≥ *high_threshold*.
        low_conf_mean_iou: Mean IoU on frames where PSR < *low_threshold*.
        optimal_threshold: PSR value that maximises the balanced accuracy
            of the binary "tracking failure" classifier.
        n_frames: Total number of evaluated frames.
        n_high_conf: Frames classified as high confidence.
        n_low_conf: Frames classified as low confidence.
    """

    tracker_name: str
    sequence_name: str
    pearson_r: float
    spearman_rho: float
    high_conf_mean_iou: float
    low_conf_mean_iou: float
    optimal_threshold: float
    n_frames: int
    n_high_conf: int = 0
    n_low_conf: int = 0
    high_threshold: float = 7.0
    low_threshold: float = 4.0

    def __str__(self) -> str:
        return (
            f"ConfidenceCalibration[{self.tracker_name} on {self.sequence_name}]\n"
            f"  Pearson r={self.pearson_r:.3f}  Spearman ρ={self.spearman_rho:.3f}\n"
            f"  high-conf mIoU={self.high_conf_mean_iou:.3f} (n={self.n_high_conf})  "
            f"low-conf mIoU={self.low_conf_mean_iou:.3f} (n={self.n_low_conf})\n"
            f"  optimal PSR threshold={self.optimal_threshold:.2f}"
        )


class ConfidenceCalibrationAnalyzer:
    """Analyze how well PSR confidence scores predict tracking accuracy.

    Args:
        high_threshold: PSR value above which a frame is "high confidence".
            Default: 7.0 (from Bolme et al., CVPR 2010).
        low_threshold: PSR value below which a frame is "low confidence".
            Default: 4.0.
        failure_iou: IoU below which a frame is considered a tracking failure.
            Used when finding the optimal PSR threshold for failure detection.
            Default: 0.1.
        n_threshold_candidates: Number of PSR threshold candidates to evaluate
            when searching for the optimal failure threshold. Default: 100.
    """

    def __init__(
        self,
        high_threshold: float = 7.0,
        low_threshold: float = 4.0,
        failure_iou: float = 0.1,
        n_threshold_candidates: int = 100,
    ) -> None:
        if high_threshold <= low_threshold:
            raise ValueError(
                f"high_threshold ({high_threshold}) must exceed low_threshold ({low_threshold})."
            )
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.failure_iou = failure_iou
        self.n_threshold_candidates = n_threshold_candidates

    def analyze(
        self,
        confidences: np.ndarray,
        ious: np.ndarray,
        tracker_name: str = "unknown",
        sequence_name: str = "unknown",
    ) -> ConfidenceCalibrationResult:
        """Compute calibration metrics for one tracker on one sequence.

        Args:
            confidences: ``(N,)`` array of per-frame PSR values.
            ious: ``(N,)`` array of per-frame IoU values (ground-truth aligned).
            tracker_name: Label for the result object.
            sequence_name: Label for the result object.

        Returns:
            :class:`ConfidenceCalibrationResult` with all metrics populated.

        Raises:
            ValueError: If ``confidences`` and ``ious`` have different lengths
                or fewer than 2 frames.
        """
        confs = np.asarray(confidences, dtype=np.float64)
        iou_arr = np.asarray(ious, dtype=np.float64)

        if confs.shape != iou_arr.shape:
            raise ValueError(
                f"confidences and ious must have the same shape, "
                f"got {confs.shape} vs {iou_arr.shape}."
            )
        if len(confs) < 2:
            raise ValueError("At least 2 frames are required for calibration analysis.")

        pearson_r = self._pearson(confs, iou_arr)
        spearman_rho = self._spearman(confs, iou_arr)

        high_mask = confs >= self.high_threshold
        low_mask = confs < self.low_threshold
        high_iou = float(iou_arr[high_mask].mean()) if high_mask.any() else float("nan")
        low_iou = float(iou_arr[low_mask].mean()) if low_mask.any() else float("nan")

        opt_threshold = self.optimal_failure_threshold(confs, iou_arr)

        return ConfidenceCalibrationResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            pearson_r=pearson_r,
            spearman_rho=spearman_rho,
            high_conf_mean_iou=high_iou,
            low_conf_mean_iou=low_iou,
            optimal_threshold=opt_threshold,
            n_frames=int(len(confs)),
            n_high_conf=int(high_mask.sum()),
            n_low_conf=int(low_mask.sum()),
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold,
        )

    def optimal_failure_threshold(
        self,
        confidences: np.ndarray,
        ious: np.ndarray,
    ) -> float:
        """Find the PSR threshold that best separates successful from failed frames.

        Sweeps candidate PSR thresholds and selects the one that maximises
        **balanced accuracy** — the mean of sensitivity (recall on failure
        frames) and specificity (recall on success frames) — for the binary
        classification task "IoU < failure_iou".

        Args:
            confidences: ``(N,)`` array of per-frame PSR values.
            ious: ``(N,)`` array of per-frame IoU values.

        Returns:
            Optimal PSR threshold as a float.  A tracker frame with PSR below
            this value should trigger a re-detection or alert.
            Returns the median PSR when the data has no clear separation.
        """
        confs = np.asarray(confidences, dtype=np.float64)
        iou_arr = np.asarray(ious, dtype=np.float64)

        # Binary failure labels (True = failure)
        failures = iou_arr < self.failure_iou

        if failures.sum() == 0 or (~failures).sum() == 0:
            return float(np.median(confs))

        psr_min, psr_max = float(confs.min()), float(confs.max())
        if psr_min >= psr_max:
            return psr_min

        candidates = np.linspace(psr_min, psr_max, self.n_threshold_candidates)
        best_thresh = float(np.median(confs))
        best_ba = -1.0

        for thresh in candidates:
            predicted_failure = confs < thresh
            tp = (predicted_failure & failures).sum()
            tn = (~predicted_failure & ~failures).sum()
            fn = (~predicted_failure & failures).sum()
            fp = (predicted_failure & ~failures).sum()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ba = (sensitivity + specificity) / 2.0
            if ba > best_ba:
                best_ba = ba
                best_thresh = float(thresh)

        return best_thresh

    def aggregate(
        self,
        results: List[ConfidenceCalibrationResult],
    ) -> dict:
        """Aggregate calibration results across multiple sequences.

        Args:
            results: List of per-sequence :class:`ConfidenceCalibrationResult`.

        Returns:
            Dict with keys ``pearson_r``, ``spearman_rho``,
            ``high_conf_mean_iou``, ``low_conf_mean_iou``,
            ``mean_optimal_threshold``, ``n_sequences``.
        """
        if not results:
            return {}
        valid_pearson = [r.pearson_r for r in results if not np.isnan(r.pearson_r)]
        valid_spearman = [r.spearman_rho for r in results if not np.isnan(r.spearman_rho)]
        valid_high = [r.high_conf_mean_iou for r in results if not np.isnan(r.high_conf_mean_iou)]
        valid_low = [r.low_conf_mean_iou for r in results if not np.isnan(r.low_conf_mean_iou)]
        thresholds = [r.optimal_threshold for r in results]
        return {
            "pearson_r": float(np.mean(valid_pearson)) if valid_pearson else float("nan"),
            "spearman_rho": float(np.mean(valid_spearman)) if valid_spearman else float("nan"),
            "high_conf_mean_iou": float(np.mean(valid_high)) if valid_high else float("nan"),
            "low_conf_mean_iou": float(np.mean(valid_low)) if valid_low else float("nan"),
            "mean_optimal_threshold": float(np.mean(thresholds)),
            "n_sequences": len(results),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        """Pearson correlation coefficient between x and y."""
        if len(x) < 2:
            return 0.0
        xm = x - x.mean()
        ym = y - y.mean()
        denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
        if denom < 1e-12:
            return 0.0
        return float((xm * ym).sum() / denom)

    @staticmethod
    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        """Spearman rank correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        # Equivalent to Pearson correlation of rank arrays
        rxm = rx - rx.mean()
        rym = ry - ry.mean()
        denom = np.sqrt((rxm ** 2).sum() * (rym ** 2).sum())
        if denom < 1e-12:
            return 0.0
        return float((rxm * rym).sum() / denom)
