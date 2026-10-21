"""Tracker reliability analysis: confidence calibration and score–IoU correlation.

A well-calibrated tracker reports high confidence when it is accurate (high IoU)
and low confidence when it has drifted.  This module quantifies that property so
researchers can compare tracker confidence quality — a critical metric for edge
deployment, where the system must decide *when* to trigger re-detection without
constant human supervision.

Metrics
-------
* **Spearman rank correlation** — rank correlation between per-frame confidence
  scores and IoU values.  A value > 0.5 indicates the score is a useful proxy
  for tracking accuracy.
* **Calibration curve** — reliability diagram: confidence bins vs. mean IoU
  within each bin.  A perfectly calibrated tracker lies on the diagonal.
* **Expected Calibration Error (ECE)** — weighted mean absolute deviation
  between bin confidence midpoint and bin mean IoU.  Lower is better.
* **Optimal threshold** — confidence threshold that maximises the F1-like
  detection of "good tracking" frames (IoU ≥ ``iou_good_threshold``).

These metrics are intentionally compatible with
:class:`~eovot.trackers.confident_base.ConfidentBaseTracker` output: the
``scores`` argument should be the per-frame confidence values returned by
:meth:`~eovot.trackers.confident_base.ConfidentBaseTracker.update_with_score`.

Typical usage::

    from eovot.metrics.reliability import TrackerReliabilityAnalyzer

    analyzer = TrackerReliabilityAnalyzer()
    scores = [0.9, 0.8, 0.3, 0.1, 0.7]   # confidence per frame
    ious   = [0.85, 0.75, 0.2, 0.05, 0.7] # ground-truth IoU per frame
    result = analyzer.analyze(scores, ious, tracker_name="PSR-MOSSE")
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ReliabilityResult:
    """Reliability analysis summary for a single tracker.

    Attributes:
        tracker_name:       Human-readable tracker identifier.
        spearman_r:         Spearman rank correlation between confidence and IoU.
                            Range ``[-1, 1]``; higher is better.
        ece:                Expected Calibration Error.  Lower is better, 0 = perfect.
        optimal_threshold:  Confidence threshold maximising F1 for detecting
                            "good tracking" frames (IoU ≥ ``iou_good_threshold``).
        f1_at_optimal:      F1 score at ``optimal_threshold``.
        bin_confidences:    Midpoint confidence for each calibration bin.
        bin_accuracies:     Mean IoU within each calibration bin.
        bin_weights:        Fraction of frames in each bin (sums to 1).
        iou_good_threshold: IoU above which a frame is "accurately tracked".
        n_frames:           Total frames analysed.
    """

    tracker_name: str
    spearman_r: float
    ece: float
    optimal_threshold: float
    f1_at_optimal: float
    bin_confidences: np.ndarray
    bin_accuracies: np.ndarray
    bin_weights: np.ndarray
    iou_good_threshold: float
    n_frames: int

    def __str__(self) -> str:
        return (
            f"ReliabilityResult[{self.tracker_name}] "
            f"Spearman_r={self.spearman_r:.3f}  "
            f"ECE={self.ece:.4f}  "
            f"opt_threshold={self.optimal_threshold:.2f}  "
            f"F1@opt={self.f1_at_optimal:.3f}  "
            f"n={self.n_frames}"
        )


class TrackerReliabilityAnalyzer:
    """Analyze how well a tracker's confidence scores correlate with accuracy.

    This class supports trackers that implement
    :class:`~eovot.trackers.confident_base.ConfidentBaseTracker`, though any
    sequence of ``(confidence, iou)`` pairs can be passed.

    Args:
        n_bins:              Number of equal-width bins for the calibration
                             curve.  Default: ``10``.
        iou_good_threshold:  IoU above which a frame is "accurately tracked"
                             for the F1 threshold search.  Default: ``0.5``
                             (standard VOT overlap criterion).
    """

    def __init__(
        self,
        n_bins: int = 10,
        iou_good_threshold: float = 0.5,
    ) -> None:
        self.n_bins = n_bins
        self.iou_good_threshold = iou_good_threshold

    # ------------------------------------------------------------------ #
    # Individual metric computations                                       #
    # ------------------------------------------------------------------ #

    def spearman_r(self, scores: np.ndarray, ious: np.ndarray) -> float:
        """Spearman rank correlation between confidence scores and IoU.

        Uses the standard rank-order formula without any external library
        dependency.

        Args:
            scores: Per-frame confidence values (any non-negative scale).
            ious:   Per-frame IoU values.

        Returns:
            Spearman *r* in ``[-1, 1]``.  Returns ``0.0`` for fewer than
            two valid frames.
        """
        n = min(len(scores), len(ious))
        if n < 2:
            return 0.0
        s = np.asarray(scores[:n], dtype=np.float64)
        v = np.asarray(ious[:n], dtype=np.float64)
        s_rank = np.argsort(np.argsort(s)).astype(np.float64)
        v_rank = np.argsort(np.argsort(v)).astype(np.float64)
        r = np.corrcoef(s_rank, v_rank)[0, 1]
        return 0.0 if np.isnan(r) else float(r)

    def calibration_curve(
        self, scores: np.ndarray, ious: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute confidence calibration curve data (reliability diagram).

        Scores are binned into ``n_bins`` equal-width buckets over ``[0, 1]``.
        For each bucket the mean IoU of all frames in that bucket is returned
        along with the bucket's weight (fraction of total frames).

        Args:
            scores: Per-frame confidence values in ``[0, 1]``.
            ious:   Per-frame IoU values.

        Returns:
            ``(bin_confidences, bin_accuracies, bin_weights)`` — each shape
            ``(n_bins,)``.  Buckets with no samples get accuracy ``0.0`` and
            weight ``0.0``.
        """
        n = min(len(scores), len(ious))
        s = np.asarray(scores[:n], dtype=np.float64)
        v = np.asarray(ious[:n], dtype=np.float64)

        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        midpoints = (edges[:-1] + edges[1:]) / 2.0
        bin_acc = np.zeros(self.n_bins, dtype=np.float64)
        bin_wts = np.zeros(self.n_bins, dtype=np.float64)

        for k in range(self.n_bins):
            lo, hi = edges[k], edges[k + 1]
            # last bin is inclusive on the right to capture score == 1.0
            if k == self.n_bins - 1:
                mask = (s >= lo) & (s <= hi)
            else:
                mask = (s >= lo) & (s < hi)
            if mask.sum() > 0:
                bin_acc[k] = float(v[mask].mean())
                bin_wts[k] = float(mask.sum()) / n

        return midpoints, bin_acc, bin_wts

    def expected_calibration_error(
        self, scores: np.ndarray, ious: np.ndarray
    ) -> float:
        """Expected Calibration Error (ECE).

        ECE = sum over bins of weight × |bin_confidence_midpoint − bin_mean_iou|.

        Args:
            scores: Per-frame confidence values.
            ious:   Per-frame IoU values.

        Returns:
            ECE in ``[0, 1]``.  0 = perfect calibration.
        """
        mids, accs, wts = self.calibration_curve(scores, ious)
        return float(np.sum(wts * np.abs(mids - accs)))

    def optimal_threshold(
        self, scores: np.ndarray, ious: np.ndarray
    ) -> Tuple[float, float]:
        """Find the confidence threshold maximising F1 for good-frame detection.

        A frame is "good" if its IoU ≥ ``iou_good_threshold``.  A prediction
        is "positive" when its confidence ≥ *threshold*.  F1 is maximised by
        scanning 51 evenly-spaced candidate thresholds in ``[0, 1]``.

        Args:
            scores: Per-frame confidence values.
            ious:   Per-frame IoU values.

        Returns:
            ``(optimal_threshold, f1_at_threshold)``.
        """
        n = min(len(scores), len(ious))
        if n < 2:
            return 0.5, 0.0
        s = np.asarray(scores[:n], dtype=np.float64)
        v = np.asarray(ious[:n], dtype=np.float64)
        labels = (v >= self.iou_good_threshold).astype(np.float64)

        best_thr, best_f1 = 0.5, 0.0
        for thr in np.linspace(0.0, 1.0, 51):
            preds = (s >= thr).astype(np.float64)
            tp = float((preds * labels).sum())
            fp = float((preds * (1.0 - labels)).sum())
            fn = float(((1.0 - preds) * labels).sum())
            denom = 2.0 * tp + fp + fn
            f1 = (2.0 * tp / denom) if denom > 0.0 else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)

        return best_thr, best_f1

    # ------------------------------------------------------------------ #
    # High-level entry point                                               #
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        scores: List[float],
        ious: List[float],
        tracker_name: str = "",
    ) -> ReliabilityResult:
        """Full reliability analysis for a tracker on a sequence or benchmark run.

        Args:
            scores:       Per-frame confidence values (floats in ``[0, 1]``).
            ious:         Per-frame IoU values.
            tracker_name: Identifier stored in the result.

        Returns:
            :class:`ReliabilityResult` populated with all reliability statistics.
        """
        s = np.asarray(scores, dtype=np.float64)
        v = np.asarray(ious, dtype=np.float64)
        n = min(len(s), len(v))

        r = self.spearman_r(s, v)
        mids, accs, wts = self.calibration_curve(s, v)
        ece = float(np.sum(wts * np.abs(mids - accs)))
        thr, f1 = self.optimal_threshold(s, v)

        return ReliabilityResult(
            tracker_name=tracker_name,
            spearman_r=r,
            ece=ece,
            optimal_threshold=thr,
            f1_at_optimal=f1,
            bin_confidences=mids,
            bin_accuracies=accs,
            bin_weights=wts,
            iou_good_threshold=self.iou_good_threshold,
            n_frames=n,
        )
