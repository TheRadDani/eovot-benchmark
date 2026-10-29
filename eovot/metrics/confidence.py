"""Confidence-IoU analysis for response-quality-aware trackers.

Evaluates how well a tracker's internal confidence signal (e.g. PSR from
:class:`~eovot.trackers.adaptive_kcf.AdaptiveKCFTracker`) predicts tracking
quality and failure events — bridging the gap between a tracker's
self-assessment and objectively measured accuracy.

Core analyses
~~~~~~~~~~~~~

**Pearson correlation** — linear relationship between confidence and per-frame
IoU.  A strong positive correlation (r > 0.5) indicates the confidence signal
is genuinely informative about tracking quality.

**Failure-prediction AUROC** — binary classification accuracy of confidence as
a predictor of tracking failures (IoU < ``failure_iou_threshold``).
AUROC = 1.0 is a perfect failure predictor; 0.5 is random.  This metric is
essential for calibrating re-initialization triggers on the edge.

**Optimal threshold sweep** — sweeps candidate thresholds on the confidence
signal and selects the one that maximises the F1-score for failure detection.
The result is a principled, data-driven value for the ``min_psr``
hyperparameter of :class:`~eovot.trackers.adaptive_kcf.AdaptiveKCFTracker`.

Typical usage::

    from eovot.trackers.adaptive_kcf import AdaptiveKCFTracker
    from eovot.metrics.confidence import ConfidenceAnalyzer

    tracker = AdaptiveKCFTracker()
    # ... run benchmark to collect ious and tracker.psr_history ...

    analyzer = ConfidenceAnalyzer()
    result = analyzer.analyze(tracker.psr_history, ious,
                              tracker_name="AdaptiveKCF", sequence_name="car1")
    print(result)
    # ConfidenceResult[AdaptiveKCF on car1]  r=0.72  AUROC=0.81
    #   best_thr=6.2  F1=0.74  frames=300  failures=12
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ConfidenceResult:
    """Confidence-IoU analysis summary for one sequence.

    Attributes:
        tracker_name: Tracker identifier.
        sequence_name: Sequence identifier.
        pearson_r: Pearson correlation between confidence scores and per-frame IoU.
        auroc: AUROC of confidence as a binary failure detector.
        best_threshold: Confidence threshold maximising F1 for failure detection.
        best_f1: F1-score at ``best_threshold``.
        n_frames: Number of evaluated frames.
        n_failures: Ground-truth failure frames (IoU < ``failure_iou_threshold``).
    """

    tracker_name: str
    sequence_name: str
    pearson_r: float
    auroc: float
    best_threshold: float
    best_f1: float
    n_frames: int
    n_failures: int

    def __str__(self) -> str:
        return (
            f"ConfidenceResult[{self.tracker_name} on {self.sequence_name}]  "
            f"r={self.pearson_r:.3f}  AUROC={self.auroc:.3f}  "
            f"best_thr={self.best_threshold:.2f}  F1={self.best_f1:.3f}  "
            f"frames={self.n_frames}  failures={self.n_failures}"
        )

    def to_dict(self) -> Dict:
        return {
            "tracker_name": self.tracker_name,
            "sequence_name": self.sequence_name,
            "pearson_r": round(self.pearson_r, 4),
            "auroc": round(self.auroc, 4),
            "best_threshold": round(self.best_threshold, 3),
            "best_f1": round(self.best_f1, 4),
            "n_frames": self.n_frames,
            "n_failures": self.n_failures,
        }


class ConfidenceAnalyzer:
    """Evaluate a tracker's confidence signal against ground-truth per-frame IoU.

    Args:
        failure_iou_threshold: IoU below which a frame is labelled a tracking
            failure.  Default: ``0.1`` (standard VOT criterion).
        n_thresholds: Number of candidate confidence thresholds for the F1
            sweep.  Default: ``100``.
    """

    def __init__(
        self,
        failure_iou_threshold: float = 0.1,
        n_thresholds: int = 100,
    ) -> None:
        if not 0.0 <= failure_iou_threshold <= 1.0:
            raise ValueError(
                f"failure_iou_threshold must be in [0, 1], got {failure_iou_threshold}"
            )
        self.failure_iou_threshold = failure_iou_threshold
        self.n_thresholds = n_thresholds

    # ------------------------------------------------------------------
    # Core metric computations
    # ------------------------------------------------------------------

    def compute_auroc(
        self, confidence: np.ndarray, is_failure: np.ndarray
    ) -> float:
        """AUROC of confidence as a binary failure predictor.

        Low confidence predicts failure; high confidence predicts success.
        The score is computed using the trapezoidal rule over the ROC curve.

        Args:
            confidence: Per-frame confidence scores (higher = more confident).
            is_failure: Boolean array, ``True`` on failure frames.

        Returns:
            AUROC in ``[0, 1]``.  Returns ``0.5`` (chance) when there are
            no positives or no negatives in ``is_failure``.
        """
        n_pos = int(is_failure.sum())
        n_neg = int((~is_failure).sum())
        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Invert: high failure score → low confidence.
        scores = -confidence
        order = np.argsort(scores)
        is_fail_sorted = is_failure[order]

        cum_tp = np.concatenate([[0.0], np.cumsum(is_fail_sorted) / n_pos])
        cum_fp = np.concatenate([[0.0], np.cumsum(~is_fail_sorted) / n_neg])

        try:
            return float(np.trapezoid(cum_tp, cum_fp))
        except AttributeError:  # numpy < 2.0
            return float(np.trapz(cum_tp, cum_fp))

    def find_best_threshold(
        self, confidence: np.ndarray, is_failure: np.ndarray
    ) -> Tuple[float, float]:
        """Find the confidence threshold that maximises F1 for failure detection.

        A frame is predicted as a failure when ``confidence < threshold``.

        Args:
            confidence: Per-frame confidence scores.
            is_failure: Boolean array of ground-truth failures.

        Returns:
            ``(best_threshold, best_f1)`` tuple.  Returns
            ``(confidence.min(), 0.0)`` when there are no failures.
        """
        if is_failure.sum() == 0:
            return float(confidence.min()), 0.0

        thresholds = np.linspace(
            float(confidence.min()), float(confidence.max()), self.n_thresholds
        )
        best_thr, best_f1 = float(thresholds[0]), 0.0

        for thr in thresholds:
            pred_fail = confidence < thr
            tp = int(np.logical_and(pred_fail, is_failure).sum())
            fp = int(np.logical_and(pred_fail, ~is_failure).sum())
            fn = int(np.logical_and(~pred_fail, is_failure).sum())
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        return best_thr, best_f1

    # ------------------------------------------------------------------
    # High-level analysis entry points
    # ------------------------------------------------------------------

    def analyze(
        self,
        confidence: List[float],
        ious: List[float],
        tracker_name: str = "",
        sequence_name: str = "",
    ) -> ConfidenceResult:
        """Run full confidence-IoU analysis for one sequence.

        Args:
            confidence: Per-frame confidence scores (e.g. PSR values from
                :class:`~eovot.trackers.adaptive_kcf.AdaptiveKCFTracker`).
            ious: Per-frame IoU values against ground truth.
            tracker_name: Identifier stored in the result.
            sequence_name: Identifier stored in the result.

        Returns:
            :class:`ConfidenceResult` with all statistics populated.

        Raises:
            ValueError: If fewer than 2 frames are provided or the arrays
                have incompatible lengths after truncation.
        """
        conf_arr = np.asarray(confidence, dtype=np.float64)
        iou_arr = np.asarray(ious, dtype=np.float64)
        n = min(len(conf_arr), len(iou_arr))
        if n < 2:
            raise ValueError(
                f"At least 2 frames are required for analysis, got {n}."
            )
        conf_arr = conf_arr[:n]
        iou_arr = iou_arr[:n]

        # Pearson correlation between confidence and IoU.
        r = float(np.corrcoef(conf_arr, iou_arr)[0, 1])
        if not np.isfinite(r):
            r = 0.0

        is_failure = iou_arr < self.failure_iou_threshold
        auroc = self.compute_auroc(conf_arr, is_failure)
        best_thr, best_f1 = self.find_best_threshold(conf_arr, is_failure)

        return ConfidenceResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            pearson_r=r,
            auroc=auroc,
            best_threshold=best_thr,
            best_f1=best_f1,
            n_frames=n,
            n_failures=int(is_failure.sum()),
        )

    def analyze_benchmark(
        self,
        confidence_by_sequence: Dict[str, List[float]],
        ious_by_sequence: Dict[str, List[float]],
        tracker_name: str = "",
    ) -> Dict:
        """Aggregate confidence analysis across a full benchmark run.

        Args:
            confidence_by_sequence: ``{seq_name: [score0, score1, …]}``.
            ious_by_sequence: ``{seq_name: [iou0, iou1, …]}``.
            tracker_name: Tracker identifier.

        Returns:
            Dict with two keys:

            * ``"per_sequence"`` — ``{seq_name: ConfidenceResult}``
            * ``"aggregate"`` — summary scalars across all sequences
        """
        per_seq: Dict[str, ConfidenceResult] = {}
        for seq_name, conf in confidence_by_sequence.items():
            if seq_name not in ious_by_sequence:
                continue
            try:
                per_seq[seq_name] = self.analyze(
                    conf,
                    ious_by_sequence[seq_name],
                    tracker_name=tracker_name,
                    sequence_name=seq_name,
                )
            except ValueError:
                pass

        n = len(per_seq)
        if n == 0:
            return {"per_sequence": {}, "aggregate": {
                "tracker_name": tracker_name,
                "num_sequences": 0,
            }}

        mean_r = float(np.mean([r.pearson_r for r in per_seq.values()]))
        mean_auroc = float(np.mean([r.auroc for r in per_seq.values()]))
        mean_f1 = float(np.mean([r.best_f1 for r in per_seq.values()]))
        median_thr = float(
            np.median([r.best_threshold for r in per_seq.values()])
        )

        return {
            "per_sequence": per_seq,
            "aggregate": {
                "tracker_name": tracker_name,
                "num_sequences": n,
                "mean_pearson_r": round(mean_r, 4),
                "mean_auroc": round(mean_auroc, 4),
                "mean_best_f1": round(mean_f1, 4),
                "median_best_threshold": round(median_thr, 2),
            },
        }
