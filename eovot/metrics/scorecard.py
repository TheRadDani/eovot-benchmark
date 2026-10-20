"""Unified Edge Deployment Scorecard — combines all EOVOT metrics into one score.

The core thesis of EOVOT is that trackers must be evaluated holistically: accuracy
alone is not enough for edge deployment.  Yet until this module, the project
produced separate numbers for accuracy, robustness, temporal consistency, and
efficiency with no mechanism to combine them.  Practitioners and paper authors
need a single comparable scalar.

:class:`EdgeScorecardEngine` normalises each metric dimension to [0, 1] and
produces a weighted sum called the **Edge Suitability Index (ESI)**:

    ESI = w_acc   × score_accuracy
        + w_rob   × score_robustness
        + w_smooth × score_smoothness
        + w_eff   × score_efficiency

where each score is derived from the standard EOVOT metric outputs:

+---------------+----------------------------+-----------------------------+
| Dimension     | Source metric              | Normalisation               |
+===============+============================+=============================+
| Accuracy      | success_auc                | raw ∈ [0, 1]                |
+---------------+----------------------------+-----------------------------+
| Robustness    | EAO = 1 − failure_rate     | 1 − (failures / N_frames)   |
+---------------+----------------------------+-----------------------------+
| Smoothness    | smoothness_score           | raw ∈ (0, 1] already        |
+---------------+----------------------------+-----------------------------+
| Efficiency    | Edge Efficiency Score      | logistic normalisation      |
+---------------+----------------------------+-----------------------------+

The ESI is scaled to 0–100 for intuitive readability.  Dimension scores are
also exposed individually so researchers can understand the breakdown.

Default weights (equal):  0.25 each.  Pass custom weights to emphasise
deployment constraints — e.g. weight efficiency more for battery devices.

Typical usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.kcf import KCFTracker
    from eovot.metrics.scorecard import EdgeScorecardEngine
    from eovot.metrics.robustness import RobustnessAnalyzer
    from eovot.metrics.temporal import TemporalConsistencyAnalyzer

    dataset = SyntheticDataset(num_sequences=5)
    engine  = BenchmarkEngine(verbose=False)
    result  = engine.run(KCFTracker(), dataset, "Synthetic")

    robustness_data = {
        r.sequence_name: r.ious for r in result.sequence_results
    }
    temporal_data = {
        r.sequence_name: r.predictions for r in result.sequence_results
        if r.predictions is not None
    }

    scorecard = EdgeScorecardEngine()
    entry = scorecard.score(result, robustness_data, temporal_data)
    print(entry)
    print(scorecard.to_markdown([entry]))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .robustness import RobustnessAnalyzer
from .temporal import TemporalConsistencyAnalyzer

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Weights configuration
# ---------------------------------------------------------------------------

@dataclass
class ScorecardWeights:
    """Configurable dimension weights for the Edge Suitability Index.

    All weights are normalised internally, so they only need to be relative
    to each other.  The defaults give equal weight to all four dimensions.

    Attributes:
        accuracy: Weight for the accuracy dimension (success AUC).
        robustness: Weight for the robustness dimension (failure-adjusted EAO).
        smoothness: Weight for temporal consistency (smoothness score).
        efficiency: Weight for efficiency (normalised EES).
    """

    accuracy: float = 0.25
    robustness: float = 0.25
    smoothness: float = 0.25
    efficiency: float = 0.25

    def normalised(self) -> "ScorecardWeights":
        """Return a copy with weights summing to 1.0."""
        total = self.accuracy + self.robustness + self.smoothness + self.efficiency
        if total <= 0:
            raise ValueError("All weights are zero; cannot normalise.")
        return ScorecardWeights(
            accuracy=self.accuracy / total,
            robustness=self.robustness / total,
            smoothness=self.smoothness / total,
            efficiency=self.efficiency / total,
        )


# ---------------------------------------------------------------------------
# Per-tracker scorecard entry
# ---------------------------------------------------------------------------

@dataclass
class ScorecardEntry:
    """Full scorecard for one tracker/dataset combination.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        dataset_name: Dataset on which the tracker was evaluated.
        esi: Edge Suitability Index — weighted composite score in [0, 100].
        accuracy_score: Normalised accuracy dimension score ∈ [0, 1].
        robustness_score: Normalised robustness dimension score ∈ [0, 1].
        smoothness_score: Normalised temporal smoothness dimension score ∈ [0, 1].
        efficiency_score: Normalised efficiency dimension score ∈ [0, 1].
        source_metrics: Raw metric values used to compute the scores.
    """

    tracker_name: str
    dataset_name: str
    esi: float
    accuracy_score: float
    robustness_score: float
    smoothness_score: float
    efficiency_score: float
    source_metrics: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"ScorecardEntry[{self.tracker_name} on {self.dataset_name}]  "
            f"ESI={self.esi:.1f}/100  "
            f"acc={self.accuracy_score:.3f}  rob={self.robustness_score:.3f}  "
            f"smooth={self.smoothness_score:.3f}  eff={self.efficiency_score:.3f}"
        )

    def to_dict(self) -> Dict:
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "esi": round(self.esi, 2),
            "accuracy_score": round(self.accuracy_score, 4),
            "robustness_score": round(self.robustness_score, 4),
            "smoothness_score": round(self.smoothness_score, 4),
            "efficiency_score": round(self.efficiency_score, 4),
            "source_metrics": {k: round(v, 6) for k, v in self.source_metrics.items()},
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EdgeScorecardEngine:
    """Compute the Edge Suitability Index (ESI) from EOVOT benchmark results.

    Args:
        weights: Dimension weights for the composite score.  Defaults to
            equal weights across all four dimensions.
        memory_budget_mb: Memory ceiling for the efficiency dimension's
            normalisation.  Trackers within this budget receive full credit.
            Default: ``512.0`` MB.
        max_ees_ref: Reference EES value used to normalise the efficiency
            score.  A tracker achieving ``max_ees_ref`` scores 1.0 in
            efficiency.  Default: ``2.5`` (typical good edge tracker value
            at ~30 FPS / 0.5 mIoU / 256 MB; tune for your device fleet).

    Example::

        engine = EdgeScorecardEngine(weights=ScorecardWeights(efficiency=0.4))
        entries = [engine.score(r, rob_data, temp_data) for r in results]
        print(engine.to_markdown(entries))
    """

    def __init__(
        self,
        weights: Optional[ScorecardWeights] = None,
        memory_budget_mb: float = 512.0,
        max_ees_ref: float = 2.5,
    ) -> None:
        self._weights = (weights or ScorecardWeights()).normalised()
        self._memory_budget_mb = memory_budget_mb
        self._max_ees_ref = max(max_ees_ref, 1e-6)
        self._rob_analyzer = RobustnessAnalyzer()
        self._temp_analyzer = TemporalConsistencyAnalyzer()

    # ------------------------------------------------------------------
    # Main scoring entry point
    # ------------------------------------------------------------------

    def score(
        self,
        result: "BenchmarkResult",
        sequence_ious: Optional[Dict[str, np.ndarray]] = None,
        sequence_predictions: Optional[Dict[str, np.ndarray]] = None,
    ) -> ScorecardEntry:
        """Compute a full :class:`ScorecardEntry` for one benchmark result.

        Args:
            result: Output of :class:`~eovot.benchmark.engine.BenchmarkEngine.run`.
            sequence_ious: Mapping ``{sequence_name: ious_array}`` for robustness
                analysis.  If ``None``, inferred from ``result.sequence_results``.
            sequence_predictions: Mapping ``{sequence_name: predictions_array}``
                for temporal consistency analysis.  If ``None``, inferred from
                ``result.sequence_results`` when predictions are stored.

        Returns:
            :class:`ScorecardEntry` with ESI and per-dimension scores.
        """
        # Resolve IoU and prediction data from the result if not supplied
        if sequence_ious is None:
            sequence_ious = {
                r.sequence_name: r.ious for r in result.sequence_results
            }
        if sequence_predictions is None:
            sequence_predictions = {
                r.sequence_name: r.predictions
                for r in result.sequence_results
                if r.predictions is not None
            }

        acc_score, acc_raw = self._accuracy_score(result)
        rob_score, rob_raw = self._robustness_score(result.tracker_name, sequence_ious)
        smooth_score, smooth_raw = self._smoothness_score(
            result.tracker_name, sequence_predictions
        )
        eff_score, eff_raw = self._efficiency_score(result)

        w = self._weights
        esi = 100.0 * (
            w.accuracy   * acc_score
            + w.robustness * rob_score
            + w.smoothness * smooth_score
            + w.efficiency * eff_score
        )

        source = {
            "success_auc": acc_raw,
            "mean_eao": rob_raw,
            "mean_smoothness": smooth_raw,
            "ees": eff_raw,
            "mean_iou": result.mean_iou,
            "mean_fps": result.mean_fps,
            "peak_memory_mb": result.peak_memory_mb,
        }

        return ScorecardEntry(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            esi=round(esi, 2),
            accuracy_score=round(acc_score, 4),
            robustness_score=round(rob_score, 4),
            smoothness_score=round(smooth_score, 4),
            efficiency_score=round(eff_score, 4),
            source_metrics=source,
        )

    def rank(
        self,
        results: List["BenchmarkResult"],
        sequence_ious: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        sequence_predictions: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> List[ScorecardEntry]:
        """Score multiple trackers and return them ranked by ESI (highest first).

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult` per tracker.
            sequence_ious: Per-tracker IoU data; keyed by tracker name.
                If ``None``, inferred from each result's sequence data.
            sequence_predictions: Per-tracker prediction arrays; keyed by tracker name.
                If ``None``, inferred from each result's sequence data.

        Returns:
            List of :class:`ScorecardEntry` objects sorted by ESI descending.
        """
        entries: List[ScorecardEntry] = []
        for r in results:
            iou_data = (sequence_ious or {}).get(r.tracker_name)
            pred_data = (sequence_predictions or {}).get(r.tracker_name)
            entries.append(self.score(r, iou_data, pred_data))
        entries.sort(key=lambda e: e.esi, reverse=True)
        return entries

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _accuracy_score(
        self, result: "BenchmarkResult"
    ) -> Tuple[float, float]:
        """Accuracy dimension: success AUC ∈ [0, 1], already normalised."""
        auc = result.mean_success_auc
        if auc is None:
            auc = result.mean_iou  # fallback to raw IoU
        auc = max(0.0, min(1.0, float(auc)))
        return auc, auc

    def _robustness_score(
        self,
        tracker_name: str,
        sequence_ious: Dict[str, np.ndarray],
    ) -> Tuple[float, float]:
        """Robustness dimension: mean EAO across sequences ∈ [0, 1]."""
        if not sequence_ious:
            return 0.5, 0.5  # no data — neutral score
        agg = self._rob_analyzer.analyze_benchmark(sequence_ious, tracker_name)
        mean_eao = float(agg["aggregate"].get("mean_eao", 0.5))
        mean_eao = max(0.0, min(1.0, mean_eao))
        return mean_eao, mean_eao

    def _smoothness_score(
        self,
        tracker_name: str,
        sequence_predictions: Dict[str, np.ndarray],
    ) -> Tuple[float, float]:
        """Temporal consistency dimension: mean smoothness score ∈ (0, 1]."""
        if not sequence_predictions:
            return 0.5, 0.5  # no prediction data — neutral score
        valid = {k: v for k, v in sequence_predictions.items() if v is not None and len(v) >= 2}
        if not valid:
            return 0.5, 0.5
        agg = self._temp_analyzer.analyze_benchmark(valid, tracker_name)
        mean_smooth = float(agg["aggregate"].get("mean_smoothness_score", 0.5))
        mean_smooth = max(0.0, min(1.0, mean_smooth))
        return mean_smooth, mean_smooth

    def _efficiency_score(
        self, result: "BenchmarkResult"
    ) -> Tuple[float, float]:
        """Efficiency dimension: normalised EES using logistic-like scaling.

        EES = mean_iou × log1p(fps) / (1 + memory_mb / budget_mb)

        Normalised to [0, 1] by dividing by max_ees_ref and clipping.
        """
        fps = max(result.mean_fps, 0.0)
        iou = max(result.mean_iou, 0.0)
        mem = max(result.peak_memory_mb, 0.0)
        ees = iou * math.log1p(fps) / (1.0 + mem / self._memory_budget_mb)
        norm_ees = min(ees / self._max_ees_ref, 1.0)
        return norm_ees, ees

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown(self, entries: List[ScorecardEntry]) -> str:
        """Format a ranked scorecard table as a Markdown string.

        Args:
            entries: Output of :meth:`rank` or a list of :meth:`score` results.

        Returns:
            Multi-line Markdown table, sorted by ESI descending.
        """
        sorted_entries = sorted(entries, key=lambda e: e.esi, reverse=True)
        lines = [
            "| Rank | Tracker | Dataset | ESI | Accuracy | Robustness | Smoothness | Efficiency |",
            "|------|---------|---------|----:|---------:|-----------:|-----------:|-----------:|",
        ]
        for rank, e in enumerate(sorted_entries, start=1):
            lines.append(
                f"| {rank} | {e.tracker_name} | {e.dataset_name} "
                f"| **{e.esi:.1f}** "
                f"| {e.accuracy_score:.3f} "
                f"| {e.robustness_score:.3f} "
                f"| {e.smoothness_score:.3f} "
                f"| {e.efficiency_score:.3f} |"
            )
        return "\n".join(lines)

    def to_csv(self, entries: List[ScorecardEntry]) -> str:
        """Format the scorecard as a CSV string.

        Args:
            entries: Output of :meth:`rank` or a list of :meth:`score` results.

        Returns:
            CSV string with header row and one data row per tracker.
        """
        sorted_entries = sorted(entries, key=lambda e: e.esi, reverse=True)
        lines = ["tracker_name,dataset_name,esi,accuracy_score,robustness_score,"
                 "smoothness_score,efficiency_score,mean_iou,mean_fps,peak_memory_mb"]
        for e in sorted_entries:
            src = e.source_metrics
            lines.append(
                f"{e.tracker_name},{e.dataset_name},{e.esi:.2f},"
                f"{e.accuracy_score:.4f},{e.robustness_score:.4f},"
                f"{e.smoothness_score:.4f},{e.efficiency_score:.4f},"
                f"{src.get('mean_iou', 0.0):.4f},"
                f"{src.get('mean_fps', 0.0):.1f},"
                f"{src.get('peak_memory_mb', 0.0):.1f}"
            )
        return "\n".join(lines)
