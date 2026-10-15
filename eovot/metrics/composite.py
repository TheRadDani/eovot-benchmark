"""Composite multi-dimensional tracker scoring for EOVOT.

Standard VOT evaluations rank trackers by a single scalar (e.g. success AUC).
Edge deployment, however, demands simultaneous optimisation across three
orthogonal dimensions — **accuracy**, **efficiency**, and **robustness** — none
of which alone tells the full story:

* A tracker with high IoU but 2 FPS is unusable on Raspberry Pi 4.
* A tracker running at 120 FPS but failing every 30 frames is equally useless.
* A tracker with low memory use but poor temporal smoothness causes jitter in
  downstream controllers.

The :class:`CompositeScoreEngine` addresses this gap with the **EOVOT-Score**:
a principled, publication-ready composite that combines all three dimensions
through min–max normalisation and configurable per-dimension weights.

EOVOT-Score Formula
~~~~~~~~~~~~~~~~~~~
Each dimension is aggregated from its constituent metrics, normalised to
``[0, 1]`` across the tracker set, and then combined::

    accuracy_score    = mean_norm(mIoU, success_AUC, norm_precision_AUC)
    efficiency_score  = mean_norm(EES, norm_fps, 1 - norm_memory)
    robustness_score  = mean_norm(EAO, survival_rate, 1 - failure_rate)

    EOVOT-Score = w_acc × accuracy_score
                + w_eff × efficiency_score
                + w_rob × robustness_score

Default weights ``(0.4, 0.35, 0.25)`` reflect that accuracy is the primary
criterion, efficiency matters most for edge deployment, and robustness breaks
ties.  All weights are configurable.

Typical usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.metrics.composite import CompositeScoreEngine

    ds = SyntheticDataset(num_sequences=5, num_frames=100)
    engine = BenchmarkEngine(verbose=False)
    results = [
        engine.run(MOSSETracker(), ds, dataset_name="Synthetic"),
        engine.run(KCFTracker(),   ds, dataset_name="Synthetic"),
    ]

    scorer = CompositeScoreEngine()
    ranking = scorer.rank(results)
    print(scorer.to_markdown(ranking))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


@dataclass
class DimensionScores:
    """Normalised per-dimension scores for one tracker."""

    accuracy: float
    """Mean of normalised mIoU, success AUC, normalised precision AUC — in [0, 1]."""
    efficiency: float
    """Mean of normalised EES, FPS, and inverse-memory scores — in [0, 1]."""
    robustness: float
    """Mean of normalised EAO, survival rate, and inverse-failure-rate — in [0, 1]."""

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": round(self.accuracy, 4),
            "efficiency": round(self.efficiency, 4),
            "robustness": round(self.robustness, 4),
        }


@dataclass
class CompositeEntry:
    """Full composite scoring record for one tracker.

    Attributes:
        tracker_name:     Human-readable tracker identifier.
        dataset_name:     Dataset on which the tracker was evaluated.
        eovot_score:      Composite EOVOT-Score in ``[0, 1]``.  Higher is better.
        dimensions:       Per-dimension normalised scores.
        mean_iou:         Raw mean IoU (not normalised) for reference.
        success_auc:      Raw success-curve AUC.
        norm_precision_auc: Raw normalised-precision AUC (GOT-10k protocol).
        fps:              Raw mean FPS.
        peak_memory_mb:   Raw peak memory in MiB.
        ees:              Raw Edge Efficiency Score.
        eao:              Raw Expected Average Overlap (robustness).
        survival_rate:    Fraction of non-burn-in frames where IoU ≥ threshold.
        failure_rate:     Mean failures per sequence.
        rank:             1-based rank in the comparison set (1 = best).
    """

    tracker_name: str
    dataset_name: str
    eovot_score: float
    dimensions: DimensionScores
    mean_iou: float
    success_auc: float
    norm_precision_auc: float
    fps: float
    peak_memory_mb: float
    ees: float
    eao: float
    survival_rate: float
    failure_rate: float
    rank: int = field(default=0)

    def __str__(self) -> str:
        return (
            f"CompositeEntry(rank={self.rank}  {self.tracker_name}  "
            f"EOVOT={self.eovot_score:.4f}  "
            f"acc={self.dimensions.accuracy:.3f}  "
            f"eff={self.dimensions.efficiency:.3f}  "
            f"rob={self.dimensions.robustness:.3f})"
        )


class CompositeScoreEngine:
    """Compute the multi-dimensional EOVOT composite tracker score.

    Combines accuracy, efficiency, and robustness into a single comparable
    scalar using min–max normalisation across the provided tracker set.

    Args:
        weight_accuracy:   Contribution of the accuracy dimension (default 0.40).
        weight_efficiency: Contribution of the efficiency dimension (default 0.35).
        weight_robustness: Contribution of the robustness dimension (default 0.25).
        memory_budget_mb:  Memory ceiling used to compute the EES via
            :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`.
        robustness_failure_threshold: IoU below which a frame is a failure
            (passed to :class:`~eovot.metrics.robustness.RobustnessAnalyzer`).

    Raises:
        ValueError: If weights do not sum to 1 (±1e-6) or any weight is negative.
    """

    def __init__(
        self,
        weight_accuracy: float = 0.40,
        weight_efficiency: float = 0.35,
        weight_robustness: float = 0.25,
        memory_budget_mb: float = 512.0,
        robustness_failure_threshold: float = 0.1,
    ) -> None:
        total = weight_accuracy + weight_efficiency + weight_robustness
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Dimension weights must sum to 1.0, got {total:.6f}. "
                f"Adjust weight_accuracy + weight_efficiency + weight_robustness."
            )
        for name, w in [
            ("weight_accuracy", weight_accuracy),
            ("weight_efficiency", weight_efficiency),
            ("weight_robustness", weight_robustness),
        ]:
            if w < 0:
                raise ValueError(f"{name} must be non-negative, got {w}.")

        self.weight_accuracy = weight_accuracy
        self.weight_efficiency = weight_efficiency
        self.weight_robustness = weight_robustness
        self.memory_budget_mb = memory_budget_mb
        self.robustness_failure_threshold = robustness_failure_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        results: List["BenchmarkResult"],
    ) -> List[CompositeEntry]:
        """Compute EOVOT-Scores and rank a set of tracker results.

        Normalises all metric dimensions jointly across the provided
        ``results`` list, then combines them per the configured weights.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult`
                per tracker / dataset combination.  Must contain at least
                two entries for normalisation to be meaningful (single-tracker
                comparison degenerates to all-zeros normalised scores).

        Returns:
            List of :class:`CompositeEntry` sorted by EOVOT-Score descending
            (rank 1 = best overall edge tracker).
        """
        if not results:
            return []

        raw = [self._extract_raw(r) for r in results]
        entries = self._normalize_and_combine(raw)
        entries.sort(key=lambda e: e.eovot_score, reverse=True)
        for i, e in enumerate(entries):
            e.rank = i + 1
        return entries

    def to_markdown(self, entries: List[CompositeEntry]) -> str:
        """Format a composite ranking table as a Markdown string.

        Args:
            entries: Output of :meth:`rank`.

        Returns:
            Multi-line Markdown table ready for README/report embedding.
        """
        header = (
            "| Rank | Tracker | EOVOT↑ | Acc↑ | Eff↑ | Rob↑ "
            "| mIoU | S-AUC | FPS | Mem(MB) | EAO |"
        )
        sep = "|------|---------|--------|------|------|------|------|-------|-----|---------|-----|"
        rows = [header, sep]
        for e in entries:
            rows.append(
                f"| {e.rank} | {e.tracker_name} "
                f"| **{e.eovot_score:.4f}** "
                f"| {e.dimensions.accuracy:.3f} "
                f"| {e.dimensions.efficiency:.3f} "
                f"| {e.dimensions.robustness:.3f} "
                f"| {e.mean_iou:.4f} "
                f"| {e.success_auc:.4f} "
                f"| {e.fps:.1f} "
                f"| {e.peak_memory_mb:.1f} "
                f"| {e.eao:.4f} |"
            )
        return "\n".join(rows)

    def to_dict(self, entries: List[CompositeEntry]) -> List[Dict]:
        """Serialise the ranking to a list of dicts for JSON export.

        Args:
            entries: Output of :meth:`rank`.

        Returns:
            List of dicts, one per tracker, with all raw and normalised scores.
        """
        out = []
        for e in entries:
            out.append({
                "rank": e.rank,
                "tracker_name": e.tracker_name,
                "dataset_name": e.dataset_name,
                "eovot_score": round(e.eovot_score, 6),
                "dimensions": e.dimensions.as_dict(),
                "raw": {
                    "mean_iou": round(e.mean_iou, 4),
                    "success_auc": round(e.success_auc, 4),
                    "norm_precision_auc": round(e.norm_precision_auc, 4),
                    "fps": round(e.fps, 2),
                    "peak_memory_mb": round(e.peak_memory_mb, 2),
                    "ees": round(e.ees, 4),
                    "eao": round(e.eao, 4),
                    "survival_rate": round(e.survival_rate, 4),
                    "failure_rate": round(e.failure_rate, 4),
                },
            })
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_raw(self, result: "BenchmarkResult") -> Dict:
        """Extract raw metric scalars from a BenchmarkResult."""
        from ..metrics.efficiency import EfficiencyMetricsEngine
        from ..metrics.robustness import RobustnessAnalyzer

        eff_engine = EfficiencyMetricsEngine(memory_budget_mb=self.memory_budget_mb)
        ees = eff_engine.edge_efficiency_score(
            mean_iou=result.mean_iou,
            fps=result.mean_fps,
            peak_memory_mb=result.peak_memory_mb,
        )

        rob_analyzer = RobustnessAnalyzer(
            failure_threshold=self.robustness_failure_threshold,
        )
        seq_ious = {r.sequence_name: r.ious for r in result.sequence_results}
        rob_agg = rob_analyzer.analyze_benchmark(seq_ious, tracker_name=result.tracker_name)
        agg = rob_agg["aggregate"]

        n_seq = max(agg["num_sequences"], 1)
        failure_rate = agg["mean_failures_per_sequence"]

        return {
            "tracker_name": result.tracker_name,
            "dataset_name": result.dataset_name,
            # accuracy
            "mean_iou": result.mean_iou,
            "success_auc": result.mean_success_auc or result.mean_iou,
            "norm_precision_auc": result.mean_normalized_precision_auc or 0.0,
            # efficiency
            "fps": result.mean_fps,
            "peak_memory_mb": result.peak_memory_mb,
            "ees": ees,
            # robustness
            "eao": float(agg["mean_eao"]),
            "survival_rate": float(agg["mean_survival_rate"]),
            "failure_rate": float(failure_rate),
        }

    def _normalize_and_combine(
        self, raw_list: List[Dict]
    ) -> List[CompositeEntry]:
        """Min-max normalise across the tracker set and produce CompositeEntry objects."""

        def _norm_col(values: List[float], higher_is_better: bool) -> List[float]:
            """Min-max normalise; invert when lower is better."""
            lo, hi = min(values), max(values)
            span = hi - lo
            if span < 1e-12:
                return [1.0] * len(values)
            normed = [(v - lo) / span for v in values]
            if not higher_is_better:
                normed = [1.0 - v for v in normed]
            return normed

        n = len(raw_list)

        # Accuracy columns (all higher-is-better)
        miou_n   = _norm_col([r["mean_iou"]          for r in raw_list], True)
        sauc_n   = _norm_col([r["success_auc"]        for r in raw_list], True)
        npauc_n  = _norm_col([r["norm_precision_auc"] for r in raw_list], True)

        # Efficiency columns
        fps_n    = _norm_col([r["fps"]           for r in raw_list], True)
        mem_n    = _norm_col([r["peak_memory_mb"] for r in raw_list], False)  # lower is better
        ees_n    = _norm_col([r["ees"]            for r in raw_list], True)

        # Robustness columns
        eao_n     = _norm_col([r["eao"]          for r in raw_list], True)
        surv_n    = _norm_col([r["survival_rate"] for r in raw_list], True)
        fail_n    = _norm_col([r["failure_rate"]  for r in raw_list], False)  # lower is better

        entries = []
        for i, raw in enumerate(raw_list):
            acc_score  = (miou_n[i] + sauc_n[i] + npauc_n[i]) / 3.0
            eff_score  = (fps_n[i] + mem_n[i] + ees_n[i]) / 3.0
            rob_score  = (eao_n[i] + surv_n[i] + fail_n[i]) / 3.0

            eovot = (
                self.weight_accuracy   * acc_score
                + self.weight_efficiency * eff_score
                + self.weight_robustness * rob_score
            )

            entries.append(CompositeEntry(
                tracker_name=raw["tracker_name"],
                dataset_name=raw["dataset_name"],
                eovot_score=eovot,
                dimensions=DimensionScores(
                    accuracy=acc_score,
                    efficiency=eff_score,
                    robustness=rob_score,
                ),
                mean_iou=raw["mean_iou"],
                success_auc=raw["success_auc"],
                norm_precision_auc=raw["norm_precision_auc"],
                fps=raw["fps"],
                peak_memory_mb=raw["peak_memory_mb"],
                ees=raw["ees"],
                eao=raw["eao"],
                survival_rate=raw["survival_rate"],
                failure_rate=raw["failure_rate"],
            ))

        return entries
