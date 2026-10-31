"""Unified multi-dimensional tracker scorecard for EOVOT.

Aggregates accuracy, robustness, efficiency, and temporal consistency metrics
into a single normalized, comparable score per tracker — enabling publication-
quality ranking tables that reflect the full multi-objective nature of edge
visual object tracking.

Design
------
Each metric dimension is min-max normalized across the evaluation cohort
(the set of trackers being compared) so that scores are always in ``[0, 1]``
and directly comparable.  A weighted sum produces the composite score.

Default weights reflect the EOVOT thesis priority order:

- **Accuracy** (success AUC): 0.35 — primary research metric
- **Robustness** (EAO): 0.25 — deployment reliability
- **Efficiency** (EES): 0.25 — core edge constraint
- **Temporal consistency** (smoothness): 0.15 — downstream system quality

Weights sum to 1.0 and can be overridden at construction time to match a
specific evaluation protocol or paper's methodology.

Typical usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.metrics.scorecard import TrackerScorecard
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker

    dataset = SyntheticDataset(num_sequences=5, num_frames=120)
    engine  = BenchmarkEngine(verbose=False)
    results = [
        engine.run(MOSSETracker(), dataset, dataset_name="Synthetic"),
        engine.run(KCFTracker(),   dataset, dataset_name="Synthetic"),
    ]

    scorecard = TrackerScorecard()
    scores    = scorecard.compute(results)
    print(scorecard.to_markdown_table(scores))
    print(scorecard.to_latex_table(scores))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from .efficiency import EfficiencyMetricsEngine
from .robustness import RobustnessAnalyzer
from .temporal import TemporalConsistencyAnalyzer

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


@dataclass
class TrackerScore:
    """Normalized multi-dimensional score for one tracker on one dataset.

    All ``*_score`` fields are min-max normalized relative to the cohort
    passed to :meth:`TrackerScorecard.compute`, so they are *not* absolute
    measures — a score of ``1.0`` means best-in-cohort, ``0.0`` worst.

    Attributes:
        tracker_name:     Human-readable tracker identifier.
        dataset_name:     Dataset on which the tracker was evaluated.
        accuracy_score:   Normalized success-AUC score.
        robustness_score: Normalized Expected Average Overlap (EAO) score.
        efficiency_score: Normalized Edge Efficiency Score (EES).
        temporal_score:   Normalized smoothness score.
        composite_score:  Weighted sum of the four dimension scores.
        mean_iou:         Raw mean IoU across all frames.
        success_auc:      Raw success-curve AUC.
        eao:              Raw Expected Average Overlap.
        ees:              Raw Edge Efficiency Score.
        smoothness:       Raw temporal smoothness scalar.
        fps:              Raw mean frames-per-second.
    """

    tracker_name: str
    dataset_name: str

    # Normalized dimension scores in [0, 1]
    accuracy_score: float
    robustness_score: float
    efficiency_score: float
    temporal_score: float
    composite_score: float

    # Raw metrics preserved for reproducibility and external reporting
    mean_iou: float
    success_auc: float
    eao: float
    ees: float
    smoothness: float
    fps: float

    def to_dict(self) -> Dict:
        """Return a flat dict suitable for JSON export or DataFrame construction."""
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "composite_score": round(self.composite_score, 4),
            "accuracy_score": round(self.accuracy_score, 4),
            "robustness_score": round(self.robustness_score, 4),
            "efficiency_score": round(self.efficiency_score, 4),
            "temporal_score": round(self.temporal_score, 4),
            "mean_iou": round(self.mean_iou, 4),
            "success_auc": round(self.success_auc, 4),
            "eao": round(self.eao, 4),
            "ees": round(self.ees, 4),
            "smoothness": round(self.smoothness, 4),
            "fps": round(self.fps, 2),
        }

    def __str__(self) -> str:
        return (
            f"TrackerScore[{self.tracker_name}] "
            f"composite={self.composite_score:.4f} "
            f"(acc={self.accuracy_score:.3f}, "
            f"rob={self.robustness_score:.3f}, "
            f"eff={self.efficiency_score:.3f}, "
            f"temp={self.temporal_score:.3f})"
        )


class TrackerScorecard:
    """Normalize and aggregate multi-metric benchmark results into a scorecard.

    Takes a list of :class:`~eovot.benchmark.engine.BenchmarkResult` objects
    (one per tracker, typically all on the same dataset) and produces a
    :class:`TrackerScore` for each, ranked by composite score.

    Args:
        accuracy_weight:   Weight for the accuracy dimension.  Default ``0.35``.
        robustness_weight: Weight for the robustness dimension.  Default ``0.25``.
        efficiency_weight: Weight for the efficiency dimension.  Default ``0.25``.
        temporal_weight:   Weight for the temporal consistency dimension.
                           Default ``0.15``.
        memory_budget_mb:  Memory budget (MB) passed to
                           :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`
                           for EES computation.  Default ``512.0``.

    Raises:
        ValueError: If the four weights do not sum to ``1.0`` (tolerance 1e-6).
    """

    def __init__(
        self,
        accuracy_weight: float = 0.35,
        robustness_weight: float = 0.25,
        efficiency_weight: float = 0.25,
        temporal_weight: float = 0.15,
        memory_budget_mb: float = 512.0,
    ) -> None:
        total = accuracy_weight + robustness_weight + efficiency_weight + temporal_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")
        self.accuracy_weight = accuracy_weight
        self.robustness_weight = robustness_weight
        self.efficiency_weight = efficiency_weight
        self.temporal_weight = temporal_weight
        self._robustness = RobustnessAnalyzer()
        self._temporal = TemporalConsistencyAnalyzer()
        self._efficiency = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _minmax(values: List[float]) -> List[float]:
        """Min-max normalize a list to [0, 1]; returns 0.5 for constant lists."""
        arr = np.asarray(values, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-12:
            return [0.5] * len(values)
        return list((arr - lo) / (hi - lo))

    def _raw_metrics(
        self, result: "BenchmarkResult"
    ) -> Tuple[float, float, float, float, float, float]:
        """Compute (success_auc, eao, ees, smoothness, fps, mean_iou) for one result."""
        # Accuracy: prefer per-sequence mean success AUC if available
        success_auc = result.mean_success_auc if result.mean_success_auc is not None else result.mean_iou

        # Robustness: mean EAO via RobustnessAnalyzer
        seq_ious = {r.sequence_name: r.ious for r in result.sequence_results}
        rob_out = self._robustness.analyze_benchmark(seq_ious, tracker_name=result.tracker_name)
        eao: float = rob_out["aggregate"]["mean_eao"]

        # Efficiency: Edge Efficiency Score
        eff_entries = self._efficiency.rank_trackers([result])
        ees: float = eff_entries[0].ees if eff_entries else 0.0

        # Temporal: mean smoothness score across sequences
        seq_preds = {
            r.sequence_name: r.predictions
            for r in result.sequence_results
            if r.predictions is not None and len(r.predictions) > 1
        }
        if seq_preds:
            temp_out = self._temporal.analyze_benchmark(seq_preds, tracker_name=result.tracker_name)
            smoothness: float = temp_out["aggregate"].get("mean_smoothness_score", 0.5)
        else:
            smoothness = 0.5

        return success_auc, eao, ees, smoothness, result.mean_fps, result.mean_iou

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, results: "List[BenchmarkResult]") -> List[TrackerScore]:
        """Compute normalized multi-dimensional scores for a cohort of trackers.

        Scores are min-max normalized *within* the cohort, so the result
        depends on which trackers are compared together.  To compare trackers
        across sessions, ensure all relevant results are passed in one call.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult`
                per tracker.  All results should cover the same dataset for
                meaningful cross-tracker comparison.

        Returns:
            List of :class:`TrackerScore` sorted by ``composite_score``
            descending (best tracker first).

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("results must contain at least one BenchmarkResult.")

        raw = [self._raw_metrics(r) for r in results]

        success_aucs = [x[0] for x in raw]
        eaos         = [x[1] for x in raw]
        eess         = [x[2] for x in raw]
        smoothnesses = [x[3] for x in raw]

        norm_acc  = self._minmax(success_aucs)
        norm_rob  = self._minmax(eaos)
        norm_eff  = self._minmax(eess)
        norm_temp = self._minmax(smoothnesses)

        scores: List[TrackerScore] = []
        for i, result in enumerate(results):
            composite = (
                self.accuracy_weight   * norm_acc[i]
                + self.robustness_weight * norm_rob[i]
                + self.efficiency_weight * norm_eff[i]
                + self.temporal_weight   * norm_temp[i]
            )
            scores.append(TrackerScore(
                tracker_name=result.tracker_name,
                dataset_name=result.dataset_name,
                accuracy_score=norm_acc[i],
                robustness_score=norm_rob[i],
                efficiency_score=norm_eff[i],
                temporal_score=norm_temp[i],
                composite_score=composite,
                mean_iou=raw[i][5],
                success_auc=raw[i][0],
                eao=raw[i][1],
                ees=raw[i][2],
                smoothness=raw[i][3],
                fps=raw[i][4],
            ))

        scores.sort(key=lambda s: s.composite_score, reverse=True)
        return scores

    def to_markdown_table(self, scores: List[TrackerScore]) -> str:
        """Format a ranked scorecard as a GitHub-flavored Markdown table.

        Args:
            scores: Output of :meth:`compute`.

        Returns:
            Multi-line Markdown string ready to embed in READMEs or reports.
        """
        header = (
            "| Rank | Tracker | Dataset | Composite ↑ | Accuracy ↑ | "
            "Robustness ↑ | Efficiency ↑ | Temporal ↑ | mIoU | FPS |"
        )
        sep = (
            "|------|---------|---------|------------:|-----------:|"
            "-------------:|-------------:|-----------:|-----:|----:|"
        )
        rows = [header, sep]
        for rank, s in enumerate(scores, start=1):
            rows.append(
                f"| {rank} | {s.tracker_name} | {s.dataset_name} "
                f"| **{s.composite_score:.4f}** "
                f"| {s.accuracy_score:.3f} "
                f"| {s.robustness_score:.3f} "
                f"| {s.efficiency_score:.3f} "
                f"| {s.temporal_score:.3f} "
                f"| {s.mean_iou:.4f} "
                f"| {s.fps:.1f} |"
            )
        return "\n".join(rows)

    def to_latex_table(self, scores: List[TrackerScore]) -> str:
        """Format a ranked scorecard as a LaTeX ``tabular`` for paper inclusion.

        The table uses ``booktabs`` rules (``\\toprule`` / ``\\midrule`` /
        ``\\bottomrule``) — add ``\\usepackage{booktabs}`` to your preamble.

        Args:
            scores: Output of :meth:`compute`.

        Returns:
            Multi-line LaTeX string.
        """
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{EOVOT Multi-Dimensional Tracker Comparison}",
            r"\label{tab:eovot-scorecard}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            (
                r"Tracker & Composite$\uparrow$ & Accuracy$\uparrow$ "
                r"& Robustness$\uparrow$ & Efficiency$\uparrow$ "
                r"& Temporal$\uparrow$ & FPS$\uparrow$ \\"
            ),
            r"\midrule",
        ]
        for s in scores:
            lines.append(
                f"{s.tracker_name} & {s.composite_score:.4f} & "
                f"{s.accuracy_score:.3f} & {s.robustness_score:.3f} & "
                f"{s.efficiency_score:.3f} & {s.temporal_score:.3f} & "
                f"{s.fps:.1f} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)
