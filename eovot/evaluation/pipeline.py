"""Unified evaluation pipeline for EOVOT.

Chains five metric layers on top of a standard
:class:`~eovot.benchmark.engine.BenchmarkEngine` run:

1. **Robustness** — failure detection, EAO, survival rate
   (:class:`~eovot.metrics.robustness.RobustnessAnalyzer`)
2. **Temporal consistency** — position / scale jitter, smoothness score
   (:class:`~eovot.metrics.temporal.TemporalConsistencyAnalyzer`)
3. **Attribute analysis** — per-challenge-attribute accuracy breakdown
   (:class:`~eovot.metrics.attributes.AttributeAnalyzer`)
4. **Efficiency** — Edge Efficiency Score and Pareto-front ranking
   (:class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`)
5. **Statistical testing** — pairwise Wilcoxon tests with Bonferroni
   correction when two or more trackers are evaluated
   (:class:`~eovot.metrics.statistical.StatisticalTestEngine`)

Typical usage::

    from eovot.evaluation import EvaluationPipeline
    from eovot.trackers.registry import build_tracker
    from eovot.datasets.synthetic import SyntheticDataset

    pipeline = EvaluationPipeline(output_dir="results/eval")
    report = pipeline.run(
        trackers=[build_tracker("MOSSE"), build_tracker("KCF"), build_tracker("DSST")],
        dataset=SyntheticDataset(num_sequences=10, num_frames=120, motion="linear"),
        dataset_name="Synthetic-Linear",
    )
    print(report.to_markdown())
    paths = report.save("results/eval", name="linear_eval")
    # paths["markdown"] → results/eval/linear_eval.md
    # paths["json"]     → results/eval/linear_eval.json

You can also analyze pre-computed results without re-running the benchmark::

    report = pipeline.run_from_results([result_mosse, result_kcf, result_dsst])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..metrics.attributes import AttributeAnalyzer, AttributePerformanceTable
from ..metrics.efficiency import EfficiencyEntry, EfficiencyMetricsEngine
from ..metrics.robustness import RobustnessAnalyzer
from ..metrics.statistical import PairwiseSummary, StatisticalTestEngine
from ..metrics.temporal import TemporalConsistencyAnalyzer

if TYPE_CHECKING:
    from ..datasets.base import BaseDataset
    from ..trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Aggregated output of the full :class:`EvaluationPipeline`.

    Attributes:
        tracker_names: Ordered list of evaluated tracker names.
        dataset_name: Name of the dataset used for evaluation.
        benchmark_results: Raw :class:`~eovot.benchmark.engine.BenchmarkResult`
            objects, one per tracker.
        robustness: Per-tracker robustness summary dicts as returned by
            :meth:`~eovot.metrics.robustness.RobustnessAnalyzer.analyze_benchmark`.
        temporal: Per-tracker temporal consistency summary dicts as returned by
            :meth:`~eovot.metrics.temporal.TemporalConsistencyAnalyzer.analyze_benchmark`.
        attributes: Per-tracker :class:`~eovot.metrics.attributes.AttributePerformanceTable`.
        efficiency: Trackers ranked by Edge Efficiency Score (highest first),
            with Pareto-front flags set.
        statistical: Pairwise Wilcoxon comparison, or ``None`` when only one
            tracker was evaluated.
    """

    tracker_names: List[str]
    dataset_name: str
    benchmark_results: List[BenchmarkResult]
    robustness: Dict[str, Dict] = field(default_factory=dict)
    temporal: Dict[str, Dict] = field(default_factory=dict)
    attributes: Dict[str, AttributePerformanceTable] = field(default_factory=dict)
    efficiency: List[EfficiencyEntry] = field(default_factory=list)
    statistical: Optional[PairwiseSummary] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the full evaluation as a Markdown document.

        Returns:
            Multi-section Markdown string suitable for saving as ``.md`` or
            embedding in a GitHub wiki page.
        """
        sections: List[str] = []

        sections.append(f"# EOVOT Evaluation Report — {self.dataset_name}")
        sections.append(
            f"**Trackers evaluated:** {', '.join(self.tracker_names)}  \n"
            f"**Sequences:** "
            f"{len(self.benchmark_results[0].sequence_results) if self.benchmark_results else 0}"
        )

        # --- Accuracy overview ---
        sections.append("## Accuracy Overview")
        sections.append(self._accuracy_table())

        # --- Robustness ---
        sections.append("## Robustness")
        sections.append(self._robustness_table())

        # --- Temporal consistency ---
        sections.append("## Temporal Consistency")
        sections.append(self._temporal_table())

        # --- Efficiency ---
        sections.append("## Edge Efficiency")
        if self.efficiency:
            eff_engine = EfficiencyMetricsEngine()
            sections.append(eff_engine.to_markdown_table(self.efficiency))
        else:
            sections.append("*No efficiency data available.*")

        # --- Attribute breakdown ---
        sections.append("## Per-Attribute Performance")
        for name in self.tracker_names:
            table = self.attributes.get(name)
            if table is not None:
                sections.append(table.to_markdown())

        # --- Statistical tests ---
        sections.append("## Statistical Significance")
        if self.statistical is not None:
            sections.append(self.statistical.to_markdown())
        else:
            sections.append("*Single-tracker evaluation — no pairwise tests run.*")

        return "\n\n".join(sections)

    def save(self, output_dir: str, name: str = "evaluation") -> Dict[str, Path]:
        """Persist the report to disk as both Markdown and JSON.

        Args:
            output_dir: Directory to write files into (created if absent).
            name: Base file name (without extension).

        Returns:
            Dict with keys ``"markdown"`` and ``"json"`` pointing to the
            written :class:`~pathlib.Path` objects.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        md_path = out / f"{name}.md"
        md_path.write_text(self.to_markdown(), encoding="utf-8")

        json_path = out / f"{name}.json"
        json_path.write_text(
            json.dumps(self._to_dict(), indent=2, default=_json_default),
            encoding="utf-8",
        )

        return {"markdown": md_path, "json": json_path}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _accuracy_table(self) -> str:
        header = (
            "| Tracker | mIoU | Success AUC | Precision AUC | FPS | Mem (MB) |\n"
            "|---------|-----:|------------:|--------------:|----:|---------:|"
        )
        rows = []
        for r in self.benchmark_results:
            sauc = r.mean_success_auc
            pauc = r.mean_precision_auc
            sauc_s = f"{sauc:.4f}" if sauc is not None else "—"
            pauc_s = f"{pauc:.4f}" if pauc is not None else "—"
            rows.append(
                f"| {r.tracker_name} "
                f"| {r.mean_iou:.4f} "
                f"| {sauc_s} "
                f"| {pauc_s} "
                f"| {r.mean_fps:.1f} "
                f"| {r.peak_memory_mb:.1f} |"
            )
        return header + "\n" + "\n".join(rows)

    def _robustness_table(self) -> str:
        header = (
            "| Tracker | Failures | Mean EAO | Survival | Recovery Lag (fr) |\n"
            "|---------|----------|------:|--------:|------:|"
        )
        rows = []
        for name in self.tracker_names:
            agg = self.robustness.get(name, {}).get("aggregate", {})
            if not agg:
                rows.append(f"| {name} | — | — | — | — |")
                continue
            rows.append(
                f"| {name} "
                f"| {agg.get('total_failures', '—')} "
                f"| {agg.get('mean_eao', 0.0):.4f} "
                f"| {agg.get('mean_survival_rate', 0.0):.4f} "
                f"| {agg.get('mean_recovery_lag_frames', 0.0):.1f} |"
            )
        return header + "\n" + "\n".join(rows)

    def _temporal_table(self) -> str:
        header = (
            "| Tracker | Smoothness | Pos Jitter | Scale Jitter | VOR |\n"
            "|---------|------:|------:|------:|------:|"
        )
        rows = []
        for name in self.tracker_names:
            agg = self.temporal.get(name, {}).get("aggregate", {})
            if not agg:
                rows.append(f"| {name} | — | — | — | — |")
                continue
            rows.append(
                f"| {name} "
                f"| {agg.get('mean_smoothness_score', 0.0):.4f} "
                f"| {agg.get('mean_position_jitter', 0.0):.6f} "
                f"| {agg.get('mean_scale_jitter', 0.0):.6f} "
                f"| {agg.get('mean_velocity_outlier_ratio', 0.0):.4f} |"
            )
        return header + "\n" + "\n".join(rows)

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "tracker_names": self.tracker_names,
            "dataset_name": self.dataset_name,
            "robustness": {
                name: {
                    "aggregate": data.get("aggregate", {}),
                }
                for name, data in self.robustness.items()
            },
            "temporal": {
                name: {
                    "aggregate": data.get("aggregate", {}),
                }
                for name, data in self.temporal.items()
            },
            "attributes": {
                name: table.to_dict()
                for name, table in self.attributes.items()
            },
            "efficiency": [
                {
                    "tracker_name": e.tracker_name,
                    "dataset_name": e.dataset_name,
                    "mean_iou": e.mean_iou,
                    "fps": e.fps,
                    "peak_memory_mb": e.peak_memory_mb,
                    "ees": e.ees,
                    "on_pareto_front": e.on_pareto_front,
                }
                for e in self.efficiency
            ],
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class EvaluationPipeline:
    """End-to-end multi-layer evaluation pipeline.

    Args:
        output_dir: Default directory for :meth:`EvaluationReport.save`.
            Not written to automatically — pass *output_dir* explicitly
            in ``save()`` calls.
        memory_budget_mb: Memory ceiling (MB) for the Edge Efficiency Score
            denominator.  Default: ``512.0`` MB.
        verbose: If ``True`` the benchmark engine prints per-sequence progress.
        n_bootstrap: Bootstrap resamples for statistical CIs.  Default: 2000
            (lower than the module default for faster pipeline runs).
        statistical_alpha: Significance level for Wilcoxon tests.

    Example::

        pipeline = EvaluationPipeline(memory_budget_mb=256.0)
        report = pipeline.run(
            trackers=[MOSSETracker(), KCFTracker(), DSSTTracker()],
            dataset=SyntheticDataset(num_sequences=10, num_frames=120),
            dataset_name="Synthetic",
        )
        report.save("results/eval", name="mosse_kcf_dsst")
    """

    def __init__(
        self,
        output_dir: str = "results/evaluation",
        memory_budget_mb: float = 512.0,
        verbose: bool = True,
        n_bootstrap: int = 2000,
        statistical_alpha: float = 0.05,
    ) -> None:
        self.output_dir = Path(output_dir)
        self._bench = BenchmarkEngine(verbose=verbose)
        self._rob = RobustnessAnalyzer()
        self._temp = TemporalConsistencyAnalyzer()
        self._attr = AttributeAnalyzer()
        self._eff = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)
        self._stat = StatisticalTestEngine(
            alpha=statistical_alpha,
            n_bootstrap=n_bootstrap,
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(
        self,
        trackers: "List[BaseTracker]",
        dataset: "BaseDataset",
        dataset_name: str = "Dataset",
    ) -> EvaluationReport:
        """Run benchmarks for all trackers then apply every analysis layer.

        Args:
            trackers: One or more :class:`~eovot.trackers.base.BaseTracker`
                instances to evaluate.
            dataset: Dataset supplying sequences and ground-truth boxes.
            dataset_name: Human-readable dataset label embedded in the report.

        Returns:
            :class:`EvaluationReport` with results from all five layers.
        """
        results: List[BenchmarkResult] = []
        for tracker in trackers:
            result = self._bench.run(tracker, dataset, dataset_name=dataset_name)
            results.append(result)
        return self.run_from_results(results)

    def run_from_results(
        self,
        results: List[BenchmarkResult],
    ) -> EvaluationReport:
        """Apply all analysis layers to pre-computed benchmark results.

        Useful when benchmarks have already been run and only the higher-level
        analysis layers need to be (re-)applied.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult` per
                tracker.  All results must share the same dataset and sequence
                names for statistical tests to be meaningful.

        Returns:
            :class:`EvaluationReport` with results from all five layers.
        """
        if not results:
            raise ValueError("run_from_results requires at least one BenchmarkResult.")

        dataset_name = results[0].dataset_name
        tracker_names = [r.tracker_name for r in results]

        # 1. Robustness
        robustness = self._run_robustness(results)

        # 2. Temporal consistency
        temporal = self._run_temporal(results)

        # 3. Attribute breakdown
        attributes = self._run_attributes(results)

        # 4. Efficiency
        efficiency = self._eff.rank_trackers(results)

        # 5. Statistical tests (only when ≥ 2 trackers share ≥ 2 sequences)
        statistical = self._run_statistical(results)

        return EvaluationReport(
            tracker_names=tracker_names,
            dataset_name=dataset_name,
            benchmark_results=results,
            robustness=robustness,
            temporal=temporal,
            attributes=attributes,
            efficiency=efficiency,
            statistical=statistical,
        )

    # ------------------------------------------------------------------
    # Private layer runners
    # ------------------------------------------------------------------

    def _run_robustness(
        self, results: List[BenchmarkResult]
    ) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for r in results:
            seq_ious = {
                sr.sequence_name: sr.ious for sr in r.sequence_results
            }
            out[r.tracker_name] = self._rob.analyze_benchmark(
                seq_ious, tracker_name=r.tracker_name
            )
        return out

    def _run_temporal(
        self, results: List[BenchmarkResult]
    ) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for r in results:
            seq_preds = {
                sr.sequence_name: sr.predictions
                for sr in r.sequence_results
                if sr.predictions is not None
            }
            out[r.tracker_name] = self._temp.analyze_benchmark(
                seq_preds, tracker_name=r.tracker_name
            )
        return out

    def _run_attributes(
        self, results: List[BenchmarkResult]
    ) -> Dict[str, AttributePerformanceTable]:
        out: Dict[str, AttributePerformanceTable] = {}
        for r in results:
            out[r.tracker_name] = self._attr.breakdown(r)
        return out

    def _run_statistical(
        self, results: List[BenchmarkResult]
    ) -> Optional[PairwiseSummary]:
        if len(results) < 2:
            return None

        # Use per-sequence mean IoU as the paired score for Wilcoxon tests.
        # Align by sequence name so only sequences present in ALL results are used.
        common_seqs = None
        seq_map: Dict[str, Dict[str, float]] = {}
        for r in results:
            seqs = {sr.sequence_name: sr.mean_iou for sr in r.sequence_results}
            seq_map[r.tracker_name] = seqs
            if common_seqs is None:
                common_seqs = set(seqs)
            else:
                common_seqs &= set(seqs)

        if not common_seqs or len(common_seqs) < 2:
            return None

        sorted_seqs = sorted(common_seqs)
        tracker_scores: Dict[str, List[float]] = {
            name: [seq_map[name][s] for s in sorted_seqs]
            for name in [r.tracker_name for r in results]
        }

        try:
            return self._stat.pairwise_report(tracker_scores)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
