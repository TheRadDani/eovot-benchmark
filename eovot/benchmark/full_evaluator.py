"""Full-pipeline benchmark evaluator integrating all EOVOT metric modules.

:class:`FullBenchmarkEvaluator` is the single entry point for a complete,
publication-ready benchmark evaluation.  It runs the core benchmark loop
and automatically applies every analysis module — robustness, temporal
consistency, sequence-attribute breakdown, and edge-efficiency ranking —
so researchers do not need to chain the modules manually.

Typical usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.benchmark.full_evaluator import FullBenchmarkEvaluator

    dataset   = SyntheticDataset(num_sequences=20, num_frames=300)
    trackers  = [MOSSETracker(), KCFTracker()]

    evaluator = FullBenchmarkEvaluator(memory_budget_mb=512.0, verbose=True)
    report    = evaluator.run(trackers, dataset, dataset_name="Synthetic")

    print(report.summary_markdown())
    report.save("results/full_eval.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..metrics.robustness import RobustnessAnalyzer
from ..metrics.temporal import TemporalConsistencyAnalyzer
from ..metrics.attributes import AttributeAnalyzer, AttributePerformanceTable
from ..metrics.efficiency import EfficiencyMetricsEngine, EfficiencyEntry
from ..trackers.base import BaseTracker


@dataclass
class TrackerFullReport:
    """All metric modules applied to one tracker on one dataset.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Name of the dataset.
        benchmark_result: Raw accuracy and profiling output from
            :class:`~eovot.benchmark.engine.BenchmarkEngine`.
        total_failures: Cumulative failure count across all sequences.
        mean_failures_per_sequence: Average failures per sequence.
        mean_eao: Mean Expected Average Overlap (simplified, no re-init).
        mean_survival_rate: Fraction of frames the tracker stayed alive.
        mean_recovery_lag_frames: Mean frames to recover after a failure.
        mean_smoothness_score: Mean temporal smoothness in ``(0, 1]``.
        mean_position_jitter: Mean normalised position jitter.
        mean_scale_jitter: Mean consecutive area-ratio std.
        mean_velocity_outlier_ratio: Mean fraction of acceleration outliers.
        attribute_table: Per-challenge attribute breakdown, or ``None``
            when no GT boxes were available for attribute detection.
        efficiency_entry: EES and Pareto-front flag (set after all trackers
            are evaluated so the Pareto computation sees all competitors).
    """

    tracker_name: str
    dataset_name: str
    benchmark_result: BenchmarkResult

    # Robustness
    total_failures: int = 0
    mean_failures_per_sequence: float = 0.0
    mean_eao: float = 0.0
    mean_survival_rate: float = 0.0
    mean_recovery_lag_frames: float = 0.0

    # Temporal consistency
    mean_smoothness_score: float = 0.0
    mean_position_jitter: float = 0.0
    mean_scale_jitter: float = 0.0
    mean_velocity_outlier_ratio: float = 0.0

    # Per-attribute breakdown
    attribute_table: Optional[AttributePerformanceTable] = None

    # Efficiency (populated after all trackers have been evaluated)
    efficiency_entry: Optional[EfficiencyEntry] = None

    def to_dict(self) -> Dict:
        """Serialise to a JSON-compatible dict."""
        d: Dict = {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "accuracy": self.benchmark_result.summary(),
            "robustness": {
                "total_failures": self.total_failures,
                "mean_failures_per_sequence": round(self.mean_failures_per_sequence, 3),
                "mean_eao": round(self.mean_eao, 4),
                "mean_survival_rate": round(self.mean_survival_rate, 4),
                "mean_recovery_lag_frames": round(self.mean_recovery_lag_frames, 2),
            },
            "temporal_consistency": {
                "mean_smoothness_score": round(self.mean_smoothness_score, 4),
                "mean_position_jitter": round(self.mean_position_jitter, 4),
                "mean_scale_jitter": round(self.mean_scale_jitter, 4),
                "mean_velocity_outlier_ratio": round(self.mean_velocity_outlier_ratio, 4),
            },
        }
        if self.attribute_table is not None:
            d["attribute_breakdown"] = self.attribute_table.to_dict()
        if self.efficiency_entry is not None:
            d["efficiency"] = {
                "ees": round(self.efficiency_entry.ees, 4),
                "on_pareto_front": self.efficiency_entry.on_pareto_front,
            }
        return d


@dataclass
class FullEvaluationReport:
    """Complete multi-tracker evaluation report from :class:`FullBenchmarkEvaluator`.

    Attributes:
        dataset_name: Name of the evaluated dataset.
        tracker_reports: One :class:`TrackerFullReport` per tracker.
        efficiency_ranking: Trackers sorted by EES (best first), with
            Pareto-front flags set.
        wall_clock_seconds: Total wall-clock time for the full evaluation.
    """

    dataset_name: str
    tracker_reports: List[TrackerFullReport] = field(default_factory=list)
    efficiency_ranking: List[EfficiencyEntry] = field(default_factory=list)
    wall_clock_seconds: float = 0.0

    def summary_markdown(self) -> str:
        """Generate a publication-ready Markdown summary covering all metric modules.

        Returns:
            Multi-section Markdown string with accuracy, robustness, temporal
            consistency, and efficiency tables — ready to paste into a README
            or research paper appendix.
        """
        lines = [
            "# EOVOT Full Evaluation Report",
            "",
            (
                f"**Dataset:** {self.dataset_name}  "
                f"**Trackers:** {len(self.tracker_reports)}  "
                f"**Time:** {self.wall_clock_seconds:.1f}s"
            ),
            "",
        ]

        # --- Accuracy & throughput table ---
        lines += [
            "## Accuracy & Throughput",
            "",
            "| Tracker | mIoU | Succ. AUC | Prec. AUC | FPS | Mem (MiB) |",
            "|---------|-----:|----------:|----------:|----:|----------:|",
        ]
        for r in sorted(
            self.tracker_reports,
            key=lambda x: x.benchmark_result.mean_iou,
            reverse=True,
        ):
            br = r.benchmark_result
            sauc = br.mean_success_auc
            pauc = br.mean_precision_auc
            sauc_str = f"{sauc:.4f}" if sauc is not None else "—"
            pauc_str = f"{pauc:.4f}" if pauc is not None else "—"
            lines.append(
                f"| {r.tracker_name} "
                f"| {br.mean_iou:.4f} "
                f"| {sauc_str} "
                f"| {pauc_str} "
                f"| {br.mean_fps:.1f} "
                f"| {br.peak_memory_mb:.1f} |"
            )
        lines.append("")

        # --- Robustness table ---
        lines += [
            "## Robustness",
            "",
            "| Tracker | Failures | Mean EAO ↑ | Survival ↑ | Recov. Lag ↓ |",
            "|---------|--------:|-----------:|-----------:|-------------:|",
        ]
        for r in sorted(self.tracker_reports, key=lambda x: x.mean_eao, reverse=True):
            lines.append(
                f"| {r.tracker_name} "
                f"| {r.total_failures} "
                f"| {r.mean_eao:.4f} "
                f"| {r.mean_survival_rate:.3f} "
                f"| {r.mean_recovery_lag_frames:.1f} fr |"
            )
        lines.append("")

        # --- Temporal consistency table ---
        lines += [
            "## Temporal Consistency",
            "",
            "| Tracker | Smoothness ↑ | Pos. Jitter ↓ | Scale Jitter ↓ | VOR ↓ |",
            "|---------|------------:|-------------:|--------------:|------:|",
        ]
        for r in sorted(
            self.tracker_reports, key=lambda x: x.mean_smoothness_score, reverse=True
        ):
            lines.append(
                f"| {r.tracker_name} "
                f"| {r.mean_smoothness_score:.4f} "
                f"| {r.mean_position_jitter:.4f} "
                f"| {r.mean_scale_jitter:.4f} "
                f"| {r.mean_velocity_outlier_ratio:.4f} |"
            )
        lines.append("")

        # --- Efficiency ranking ---
        if self.efficiency_ranking:
            lines += [
                "## Edge Efficiency Ranking",
                "",
                "| Rank | Tracker | EES ↑ | mIoU | FPS | Mem (MiB) | Pareto |",
                "|-----:|---------|------:|-----:|----:|---------:|:------:|",
            ]
            for i, e in enumerate(self.efficiency_ranking, start=1):
                pareto = "✓" if e.on_pareto_front else ""
                lines.append(
                    f"| {i} | {e.tracker_name} "
                    f"| {e.ees:.4f} "
                    f"| {e.mean_iou:.4f} "
                    f"| {e.fps:.1f} "
                    f"| {e.peak_memory_mb:.1f} "
                    f"| {pareto} |"
                )

        return "\n".join(lines)

    def save(self, path: Union[str, Path]) -> Path:
        """Serialise the full report to a JSON file.

        Args:
            path: Destination path.  A ``.json`` suffix is added if absent.

        Returns:
            The resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dataset_name": self.dataset_name,
            "wall_clock_seconds": round(self.wall_clock_seconds, 2),
            "trackers": [r.to_dict() for r in self.tracker_reports],
            "efficiency_ranking": [
                {
                    "rank": i + 1,
                    "tracker_name": e.tracker_name,
                    "ees": round(e.ees, 4),
                    "mean_iou": round(e.mean_iou, 4),
                    "fps": round(e.fps, 2),
                    "peak_memory_mb": round(e.peak_memory_mb, 2),
                    "on_pareto_front": e.on_pareto_front,
                }
                for i, e in enumerate(self.efficiency_ranking)
            ],
        }
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return p


class FullBenchmarkEvaluator:
    """Run all EOVOT metric modules on multiple trackers in one call.

    Chains together five analysis stages:

    1. :class:`~eovot.benchmark.engine.BenchmarkEngine` — accuracy + profiling + energy
    2. :class:`~eovot.metrics.robustness.RobustnessAnalyzer` — failure count, EAO, survival
    3. :class:`~eovot.metrics.temporal.TemporalConsistencyAnalyzer` — smoothness metrics
    4. :class:`~eovot.metrics.attributes.AttributeAnalyzer` — per-challenge breakdown
    5. :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine` — Pareto-front ranking

    The Pareto computation is performed **after** all trackers are evaluated so
    it sees the full comparison set.

    Args:
        memory_budget_mb: Memory ceiling (MiB) for EES computation.
            Default: ``512.0``.
        tdp_watts: Device TDP for energy profiling (``None`` disables it).
        verbose: Print per-sequence benchmark progress.  Default: ``True``.

    Example::

        evaluator = FullBenchmarkEvaluator(memory_budget_mb=256.0, tdp_watts=6.0)
        report = evaluator.run(
            [MOSSETracker(), KCFTracker()],
            SyntheticDataset(num_sequences=20, num_frames=300),
            dataset_name="Synthetic",
        )
        print(report.summary_markdown())
        report.save("results/full_eval.json")
    """

    def __init__(
        self,
        memory_budget_mb: float = 512.0,
        tdp_watts: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        self._engine = BenchmarkEngine(verbose=verbose, tdp_watts=tdp_watts)
        self._robustness = RobustnessAnalyzer()
        self._temporal = TemporalConsistencyAnalyzer()
        self._attributes = AttributeAnalyzer()
        self._efficiency = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)
        self.verbose = verbose

    def run(
        self,
        trackers: List[BaseTracker],
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> FullEvaluationReport:
        """Evaluate all *trackers* on *dataset* and return a full report.

        Each tracker is benchmarked in turn, then all analysis modules are
        applied, and finally a cross-tracker efficiency ranking is computed.

        Args:
            trackers: List of :class:`~eovot.trackers.base.BaseTracker` instances.
            dataset: Dataset to evaluate on.
            dataset_name: Human-readable dataset label used in tables.
            max_sequences: Cap number of sequences (useful for quick trials).

        Returns:
            :class:`FullEvaluationReport` with all metrics populated.
        """
        t_start = time.perf_counter()
        tracker_reports: List[TrackerFullReport] = []
        benchmark_results: List[BenchmarkResult] = []

        for tracker in trackers:
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Evaluating: {tracker.name}")
                print(f"{'=' * 60}")

            br = self._engine.run(tracker, dataset, dataset_name, max_sequences)
            benchmark_results.append(br)

            # Robustness analysis.
            seq_ious = {r.sequence_name: r.ious for r in br.sequence_results}
            rob = self._robustness.analyze_benchmark(seq_ious, tracker_name=tracker.name)
            rob_agg = rob["aggregate"]

            # Temporal consistency — only for sequences with stored predictions.
            t_scores: List[float] = []
            t_jitter: List[float] = []
            t_scale: List[float] = []
            t_vor: List[float] = []
            for sr in br.sequence_results:
                if sr.predictions is not None and len(sr.predictions) > 1:
                    tc = self._temporal.analyze(
                        sr.predictions,
                        tracker_name=tracker.name,
                        sequence_name=sr.sequence_name,
                    )
                    t_scores.append(tc.smoothness_score)
                    t_jitter.append(tc.position_jitter)
                    t_scale.append(tc.scale_jitter)
                    t_vor.append(tc.velocity_outlier_ratio)

            # Per-attribute breakdown.
            attr_table = self._attributes.breakdown(br)

            report = TrackerFullReport(
                tracker_name=tracker.name,
                dataset_name=dataset_name,
                benchmark_result=br,
                total_failures=rob_agg["total_failures"],
                mean_failures_per_sequence=rob_agg["mean_failures_per_sequence"],
                mean_eao=rob_agg["mean_eao"],
                mean_survival_rate=rob_agg["mean_survival_rate"],
                mean_recovery_lag_frames=rob_agg["mean_recovery_lag_frames"],
                mean_smoothness_score=float(np.mean(t_scores)) if t_scores else 0.0,
                mean_position_jitter=float(np.mean(t_jitter)) if t_jitter else 0.0,
                mean_scale_jitter=float(np.mean(t_scale)) if t_scale else 0.0,
                mean_velocity_outlier_ratio=float(np.mean(t_vor)) if t_vor else 0.0,
                attribute_table=attr_table,
            )
            tracker_reports.append(report)

        # Efficiency ranking across all trackers (Pareto needs all competitors).
        efficiency_ranking = self._efficiency.rank_trackers(benchmark_results)

        # Wire efficiency entries back into per-tracker reports.
        eff_map = {e.tracker_name: e for e in efficiency_ranking}
        for r in tracker_reports:
            r.efficiency_entry = eff_map.get(r.tracker_name)

        return FullEvaluationReport(
            dataset_name=dataset_name,
            tracker_reports=tracker_reports,
            efficiency_ranking=efficiency_ranking,
            wall_clock_seconds=time.perf_counter() - t_start,
        )
