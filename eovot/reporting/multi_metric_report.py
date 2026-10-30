"""Multi-metric unified experiment reporter for EOVOT.

The standard :class:`~eovot.experiment.runner.ExperimentRunner` leaderboard
ranks trackers by accuracy (success AUC / mIoU) only.  This module bridges
the gap between the existing metric engines — accuracy, robustness, temporal
consistency, and edge efficiency — and the experiment pipeline, producing a
single consolidated report for every tracker in a run.

The resulting table is publication-ready: each row shows a tracker's full
profile across all evaluation axes so researchers can make informed
accuracy-vs-efficiency tradeoffs without running separate analysis scripts.

Example
-------
::

    from eovot.reporting.multi_metric_report import MultiMetricReporter
    from eovot.benchmark.engine import BenchmarkResult

    reporter = MultiMetricReporter(memory_budget_mb=512.0)
    summaries = reporter.summarize(benchmark_results)
    print(reporter.to_markdown(summaries))
    report_dict = reporter.to_dict(summaries)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

from ..metrics.efficiency import EfficiencyMetricsEngine
from ..metrics.robustness import RobustnessAnalyzer
from ..metrics.temporal import TemporalConsistencyAnalyzer


@dataclass
class TrackerMultiMetricSummary:
    """All-axes evaluation summary for one tracker/dataset combination.

    Aggregates accuracy, robustness, temporal consistency, and edge
    efficiency into a single object so downstream code (report writers,
    leaderboard builders, serialisers) has one clear place to look.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        dataset_name: Dataset the tracker was evaluated on.
        num_sequences: Number of sequences in the benchmark run.

        mean_iou: Mean IoU across all frames (accuracy axis).
        success_auc: Area under the success curve (accuracy axis).
        precision_auc: Normalised AUC of the precision curve (accuracy axis).

        mean_fps: Mean frames-per-second throughput.
        peak_memory_mb: Peak RSS memory footprint in megabytes.
        mean_latency_ms: Mean per-frame processing latency in milliseconds.
        ees: Edge Efficiency Score — composite accuracy × throughput / memory.
        on_pareto_front: Whether this tracker lies on the accuracy–EES Pareto front.

        mean_eao: Expected Average Overlap averaged across sequences (robustness axis).
        total_failures: Total number of tracking failures across all sequences.
        mean_failures_per_sequence: Average failures per sequence.
        mean_recovery_lag_frames: Average frames to recovery after a failure.
        mean_survival_rate: Fraction of frames where IoU ≥ failure threshold.

        mean_smoothness_score: Composite temporal smoothness in ``(0, 1]``.
        mean_position_jitter: Normalised std of frame-to-frame displacement.
        mean_scale_jitter: Std of consecutive area ratios.
        mean_velocity_outlier_ratio: Fraction of frames with acceleration outliers.
    """

    tracker_name: str
    dataset_name: str
    num_sequences: int

    # Accuracy
    mean_iou: float = 0.0
    success_auc: float = 0.0
    precision_auc: float = 0.0

    # Efficiency
    mean_fps: float = 0.0
    peak_memory_mb: float = 0.0
    mean_latency_ms: float = 0.0
    ees: float = 0.0
    on_pareto_front: bool = False

    # Robustness
    mean_eao: float = 0.0
    total_failures: int = 0
    mean_failures_per_sequence: float = 0.0
    mean_recovery_lag_frames: float = 0.0
    mean_survival_rate: float = 0.0

    # Temporal consistency
    mean_smoothness_score: float = 0.0
    mean_position_jitter: float = 0.0
    mean_scale_jitter: float = 0.0
    mean_velocity_outlier_ratio: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "num_sequences": self.num_sequences,
            "accuracy": {
                "mean_iou": round(self.mean_iou, 4),
                "success_auc": round(self.success_auc, 4),
                "precision_auc": round(self.precision_auc, 4),
            },
            "efficiency": {
                "mean_fps": round(self.mean_fps, 2),
                "peak_memory_mb": round(self.peak_memory_mb, 2),
                "mean_latency_ms": round(self.mean_latency_ms, 3),
                "ees": round(self.ees, 4),
                "on_pareto_front": self.on_pareto_front,
            },
            "robustness": {
                "mean_eao": round(self.mean_eao, 4),
                "total_failures": self.total_failures,
                "mean_failures_per_sequence": round(self.mean_failures_per_sequence, 3),
                "mean_recovery_lag_frames": round(self.mean_recovery_lag_frames, 2),
                "mean_survival_rate": round(self.mean_survival_rate, 4),
            },
            "temporal": {
                "mean_smoothness_score": round(self.mean_smoothness_score, 4),
                "mean_position_jitter": round(self.mean_position_jitter, 6),
                "mean_scale_jitter": round(self.mean_scale_jitter, 6),
                "mean_velocity_outlier_ratio": round(self.mean_velocity_outlier_ratio, 4),
            },
        }


class MultiMetricReporter:
    """Compute and format multi-axis tracker evaluation reports.

    Aggregates accuracy, robustness, temporal consistency, and edge
    efficiency metrics from :class:`~eovot.benchmark.engine.BenchmarkResult`
    objects produced by the benchmark engine.

    Args:
        memory_budget_mb: Memory ceiling for Edge Efficiency Score computation.
            Default ``512.0`` MB (typical constrained edge device).
        robustness_failure_threshold: IoU threshold below which a frame is
            counted as a tracking failure.  Default ``0.1``.
        robustness_burn_in: Frames at the start of each sequence to skip
            before checking for failures.  Default ``5``.

    Example::

        reporter = MultiMetricReporter(memory_budget_mb=512.0)
        summaries = reporter.summarize(results)
        print(reporter.to_markdown(summaries))
        reporter.save_json(summaries, "results/multi_metric_report.json")
    """

    def __init__(
        self,
        memory_budget_mb: float = 512.0,
        robustness_failure_threshold: float = 0.1,
        robustness_burn_in: int = 5,
    ) -> None:
        self._efficiency_engine = EfficiencyMetricsEngine(
            memory_budget_mb=memory_budget_mb
        )
        self._robustness_analyzer = RobustnessAnalyzer(
            failure_threshold=robustness_failure_threshold,
            burn_in_frames=robustness_burn_in,
        )
        self._temporal_analyzer = TemporalConsistencyAnalyzer()

    # ------------------------------------------------------------------
    # Core summarization
    # ------------------------------------------------------------------

    def summarize(
        self, results: "List[BenchmarkResult]"
    ) -> List[TrackerMultiMetricSummary]:
        """Compute all-axes metrics from a list of benchmark results.

        Iterates over each :class:`~eovot.benchmark.engine.BenchmarkResult`,
        runs the robustness and temporal analyzers on its per-sequence data,
        and enriches it with an efficiency score.  Pareto fronts are computed
        once across the full result set.

        Args:
            results: One ``BenchmarkResult`` per tracker/dataset pair, as
                returned by :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.

        Returns:
            List of :class:`TrackerMultiMetricSummary`, one per input result,
            with ``on_pareto_front`` flags set.
        """
        summaries = []
        for result in results:
            summary = self._summarize_one(result)
            summaries.append(summary)

        self._mark_pareto_front(summaries)
        return summaries

    def _summarize_one(
        self, result: "BenchmarkResult"
    ) -> TrackerMultiMetricSummary:
        seq_results = result.sequence_results

        # --- accuracy ---
        miou = result.mean_iou
        sauc = result.mean_success_auc or miou
        pauc = result.mean_precision_auc or 0.0

        # --- efficiency ---
        mean_fps = result.mean_fps
        peak_mem = result.peak_memory_mb
        mean_lat = float(
            np.mean([r.profiling.latency_mean_ms for r in seq_results])
        ) if seq_results else 0.0
        ees = self._efficiency_engine.edge_efficiency_score(miou, mean_fps, peak_mem)

        # --- robustness ---
        seq_ious: Dict[str, np.ndarray] = {
            r.sequence_name: r.ious for r in seq_results
        }
        rob_agg = self._robustness_analyzer.analyze_benchmark(
            seq_ious, tracker_name=result.tracker_name
        )["aggregate"]

        # --- temporal consistency ---
        seq_preds: Dict[str, np.ndarray] = {}
        for r in seq_results:
            if r.predictions is not None and len(r.predictions) > 1:
                seq_preds[r.sequence_name] = r.predictions

        if seq_preds:
            temp_agg = self._temporal_analyzer.analyze_benchmark(
                seq_preds, tracker_name=result.tracker_name
            )["aggregate"]
        else:
            temp_agg = {
                "mean_smoothness_score": 0.0,
                "mean_position_jitter": 0.0,
                "mean_scale_jitter": 0.0,
                "mean_velocity_outlier_ratio": 0.0,
            }

        return TrackerMultiMetricSummary(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            num_sequences=len(seq_results),
            mean_iou=round(miou, 4),
            success_auc=round(sauc, 4),
            precision_auc=round(pauc, 4),
            mean_fps=round(mean_fps, 2),
            peak_memory_mb=round(peak_mem, 2),
            mean_latency_ms=round(mean_lat, 3),
            ees=round(ees, 4),
            mean_eao=round(float(rob_agg.get("mean_eao", 0.0)), 4),
            total_failures=int(rob_agg.get("total_failures", 0)),
            mean_failures_per_sequence=round(
                float(rob_agg.get("mean_failures_per_sequence", 0.0)), 3
            ),
            mean_recovery_lag_frames=round(
                float(rob_agg.get("mean_recovery_lag_frames", 0.0)), 2
            ),
            mean_survival_rate=round(
                float(rob_agg.get("mean_survival_rate", 0.0)), 4
            ),
            mean_smoothness_score=round(
                float(temp_agg.get("mean_smoothness_score", 0.0)), 4
            ),
            mean_position_jitter=round(
                float(temp_agg.get("mean_position_jitter", 0.0)), 6
            ),
            mean_scale_jitter=round(
                float(temp_agg.get("mean_scale_jitter", 0.0)), 6
            ),
            mean_velocity_outlier_ratio=round(
                float(temp_agg.get("mean_velocity_outlier_ratio", 0.0)), 4
            ),
        )

    def _mark_pareto_front(
        self, summaries: List[TrackerMultiMetricSummary]
    ) -> None:
        """Set ``on_pareto_front`` in-place using (mIoU, EES) objectives."""
        for i, candidate in enumerate(summaries):
            dominated = False
            for j, other in enumerate(summaries):
                if i == j:
                    continue
                if (
                    other.mean_iou >= candidate.mean_iou
                    and other.ees >= candidate.ees
                    and (other.mean_iou > candidate.mean_iou or other.ees > candidate.ees)
                ):
                    dominated = True
                    break
            candidate.on_pareto_front = not dominated

    # ------------------------------------------------------------------
    # Output formatters
    # ------------------------------------------------------------------

    def to_markdown(self, summaries: List[TrackerMultiMetricSummary]) -> str:
        """Render a comprehensive Markdown report sorted by EES descending.

        The report is structured into three Markdown tables:

        1. **Accuracy & Efficiency** — mIoU, success AUC, FPS, EES, Pareto flag
        2. **Robustness** — EAO, failures, recovery lag, survival rate
        3. **Temporal Consistency** — smoothness score, position/scale jitter, VOR

        Args:
            summaries: Output of :meth:`summarize`.

        Returns:
            Multi-line Markdown string suitable for writing to ``.md`` files.
        """
        if not summaries:
            return "No tracker results to report.\n"

        sorted_s = sorted(summaries, key=lambda s: s.ees, reverse=True)

        lines: List[str] = [
            "# EOVOT Multi-Metric Tracker Report\n",
            "## Accuracy & Efficiency\n",
            "| Rank | Tracker | Dataset | mIoU | Suc.AUC | FPS | Mem(MB) | EES | Pareto |",
            "|------|---------|---------|-----:|--------:|----:|--------:|----:|:------:|",
        ]
        for rank, s in enumerate(sorted_s, start=1):
            pareto = "✓" if s.on_pareto_front else ""
            lines.append(
                f"| {rank} | {s.tracker_name} | {s.dataset_name} "
                f"| {s.mean_iou:.4f} | {s.success_auc:.4f} "
                f"| {s.mean_fps:.1f} | {s.peak_memory_mb:.1f} "
                f"| {s.ees:.4f} | {pareto} |"
            )

        lines += [
            "",
            "## Robustness\n",
            "| Tracker | EAO | Failures | Fail/Seq | Recovery(fr) | Survival |",
            "|---------|----:|---------:|---------:|-------------:|---------:|",
        ]
        for s in sorted_s:
            lines.append(
                f"| {s.tracker_name} | {s.mean_eao:.4f} | {s.total_failures} "
                f"| {s.mean_failures_per_sequence:.2f} "
                f"| {s.mean_recovery_lag_frames:.1f} "
                f"| {s.mean_survival_rate:.4f} |"
            )

        lines += [
            "",
            "## Temporal Consistency\n",
            "| Tracker | Smoothness | Pos.Jitter | Scale.Jitter | VOR |",
            "|---------|----------:|-----------:|-------------:|----:|",
        ]
        for s in sorted_s:
            lines.append(
                f"| {s.tracker_name} | {s.mean_smoothness_score:.4f} "
                f"| {s.mean_position_jitter:.4f} | {s.mean_scale_jitter:.4f} "
                f"| {s.mean_velocity_outlier_ratio:.4f} |"
            )

        lines.append("")
        return "\n".join(lines)

    def to_dict(self, summaries: List[TrackerMultiMetricSummary]) -> Dict:
        """Serialise summaries to a JSON-serialisable dict.

        Args:
            summaries: Output of :meth:`summarize`.

        Returns:
            Dict with ``"trackers"`` list and schema metadata.
        """
        return {
            "schema_version": "1.0",
            "description": "EOVOT multi-metric tracker evaluation report",
            "trackers": [s.to_dict() for s in summaries],
        }

    def save_json(
        self,
        summaries: List[TrackerMultiMetricSummary],
        path: str,
    ) -> None:
        """Write the multi-metric report to a JSON file.

        Args:
            summaries: Output of :meth:`summarize`.
            path: Destination file path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(summaries), indent=2), encoding="utf-8"
        )
