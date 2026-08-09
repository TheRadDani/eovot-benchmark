"""Full multi-dimensional evaluation pipeline for EOVOT.

Wraps :class:`~eovot.benchmark.engine.BenchmarkEngine` with a post-run
analysis stage that computes all four metric dimensions for each tracker:

1. **Accuracy** — mIoU, success AUC, precision AUC
   (produced by :class:`~eovot.benchmark.engine.BenchmarkEngine`)
2. **Robustness** — failure count, EAO, survival rate
   (via :class:`~eovot.metrics.robustness.RobustnessAnalyzer`)
3. **Temporal consistency** — smoothness score, position/scale jitter
   (via :class:`~eovot.metrics.temporal.TemporalConsistencyAnalyzer`)
4. **Edge efficiency** — EES, Pareto-front membership
   (via :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`)

The result is a :class:`FullEvaluationReport` that bundles all four
dimensions per tracker and generates a combined ranked leaderboard.

Typical usage::

    from eovot.experiment.full_evaluation import FullEvaluationPipeline
    from eovot.trackers.registry import build_tracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset  = SyntheticDataset(num_sequences=5, num_frames=50)
    trackers = [build_tracker("MOSSE"), build_tracker("KCF")]

    pipeline = FullEvaluationPipeline(memory_budget_mb=256.0)
    report   = pipeline.run(trackers, dataset, dataset_name="Synthetic")

    print(report.to_markdown_table())
    report.save("results/full_report.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..metrics.efficiency import EfficiencyEntry, EfficiencyMetricsEngine
from ..metrics.robustness import RobustnessAnalyzer
from ..metrics.temporal import TemporalConsistencyAnalyzer
from ..trackers.base import BaseTracker


@dataclass
class TrackerDimensions:
    """All four metric dimensions for one tracker on one dataset.

    Produced by :class:`FullEvaluationPipeline` after running the benchmark
    engine and all post-run analyzers.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        dataset_name: Dataset on which the tracker was evaluated.
        mean_iou: Mean IoU across all evaluated frames.
        success_auc: Success-curve AUC, or ``None`` if not computed.
        precision_auc: Precision-curve AUC, or ``None`` if not computed.
        mean_fps: Mean frames-per-second throughput.
        peak_memory_mb: Peak RSS memory footprint in megabytes.
        total_failures: Total tracking failures detected across all sequences.
        mean_failures_per_sequence: Average failures per sequence.
        mean_eao: Mean Expected Average Overlap across sequences.
        mean_survival_rate: Fraction of frames where IoU ≥ failure threshold.
        mean_recovery_lag_frames: Mean frames-to-recovery after a failure.
        mean_smoothness_score: Composite smoothness scalar in ``(0, 1]``.
        mean_position_jitter: Normalised position jitter (lower = smoother).
        mean_scale_jitter: Scale jitter standard deviation (lower = smoother).
        mean_velocity_outlier_ratio: Fraction of acceleration-outlier frames.
        ees: Edge Efficiency Score — accuracy × throughput / memory.
        on_pareto_front: ``True`` when no other tracker dominates this one in
            both mIoU and EES.
    """

    tracker_name: str
    dataset_name: str

    # Accuracy
    mean_iou: float = 0.0
    success_auc: Optional[float] = None
    precision_auc: Optional[float] = None

    # Hardware profiling
    mean_fps: float = 0.0
    peak_memory_mb: float = 0.0

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

    # Edge efficiency
    ees: float = 0.0
    on_pareto_front: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a nested dict compatible with JSON export."""
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "accuracy": {
                "mean_iou": round(self.mean_iou, 4),
                "success_auc": (
                    round(self.success_auc, 4)
                    if self.success_auc is not None
                    else None
                ),
                "precision_auc": (
                    round(self.precision_auc, 4)
                    if self.precision_auc is not None
                    else None
                ),
            },
            "profiling": {
                "mean_fps": round(self.mean_fps, 2),
                "peak_memory_mb": round(self.peak_memory_mb, 2),
            },
            "robustness": {
                "total_failures": self.total_failures,
                "mean_failures_per_sequence": round(
                    self.mean_failures_per_sequence, 3
                ),
                "mean_eao": round(self.mean_eao, 4),
                "mean_survival_rate": round(self.mean_survival_rate, 4),
                "mean_recovery_lag_frames": round(
                    self.mean_recovery_lag_frames, 2
                ),
            },
            "temporal_consistency": {
                "mean_smoothness_score": round(self.mean_smoothness_score, 4),
                "mean_position_jitter": round(self.mean_position_jitter, 6),
                "mean_scale_jitter": round(self.mean_scale_jitter, 6),
                "mean_velocity_outlier_ratio": round(
                    self.mean_velocity_outlier_ratio, 4
                ),
            },
            "edge_efficiency": {
                "ees": round(self.ees, 4),
                "on_pareto_front": self.on_pareto_front,
            },
        }

    def __str__(self) -> str:
        sauc = f"{self.success_auc:.4f}" if self.success_auc is not None else "N/A"
        return (
            f"TrackerDimensions[{self.tracker_name} on {self.dataset_name}]\n"
            f"  Accuracy:  mIoU={self.mean_iou:.4f}  AUC={sauc}\n"
            f"  Profiling: FPS={self.mean_fps:.1f}  Mem={self.peak_memory_mb:.1f} MB\n"
            f"  Robustness: failures={self.total_failures}  EAO={self.mean_eao:.4f}  "
            f"survival={self.mean_survival_rate:.4f}\n"
            f"  Temporal:  S={self.mean_smoothness_score:.4f}  "
            f"jitter={self.mean_position_jitter:.4f}\n"
            f"  Efficiency: EES={self.ees:.4f}  Pareto={'yes' if self.on_pareto_front else 'no'}"
        )


@dataclass
class FullEvaluationReport:
    """Multi-tracker full-dimension evaluation report.

    Produced by :class:`FullEvaluationPipeline.run`.  Contains one
    :class:`TrackerDimensions` entry per tracker evaluated.

    Attributes:
        dataset_name: Name of the dataset used for evaluation.
        evaluations: Per-tracker dimension summaries, sorted by EES descending.
    """

    dataset_name: str
    evaluations: List[TrackerDimensions] = field(default_factory=list)

    def to_markdown_table(self) -> str:
        """Build a Markdown leaderboard ranked by Edge Efficiency Score.

        Returns:
            Multi-line Markdown string with a table header and one row per
            tracker, plus a legend section.
        """
        rows = sorted(self.evaluations, key=lambda e: e.ees, reverse=True)
        lines = [
            f"# EOVOT Full Evaluation — {self.dataset_name}\n",
            "| Rank | Tracker | mIoU | Succ AUC | FPS | Mem(MB) | EES "
            "| Failures | EAO | Smoothness | Pareto |",
            "|------|---------|-----:|---------:|----:|--------:|----:"
            "|---------:|----:|----------:|:------:|",
        ]
        for rank, e in enumerate(rows, 1):
            sauc = f"{e.success_auc:.4f}" if e.success_auc is not None else "—"
            pareto = "✓" if e.on_pareto_front else ""
            lines.append(
                f"| {rank} | {e.tracker_name} "
                f"| {e.mean_iou:.4f} | {sauc} "
                f"| {e.mean_fps:.1f} | {e.peak_memory_mb:.1f} "
                f"| {e.ees:.4f} | {e.total_failures} "
                f"| {e.mean_eao:.4f} | {e.mean_smoothness_score:.4f} "
                f"| {pareto} |"
            )
        lines.append(
            "\n**EES** = mean_iou × log1p(fps) / (1 + peak_mem / budget). "
            "**Pareto** = no other tracker dominates in both mIoU and EES."
        )
        return "\n".join(lines)

    def save(self, path: Union[str, Path]) -> Path:
        """Serialise the report to a JSON file.

        Args:
            path: Destination file path.  Parent directories are created
                automatically.  A ``.json`` suffix is appended when absent.

        Returns:
            Resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dataset": self.dataset_name,
            "evaluations": [e.to_dict() for e in self.evaluations],
        }
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return p

    def best_by_ees(self) -> Optional[TrackerDimensions]:
        """Return the tracker with the highest Edge Efficiency Score."""
        if not self.evaluations:
            return None
        return max(self.evaluations, key=lambda e: e.ees)

    def best_by_accuracy(self) -> Optional[TrackerDimensions]:
        """Return the tracker with the highest mIoU."""
        if not self.evaluations:
            return None
        return max(self.evaluations, key=lambda e: e.mean_iou)

    def pareto_trackers(self) -> List[TrackerDimensions]:
        """Return all Pareto-optimal trackers."""
        return [e for e in self.evaluations if e.on_pareto_front]


class FullEvaluationPipeline:
    """Run benchmark + four-dimension post-analysis for one or more trackers.

    A single call to :meth:`run` executes:

    1. :class:`~eovot.benchmark.engine.BenchmarkEngine` — accuracy + profiling.
    2. :class:`~eovot.metrics.robustness.RobustnessAnalyzer` — failure analysis.
    3. :class:`~eovot.metrics.temporal.TemporalConsistencyAnalyzer` — smoothness.
    4. :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine` — EES + Pareto.

    Args:
        memory_budget_mb: Memory ceiling for EES computation (default: 512 MB).
            Trackers within this budget receive full memory credit; those above
            it are soft-penalised proportionally.
        tdp_watts: CPU TDP in watts for energy profiling. ``None`` disables
            energy estimation (default).
        verbose: Print per-sequence benchmark progress (default: ``True``).

    Example::

        from eovot.experiment.full_evaluation import FullEvaluationPipeline
        from eovot.trackers.registry import build_tracker
        from eovot.datasets.synthetic import SyntheticDataset

        dataset  = SyntheticDataset(num_sequences=3, num_frames=30, seed=0)
        trackers = [build_tracker("MOSSE"), build_tracker("KCF")]

        report = FullEvaluationPipeline(verbose=False).run(
            trackers, dataset, dataset_name="Synthetic"
        )
        print(report.to_markdown_table())
    """

    def __init__(
        self,
        memory_budget_mb: float = 512.0,
        tdp_watts: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        if memory_budget_mb <= 0:
            raise ValueError(
                f"memory_budget_mb must be positive, got {memory_budget_mb}."
            )
        self.memory_budget_mb = memory_budget_mb
        self.tdp_watts = tdp_watts
        self.verbose = verbose

        self._robustness = RobustnessAnalyzer()
        self._temporal = TemporalConsistencyAnalyzer()
        self._efficiency = EfficiencyMetricsEngine(
            memory_budget_mb=memory_budget_mb
        )

    def run(
        self,
        trackers: Union[BaseTracker, List[BaseTracker]],
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> FullEvaluationReport:
        """Evaluate trackers on a dataset and compute all four metric dimensions.

        Args:
            trackers: A single :class:`~eovot.trackers.base.BaseTracker` or
                a list of them.  Each is evaluated independently on the same
                dataset.
            dataset: Dataset to evaluate on.
            dataset_name: Human-readable dataset label used in the report.
            max_sequences: Limit evaluation to the first *N* sequences.
                ``None`` (default) evaluates all sequences.

        Returns:
            :class:`FullEvaluationReport` with all four metric dimensions
            populated for every tracker, and Pareto flags set.
        """
        if isinstance(trackers, BaseTracker):
            trackers = [trackers]

        engine = BenchmarkEngine(verbose=self.verbose, tdp_watts=self.tdp_watts)
        benchmark_results: List[BenchmarkResult] = []

        for tracker in trackers:
            result = engine.run(
                tracker=tracker,
                dataset=dataset,
                dataset_name=dataset_name,
                max_sequences=max_sequences,
            )
            benchmark_results.append(result)

        return self._build_report(benchmark_results, dataset_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_report(
        self,
        results: List[BenchmarkResult],
        dataset_name: str,
    ) -> FullEvaluationReport:
        """Analyse all results and build the :class:`FullEvaluationReport`."""
        dimensions: List[TrackerDimensions] = [
            self._evaluate_one(r) for r in results
        ]

        # Compute Pareto front across all trackers in the comparison set.
        entries = [
            EfficiencyEntry(
                tracker_name=d.tracker_name,
                dataset_name=d.dataset_name,
                mean_iou=d.mean_iou,
                fps=d.mean_fps,
                peak_memory_mb=d.peak_memory_mb,
                ees=d.ees,
            )
            for d in dimensions
        ]
        self._efficiency.compute_pareto_front(entries)
        for dim, entry in zip(dimensions, entries):
            dim.on_pareto_front = entry.on_pareto_front

        return FullEvaluationReport(
            dataset_name=dataset_name, evaluations=dimensions
        )

    def _evaluate_one(self, result: BenchmarkResult) -> TrackerDimensions:
        """Run all post-analysis passes on a single :class:`BenchmarkResult`."""
        dim = TrackerDimensions(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
        )

        # 1. Accuracy + profiling (already in BenchmarkResult)
        dim.mean_iou = result.mean_iou
        dim.success_auc = result.mean_success_auc
        dim.precision_auc = result.mean_precision_auc
        dim.mean_fps = result.mean_fps
        dim.peak_memory_mb = result.peak_memory_mb

        # 2. Robustness
        seq_ious: Dict[str, np.ndarray] = {
            r.sequence_name: r.ious for r in result.sequence_results
        }
        rob = self._robustness.analyze_benchmark(
            seq_ious, tracker_name=result.tracker_name
        )
        agg = rob["aggregate"]
        dim.total_failures = int(agg.get("total_failures", 0))
        dim.mean_failures_per_sequence = float(
            agg.get("mean_failures_per_sequence", 0.0)
        )
        dim.mean_eao = float(agg.get("mean_eao", 0.0))
        dim.mean_survival_rate = float(agg.get("mean_survival_rate", 0.0))
        dim.mean_recovery_lag_frames = float(
            agg.get("mean_recovery_lag_frames", 0.0)
        )

        # 3. Temporal consistency (requires per-frame predictions)
        seq_preds: Dict[str, np.ndarray] = {
            r.sequence_name: r.predictions
            for r in result.sequence_results
            if r.predictions is not None and len(r.predictions) > 0
        }
        if seq_preds:
            temp = self._temporal.analyze_benchmark(
                seq_preds, tracker_name=result.tracker_name
            )
            tagg = temp.get("aggregate", {})
            dim.mean_smoothness_score = float(
                tagg.get("mean_smoothness_score", 0.0)
            )
            dim.mean_position_jitter = float(
                tagg.get("mean_position_jitter", 0.0)
            )
            dim.mean_scale_jitter = float(tagg.get("mean_scale_jitter", 0.0))
            dim.mean_velocity_outlier_ratio = float(
                tagg.get("mean_velocity_outlier_ratio", 0.0)
            )

        # 4. Edge efficiency
        dim.ees = self._efficiency.edge_efficiency_score(
            mean_iou=dim.mean_iou,
            fps=dim.mean_fps,
            peak_memory_mb=dim.peak_memory_mb,
        )

        return dim
