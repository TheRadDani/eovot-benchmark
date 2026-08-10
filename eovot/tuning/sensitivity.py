"""One-At-a-Time (OAT) sensitivity analysis for tracker hyperparameters.

OAT sensitivity analysis varies each parameter independently while holding
all others at their baseline values.  For each parameter it records how much
the target metric changes relative to the baseline, producing a **normalised
sensitivity score**:

    S_p = (metric_max_p - metric_min_p) / max(|metric_baseline|, ε)

A high |S_p| means the metric is strongly affected by parameter *p*; a low
score means the tracker is robust to changes in that dimension.

This analysis is complementary to grid search: grid search finds the optimum
while OAT tells you *which parameters matter* — useful for deciding what to
tune on a resource-constrained device.

Example::

    from eovot.tuning.sensitivity import SensitivityAnalyzer
    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, frames_per_sequence=50)
    analyzer = SensitivityAnalyzer(
        tracker_class=MOSSETracker,
        base_params={"learning_rate": 0.125, "sigma": 2.0},
        param_ranges={
            "learning_rate": [0.05, 0.075, 0.125, 0.175, 0.2],
            "sigma": [0.5, 1.0, 2.0, 3.0, 4.0],
        },
        metric="mean_iou",
    )
    report = analyzer.run(dataset, dataset_name="Synthetic")
    print(report.to_markdown())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..trackers.base import BaseTracker


@dataclass
class ParameterSensitivity:
    """Sensitivity statistics for a single hyperparameter."""

    param_name: str
    values_tested: List[Any]
    metric_values: List[float]

    baseline_value: Any
    baseline_metric: float

    sensitivity_score: float
    """Normalised sensitivity: (max_metric - min_metric) / |baseline_metric|."""

    metric_name: str

    @property
    def metric_range(self) -> float:
        """Difference between max and min metric across all tested values."""
        return float(max(self.metric_values) - min(self.metric_values))

    @property
    def best_value(self) -> Any:
        """Value of the parameter that yielded the highest metric."""
        return self.values_tested[int(np.argmax(self.metric_values))]

    @property
    def worst_value(self) -> Any:
        """Value of the parameter that yielded the lowest metric."""
        return self.values_tested[int(np.argmin(self.metric_values))]

    def to_dict(self) -> Dict:
        return {
            "param_name": self.param_name,
            "baseline_value": self.baseline_value,
            "baseline_metric": round(self.baseline_metric, 4),
            "sensitivity_score": round(self.sensitivity_score, 4),
            "metric_range": round(self.metric_range, 4),
            "best_value": self.best_value,
            "worst_value": self.worst_value,
            "values_tested": self.values_tested,
            "metric_values": [round(v, 4) for v in self.metric_values],
        }


@dataclass
class SensitivityReport:
    """Full OAT sensitivity analysis report for a tracker.

    Attributes:
        tracker_class_name: Name of the analysed tracker class.
        dataset_name:       Name of the dataset used.
        metric_name:        Optimisation metric.
        baseline_params:    The fixed baseline parameter set.
        baseline_metric:    Metric value at the baseline.
        sensitivities:      Per-parameter sensitivity, sorted by
                            ``|sensitivity_score|`` descending.
        elapsed_s:          Total wall-clock time for the analysis.
    """

    tracker_class_name: str
    dataset_name: str
    metric_name: str
    baseline_params: Dict[str, Any]
    baseline_metric: float
    sensitivities: List[ParameterSensitivity] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def most_sensitive_param(self) -> Optional[str]:
        """Name of the parameter with the highest absolute sensitivity."""
        return self.sensitivities[0].param_name if self.sensitivities else None

    def to_markdown(self) -> str:
        """Render a Markdown report with a ranked sensitivity table."""
        baseline_str = ", ".join(f"{k}={v}" for k, v in self.baseline_params.items())
        lines = [
            f"# OAT Sensitivity Analysis: {self.tracker_class_name} on {self.dataset_name}",
            f"**Metric:** {self.metric_name} | "
            f"**Baseline:** {baseline_str} | "
            f"**Baseline {self.metric_name}:** {self.baseline_metric:.4f} | "
            f"**Total time:** {self.elapsed_s:.1f}s",
            "",
            "## Sensitivity Ranking",
            "",
            "| # | Parameter | Sensitivity | Range | Best value | Worst value |",
            "|---|-----------|------------|-------|------------|-------------|",
        ]
        for rank, s in enumerate(self.sensitivities, 1):
            lines.append(
                f"| {rank} | `{s.param_name}` | {s.sensitivity_score:+.4f} | "
                f"{s.metric_range:.4f} | `{s.best_value}` | `{s.worst_value}` |"
            )

        lines += ["", "## Per-Parameter Details", ""]
        for s in self.sensitivities:
            lines.append(f"### `{s.param_name}`")
            lines.append("")
            lines.append(
                "| Value | " + self.metric_name + " | Δ from baseline |"
            )
            lines.append("|-------|---------|----------------|")
            for v, m in zip(s.values_tested, s.metric_values):
                marker = " ← baseline" if v == s.baseline_value else ""
                lines.append(
                    f"| `{v}` | {m:.4f} | {m - s.baseline_metric:+.4f}{marker} |"
                )
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "tracker": self.tracker_class_name,
            "dataset": self.dataset_name,
            "metric": self.metric_name,
            "baseline_params": self.baseline_params,
            "baseline_metric": round(self.baseline_metric, 4),
            "sensitivities": [s.to_dict() for s in self.sensitivities],
            "elapsed_s": round(self.elapsed_s, 2),
        }


class SensitivityAnalyzer:
    """One-At-a-Time (OAT) sensitivity analysis for tracker hyperparameters.

    For each parameter in *param_ranges*, the tracker is re-instantiated with
    that parameter swept across the provided values while all other parameters
    are held at their *base_params* values.  The metric is recorded for each
    value, and the range and normalised sensitivity score are computed.

    Args:
        tracker_class: Tracker class to instantiate.
        base_params:   Baseline hyperparameter dictionary (the "default" point).
        param_ranges:  Mapping from parameter name to list of values to sweep.
                       Should include the baseline value in each list so the
                       baseline run can be reused.
        metric:        Scalar attribute on
                       :class:`~eovot.benchmark.engine.BenchmarkResult` to
                       measure.  Default: ``"mean_iou"``.
        verbose:       Print per-parameter progress.  Default: ``True``.
        tdp_watts:     Forwarded to BenchmarkEngine for energy profiling.
        max_sequences: Limit sequences per run for faster iteration.

    Example::

        analyzer = SensitivityAnalyzer(
            tracker_class=KCFTracker,
            base_params={"learning_rate": 0.075, "kernel_sigma": 0.5},
            param_ranges={
                "learning_rate": [0.03, 0.05, 0.075, 0.1, 0.15],
                "kernel_sigma": [0.2, 0.35, 0.5, 0.65, 0.8],
            },
            metric="mean_iou",
        )
        report = analyzer.run(dataset)
    """

    def __init__(
        self,
        tracker_class: Type[BaseTracker],
        base_params: Dict[str, Any],
        param_ranges: Dict[str, List[Any]],
        metric: str = "mean_iou",
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
        max_sequences: Optional[int] = None,
    ) -> None:
        if not param_ranges:
            raise ValueError("param_ranges must contain at least one parameter.")
        self.tracker_class = tracker_class
        self.base_params = dict(base_params)
        self.param_ranges = param_ranges
        self.metric = metric
        self.verbose = verbose
        self.tdp_watts = tdp_watts
        self.max_sequences = max_sequences

        self._engine = BenchmarkEngine(verbose=False, tdp_watts=tdp_watts)

    def _run_tracker(self, params: Dict[str, Any], dataset: BaseDataset, dataset_name: str) -> float:
        tracker = self.tracker_class(**params)
        result = self._engine.run(
            tracker, dataset, dataset_name=dataset_name, max_sequences=self.max_sequences
        )
        val = getattr(result, self.metric, None)
        if val is None:
            raise AttributeError(
                f"BenchmarkResult has no attribute '{self.metric}'."
            )
        return float(val)

    def run(
        self,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
    ) -> SensitivityReport:
        """Execute the OAT sweep and produce a :class:`SensitivityReport`.

        Args:
            dataset:      Dataset to evaluate on.
            dataset_name: Human-readable dataset name for the report.

        Returns:
            :class:`SensitivityReport` with per-parameter sensitivities sorted
            by absolute sensitivity score (most sensitive first).
        """
        t_start = time.perf_counter()

        # Baseline run.
        baseline_metric = self._run_tracker(self.base_params, dataset, dataset_name)

        if self.verbose:
            print(
                f"\nOAT Sensitivity: {self.tracker_class.__name__} | "
                f"metric={self.metric} | baseline={baseline_metric:.4f}"
            )
            print("-" * 60)

        sensitivities: List[ParameterSensitivity] = []
        eps = max(abs(baseline_metric), 1e-8)

        for param_name, values in self.param_ranges.items():
            if self.verbose:
                print(f"  Sweeping '{param_name}' over {values}...")

            metric_values: List[float] = []
            for v in values:
                params = dict(self.base_params)
                params[param_name] = v
                m = self._run_tracker(params, dataset, dataset_name)
                metric_values.append(m)
                if self.verbose:
                    delta = m - baseline_metric
                    print(
                        f"    {param_name}={v:<8}  "
                        f"{self.metric}={m:.4f}  Δ={delta:+.4f}"
                    )

            sensitivity_score = (max(metric_values) - min(metric_values)) / eps

            # The baseline value within this sweep (may differ from exact baseline_metric
            # due to dataset randomness, but use the closest match).
            baseline_val = self.base_params.get(param_name, values[0])

            sensitivities.append(
                ParameterSensitivity(
                    param_name=param_name,
                    values_tested=list(values),
                    metric_values=metric_values,
                    baseline_value=baseline_val,
                    baseline_metric=baseline_metric,
                    sensitivity_score=sensitivity_score,
                    metric_name=self.metric,
                )
            )

        # Rank by absolute sensitivity descending.
        sensitivities.sort(key=lambda s: abs(s.sensitivity_score), reverse=True)

        elapsed = time.perf_counter() - t_start

        if self.verbose:
            print("-" * 60)
            for rank, s in enumerate(sensitivities, 1):
                print(
                    f"  [{rank}] {s.param_name:<20s} "
                    f"sensitivity={s.sensitivity_score:+.4f}  "
                    f"range={s.metric_range:.4f}"
                )

        return SensitivityReport(
            tracker_class_name=self.tracker_class.__name__,
            dataset_name=dataset_name,
            metric_name=self.metric,
            baseline_params=dict(self.base_params),
            baseline_metric=baseline_metric,
            sensitivities=sensitivities,
            elapsed_s=elapsed,
        )
