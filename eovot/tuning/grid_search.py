"""Exhaustive grid search for tracker hyperparameter optimisation.

Evaluates every combination in a discrete parameter grid using the EOVOT
benchmark engine and ranks results by a chosen scalar metric.

Supported metric names (matched against :class:`~eovot.benchmark.engine.BenchmarkResult`
attributes):

- ``"mean_iou"``         — mean IoU across all frames (higher is better)
- ``"mean_fps"``         — mean frames-per-second (higher is better)
- ``"success_auc"``      — area under the success curve
- ``"precision_auc"``    — area under the precision curve
- ``"peak_memory_mb"``   — peak resident memory (lower is better)

Example::

    from eovot.tuning.grid_search import GridSearchRunner
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, frames_per_sequence=50)
    runner = GridSearchRunner(
        tracker_class=KCFTracker,
        param_grid={
            "learning_rate": [0.05, 0.075, 0.1],
            "kernel_sigma":  [0.3, 0.5, 0.7],
        },
        metric="mean_iou",
    )
    result = runner.run(dataset, dataset_name="Synthetic")
    print(result.best_params)
    print(result.to_markdown())
"""

from __future__ import annotations

import csv
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..trackers.base import BaseTracker

# Metrics where lower is better (all others: higher is better).
_LOWER_IS_BETTER = {"peak_memory_mb", "mean_latency_ms", "total_energy_j"}


@dataclass
class TuningEntry:
    """Result for a single hyperparameter combination."""

    params: Dict[str, Any]
    """The hyperparameter values used for this run."""

    metric_value: float
    """Value of the optimisation metric."""

    metric_name: str
    """Name of the metric (e.g. ``"mean_iou"``)."""

    mean_iou: float
    mean_fps: float
    peak_memory_mb: float
    success_auc: Optional[float] = None
    precision_auc: Optional[float] = None
    run_time_s: float = 0.0
    """Wall-clock time (seconds) for this combination's benchmark run."""

    def params_str(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.params.items())

    def __lt__(self, other: TuningEntry) -> bool:
        return self.metric_value < other.metric_value


@dataclass
class TuningResult:
    """Ranked results from a completed grid search.

    Attributes:
        tracker_class_name: Name of the tracker class that was searched.
        dataset_name:       Name of the dataset used.
        metric_name:        Metric that was optimised.
        entries:            All evaluated combinations, sorted best-first.
        total_combinations: Total number of combinations evaluated.
        elapsed_s:          Total wall-clock time for the search.
    """

    tracker_class_name: str
    dataset_name: str
    metric_name: str
    entries: List[TuningEntry] = field(default_factory=list)
    total_combinations: int = 0
    elapsed_s: float = 0.0

    @property
    def best_params(self) -> Dict[str, Any]:
        """Hyperparameters of the top-ranked combination."""
        return self.entries[0].params if self.entries else {}

    @property
    def best_value(self) -> Optional[float]:
        """Metric value of the top-ranked combination."""
        return self.entries[0].metric_value if self.entries else None

    def to_markdown(self) -> str:
        """Render a Markdown table of all results, sorted best-first."""
        if not self.entries:
            return "_No results._"

        # Collect all param keys in insertion order.
        param_keys = list(self.entries[0].params.keys())
        headers = param_keys + [
            self.metric_name,
            "mean_iou",
            "mean_fps",
            "peak_memory_mb",
            "run_time_s",
        ]

        rows = []
        for e in self.entries:
            row = [str(e.params.get(k, "")) for k in param_keys]
            row += [
                f"{e.metric_value:.4f}",
                f"{e.mean_iou:.4f}",
                f"{e.mean_fps:.1f}",
                f"{e.peak_memory_mb:.2f}",
                f"{e.run_time_s:.1f}",
            ]
            rows.append(row)

        sep = ["-" * max(len(h), 6) for h in headers]
        lines = [
            f"# Grid Search: {self.tracker_class_name} on {self.dataset_name}",
            f"**Metric:** {self.metric_name} | "
            f"**Combinations:** {self.total_combinations} | "
            f"**Total time:** {self.elapsed_s:.1f}s",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")

        if self.best_params:
            lines += [
                "",
                f"**Best:** {self.entries[0].params_str()} → "
                f"{self.metric_name}={self.best_value:.4f}",
            ]
        return "\n".join(lines)

    def save_csv(self, path: str) -> None:
        """Write all results to a CSV file."""
        if not self.entries:
            return

        param_keys = list(self.entries[0].params.keys())
        fieldnames = param_keys + [
            self.metric_name,
            "mean_iou",
            "mean_fps",
            "peak_memory_mb",
            "run_time_s",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.entries:
                row = dict(e.params)
                row[self.metric_name] = e.metric_value
                row["mean_iou"] = e.mean_iou
                row["mean_fps"] = e.mean_fps
                row["peak_memory_mb"] = e.peak_memory_mb
                row["run_time_s"] = e.run_time_s
                writer.writerow(row)


class GridSearchRunner:
    """Run an exhaustive grid search over tracker hyperparameters.

    Args:
        tracker_class: The tracker class (not instance) to instantiate for
            each combination.  Must accept the parameter names in *param_grid*
            as keyword arguments to ``__init__``.
        param_grid:    Mapping from parameter name to list of values to try.
            The Cartesian product of all value lists is evaluated.
        metric:        Scalar metric to optimise.  Must be an attribute of
            :class:`~eovot.benchmark.engine.BenchmarkResult`.
            Default: ``"mean_iou"``.
        verbose:       Print progress for each combination.  Default: ``True``.
        tdp_watts:     Forwarded to :class:`~eovot.benchmark.engine.BenchmarkEngine`
            for energy profiling.  Default: ``None``.
        max_sequences: Limit the number of sequences evaluated per combination
            for faster iteration.  ``None`` uses the whole dataset.

    Example::

        runner = GridSearchRunner(
            tracker_class=MOSSETracker,
            param_grid={"learning_rate": [0.05, 0.125], "sigma": [1.5, 2.0]},
            metric="mean_iou",
        )
        result = runner.run(dataset, dataset_name="Synthetic")
    """

    def __init__(
        self,
        tracker_class: Type[BaseTracker],
        param_grid: Dict[str, List[Any]],
        metric: str = "mean_iou",
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
        max_sequences: Optional[int] = None,
    ) -> None:
        if not param_grid:
            raise ValueError("param_grid must contain at least one parameter.")
        self.tracker_class = tracker_class
        self.param_grid = param_grid
        self.metric = metric
        self.verbose = verbose
        self.tdp_watts = tdp_watts
        self.max_sequences = max_sequences

        self._engine = BenchmarkEngine(verbose=False, tdp_watts=tdp_watts)

    def _combinations(self) -> List[Dict[str, Any]]:
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _extract_metric(self, result: BenchmarkResult) -> float:
        val = getattr(result, self.metric, None)
        if val is None:
            raise AttributeError(
                f"BenchmarkResult has no attribute '{self.metric}'. "
                f"Choose from: mean_iou, mean_fps, peak_memory_mb, "
                f"mean_success_auc, mean_precision_auc."
            )
        return float(val)

    def run(
        self,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
    ) -> TuningResult:
        """Evaluate all parameter combinations and return ranked results.

        Args:
            dataset:      Dataset to benchmark on.
            dataset_name: Human-readable name for reports.

        Returns:
            :class:`TuningResult` with all combinations sorted best-first.
        """
        combinations = self._combinations()
        lower_is_better = self.metric in _LOWER_IS_BETTER

        tuning_result = TuningResult(
            tracker_class_name=self.tracker_class.__name__,
            dataset_name=dataset_name,
            metric_name=self.metric,
            total_combinations=len(combinations),
        )

        if self.verbose:
            print(
                f"\nGrid search: {self.tracker_class.__name__} | "
                f"metric={self.metric} | "
                f"{len(combinations)} combination(s)"
            )
            print("-" * 60)

        search_start = time.perf_counter()

        for i, params in enumerate(combinations, 1):
            tracker = self.tracker_class(**params)
            t0 = time.perf_counter()
            bench = self._engine.run(
                tracker,
                dataset,
                dataset_name=dataset_name,
                max_sequences=self.max_sequences,
            )
            elapsed = time.perf_counter() - t0

            metric_val = self._extract_metric(bench)
            entry = TuningEntry(
                params=params,
                metric_value=metric_val,
                metric_name=self.metric,
                mean_iou=bench.mean_iou,
                mean_fps=bench.mean_fps,
                peak_memory_mb=bench.peak_memory_mb,
                success_auc=bench.mean_success_auc,
                precision_auc=bench.mean_precision_auc,
                run_time_s=elapsed,
            )
            tuning_result.entries.append(entry)

            if self.verbose:
                print(
                    f"  [{i:>3}/{len(combinations)}] {entry.params_str():<45s} "
                    f"{self.metric}={metric_val:.4f}  ({elapsed:.1f}s)"
                )

        # Sort: best first.  For lower-is-better metrics, sort ascending.
        tuning_result.entries.sort(
            key=lambda e: e.metric_value,
            reverse=not lower_is_better,
        )

        tuning_result.elapsed_s = time.perf_counter() - search_start

        if self.verbose:
            print("-" * 60)
            best = tuning_result.entries[0]
            print(f"Best: {best.params_str()} → {self.metric}={best.metric_value:.4f}")

        return tuning_result
