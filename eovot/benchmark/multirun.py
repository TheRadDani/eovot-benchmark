"""Multi-run benchmark with statistical confidence intervals for latency and accuracy.

Single-run latency measurements are noisy: OS scheduling jitter, CPU frequency
scaling, and cold-start effects can distort per-frame timing by 20–50 % on the
first few runs.  Publication-quality benchmarks require running the tracker
multiple times and reporting confidence intervals.

:class:`MultiRunBenchmark` addresses this by executing the full benchmark
loop K times, aggregating all scalar metrics across runs, and computing
95 % bootstrap confidence intervals.  The result is a
:class:`MultiRunResult` that surfaces mean ± CI for:

* FPS
* Mean latency (ms)
* p95 latency (ms)
* p99 latency (ms)
* Mean IoU
* Success AUC

Warm-up run
~~~~~~~~~~~
The first run is optionally treated as a warm-up that is excluded from
statistics.  Warm-up eliminates the one-time overhead of branch-predictor
priming, OS page faults on the frame buffer, and Python-level object
allocations that inflate first-run latency figures.

Typical usage::

    from eovot.benchmark.multirun import MultiRunBenchmark
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, num_frames=100)
    tracker = KCFTracker()

    mr = MultiRunBenchmark(num_runs=5, warmup_run=True, seed=42)
    result = mr.run(tracker, dataset, dataset_name="Synthetic")
    print(result)
    print(result.to_markdown())
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..trackers.base import BaseTracker

if TYPE_CHECKING:
    pass


@dataclass
class ConfidenceInterval:
    """A percentile bootstrap 95 % confidence interval for a scalar metric.

    Attributes:
        mean:   Sample mean across all runs.
        lower:  2.5th percentile of bootstrap distribution.
        upper:  97.5th percentile of bootstrap distribution.
        std:    Standard deviation across runs.
    """

    mean: float
    lower: float
    upper: float
    std: float

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}] ±{self.std:.4f}"


@dataclass
class MultiRunResult:
    """Aggregated statistics over K independent benchmark runs.

    All :class:`ConfidenceInterval` fields are 95 % bootstrap CIs computed
    from ``num_measurement_runs`` repeated evaluations of the same tracker.

    Attributes:
        tracker_name: Identifier of the evaluated tracker.
        dataset_name: Dataset used for evaluation.
        num_runs: Total runs requested (including optional warm-up).
        num_measurement_runs: Runs included in statistics (num_runs - warmup).
        fps_ci: CI for mean FPS across sequences and runs.
        latency_mean_ci: CI for mean per-frame latency (ms).
        latency_p95_ci: CI for 95th-percentile per-frame latency (ms).
        latency_p99_ci: CI for 99th-percentile per-frame latency (ms).
        mean_iou_ci: CI for mean IoU across sequences and runs.
        success_auc_ci: CI for mean success-curve AUC (None if not computed).
        raw_runs: Full :class:`BenchmarkResult` objects for measurement runs.
    """

    tracker_name: str
    dataset_name: str
    num_runs: int
    num_measurement_runs: int
    fps_ci: ConfidenceInterval
    latency_mean_ci: ConfidenceInterval
    latency_p95_ci: ConfidenceInterval
    latency_p99_ci: ConfidenceInterval
    mean_iou_ci: ConfidenceInterval
    success_auc_ci: Optional[ConfidenceInterval] = None
    raw_runs: List[BenchmarkResult] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        lines = [
            f"MultiRunResult[{self.tracker_name} on {self.dataset_name}]",
            f"  runs  : {self.num_measurement_runs} (+ {self.num_runs - self.num_measurement_runs} warmup)",
            f"  FPS   : {self.fps_ci}",
            f"  lat   : {self.latency_mean_ci} ms",
            f"  p95   : {self.latency_p95_ci} ms",
            f"  p99   : {self.latency_p99_ci} ms",
            f"  mIoU  : {self.mean_iou_ci}",
        ]
        if self.success_auc_ci is not None:
            lines.append(f"  AUC   : {self.success_auc_ci}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize to a nested dict for JSON export."""
        def _ci_dict(ci: Optional[ConfidenceInterval]) -> Optional[Dict]:
            if ci is None:
                return None
            return {"mean": ci.mean, "lower": ci.lower, "upper": ci.upper, "std": ci.std}

        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "num_runs": self.num_runs,
            "num_measurement_runs": self.num_measurement_runs,
            "fps": _ci_dict(self.fps_ci),
            "latency_mean_ms": _ci_dict(self.latency_mean_ci),
            "latency_p95_ms": _ci_dict(self.latency_p95_ci),
            "latency_p99_ms": _ci_dict(self.latency_p99_ci),
            "mean_iou": _ci_dict(self.mean_iou_ci),
            "success_auc": _ci_dict(self.success_auc_ci),
        }

    def to_markdown(self) -> str:
        """Format a summary table as a Markdown string."""
        def _fmt(ci: Optional[ConfidenceInterval], fmt: str = ".4f") -> str:
            if ci is None:
                return "—"
            return f"{ci.mean:{fmt}} [{ci.lower:{fmt}}, {ci.upper:{fmt}}]"

        lines = [
            f"## Multi-Run Benchmark: {self.tracker_name} on {self.dataset_name}",
            f"*{self.num_measurement_runs} measurement runs*"
            + (f" + 1 warm-up" if self.num_runs > self.num_measurement_runs else ""),
            "",
            "| Metric | Mean [95% CI] |",
            "|--------|--------------|",
            f"| FPS | {_fmt(self.fps_ci, '.1f')} |",
            f"| Latency mean (ms) | {_fmt(self.latency_mean_ci, '.3f')} |",
            f"| Latency p95 (ms) | {_fmt(self.latency_p95_ci, '.3f')} |",
            f"| Latency p99 (ms) | {_fmt(self.latency_p99_ci, '.3f')} |",
            f"| Mean IoU | {_fmt(self.mean_iou_ci)} |",
        ]
        if self.success_auc_ci is not None:
            lines.append(f"| Success AUC | {_fmt(self.success_auc_ci)} |")
        return "\n".join(lines)


class MultiRunBenchmark:
    """Run a tracker K times and compute 95 % bootstrap CIs on performance metrics.

    Args:
        num_runs: Total number of benchmark runs to execute.  Must be ≥ 2
            to compute meaningful confidence intervals.
        warmup_run: When ``True``, the first run is discarded before computing
            statistics.  Strongly recommended: the first run is typically
            20–50 % slower due to OS-level cold-start effects.
            Default: ``True``.
        n_bootstrap: Number of bootstrap resamples for CI estimation.
            Default: ``5000`` (balances accuracy and runtime).
        seed: RNG seed for reproducible bootstrap sampling.  Default: ``42``.
        verbose: Print per-run progress.  Default: ``False``.
        tdp_watts: If provided, passes this TDP value to the underlying
            :class:`~eovot.benchmark.engine.BenchmarkEngine` for energy
            profiling.  Default: ``None``.

    Raises:
        ValueError: If ``num_runs < 2``.

    Example::

        mr = MultiRunBenchmark(num_runs=5, warmup_run=True)
        result = mr.run(KCFTracker(), dataset, "Synthetic")
        print(result.to_markdown())
    """

    def __init__(
        self,
        num_runs: int = 5,
        warmup_run: bool = True,
        n_bootstrap: int = 5_000,
        seed: int = 42,
        verbose: bool = False,
        tdp_watts: Optional[float] = None,
    ) -> None:
        if num_runs < 2:
            raise ValueError(f"num_runs must be >= 2, got {num_runs}")
        self.num_runs = num_runs
        self.warmup_run = warmup_run
        self.n_bootstrap = n_bootstrap
        self._rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.tdp_watts = tdp_watts

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> MultiRunResult:
        """Execute ``num_runs`` benchmark runs and aggregate statistics.

        Args:
            tracker: The tracker to evaluate.  Its internal state is reset
                between runs because :meth:`initialize` is called per-sequence
                by the engine — no explicit reset is needed.
            dataset: Dataset to evaluate on.
            dataset_name: Label used in reports.
            max_sequences: Cap the number of sequences per run.

        Returns:
            :class:`MultiRunResult` with per-metric 95 % confidence intervals.
        """
        engine = BenchmarkEngine(verbose=False, tdp_watts=self.tdp_watts)
        all_results: List[BenchmarkResult] = []

        for i in range(self.num_runs):
            is_warmup = self.warmup_run and i == 0
            tag = "warm-up" if is_warmup else f"run {i + 1 - int(self.warmup_run)}/{self.num_runs - int(self.warmup_run)}"
            if self.verbose:
                print(f"  [{tracker.name}] {tag}...")

            result = engine.run(
                tracker, dataset,
                dataset_name=dataset_name,
                max_sequences=max_sequences,
            )
            if not is_warmup:
                all_results.append(result)

        return self._aggregate(tracker.name, dataset_name, all_results)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        tracker_name: str,
        dataset_name: str,
        runs: List[BenchmarkResult],
    ) -> MultiRunResult:
        fps_vals = np.array([r.mean_fps for r in runs])
        lat_vals = np.array([
            np.mean([s.profiling.latency_mean_ms for s in r.sequence_results])
            for r in runs
        ])
        p95_vals = np.array([
            np.mean([s.profiling.latency_p95_ms for s in r.sequence_results])
            for r in runs
        ])
        p99_vals = np.array([
            np.mean([s.profiling.latency_p99_ms for s in r.sequence_results])
            for r in runs
        ])
        iou_vals = np.array([r.mean_iou for r in runs])

        auc_vals: Optional[np.ndarray] = None
        if all(r.mean_success_auc is not None for r in runs):
            auc_vals = np.array([r.mean_success_auc for r in runs])

        num_warmup = int(self.warmup_run)
        return MultiRunResult(
            tracker_name=tracker_name,
            dataset_name=dataset_name,
            num_runs=self.num_runs,
            num_measurement_runs=len(runs),
            fps_ci=self._bootstrap_ci(fps_vals),
            latency_mean_ci=self._bootstrap_ci(lat_vals),
            latency_p95_ci=self._bootstrap_ci(p95_vals),
            latency_p99_ci=self._bootstrap_ci(p99_vals),
            mean_iou_ci=self._bootstrap_ci(iou_vals),
            success_auc_ci=self._bootstrap_ci(auc_vals) if auc_vals is not None else None,
            raw_runs=runs,
        )

    def _bootstrap_ci(
        self,
        values: np.ndarray,
        alpha: float = 0.05,
    ) -> ConfidenceInterval:
        """Compute a percentile bootstrap CI for the mean of *values*.

        Args:
            values: 1-D array of per-run scalars.
            alpha:  Significance level; default 0.05 gives a 95 % CI.

        Returns:
            :class:`ConfidenceInterval` with mean, lower, upper, and std.
        """
        n = len(values)
        boot_means = np.array([
            self._rng.choice(values, size=n, replace=True).mean()
            for _ in range(self.n_bootstrap)
        ])
        lower = float(np.percentile(boot_means, 100 * alpha / 2))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return ConfidenceInterval(
            mean=float(values.mean()),
            lower=lower,
            upper=upper,
            std=float(values.std()),
        )
