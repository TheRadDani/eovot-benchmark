"""Measurement variance profiler for reproducible benchmark reporting.

Standard benchmarks report mean FPS, mean IoU, and mean latency — but a
single-run mean hides measurement noise that can mislead hardware comparison
studies.  On edge hardware (Raspberry Pi, Jetson, microcontrollers) system
jitter, thermal throttling, cache cold-starts, and OS scheduling variability
can make the same tracker appear 20–40 % faster or slower run-to-run.

This module runs a tracker *K* times on the same sequence set and produces:

- **Per-metric coefficient of variation (CV)** — std / mean for IoU, FPS,
  and latency.  A CV > 0.05 (5 %) indicates the metric is too noisy to
  distinguish trackers separated by less than one standard deviation.
- **95 % bootstrap confidence intervals** for each aggregate metric, so
  reported numbers can carry ± bounds rather than bare means.
- **Reliability flag** — whether a tracker's FPS variance is small enough
  to trust single-run comparisons (CV_FPS < ``reliability_threshold``).

This is essential for publication-quality reporting: results with high CV
should be accompanied by multi-run statistics rather than a single mean.

Typical usage::

    from eovot.analysis.variance_profiler import VarianceProfiler
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.kcf import KCFTracker
    from eovot.benchmark.engine import BenchmarkEngine

    dataset = SyntheticDataset(num_sequences=3, num_frames=60, seed=0)
    engine = BenchmarkEngine(verbose=False)

    profiler = VarianceProfiler(n_runs=5, seed=42)
    report = profiler.profile(KCFTracker(), dataset, dataset_name="Synthetic", engine=engine)

    print(report)
    print(report.to_markdown())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
    from ..trackers.base import BaseTracker
    from ..datasets.base import BaseDataset


@dataclass
class MetricStats:
    """Descriptive statistics for one metric across K benchmark runs.

    Attributes:
        name:   Metric identifier (e.g. ``"fps"``, ``"mean_iou"``).
        mean:   Sample mean across K runs.
        std:    Sample standard deviation.
        cv:     Coefficient of variation (std / mean).  Zero when mean is 0.
        ci_lo:  Lower bound of the 95 % bootstrap confidence interval.
        ci_hi:  Upper bound of the 95 % bootstrap confidence interval.
        values: Raw per-run values (length K).
    """

    name: str
    mean: float
    std: float
    cv: float
    ci_lo: float
    ci_hi: float
    values: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.mean:.4f} ± {self.std:.4f}  "
            f"CV={self.cv:.3f}  95%CI=[{self.ci_lo:.4f}, {self.ci_hi:.4f}]"
        )


@dataclass
class VarianceReport:
    """Multi-run variance summary for one tracker on one dataset.

    Attributes:
        tracker_name:       Tracker identifier.
        dataset_name:       Dataset identifier.
        n_runs:             Number of independent benchmark runs.
        metrics:            Mapping from metric name to :class:`MetricStats`.
        reliable:           ``True`` when ``CV_FPS < reliability_threshold``.
        reliability_threshold: CV threshold used for the ``reliable`` flag.
    """

    tracker_name: str
    dataset_name: str
    n_runs: int
    metrics: Dict[str, MetricStats]
    reliable: bool
    reliability_threshold: float

    def __str__(self) -> str:
        status = "RELIABLE" if self.reliable else "NOISY"
        lines = [
            f"VarianceReport [{self.tracker_name} on {self.dataset_name}]  "
            f"runs={self.n_runs}  [{status}]",
        ]
        for s in self.metrics.values():
            lines.append(f"  {s}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Return a Markdown table of variance statistics."""
        header = "| Metric | Mean | Std | CV | 95% CI Lo | 95% CI Hi |"
        sep = "|--------|-----:|----:|---:|----------:|----------:|"
        rows = [header, sep]
        for s in self.metrics.values():
            rows.append(
                f"| {s.name} | {s.mean:.4f} | {s.std:.4f} | {s.cv:.3f} "
                f"| {s.ci_lo:.4f} | {s.ci_hi:.4f} |"
            )
        rows.append("")
        status = "reliable (CV_FPS < threshold)" if self.reliable else "noisy — multi-run reporting recommended"
        rows.append(f"*Measurement reliability: **{status}***")
        return "\n".join(rows)


class VarianceProfiler:
    """Run a tracker K times and measure metric variance for reliability assessment.

    Each run re-creates a fresh :class:`~eovot.benchmark.engine.BenchmarkEngine`
    and re-evaluates the same tracker on the same dataset, capturing the natural
    run-to-run variability of hardware timers, OS scheduling, and tracker
    internal state.

    Args:
        n_runs:               Number of independent runs.  Default: ``5``.
        seed:                 NumPy random seed for bootstrap resampling.
                              Does **not** affect the tracker or dataset.
        reliability_threshold: CV threshold below which the FPS measurement
                               is considered reliable for single-run comparisons.
                               Default: ``0.05`` (5 %).
        n_bootstrap:          Bootstrap resampling iterations for CI estimation.
                              Default: ``2000``.
    """

    def __init__(
        self,
        n_runs: int = 5,
        seed: int = 42,
        reliability_threshold: float = 0.05,
        n_bootstrap: int = 2000,
    ) -> None:
        if n_runs < 2:
            raise ValueError(f"n_runs must be ≥ 2, got {n_runs}.")
        self.n_runs = n_runs
        self.seed = seed
        self.reliability_threshold = reliability_threshold
        self.n_bootstrap = n_bootstrap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        tracker: "BaseTracker",
        dataset: "BaseDataset",
        dataset_name: str = "unknown",
        engine: Optional["BenchmarkEngine"] = None,
        max_sequences: Optional[int] = None,
    ) -> VarianceReport:
        """Run *tracker* K times on *dataset* and return a :class:`VarianceReport`.

        Args:
            tracker:       Tracker to evaluate.
            dataset:       Dataset to evaluate on.
            dataset_name:  Human-readable dataset identifier.
            engine:        :class:`~eovot.benchmark.engine.BenchmarkEngine` to
                           reuse.  When ``None``, a default engine with
                           ``verbose=False`` is created for each run.
            max_sequences: Limit sequences per run (useful for quick smoke tests).

        Returns:
            :class:`VarianceReport` with per-metric statistics and reliability flag.
        """
        from ..benchmark.engine import BenchmarkEngine

        results: List["BenchmarkResult"] = []
        for _ in range(self.n_runs):
            eng = engine if engine is not None else BenchmarkEngine(verbose=False)
            r = eng.run(tracker, dataset, dataset_name=dataset_name,
                        max_sequences=max_sequences)
            results.append(r)

        return self._build_report(tracker.name, dataset_name, results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bootstrap_ci(
        self, values: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Percentile bootstrap CI with ``n_bootstrap`` resamples."""
        rng = np.random.default_rng(self.seed)
        resampled_means = np.array(
            [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(self.n_bootstrap)]
        )
        alpha = 1.0 - confidence
        lo = float(np.percentile(resampled_means, 100 * alpha / 2))
        hi = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
        return lo, hi

    def _metric_stats(self, name: str, values: List[float]) -> MetricStats:
        arr = np.array(values, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        cv = std / mean if mean != 0.0 else 0.0
        ci_lo, ci_hi = self._bootstrap_ci(arr)
        return MetricStats(
            name=name,
            mean=mean,
            std=std,
            cv=cv,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            values=values,
        )

    def _build_report(
        self,
        tracker_name: str,
        dataset_name: str,
        results: List["BenchmarkResult"],
    ) -> VarianceReport:
        fps_vals = [r.mean_fps for r in results]
        iou_vals = [r.mean_iou for r in results]
        mem_vals = [r.peak_memory_mb for r in results]

        metrics: Dict[str, MetricStats] = {
            "mean_iou": self._metric_stats("mean_iou", iou_vals),
            "fps": self._metric_stats("fps", fps_vals),
            "peak_memory_mb": self._metric_stats("peak_memory_mb", mem_vals),
        }

        # Optional metrics (present only when all runs include them)
        sauc_vals = [r.mean_success_auc for r in results if r.mean_success_auc is not None]
        if len(sauc_vals) == len(results):
            metrics["success_auc"] = self._metric_stats("success_auc", sauc_vals)

        energy_vals = [r.mean_energy_per_frame_mj for r in results
                       if r.mean_energy_per_frame_mj is not None]
        if len(energy_vals) == len(results):
            metrics["energy_per_frame_mj"] = self._metric_stats(
                "energy_per_frame_mj", energy_vals
            )

        fps_cv = metrics["fps"].cv
        reliable = fps_cv < self.reliability_threshold

        return VarianceReport(
            tracker_name=tracker_name,
            dataset_name=dataset_name,
            n_runs=self.n_runs,
            metrics=metrics,
            reliable=reliable,
            reliability_threshold=self.reliability_threshold,
        )
