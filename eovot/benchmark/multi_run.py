"""Multi-run statistical benchmark for measuring tracker performance repeatability.

Running a tracker once per sequence gives a single noisy estimate of FPS and
latency.  On edge hardware, thermal throttling, OS scheduler jitter, and CPU
cache cold-starts can produce 10–30% run-to-run variance.

:class:`MultiRunBenchmark` repeats each sequence ``n_runs`` times, optionally
discards a warm-up run, and reports mean, standard deviation, and 95%
confidence intervals for all hardware metrics.

This is required for publishable benchmarking claims — "X fps" must be reported
as "X ± Y fps" with an explicit number of runs and confidence interval.

Example::

    from eovot.benchmark.multi_run import MultiRunBenchmark
    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, sequence_length=50)
    tracker = MOSSETracker()
    benchmark = MultiRunBenchmark(n_runs=3, warmup_runs=1)
    report = benchmark.run(tracker, dataset, dataset_name="synthetic")
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..trackers.base import BaseTracker

# ---------------------------------------------------------------------------
# T-distribution critical values for 95 % two-tailed CI (no scipy dependency)
# ---------------------------------------------------------------------------

_T_CRIT_TABLE: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
    40: 2.021, 60: 2.000, 120: 1.980,
}


def _t_crit_95(df: int) -> float:
    """Return the two-tailed 95 % t-critical value for *df* degrees of freedom."""
    if df >= 120:
        return 1.960  # standard normal approximation
    # Find the two closest keys and linearly interpolate.
    keys = sorted(_T_CRIT_TABLE.keys())
    if df in _T_CRIT_TABLE:
        return _T_CRIT_TABLE[df]
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    t_lo, t_hi = _T_CRIT_TABLE[lo], _T_CRIT_TABLE[hi]
    alpha = (df - lo) / (hi - lo)
    return t_lo + alpha * (t_hi - t_lo)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RunStats:
    """Statistics over N independent runs for a single scalar metric.

    Attributes:
        mean:      Sample mean.
        std:       Sample standard deviation (ddof=1).
        min:       Minimum observed value.
        max:       Maximum observed value.
        ci95_low:  Lower bound of the 95 % confidence interval.
        ci95_high: Upper bound of the 95 % confidence interval.
        n_runs:    Number of runs that produced these statistics.
    """

    mean: float
    std: float
    min: float
    max: float
    ci95_low: float
    ci95_high: float
    n_runs: int

    def __str__(self) -> str:
        return (
            f"{self.mean:.3f} ± {self.std:.3f}  "
            f"95 CI=[{self.ci95_low:.3f}, {self.ci95_high:.3f}]  "
            f"range=[{self.min:.3f}, {self.max:.3f}]  n={self.n_runs}"
        )


def _run_stats(values: List[float]) -> RunStats:
    """Compute :class:`RunStats` from a list of scalar measurements."""
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        sem = std / float(np.sqrt(n))
        t = _t_crit_95(n - 1)
        ci_low, ci_high = mean - t * sem, mean + t * sem
    else:
        ci_low = ci_high = mean
    return RunStats(
        mean=mean, std=std,
        min=float(arr.min()), max=float(arr.max()),
        ci95_low=ci_low, ci95_high=ci_high,
        n_runs=n,
    )


@dataclass
class MultiRunSummary:
    """Aggregated results from a multi-run benchmark.

    Contains per-metric :class:`RunStats` (mean, std, 95 % CI) and optional
    diagnostic flags for warm-up overhead and thermal throttling.

    Attributes:
        tracker_name:             Name of the evaluated tracker.
        dataset_name:             Name of the dataset used.
        n_runs:                   Number of measurement runs (excluding warm-up).
        warmup_runs:              Number of discarded warm-up runs.
        fps:                      FPS statistics across runs.
        latency_mean_ms:          Mean per-frame latency statistics.
        latency_p95_ms:           P95 latency statistics.
        peak_memory_mb:           Peak RSS memory statistics.
        mean_iou:                 Mean IoU statistics.
        thermal_trend_ms_per_run: Slope of mean latency vs. run index (ms/run).
                                  Positive → latency increases over successive runs
                                  (possible thermal throttling).  ``None`` when
                                  fewer than two runs were recorded.
        warmup_overhead_ratio:    First-run mean latency divided by the
                                  steady-state mean.  Values >2 indicate
                                  significant JIT / cache cold-start effects.
        per_run_results:          Individual :class:`BenchmarkResult` for each
                                  measurement run (warm-up runs excluded).
    """

    tracker_name: str
    dataset_name: str
    n_runs: int
    warmup_runs: int

    fps: RunStats
    latency_mean_ms: RunStats
    latency_p95_ms: RunStats
    peak_memory_mb: RunStats
    mean_iou: RunStats

    thermal_trend_ms_per_run: Optional[float] = None
    warmup_overhead_ratio: Optional[float] = None
    per_run_results: List[BenchmarkResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived flags
    # ------------------------------------------------------------------

    @property
    def has_thermal_throttling(self) -> bool:
        """True when the thermal trend exceeds 2 % of the mean latency per run."""
        if self.thermal_trend_ms_per_run is None or self.latency_mean_ms.mean == 0:
            return False
        return abs(self.thermal_trend_ms_per_run) / self.latency_mean_ms.mean > 0.02

    @property
    def has_warmup_penalty(self) -> bool:
        """True when the warm-up overhead ratio exceeds 1.5 (50 % overhead)."""
        return self.warmup_overhead_ratio is not None and self.warmup_overhead_ratio > 1.5

    @property
    def fps_cv(self) -> float:
        """Coefficient of variation for FPS (std / mean).  Values >0.1 are high."""
        return self.fps.std / self.fps.mean if self.fps.mean > 0 else 0.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return a flat dict suitable for logging or JSON export."""
        d: Dict = {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "n_runs": self.n_runs,
            "warmup_runs": self.warmup_runs,
            "fps_mean": round(self.fps.mean, 2),
            "fps_std": round(self.fps.std, 2),
            "fps_ci95": [round(self.fps.ci95_low, 2), round(self.fps.ci95_high, 2)],
            "fps_cv": round(self.fps_cv, 4),
            "latency_mean_ms": round(self.latency_mean_ms.mean, 3),
            "latency_std_ms": round(self.latency_mean_ms.std, 3),
            "latency_p95_ms_mean": round(self.latency_p95_ms.mean, 3),
            "peak_memory_mb_mean": round(self.peak_memory_mb.mean, 2),
            "mean_iou_mean": round(self.mean_iou.mean, 4),
            "mean_iou_std": round(self.mean_iou.std, 4),
        }
        if self.thermal_trend_ms_per_run is not None:
            d["thermal_trend_ms_per_run"] = round(self.thermal_trend_ms_per_run, 5)
            d["thermal_throttling_detected"] = self.has_thermal_throttling
        if self.warmup_overhead_ratio is not None:
            d["warmup_overhead_ratio"] = round(self.warmup_overhead_ratio, 3)
            d["warmup_penalty_detected"] = self.has_warmup_penalty
        return d

    def __str__(self) -> str:
        s = self.summary()
        ci = s["fps_ci95"]
        base = (
            f"MultiRunSummary[{s['tracker']} on {s['dataset']}] "
            f"FPS={s['fps_mean']}±{s['fps_std']}  "
            f"95 CI=[{ci[0]}, {ci[1]}]  "
            f"mIoU={s['mean_iou_mean']}±{s['mean_iou_std']}  "
            f"({s['n_runs']} runs)"
        )
        flags = []
        if s.get("thermal_throttling_detected"):
            flags.append("⚠️ thermal-throttle")
        if s.get("warmup_penalty_detected"):
            flags.append("⚠️ warmup-penalty")
        return base + ("  " + "  ".join(flags) if flags else "")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MultiRunBenchmark:
    """Evaluate a tracker over multiple independent runs for reliable statistics.

    A single-pass benchmark cannot distinguish tracker speed from OS noise.
    :class:`MultiRunBenchmark` performs ``warmup_runs + n_runs`` full passes,
    discards the warm-up passes, and reports sample statistics with 95 %
    confidence intervals.  Two diagnostic flags are also computed:

    * **Warm-up overhead ratio** — first-run latency vs. steady-state mean.
      Flags potential JIT / CPU-cache cold-start distortion in single-run results.
    * **Thermal trend** — linear slope of per-run mean latency.  A positive
      slope suggests the CPU is throttling over successive runs.

    Args:
        n_runs:         Number of measurement runs (warm-up excluded).  Minimum
                        2 for a meaningful standard deviation; 5 recommended for
                        publication-quality results.  Default: ``3``.
        warmup_runs:    Runs to discard before measurement starts.  One warm-up
                        run is typically sufficient to prime JIT and CPU caches.
                        Default: ``1``.
        tdp_watts:      Enables energy profiling in each run when set.  Pass the
                        device CPU TDP in Watts.  Default: ``None``.
        verbose:        Print per-run progress.  Default: ``True``.
    """

    def __init__(
        self,
        n_runs: int = 3,
        warmup_runs: int = 1,
        tdp_watts: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")
        if warmup_runs < 0:
            raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
        self.n_runs = n_runs
        self.warmup_runs = warmup_runs
        self.tdp_watts = tdp_watts
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> MultiRunSummary:
        """Evaluate *tracker* on *dataset* for ``warmup_runs + n_runs`` passes.

        Args:
            tracker:       Tracker under evaluation.
            dataset:       Dataset to evaluate on.
            dataset_name:  Name embedded in the returned summary.
            max_sequences: Limit sequences evaluated per run (useful for quick
                           exploratory benchmarks).

        Returns:
            :class:`MultiRunSummary` with mean/std/CI statistics over all
            measurement runs.
        """
        total = self.warmup_runs + self.n_runs
        measurement_results: List[BenchmarkResult] = []
        warmup_fps_values: List[float] = []

        for run_idx in range(total):
            is_warmup = run_idx < self.warmup_runs
            label = (
                f"warm-up {run_idx + 1}/{self.warmup_runs}"
                if is_warmup
                else f"run {run_idx - self.warmup_runs + 1}/{self.n_runs}"
            )
            if self.verbose:
                print(f"[MultiRunBenchmark] {tracker.name}  {label}")

            engine = BenchmarkEngine(verbose=False, tdp_watts=self.tdp_watts)
            result = engine.run(
                tracker=tracker,
                dataset=dataset,
                dataset_name=dataset_name,
                max_sequences=max_sequences,
            )

            if is_warmup:
                warmup_fps_values.append(result.mean_fps)
            else:
                measurement_results.append(result)

        # -----------------------------------------------------------------
        # Aggregate statistics
        # -----------------------------------------------------------------
        fps_vals = [r.mean_fps for r in measurement_results]
        lat_vals = [
            float(np.mean([s.profiling.latency_mean_ms for s in r.sequence_results]))
            for r in measurement_results
        ]
        p95_vals = [
            float(np.mean([s.profiling.latency_p95_ms for s in r.sequence_results]))
            for r in measurement_results
        ]
        mem_vals = [r.peak_memory_mb for r in measurement_results]
        iou_vals = [r.mean_iou for r in measurement_results]

        fps_stats = _run_stats(fps_vals)
        lat_stats = _run_stats(lat_vals)
        p95_stats = _run_stats(p95_vals)
        mem_stats = _run_stats(mem_vals)
        iou_stats = _run_stats(iou_vals)

        # Thermal throttling: linear trend of latency over run index
        thermal_trend: Optional[float] = None
        if len(lat_vals) >= 2:
            x = np.arange(len(lat_vals), dtype=np.float64)
            coeffs = np.polyfit(x, np.array(lat_vals), 1)
            thermal_trend = float(coeffs[0])

        # Warm-up overhead ratio: mean warm-up FPS vs. steady-state mean FPS
        warmup_ratio: Optional[float] = None
        if warmup_fps_values and fps_stats.mean > 0:
            mean_warmup_fps = float(np.mean(warmup_fps_values))
            # Overhead ratio expressed as latency ratio (inverse of FPS ratio)
            warmup_ratio = fps_stats.mean / mean_warmup_fps if mean_warmup_fps > 0 else None

        if self.verbose:
            summary = MultiRunSummary(
                tracker_name=tracker.name,
                dataset_name=dataset_name,
                n_runs=self.n_runs,
                warmup_runs=self.warmup_runs,
                fps=fps_stats,
                latency_mean_ms=lat_stats,
                latency_p95_ms=p95_stats,
                peak_memory_mb=mem_stats,
                mean_iou=iou_stats,
                thermal_trend_ms_per_run=thermal_trend,
                warmup_overhead_ratio=warmup_ratio,
                per_run_results=measurement_results,
            )
            print(f"\n[MultiRunBenchmark] {summary}")

        return MultiRunSummary(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
            n_runs=self.n_runs,
            warmup_runs=self.warmup_runs,
            fps=fps_stats,
            latency_mean_ms=lat_stats,
            latency_p95_ms=p95_stats,
            peak_memory_mb=mem_stats,
            mean_iou=iou_stats,
            thermal_trend_ms_per_run=thermal_trend,
            warmup_overhead_ratio=warmup_ratio,
            per_run_results=measurement_results,
        )

    def compare(
        self,
        trackers: List[BaseTracker],
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> List[MultiRunSummary]:
        """Evaluate multiple trackers and return their multi-run summaries.

        Trackers are evaluated sequentially.  Results are returned in the
        same order as the input list.

        Args:
            trackers:      List of trackers to compare.
            dataset:       Dataset for evaluation.
            dataset_name:  Name embedded in each summary.
            max_sequences: Limit on sequences per tracker per run.

        Returns:
            List of :class:`MultiRunSummary`, one per tracker.
        """
        return [
            self.run(t, dataset, dataset_name=dataset_name, max_sequences=max_sequences)
            for t in trackers
        ]
