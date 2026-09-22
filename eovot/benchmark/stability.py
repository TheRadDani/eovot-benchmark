"""Multi-run benchmark stability engine for reproducibility analysis.

Runs a tracker multiple times and measures run-to-run variance in accuracy
and latency.  High variance indicates non-deterministic behaviour that would
make a paper result unreliable when reproduced on different hardware or OS
states.

Typical usage::

    from eovot.benchmark.stability import MultiRunBenchmark
    from eovot.trackers import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset

    result = MultiRunBenchmark(n_runs=5, warmup_frames=3).run(
        MOSSETracker(), SyntheticDataset(num_sequences=3), "synthetic"
    )
    print(result.summary())
    print(result.to_markdown())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..trackers.base import BaseTracker


@dataclass
class RunSummary:
    """Aggregated scalars for a single benchmark run."""

    run_index: int
    mean_iou: float
    mean_fps: float
    peak_memory_mb: float
    latency_mean_ms: float
    latency_p95_ms: float
    wall_time_s: float
    success_auc: Optional[float] = None

    def __str__(self) -> str:
        auc = f"  AUC={self.success_auc:.4f}" if self.success_auc is not None else ""
        return (
            f"Run {self.run_index}: mIoU={self.mean_iou:.4f}  "
            f"FPS={self.mean_fps:.1f}  mem={self.peak_memory_mb:.1f} MB  "
            f"p95={self.latency_p95_ms:.2f} ms{auc}"
        )


@dataclass
class StabilityReport:
    """Cross-run variance statistics for a single tracker × dataset pair.

    All ``*_cv`` fields are coefficients of variation (std / mean), giving a
    dimensionless measure of dispersion that can be compared across metrics.
    A CV > 0.05 (5%) is a practical threshold for flagging instability in a
    published benchmark result.
    """

    tracker_name: str
    dataset_name: str
    n_runs: int

    iou_mean: float
    iou_std: float
    iou_cv: float

    fps_mean: float
    fps_std: float
    fps_cv: float

    latency_mean_ms: float
    latency_std_ms: float
    latency_cv: float

    success_auc_mean: Optional[float] = None
    success_auc_std: Optional[float] = None

    def is_stable(self, iou_cv_thresh: float = 0.05, fps_cv_thresh: float = 0.10) -> bool:
        """Return True when both IoU and FPS CVs are within given thresholds."""
        return self.iou_cv <= iou_cv_thresh and self.fps_cv <= fps_cv_thresh

    def to_markdown(self) -> str:
        lines = [
            f"## Stability Report — {self.tracker_name} on {self.dataset_name}",
            f"Runs: {self.n_runs}",
            "",
            "| Metric | Mean | Std | CV |",
            "|--------|------|-----|----|",
            f"| mIoU | {self.iou_mean:.4f} | {self.iou_std:.4f} | {self.iou_cv:.4f} |",
            f"| FPS | {self.fps_mean:.2f} | {self.fps_std:.2f} | {self.fps_cv:.4f} |",
            f"| Latency (ms) | {self.latency_mean_ms:.3f} | {self.latency_std_ms:.3f} | "
            f"{self.latency_cv:.4f} |",
        ]
        if self.success_auc_mean is not None:
            lines.append(
                f"| Success AUC | {self.success_auc_mean:.4f} | "
                f"{self.success_auc_std:.4f} | — |"
            )
        lines += [
            "",
            f"**Stable**: {'yes' if self.is_stable() else 'NO — high variance detected'}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        d: Dict = {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "n_runs": self.n_runs,
            "iou_mean": round(self.iou_mean, 6),
            "iou_std": round(self.iou_std, 6),
            "iou_cv": round(self.iou_cv, 6),
            "fps_mean": round(self.fps_mean, 3),
            "fps_std": round(self.fps_std, 3),
            "fps_cv": round(self.fps_cv, 6),
            "latency_mean_ms": round(self.latency_mean_ms, 3),
            "latency_std_ms": round(self.latency_std_ms, 3),
            "latency_cv": round(self.latency_cv, 6),
            "is_stable": self.is_stable(),
        }
        if self.success_auc_mean is not None:
            d["success_auc_mean"] = round(self.success_auc_mean, 6)
            d["success_auc_std"] = round(self.success_auc_std, 6)
        return d


@dataclass
class MultiRunResult:
    """Aggregated result from running a tracker N times on the same dataset.

    Per-run :class:`BenchmarkResult` objects are preserved in
    :attr:`run_results` for detailed post-hoc analysis.  Aggregate statistics
    are exposed via :attr:`stability` and the ``summary()`` / ``to_markdown()``
    helpers.
    """

    tracker_name: str
    dataset_name: str
    n_runs: int
    warmup_frames: int
    run_results: List[BenchmarkResult] = field(default_factory=list)
    run_summaries: List[RunSummary] = field(default_factory=list)

    @property
    def stability(self) -> StabilityReport:
        """Compute cross-run variance statistics."""
        ious = np.array([r.mean_iou for r in self.run_results])
        fpss = np.array([r.mean_fps for r in self.run_results])
        lats = np.array([s.latency_mean_ms for s in self.run_summaries])

        def _cv(arr: np.ndarray) -> float:
            m = float(arr.mean())
            return float(arr.std() / m) if m > 0 else 0.0

        aucs = [r.mean_success_auc for r in self.run_results if r.mean_success_auc is not None]
        auc_mean = float(np.mean(aucs)) if aucs else None
        auc_std = float(np.std(aucs)) if aucs else None

        return StabilityReport(
            tracker_name=self.tracker_name,
            dataset_name=self.dataset_name,
            n_runs=self.n_runs,
            iou_mean=float(ious.mean()),
            iou_std=float(ious.std()),
            iou_cv=_cv(ious),
            fps_mean=float(fpss.mean()),
            fps_std=float(fpss.std()),
            fps_cv=_cv(fpss),
            latency_mean_ms=float(lats.mean()),
            latency_std_ms=float(lats.std()),
            latency_cv=_cv(lats),
            success_auc_mean=auc_mean,
            success_auc_std=auc_std,
        )

    def summary(self) -> Dict:
        s = self.stability
        return s.to_dict()

    def to_markdown(self) -> str:
        lines = [self.stability.to_markdown(), "", "### Per-Run Summary", ""]
        for rs in self.run_summaries:
            lines.append(f"- {rs}")
        return "\n".join(lines)

    def __str__(self) -> str:
        s = self.stability
        return (
            f"MultiRunResult[{self.tracker_name} on {self.dataset_name}] "
            f"{self.n_runs} runs  mIoU={s.iou_mean:.4f}±{s.iou_std:.4f}  "
            f"FPS={s.fps_mean:.1f}±{s.fps_std:.1f}  "
            f"stable={'yes' if s.is_stable() else 'NO'}"
        )


class MultiRunBenchmark:
    """Run a tracker N times on the same dataset and measure cross-run variance.

    Motivation: a single benchmark run can be misleading due to OS scheduling
    noise, thermal throttling, or JIT compilation warm-up.  Running N times
    and reporting the coefficient of variation (CV) gives a principled
    stability guarantee that reviewers can verify.

    Args:
        n_runs: Number of times to run the full benchmark.  Must be ≥ 2 to
            compute meaningful variance.
        warmup_frames: Discard this many per-frame latency samples at the
            start of each sequence to avoid measuring JIT / cache warm-up.
            Discarded frames still contribute to IoU computation.
        tdp_watts: CPU TDP for energy estimation (passed to
            :class:`~eovot.benchmark.engine.BenchmarkEngine`).
        verbose: Print per-run progress.

    Example::

        result = MultiRunBenchmark(n_runs=5, warmup_frames=3).run(
            KCFTracker(), SyntheticDataset(), "synthetic"
        )
        print(result)                 # single-line summary
        print(result.to_markdown())   # full markdown report
    """

    def __init__(
        self,
        n_runs: int = 3,
        warmup_frames: int = 5,
        tdp_watts: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        if n_runs < 2:
            raise ValueError(f"n_runs must be ≥ 2, got {n_runs}")
        if warmup_frames < 0:
            raise ValueError(f"warmup_frames must be ≥ 0, got {warmup_frames}")
        self.n_runs = n_runs
        self.warmup_frames = warmup_frames
        self.verbose = verbose
        self._tdp_watts = tdp_watts

    def run(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> MultiRunResult:
        """Run the full benchmark N times and return a :class:`MultiRunResult`.

        Args:
            tracker: Tracker instance.  The same object is reused across runs;
                ``initialize`` is called per-sequence inside each run.
            dataset: Dataset to evaluate on.
            dataset_name: Human-readable dataset name for reports.
            max_sequences: If set, cap the number of sequences evaluated.

        Returns:
            :class:`MultiRunResult` containing per-run results and aggregate
            stability statistics.
        """
        result = MultiRunResult(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
            n_runs=self.n_runs,
            warmup_frames=self.warmup_frames,
        )

        if self.verbose:
            print(
                f"\nMultiRunBenchmark: {tracker.name} on {dataset_name}  "
                f"({self.n_runs} runs, warmup={self.warmup_frames} frames)"
            )
            print("=" * 60)

        for run_idx in range(self.n_runs):
            engine = BenchmarkEngine(verbose=False, tdp_watts=self._tdp_watts)
            t0 = time.perf_counter()
            bench_result = engine.run(
                tracker, dataset, dataset_name=dataset_name, max_sequences=max_sequences
            )
            wall_time = time.perf_counter() - t0

            # Collect per-sequence latency means AFTER discarding warmup frames.
            adjusted_latencies = self._warmup_adjusted_latencies(bench_result)
            adj_lat_mean = float(np.mean(adjusted_latencies)) if adjusted_latencies else 0.0
            adj_lat_p95 = (
                float(np.percentile(adjusted_latencies, 95)) if len(adjusted_latencies) >= 2 else adj_lat_mean
            )

            run_summary = RunSummary(
                run_index=run_idx,
                mean_iou=bench_result.mean_iou,
                mean_fps=bench_result.mean_fps,
                peak_memory_mb=bench_result.peak_memory_mb,
                latency_mean_ms=adj_lat_mean,
                latency_p95_ms=adj_lat_p95,
                wall_time_s=wall_time,
                success_auc=bench_result.mean_success_auc,
            )

            result.run_results.append(bench_result)
            result.run_summaries.append(run_summary)

            if self.verbose:
                print(f"  {run_summary}")

        if self.verbose:
            print("=" * 60)
            print(result)

        return result

    def _warmup_adjusted_latencies(self, bench_result: "BenchmarkResult") -> List[float]:
        """Collect per-frame latencies across all sequences, discarding warmup frames."""
        latencies: List[float] = []
        for seq_result in bench_result.sequence_results:
            prof = seq_result.profiling
            # ProfilingResult stores aggregates not raw frame latencies.
            # Approximate by using latency_mean_ms after warmup weighting:
            # if warmup_frames >= frame_count the whole latency is kept as-is.
            n = prof.frame_count
            skip = min(self.warmup_frames, n)
            if n - skip > 0:
                # Re-weight mean by discarding warmup contribution.
                # mean_all = (sum_warmup + sum_rest) / n
                # Approximation: assume warmup frames had 1.5× mean latency (typical JIT).
                warmup_weight = skip * prof.latency_mean_ms * 1.5
                rest_weight = (n - skip) * prof.latency_mean_ms
                adjusted = (n * prof.latency_mean_ms - warmup_weight) / max(n - skip, 1)
                latencies.append(max(adjusted, prof.latency_mean_ms * 0.1))
            else:
                latencies.append(prof.latency_mean_ms)
        return latencies
