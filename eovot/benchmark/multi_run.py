"""Multi-run benchmark engine for publication-quality tracker evaluation.

Runs each tracker K times on the same dataset, then aggregates results
with per-metric mean, standard deviation, and coefficient of variation.
This separates true algorithm performance from measurement noise caused
by OS scheduling, memory allocation jitter, and CPU frequency scaling —
all of which distort single-run FPS/latency numbers.

The module also produces a ``stability_score`` for each tracker, ranking
how reproducible its performance is across repeated evaluations.

Typical usage::

    from eovot.benchmark.multi_run import MultiRunBenchmark
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker

    dataset = SyntheticDataset(num_sequences=10, num_frames=150)
    benchmark = MultiRunBenchmark(num_runs=5, verbose=True)

    summary = benchmark.run(
        trackers=[MOSSETracker(), KCFTracker()],
        dataset=dataset,
        dataset_name="Synthetic-Linear",
    )

    for entry in summary:
        print(entry)

    print(benchmark.to_markdown(summary))
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..trackers.base import BaseTracker
from ..datasets.base import BaseDataset
from .engine import BenchmarkEngine, BenchmarkResult


@dataclass
class MultiRunEntry:
    """Aggregated statistics for one tracker evaluated over K independent runs.

    All scalar metrics are reported as ``mean ± std`` with coefficient of
    variation (CV = std/mean) so the reader can judge how stable the
    measurement is.  High CV on FPS indicates thermal throttling or
    background OS load; high CV on mIoU signals non-determinism in the
    tracker itself.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        dataset_name: Dataset on which the tracker was evaluated.
        num_runs: Number of independent evaluation runs.
        iou_mean: Mean mIoU across runs.
        iou_std: Standard deviation of mIoU across runs.
        iou_cv: Coefficient of variation (iou_std / iou_mean).
        fps_mean: Mean FPS across runs.
        fps_std: Standard deviation of FPS across runs.
        fps_cv: Coefficient of variation for FPS.
        latency_mean_ms: Mean of per-run mean latency (milliseconds).
        latency_std_ms: Pooled standard deviation of latency across runs.
        memory_peak_mb_mean: Mean peak RSS across runs.
        memory_peak_mb_std: Standard deviation of peak RSS.
        success_auc_mean: Mean success-curve AUC across runs (or None).
        success_auc_std: Standard deviation of success AUC (or None).
        stability_score: Composite stability index in [0, 1].  Higher
            means the tracker produces consistent results across runs.
            Computed as the geometric mean of (1 - CV_iou, 1 - CV_fps)
            clamped to [0, 1].
    """

    tracker_name: str
    dataset_name: str
    num_runs: int
    iou_mean: float
    iou_std: float
    iou_cv: float
    fps_mean: float
    fps_std: float
    fps_cv: float
    latency_mean_ms: float
    latency_std_ms: float
    memory_peak_mb_mean: float
    memory_peak_mb_std: float
    success_auc_mean: Optional[float] = None
    success_auc_std: Optional[float] = None
    stability_score: float = 0.0

    def __str__(self) -> str:
        sauc = (
            f"  succ_AUC={self.success_auc_mean:.4f}±{self.success_auc_std:.4f}"
            if self.success_auc_mean is not None
            else ""
        )
        return (
            f"MultiRunEntry[{self.tracker_name} on {self.dataset_name}  K={self.num_runs}]\n"
            f"  mIoU   = {self.iou_mean:.4f} ± {self.iou_std:.4f}  (CV={self.iou_cv:.3f})\n"
            f"  FPS    = {self.fps_mean:.2f} ± {self.fps_std:.2f}  (CV={self.fps_cv:.3f})\n"
            f"  latency= {self.latency_mean_ms:.2f} ± {self.latency_std_ms:.2f} ms\n"
            f"  mem    = {self.memory_peak_mb_mean:.1f} ± {self.memory_peak_mb_std:.1f} MiB"
            + sauc
            + f"\n  stability={self.stability_score:.4f}"
        )


def _cv(values: List[float]) -> float:
    """Coefficient of variation: std/mean, or 0 when mean is zero."""
    if len(values) < 2:
        return 0.0
    m = statistics.mean(values)
    if m == 0.0:
        return 0.0
    return statistics.stdev(values) / m


def _aggregate(tracker_name: str, dataset_name: str, runs: List[BenchmarkResult]) -> MultiRunEntry:
    """Collapse K BenchmarkResult objects into one MultiRunEntry."""
    iou_vals = [r.mean_iou for r in runs]
    fps_vals = [r.mean_fps for r in runs]
    lat_vals = [r.sequence_results[0].profiling.latency_mean_ms
                for r in runs if r.sequence_results]
    mem_vals = [r.peak_memory_mb for r in runs]

    sauc_vals = [r.mean_success_auc for r in runs if r.mean_success_auc is not None]

    iou_m = float(np.mean(iou_vals))
    iou_s = float(np.std(iou_vals, ddof=1)) if len(iou_vals) > 1 else 0.0
    fps_m = float(np.mean(fps_vals))
    fps_s = float(np.std(fps_vals, ddof=1)) if len(fps_vals) > 1 else 0.0

    iou_cv = iou_s / iou_m if iou_m > 0 else 0.0
    fps_cv = fps_s / fps_m if fps_m > 0 else 0.0

    # Stability: geometric mean of (1 - CV_iou) and (1 - CV_fps), clamped.
    s_iou = max(0.0, 1.0 - iou_cv)
    s_fps = max(0.0, 1.0 - fps_cv)
    stability = float(np.sqrt(s_iou * s_fps))

    return MultiRunEntry(
        tracker_name=tracker_name,
        dataset_name=dataset_name,
        num_runs=len(runs),
        iou_mean=round(iou_m, 4),
        iou_std=round(iou_s, 4),
        iou_cv=round(iou_cv, 4),
        fps_mean=round(fps_m, 2),
        fps_std=round(fps_s, 2),
        fps_cv=round(fps_cv, 4),
        latency_mean_ms=round(float(np.mean(lat_vals)), 3) if lat_vals else 0.0,
        latency_std_ms=round(float(np.std(lat_vals, ddof=1)), 3)
        if len(lat_vals) > 1
        else 0.0,
        memory_peak_mb_mean=round(float(np.mean(mem_vals)), 2),
        memory_peak_mb_std=round(float(np.std(mem_vals, ddof=1)), 2)
        if len(mem_vals) > 1
        else 0.0,
        success_auc_mean=round(float(np.mean(sauc_vals)), 4) if sauc_vals else None,
        success_auc_std=round(float(np.std(sauc_vals, ddof=1)), 4)
        if len(sauc_vals) > 1
        else (0.0 if sauc_vals else None),
        stability_score=round(stability, 4),
    )


class MultiRunBenchmark:
    """Evaluate trackers over K repeated runs to quantify measurement noise.

    Args:
        num_runs: Number of times each tracker is run on the dataset.
            A minimum of 3 is recommended; 5 gives reliable statistics.
            Default: ``5``.
        verbose: Print per-run progress to stdout.  Default: ``True``.
        tdp_watts: If provided, enables CPU energy estimation for each run.
            Pass the device's CPU TDP in Watts.  Default: ``None``.

    Example::

        benchmark = MultiRunBenchmark(num_runs=5)
        summary = benchmark.run(
            trackers=[MOSSETracker(), KCFTracker()],
            dataset=dataset,
            dataset_name="Synthetic",
        )
        print(benchmark.to_markdown(summary))
    """

    def __init__(
        self,
        num_runs: int = 5,
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
    ) -> None:
        if num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {num_runs}")
        self.num_runs = num_runs
        self.verbose = verbose
        self.tdp_watts = tdp_watts

    def run(
        self,
        trackers: Sequence[BaseTracker],
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> List[MultiRunEntry]:
        """Run all trackers K times and return aggregated statistics.

        Args:
            trackers:      One or more tracker instances to evaluate.
            dataset:       Dataset to benchmark on.
            dataset_name:  Human-readable dataset label.
            max_sequences: Limit the number of sequences per run.

        Returns:
            List of :class:`MultiRunEntry`, one per tracker, sorted by
            ``stability_score`` descending (most stable first).
        """
        engine = BenchmarkEngine(verbose=False, tdp_watts=self.tdp_watts)
        entries: List[MultiRunEntry] = []

        for tracker in trackers:
            if self.verbose:
                print(
                    f"\n[MultiRun] {tracker.name} × {self.num_runs} runs "
                    f"on {dataset_name}"
                )

            runs: List[BenchmarkResult] = []
            for k in range(self.num_runs):
                result = engine.run(
                    tracker=tracker,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    max_sequences=max_sequences,
                )
                runs.append(result)
                if self.verbose:
                    print(
                        f"  run {k + 1:>2}/{self.num_runs}  "
                        f"mIoU={result.mean_iou:.4f}  FPS={result.mean_fps:.1f}"
                    )

            entry = _aggregate(tracker.name, dataset_name, runs)
            entries.append(entry)
            if self.verbose:
                print(entry)

        entries.sort(key=lambda e: e.stability_score, reverse=True)
        return entries

    def to_markdown(
        self,
        entries: List[MultiRunEntry],
        title: Optional[str] = None,
    ) -> str:
        """Format multi-run results as a GitHub-flavoured Markdown table.

        Each metric is rendered as ``mean ± std (CV=x.xxx)`` so readers
        can immediately judge measurement reliability.

        Args:
            entries: Output of :meth:`run`.
            title:   Optional heading line above the table.

        Returns:
            Multi-line Markdown string.
        """
        lines: List[str] = []
        if title:
            lines.append(f"## {title}")
            lines.append("")

        has_sauc = any(e.success_auc_mean is not None for e in entries)

        header = ["Rank", "Tracker", "Runs", "mIoU (mean±std)", "FPS (mean±std)",
                  "Latency ms", "Mem MiB"]
        if has_sauc:
            header.append("Succ. AUC (mean±std)")
        header.append("Stability")

        sep = [":---:"] * len(header)
        sep[1] = ":---"

        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")

        for rank, e in enumerate(entries, start=1):
            sauc_cell = (
                f"{e.success_auc_mean:.4f}±{e.success_auc_std:.4f}"
                if e.success_auc_mean is not None
                else "—"
            )
            row = [
                str(rank),
                e.tracker_name,
                str(e.num_runs),
                f"{e.iou_mean:.4f}±{e.iou_std:.4f} (CV={e.iou_cv:.3f})",
                f"{e.fps_mean:.1f}±{e.fps_std:.1f} (CV={e.fps_cv:.3f})",
                f"{e.latency_mean_ms:.2f}±{e.latency_std_ms:.2f}",
                f"{e.memory_peak_mb_mean:.1f}±{e.memory_peak_mb_std:.1f}",
            ]
            if has_sauc:
                row.append(sauc_cell)
            row.append(f"{e.stability_score:.4f}")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def to_dict(self, entries: List[MultiRunEntry]) -> Dict:
        """Serialize multi-run results to a JSON-compatible dict.

        Args:
            entries: Output of :meth:`run`.

        Returns:
            Dict with ``"num_runs"`` and ``"trackers"`` list.
        """
        return {
            "num_runs": self.num_runs,
            "trackers": [
                {
                    "tracker_name": e.tracker_name,
                    "dataset_name": e.dataset_name,
                    "num_runs": e.num_runs,
                    "iou_mean": e.iou_mean,
                    "iou_std": e.iou_std,
                    "iou_cv": e.iou_cv,
                    "fps_mean": e.fps_mean,
                    "fps_std": e.fps_std,
                    "fps_cv": e.fps_cv,
                    "latency_mean_ms": e.latency_mean_ms,
                    "latency_std_ms": e.latency_std_ms,
                    "memory_peak_mb_mean": e.memory_peak_mb_mean,
                    "memory_peak_mb_std": e.memory_peak_mb_std,
                    "success_auc_mean": e.success_auc_mean,
                    "success_auc_std": e.success_auc_std,
                    "stability_score": e.stability_score,
                }
                for e in entries
            ],
        }
