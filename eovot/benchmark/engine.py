"""Core benchmark engine for EOVOT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import MetricsEngine
from ..profiling.profiler import Profiler, ProfilingResult
from ..trackers.base import BaseTracker


@dataclass
class SequenceResult:
    sequence_name: str
    ious: np.ndarray
    profiling: ProfilingResult

    @property
    def mean_iou(self) -> float:
        return float(self.ious.mean()) if len(self.ious) else 0.0


@dataclass
class BenchmarkResult:
    tracker_name: str
    dataset_name: str
    sequence_results: List[SequenceResult] = field(default_factory=list)

    @property
    def mean_iou(self) -> float:
        all_ious = np.concatenate([r.ious for r in self.sequence_results])
        return float(all_ious.mean()) if len(all_ious) else 0.0

    @property
    def mean_fps(self) -> float:
        return float(np.mean([r.profiling.fps for r in self.sequence_results]))

    @property
    def peak_memory_mb(self) -> float:
        return float(np.max([r.profiling.peak_memory_mb for r in self.sequence_results]))

    def summary(self) -> Dict:
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "num_sequences": len(self.sequence_results),
            "mean_iou": round(self.mean_iou, 4),
            "mean_fps": round(self.mean_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }

    def to_dict(self) -> Dict:
        """Serialise to the dict format consumed by :class:`~eovot.reporting.reporter.BenchmarkReporter`.

        Returns a dict with two keys:

        * ``"summary"`` — aggregate scalar metrics (same as :meth:`summary`).
        * ``"sequences"`` — list of per-sequence metric dicts.
        """
        return {
            "summary": self.summary(),
            "sequences": [
                {
                    "sequence_name": sr.sequence_name,
                    "mean_iou": round(sr.mean_iou, 4),
                    "fps": round(sr.profiling.fps, 2),
                    "mean_latency_ms": round(sr.profiling.latency_mean_ms, 3),
                    "peak_memory_mb": round(sr.profiling.peak_memory_mb, 2),
                }
                for sr in self.sequence_results
            ],
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"BenchmarkResult[{s['tracker']} on {s['dataset']}] "
            f"mIoU={s['mean_iou']}  FPS={s['mean_fps']}  "
            f"mem={s['peak_memory_mb']} MiB  ({s['num_sequences']} sequences)"
        )


class BenchmarkEngine:
    """Run a tracker against a dataset and collect accuracy + profiling data."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self._metrics = MetricsEngine()
        self._profiler = Profiler()

    def run(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> BenchmarkResult:
        """Evaluate *tracker* on every sequence in *dataset*."""
        result = BenchmarkResult(tracker_name=tracker.name, dataset_name=dataset_name)
        n = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)

        if self.verbose:
            print(f"\nEvaluating {tracker.name} on {dataset_name} ({n} sequences)")
            print("-" * 60)

        for idx in range(n):
            seq = dataset[idx]
            seq_result = self._run_sequence(tracker, seq)
            result.sequence_results.append(seq_result)
            if self.verbose:
                print(
                    f"  [{idx + 1:>3}/{n}] {seq_result.sequence_name:<30s} "
                    f"mIoU={seq_result.mean_iou:.3f}  "
                    f"FPS={seq_result.profiling.fps:.1f}"
                )

        if self.verbose:
            print("-" * 60)
            print(result)

        return result

    def _run_sequence(self, tracker: BaseTracker, seq: Sequence) -> SequenceResult:
        self._profiler.reset()
        frames = list(seq)
        gt = seq.ground_truth
        preds: List = []

        for i, frame in enumerate(frames):
            if i == 0:
                tracker.initialize(frame, seq.init_bbox)
                preds.append(seq.init_bbox)
            else:
                self._profiler.start_frame()
                bbox = tracker.update(frame)
                self._profiler.end_frame()
                preds.append(bbox)

        preds_arr = np.array(preds, dtype=np.float64)
        n_eval = min(len(preds_arr), len(gt))
        ious = self._metrics.batch_iou(preds_arr[:n_eval], gt[:n_eval])

        return SequenceResult(
            sequence_name=seq.name,
            ious=ious,
            profiling=self._profiler.summary(tracker.name),
        )
