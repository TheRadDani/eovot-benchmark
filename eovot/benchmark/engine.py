"""Core benchmark engine for EOVOT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import MetricsEngine
from ..profiling.profiler import Profiler, ProfilingResult
from ..trackers.base import BaseTracker


@dataclass
class BenchmarkConfig:
    """Runtime configuration for :class:`BenchmarkEngine`.

    Args:
        max_sequences: Cap on the number of sequences to evaluate.
            ``None`` evaluates the entire dataset.  Useful for quick smoke
            tests without running the full benchmark.
        verbose: Print per-sequence progress and a final summary to stdout.
    """

    max_sequences: Optional[int] = None
    verbose: bool = True


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

    def summary(self) -> Dict[str, Any]:
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "num_sequences": len(self.sequence_results),
            "mean_iou": round(self.mean_iou, 4),
            "mean_fps": round(self.mean_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a reporter-compatible dict.

        Returns a dict with two keys:

        * ``"summary"`` — scalar metrics (tracker name, mIoU, FPS, …).
        * ``"sequences"`` — list of per-sequence breakdowns, each containing
          the raw per-frame IoU array so downstream tools can compute
          success / precision curves without re-running the tracker.
        """
        return {
            "summary": self.summary(),
            "sequences": [
                {
                    "sequence_name": r.sequence_name,
                    "mean_iou": round(r.mean_iou, 4),
                    "ious": r.ious.tolist(),
                    "fps": round(r.profiling.fps, 2),
                    "mean_latency_ms": round(r.profiling.latency_mean_ms, 3),
                    "latency_p95_ms": round(r.profiling.latency_p95_ms, 3),
                    "peak_memory_mb": round(r.profiling.peak_memory_mb, 2),
                }
                for r in self.sequence_results
            ],
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"BenchmarkResult[{s['tracker_name']} on {s['dataset_name']}] "
            f"mIoU={s['mean_iou']}  FPS={s['mean_fps']}  "
            f"mem={s['peak_memory_mb']} MiB  ({s['num_sequences']} sequences)"
        )


class BenchmarkEngine:
    """Run a tracker against a dataset and collect accuracy + profiling data.

    Args:
        verbose: Print per-sequence progress.  Overridden by *config* when
            *config* is provided.
        config: Optional :class:`BenchmarkConfig` controlling run behaviour.
            When supplied it takes precedence over the *verbose* argument and
            also provides the default ``max_sequences`` cap.

    Example::

        from eovot.benchmark.engine import BenchmarkEngine, BenchmarkConfig
        from eovot.trackers.mosse import MOSSETracker
        from eovot.datasets.base import OTBDataset

        cfg = BenchmarkConfig(max_sequences=10, verbose=True)
        engine = BenchmarkEngine(config=cfg)
        result = engine.run(MOSSETracker(), OTBDataset("/data/OTB100"), dataset_name="OTB100")
        print(result["summary"])
    """

    def __init__(
        self,
        verbose: bool = True,
        config: Optional[BenchmarkConfig] = None,
    ) -> None:
        if config is not None:
            self.verbose = config.verbose
            self._config = config
        else:
            self.verbose = verbose
            self._config = BenchmarkConfig(verbose=verbose)
        self._metrics = MetricsEngine()
        self._profiler = Profiler()

    def run(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate *tracker* on every sequence in *dataset*.

        Args:
            tracker: Any :class:`~eovot.trackers.base.BaseTracker` instance.
            dataset: Any :class:`~eovot.datasets.base.BaseDataset` instance.
            dataset_name: Human-readable dataset label used in reports.
            max_sequences: Override the sequence cap for this run.  Falls back
                to :attr:`BenchmarkConfig.max_sequences` when not given.

        Returns:
            A dict with ``"summary"`` (scalar metrics) and ``"sequences"``
            (per-sequence breakdown including raw IoU arrays), compatible with
            :class:`~eovot.reporting.reporter.BenchmarkReporter`.
        """
        if max_sequences is None:
            max_sequences = self._config.max_sequences

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

        return result.to_dict()

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
