"""Core benchmark engine for EOVOT.

Orchestrates tracker evaluation over a dataset:
1. Iterates sequences.
2. Initialises the tracker on the first frame.
3. Runs the tracker on subsequent frames while profiling each update call.
4. Computes accuracy metrics against ground-truth boxes.
5. Aggregates results into a :class:`BenchmarkResult`.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.base import OTBDataset
    from eovot.benchmark.engine import BenchmarkEngine

    tracker = MOSSETracker()
    dataset = OTBDataset("/data/OTB100")
    engine  = BenchmarkEngine(verbose=True)
    result  = engine.run(tracker, dataset, dataset_name="OTB100")
    print(result.summary())
"""

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
    """Accuracy + profiling outcome for a single sequence."""

    sequence_name: str
    ious: np.ndarray          # shape (N,)
    profiling: ProfilingResult

    @property
    def mean_iou(self) -> float:
        return float(self.ious.mean()) if len(self.ious) else 0.0


@dataclass
class BenchmarkResult:
    """Aggregated results for one tracker × dataset evaluation."""

    tracker_name: str
    dataset_name: str
    sequence_results: List[SequenceResult] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Aggregate properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def mean_iou(self) -> float:
        """Mean IoU across all frames in all sequences."""
        all_ious = np.concatenate([r.ious for r in self.sequence_results])
        return float(all_ious.mean()) if len(all_ious) else 0.0

    @property
    def mean_fps(self) -> float:
        """Mean per-sequence FPS."""
        return float(np.mean([r.profiling.fps for r in self.sequence_results]))

    @property
    def peak_memory_mb(self) -> float:
        """Maximum peak memory (MiB) observed across sequences."""
        return float(
            np.max([r.profiling.peak_memory_mb for r in self.sequence_results])
        )

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def summary(self) -> Dict:
        """Return a dict suitable for logging / JSON export."""
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "num_sequences": len(self.sequence_results),
            "mean_iou": round(self.mean_iou, 4),
            "mean_fps": round(self.mean_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"BenchmarkResult[{s['tracker']} on {s['dataset']}] "
            f"mIoU={s['mean_iou']}  FPS={s['mean_fps']}  "
            f"mem={s['peak_memory_mb']} MiB  "
            f"({s['num_sequences']} sequences)"
        )


class BenchmarkEngine:
    """Run a tracker against a dataset and collect accuracy + profiling data.

    Args:
        verbose: If ``True``, print per-sequence progress to stdout.

    The engine is stateless between :meth:`run` calls — you can reuse the
    same instance across multiple (tracker, dataset) combinations.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self._metrics = MetricsEngine()
        self._profiler = Profiler()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> BenchmarkResult:
        """Evaluate *tracker* on every sequence in *dataset*.

        Args:
            tracker:       Tracker instance (will be re-initialised per sequence).
            dataset:       Dataset to evaluate on.
            dataset_name:  Label used in the result report.
            max_sequences: Evaluate only the first *N* sequences (useful for
                           quick sanity checks).

        Returns:
            :class:`BenchmarkResult` with per-sequence and aggregate metrics.
        """
        result = BenchmarkResult(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
        )
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

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _run_sequence(self, tracker: BaseTracker, seq: Sequence) -> SequenceResult:
        """Run the tracker on a single sequence and return metrics + profiling."""
        self._profiler.reset()
        frames = list(seq)          # load all frames into memory once
        gt = seq.ground_truth
        preds: List = []

        for i, frame in enumerate(frames):
            if i == 0:
                # Initialise — not timed (one-off setup cost)
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
