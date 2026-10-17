"""VOT reset-based evaluation protocol for EOVOT.

The :class:`~eovot.benchmark.engine.BenchmarkEngine` performs *One-Pass
Evaluation* (OPE): the tracker is initialized once and runs through the
entire sequence without any intervention.  While simple, OPE over-rewards
trackers that drift silently rather than failing outright.

The official VOT challenge uses a *reset-based* protocol instead:

1. The tracker is initialized on frame 0.
2. At every subsequent frame, if the predicted IoU with ground truth drops
   to **zero**, a *failure* is recorded.
3. The tracker is **not** queried for the next ``gap`` frames (default 5).
   These frames are assigned an overlap of 0.
4. After the gap, the tracker is **re-initialized** using the ground-truth
   box at that frame, and evaluation continues.

Expected Average Overlap (EAO) is then computed as the mean of all per-frame
overlaps (including the forced zeros during gap windows).  This penalizes
fragile trackers proportionally to how often and how badly they fail.

References
----------
- Kristan et al., "The Visual Object Tracking VOT2018 Challenge Results",
  ECCV 2018 Workshops.
- Čehovin et al., "Visual object tracking performance measures revisited",
  IEEE TIP 2016.

Example::

    from eovot.benchmark.vot_engine import VOTEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker

    dataset = SyntheticDataset(num_sequences=5, num_frames=150, motion="random")
    engine  = VOTEngine(gap=5, failure_threshold=0.0, verbose=True)

    for idx in range(len(dataset)):
        seq = dataset[idx]
        result = engine.run_sequence(MOSSETracker(), seq)
        print(result)

    # Or evaluate all sequences at once
    from eovot.datasets.synthetic import SyntheticDataset
    results = engine.run_dataset(MOSSETracker(), dataset, dataset_name="Synthetic")
    print(f"EAO={results.eao:.4f}  failures={results.total_failures}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import MetricsEngine
from ..profiling.profiler import Profiler, ProfilingResult
from ..trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VOTSequenceResult:
    """Per-sequence outcome from a VOT reset-based evaluation run.

    Attributes:
        sequence_name:    Name of the evaluated sequence.
        overlaps:         Per-frame overlap array; zeros cover gap windows.
        failure_frames:   Frame indices at which failures were detected.
        reinit_frames:    Frame indices at which the tracker was re-initialized.
        num_failures:     Number of failure events (== len(failure_frames)).
        eao:              Mean overlap across *all* frames (including gap zeros).
        accuracy:         Mean overlap over *non-gap* frames only.
        robustness_rate:  Failures per 100 evaluated frames.
        profiling:        Latency / memory profiling summary.
        predictions:      Predicted bounding boxes, shape ``(N, 4)``; gap frames
                          hold the last valid prediction.
        ground_truths:    Ground-truth bounding boxes, shape ``(N, 4)``.
    """

    sequence_name: str
    overlaps: np.ndarray
    failure_frames: List[int]
    reinit_frames: List[int]
    num_failures: int
    eao: float
    accuracy: float
    robustness_rate: float
    profiling: ProfilingResult
    predictions: Optional[np.ndarray] = None
    ground_truths: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return (
            f"VOTSequenceResult[{self.sequence_name}] "
            f"EAO={self.eao:.4f}  acc={self.accuracy:.4f}  "
            f"failures={self.num_failures}  "
            f"FPS={self.profiling.fps:.1f}"
        )


@dataclass
class VOTDatasetResult:
    """Aggregate VOT evaluation outcome over all sequences in a dataset.

    Attributes:
        tracker_name:  Name of the evaluated tracker.
        dataset_name:  Name of the dataset.
        sequence_results:  Per-sequence results.
        eao:           Mean EAO across all sequences (primary VOT metric).
        accuracy:      Mean per-sequence accuracy (non-gap frames only).
        mean_failures_per_sequence:  Average failure count per sequence.
        mean_fps:      Mean frames per second across sequences.
        peak_memory_mb:  Maximum memory usage across sequences.
    """

    tracker_name: str
    dataset_name: str
    sequence_results: List[VOTSequenceResult] = field(default_factory=list)

    @property
    def eao(self) -> float:
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.eao for r in self.sequence_results]))

    @property
    def accuracy(self) -> float:
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.accuracy for r in self.sequence_results]))

    @property
    def total_failures(self) -> int:
        return sum(r.num_failures for r in self.sequence_results)

    @property
    def mean_failures_per_sequence(self) -> float:
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.num_failures for r in self.sequence_results]))

    @property
    def mean_fps(self) -> float:
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.profiling.fps for r in self.sequence_results]))

    @property
    def peak_memory_mb(self) -> float:
        if not self.sequence_results:
            return 0.0
        return float(np.max([r.profiling.peak_memory_mb for r in self.sequence_results]))

    def summary(self) -> Dict:
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "num_sequences": len(self.sequence_results),
            "eao": round(self.eao, 4),
            "accuracy": round(self.accuracy, 4),
            "total_failures": self.total_failures,
            "mean_failures_per_sequence": round(self.mean_failures_per_sequence, 3),
            "mean_fps": round(self.mean_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"VOTDatasetResult[{s['tracker']} on {s['dataset']}] "
            f"EAO={s['eao']}  acc={s['accuracy']}  "
            f"failures={s['total_failures']}  FPS={s['mean_fps']}  "
            f"({s['num_sequences']} sequences)"
        )


# ---------------------------------------------------------------------------
# VOT engine
# ---------------------------------------------------------------------------

class VOTEngine:
    """Evaluate a tracker using the VOT reset-based protocol.

    Args:
        gap:               Number of frames to skip after a failure before
                           re-initializing.  VOT standard is 5.  Must be >= 0.
        failure_threshold: IoU value at or below which a failure is declared.
                           VOT standard is 0.0 (prediction does not overlap GT
                           at all).  Raising this (e.g. to 0.1) makes the
                           protocol stricter.
        verbose:           Print per-sequence progress to stdout.
    """

    def __init__(
        self,
        gap: int = 5,
        failure_threshold: float = 0.0,
        verbose: bool = True,
    ) -> None:
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")
        self.gap = gap
        self.failure_threshold = failure_threshold
        self.verbose = verbose
        self._metrics = MetricsEngine()
        self._profiler = Profiler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_sequence(self, tracker: BaseTracker, seq: Sequence) -> VOTSequenceResult:
        """Evaluate *tracker* on one sequence using the reset protocol.

        Args:
            tracker: Any :class:`~eovot.trackers.base.BaseTracker` instance.
            seq:     A sequence from a :class:`~eovot.datasets.base.BaseDataset`.

        Returns:
            :class:`VOTSequenceResult` with per-frame overlaps, failure info,
            and profiling data.
        """
        self._profiler.reset()

        frames = list(seq)
        gt = seq.ground_truth
        n = min(len(frames), len(gt))

        overlaps: List[float] = []
        predictions: List[BBox] = []
        failure_frames: List[int] = []
        reinit_frames: List[int] = []
        gap_mask: List[bool] = []  # True for frames in a gap window

        # Initialize on frame 0
        tracker.initialize(frames[0], seq.init_bbox)
        first_iou = float(
            self._metrics.batch_iou(
                np.array([seq.init_bbox]),
                np.array([gt[0]])
            )[0]
        )
        overlaps.append(first_iou)
        predictions.append(seq.init_bbox)
        gap_mask.append(False)

        skip_until = -1  # frame index after which we re-initialize

        for i in range(1, n):
            if i <= skip_until:
                # Inside a gap window — assign zero overlap, no tracker call.
                overlaps.append(0.0)
                predictions.append(predictions[-1])  # carry last known box
                gap_mask.append(True)
                continue

            if i == skip_until + 1 and skip_until >= 0:
                # Gap window just ended — re-initialize with GT box.
                tracker.initialize(frames[i], tuple(gt[i]))  # type: ignore[arg-type]
                reinit_frames.append(i)
                skip_until = -1
                iou_val = 1.0  # initialization frame: overlap = 1 by convention
                overlaps.append(iou_val)
                predictions.append(tuple(gt[i]))  # type: ignore[misc]
                gap_mask.append(False)
                continue

            self._profiler.start_frame()
            pred = tracker.update(frames[i])
            self._profiler.end_frame()

            gt_box = tuple(gt[i])  # type: ignore[misc]
            iou_val = float(
                self._metrics.batch_iou(
                    np.array([pred]),
                    np.array([gt_box])
                )[0]
            )

            if iou_val <= self.failure_threshold:
                # Failure detected
                failure_frames.append(i)
                overlaps.append(0.0)
                predictions.append(pred)
                gap_mask.append(False)
                skip_until = i + self.gap  # next `gap` frames are skipped
            else:
                overlaps.append(iou_val)
                predictions.append(pred)
                gap_mask.append(False)

        overlaps_arr = np.array(overlaps, dtype=np.float64)
        gap_arr = np.array(gap_mask)

        # EAO: mean over ALL frames (gap zeros included) — VOT standard.
        eao = float(overlaps_arr.mean()) if len(overlaps_arr) > 0 else 0.0

        # Accuracy: mean over non-gap, non-failure frames only.
        non_gap = overlaps_arr[~gap_arr]
        accuracy = float(non_gap[non_gap > 0].mean()) if (non_gap > 0).any() else 0.0

        # Robustness: failures per 100 tracking frames (excluding gap windows).
        n_tracking = int((~gap_arr).sum())
        robustness_rate = (len(failure_frames) / n_tracking * 100.0) if n_tracking > 0 else 0.0

        try:
            prof = self._profiler.summary(tracker.name)
        except ValueError:
            # No update frames were profiled (very short sequence).
            from ..profiling.profiler import ProfilingResult
            prof = ProfilingResult(
                tracker_name=tracker.name,
                frame_count=0,
                fps=0.0,
                latency_mean_ms=0.0,
                latency_std_ms=0.0,
                latency_p95_ms=0.0,
                peak_memory_mb=0.0,
            )

        preds_arr = np.array(predictions, dtype=np.float64)
        gt_arr = np.array(gt[:n], dtype=np.float64)

        return VOTSequenceResult(
            sequence_name=seq.name,
            overlaps=overlaps_arr,
            failure_frames=failure_frames,
            reinit_frames=reinit_frames,
            num_failures=len(failure_frames),
            eao=eao,
            accuracy=accuracy,
            robustness_rate=robustness_rate,
            profiling=prof,
            predictions=preds_arr,
            ground_truths=gt_arr,
        )

    def run_dataset(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> VOTDatasetResult:
        """Evaluate *tracker* over all (or up to *max_sequences*) sequences.

        Args:
            tracker:        Tracker to evaluate.
            dataset:        Dataset to iterate over.
            dataset_name:   Human-readable label used in the result summary.
            max_sequences:  Cap on number of sequences to evaluate.

        Returns:
            :class:`VOTDatasetResult` aggregating all per-sequence outcomes.
        """
        n = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)
        agg = VOTDatasetResult(tracker_name=tracker.name, dataset_name=dataset_name)

        if self.verbose:
            print(
                f"\n[VOT] Evaluating {tracker.name} on {dataset_name} "
                f"({n} sequences, gap={self.gap}, "
                f"fail_thr={self.failure_threshold})"
            )
            print("-" * 60)

        for idx in range(n):
            seq = dataset[idx]
            seq_result = self.run_sequence(tracker, seq)
            agg.sequence_results.append(seq_result)

            if self.verbose:
                print(
                    f"  [{idx + 1:>3}/{n}] {seq_result.sequence_name:<30s} "
                    f"EAO={seq_result.eao:.3f}  "
                    f"acc={seq_result.accuracy:.3f}  "
                    f"fail={seq_result.num_failures}  "
                    f"FPS={seq_result.profiling.fps:.1f}"
                )

        if self.verbose:
            print("-" * 60)
            print(agg)

        return agg
