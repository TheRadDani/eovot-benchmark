"""VOT Challenge Re-initialization Protocol for EOVOT.

Implements the standard evaluation protocol used by the VOT challenge series
(Kristan et al., 2016-2024):

1. Run the tracker on a sequence until a **failure** is detected
   (per-frame IoU drops below ``failure_threshold``).
2. Skip ``skip_frames`` frames after the failure (the tracker is considered
   "lost" during this gap — IoU = 0 for those frames).
3. **Re-initialize** the tracker at the first frame after the gap, using the
   ground-truth bounding box.
4. Repeat until the sequence ends.

This produces three key metrics:

- **Accuracy (A)** — mean IoU over all *non-gap* frames (frames where the
  tracker is active).  Equivalent to the VOT per-sequence accuracy scalar.
- **Reliability (R)** — total number of failures normalized by sequence
  length: ``num_failures / num_frames``.  Lower is better.
- **Expected Average Overlap (EAO)** — mean IoU across *all* frames
  (including gap frames counted as 0), providing a single scalar that
  penalises both low accuracy and frequent failures.

These match the scalar definitions from the official VOT toolkit for single-
object re-detection experiments, making EOVOT results directly comparable to
published VOT benchmarks.

Usage::

    from eovot.benchmark.reinit_protocol import ReinitEvaluator, ReinitResult

    evaluator = ReinitEvaluator(failure_threshold=0.1, skip_frames=5)

    # Per-sequence
    result = evaluator.evaluate_sequence(tracker, seq)
    print(result.eao, result.accuracy, result.num_failures)

    # Whole dataset
    summary = evaluator.evaluate_dataset(tracker, dataset, dataset_name="OTB100")
    print(summary)

    # Aggregate multiple trackers
    from eovot.benchmark.reinit_protocol import compare_reinit_results
    table = compare_reinit_results({"MOSSE": mosse_summary, "KCF": kcf_summary})
    print(table)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import iou as _iou
from ..trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Per-sequence result
# ---------------------------------------------------------------------------

@dataclass
class ReinitSequenceResult:
    """Re-initialization protocol result for a single sequence.

    Attributes:
        sequence_name:   Name of the evaluated sequence.
        num_frames:      Total frames in the sequence.
        num_failures:    Number of times IoU dropped below the threshold.
        failure_frames:  Frame indices at which each failure was detected.
        ious:            Per-frame IoU array; gap frames have value 0.
        accuracy:        Mean IoU over all *active* (non-gap) frames.
        reliability:     Failures-per-frame ratio.  Lower is better.
        eao:             Expected Average Overlap — mean IoU over all frames
                         (including gap zeros).  Primary scalar metric.
        mean_fps:        Mean tracker speed in frames-per-second (update only,
                         excluding re-initialisation overhead).
    """

    sequence_name: str
    num_frames: int
    num_failures: int
    failure_frames: List[int]
    ious: np.ndarray
    accuracy: float
    reliability: float
    eao: float
    mean_fps: float

    def __str__(self) -> str:
        return (
            f"ReinitSequenceResult({self.sequence_name!r}: "
            f"EAO={self.eao:.4f}  A={self.accuracy:.4f}  "
            f"R={self.reliability:.4f}  failures={self.num_failures}  "
            f"FPS={self.mean_fps:.1f})"
        )


# ---------------------------------------------------------------------------
# Dataset-level aggregate
# ---------------------------------------------------------------------------

@dataclass
class ReinitBenchmarkResult:
    """Aggregate re-initialization protocol result over a full dataset.

    Attributes:
        tracker_name:    Tracker identifier.
        dataset_name:    Dataset identifier.
        sequence_results: One :class:`ReinitSequenceResult` per sequence.
        mean_eao:        Mean EAO across all sequences.
        mean_accuracy:   Mean accuracy across all sequences.
        mean_reliability: Mean reliability (failures/frame) across sequences.
        total_failures:  Total failure count across all sequences.
        mean_fps:        Mean FPS across all sequences.
    """

    tracker_name: str
    dataset_name: str
    sequence_results: List[ReinitSequenceResult] = field(default_factory=list)

    @property
    def mean_eao(self) -> float:
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.eao for r in self.sequence_results]))

    @property
    def mean_accuracy(self) -> float:
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.accuracy for r in self.sequence_results]))

    @property
    def mean_reliability(self) -> float:
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.reliability for r in self.sequence_results]))

    @property
    def total_failures(self) -> int:
        return sum(r.num_failures for r in self.sequence_results)

    @property
    def mean_fps(self) -> float:
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.mean_fps for r in self.sequence_results]))

    def summary(self) -> Dict:
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "num_sequences": len(self.sequence_results),
            "mean_eao": round(self.mean_eao, 4),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "mean_reliability": round(self.mean_reliability, 4),
            "total_failures": self.total_failures,
            "mean_fps": round(self.mean_fps, 2),
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"ReinitBenchmarkResult[{s['tracker']} on {s['dataset']}] "
            f"EAO={s['mean_eao']}  A={s['mean_accuracy']}  "
            f"R={s['mean_reliability']}  failures={s['total_failures']}  "
            f"({s['num_sequences']} sequences)"
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ReinitEvaluator:
    """Evaluate a tracker using the VOT re-initialization protocol.

    The evaluator runs a tracker on each sequence, detects failures
    (IoU < ``failure_threshold``), skips ``skip_frames`` gap frames, then
    re-initialises the tracker at the next ground-truth position.

    Args:
        failure_threshold: IoU below which a frame is considered a failure.
            VOT standard is 0 (any overlap loss), but 0.1 is more practical
            for soft failures.  Default: ``0.1``.
        skip_frames: Number of frames to skip after a failure before
            re-initialising.  VOT standard is 5.  Default: ``5``.
        verbose: Print per-sequence progress.  Default: ``False``.

    Example::

        evaluator = ReinitEvaluator(failure_threshold=0.1, skip_frames=5)
        result = evaluator.evaluate_dataset(tracker, dataset, "OTB100")
        print(result.mean_eao)
    """

    def __init__(
        self,
        failure_threshold: float = 0.1,
        skip_frames: int = 5,
        verbose: bool = False,
    ) -> None:
        self.failure_threshold = float(failure_threshold)
        self.skip_frames = int(skip_frames)
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Per-sequence evaluation
    # ------------------------------------------------------------------

    def evaluate_sequence(
        self,
        tracker: BaseTracker,
        seq: Sequence,
    ) -> ReinitSequenceResult:
        """Run the re-initialization protocol on a single sequence.

        Args:
            tracker: Any :class:`~eovot.trackers.base.BaseTracker` instance.
            seq:     The sequence to evaluate.

        Returns:
            :class:`ReinitSequenceResult` with per-frame IoU, EAO, accuracy,
            reliability, and speed.
        """
        frames = list(seq)
        gt = seq.ground_truth
        n = min(len(frames), len(gt))

        ious = np.zeros(n, dtype=np.float64)
        latencies: List[float] = []
        failure_frames: List[int] = []

        # --- State machine ---
        # gap_remaining > 0: tracker is in a failure gap (skipping frames)
        # otherwise: tracker is active
        gap_remaining: int = 0
        initialized: bool = False

        for i in range(n):
            frame = frames[i]

            if gap_remaining > 0:
                # Still in the failure gap — skip this frame, IoU stays 0
                gap_remaining -= 1
                if gap_remaining == 0:
                    # Re-initialize at the next frame's GT
                    tracker.initialize(frame, tuple(gt[i]))  # type: ignore[arg-type]
                    initialized = True
                    ious[i] = 1.0  # initialization frame counts as perfect overlap
                continue

            if not initialized:
                # First frame: initialize
                tracker.initialize(frame, tuple(gt[i]))  # type: ignore[arg-type]
                initialized = True
                ious[i] = 1.0  # initialization frame
                continue

            # Active tracking frame
            t0 = time.perf_counter()
            pred_bbox = tracker.update(frame)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            frame_iou = _iou(pred_bbox, tuple(gt[i]))  # type: ignore[arg-type]
            ious[i] = frame_iou

            if frame_iou < self.failure_threshold:
                # Failure detected
                failure_frames.append(i)
                ious[i] = 0.0
                gap_remaining = self.skip_frames
                initialized = False

        # --- Compute scalars ---
        # Active frames: all frames that are NOT inside the skip gap
        # A gap of `skip_frames` frames following a failure at index f
        # means frames [f+1, f+skip_frames] are gap frames.
        gap_mask = self._build_gap_mask(n, failure_frames, self.skip_frames)
        active_ious = ious[~gap_mask]

        accuracy = float(active_ious.mean()) if len(active_ious) > 0 else 0.0
        reliability = len(failure_frames) / n if n > 0 else 0.0
        eao = float(ious.mean()) if n > 0 else 0.0
        mean_fps = (1000.0 / np.mean(latencies)) if latencies else 0.0

        return ReinitSequenceResult(
            sequence_name=seq.name,
            num_frames=n,
            num_failures=len(failure_frames),
            failure_frames=failure_frames,
            ious=ious,
            accuracy=accuracy,
            reliability=reliability,
            eao=eao,
            mean_fps=mean_fps,
        )

    # ------------------------------------------------------------------
    # Dataset-level evaluation
    # ------------------------------------------------------------------

    def evaluate_dataset(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> ReinitBenchmarkResult:
        """Run the re-initialization protocol on every sequence in a dataset.

        Args:
            tracker:       Tracker to evaluate.
            dataset:       Dataset providing sequences.
            dataset_name:  Human-readable label for reports.
            max_sequences: Limit evaluation to the first N sequences.

        Returns:
            :class:`ReinitBenchmarkResult` aggregating all per-sequence results.
        """
        n = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)
        result = ReinitBenchmarkResult(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
        )

        if self.verbose:
            print(
                f"\nReinit evaluation: {tracker.name} on {dataset_name} "
                f"({n} seqs, threshold={self.failure_threshold}, "
                f"skip={self.skip_frames})"
            )
            print("-" * 65)

        for idx in range(n):
            seq = dataset[idx]
            seq_result = self.evaluate_sequence(tracker, seq)
            result.sequence_results.append(seq_result)

            if self.verbose:
                print(
                    f"  [{idx + 1:>3}/{n}] {seq_result.sequence_name:<28s} "
                    f"EAO={seq_result.eao:.3f}  A={seq_result.accuracy:.3f}  "
                    f"fail={seq_result.num_failures}  "
                    f"FPS={seq_result.mean_fps:.0f}"
                )

        if self.verbose:
            print("-" * 65)
            print(result)

        return result

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_gap_mask(
        n: int,
        failure_frames: List[int],
        skip_frames: int,
    ) -> np.ndarray:
        """Return a boolean mask of length *n* where True marks gap frames."""
        mask = np.zeros(n, dtype=bool)
        for f in failure_frames:
            start = f + 1
            end = min(f + 1 + skip_frames, n)
            mask[start:end] = True
        return mask


# ---------------------------------------------------------------------------
# Multi-tracker comparison utility
# ---------------------------------------------------------------------------

def compare_reinit_results(
    results: Dict[str, ReinitBenchmarkResult],
) -> str:
    """Format a Markdown comparison table from multiple tracker results.

    Args:
        results: Mapping from tracker name to its
            :class:`ReinitBenchmarkResult`.

    Returns:
        A Markdown table string sorted by descending EAO.

    Example::

        table = compare_reinit_results({
            "MOSSE": mosse_result,
            "KCF": kcf_result,
            "KalmanFilter": kalman_result,
        })
        print(table)
    """
    rows = [
        (name, r.mean_eao, r.mean_accuracy, r.mean_reliability, r.total_failures, r.mean_fps)
        for name, r in results.items()
    ]
    rows.sort(key=lambda x: x[1], reverse=True)

    header = "| Tracker | EAO ↑ | Accuracy ↑ | Reliability ↓ | Failures | FPS ↑ |"
    sep    = "|---------|-------|------------|---------------|----------|-------|"
    lines  = [header, sep]
    for name, eao, acc, rel, fail, fps in rows:
        lines.append(
            f"| {name} | {eao:.4f} | {acc:.4f} | {rel:.4f} | {fail} | {fps:.1f} |"
        )
    return "\n".join(lines)
