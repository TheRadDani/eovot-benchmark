"""VOT Reset-Based Benchmark Engine for EOVOT.

Implements the Visual Object Tracking (VOT) challenge reset protocol, which
differs from the standard one-pass evaluation used by :class:`BenchmarkEngine`:

**Standard one-pass evaluation (OPE)**:
  Track from frame 0 to end; count all failures but *never* re-initialise.

**VOT reset protocol**:
  1. Initialise the tracker from the ground-truth at frame 0.
  2. Track frame by frame.
  3. On failure (IoU < ``failure_threshold``):
     - Record the failure frame.
     - Assign IoU = 0 for the next ``gap_frames`` frames (penalty period).
     - Re-initialise the tracker from the GT at frame ``failure + gap_frames + 1``.
  4. Repeat until the sequence ends.

**Expected Average Overlap (EAO)**:
  The full per-frame IoU array (with 0s during gap periods) is averaged to
  produce a single scalar that jointly captures accuracy *and* robustness.
  This is the canonical VOT benchmark scalar reported in challenge papers.

Why does this matter?  A tracker that fails gracefully and recovers quickly
has a much higher EAO than one that drifts silently — yet both may look
identical under plain mean-IoU.  The reset protocol makes this distinction
explicit and is required for fair comparison against published VOT results.

Typical usage::

    from eovot.benchmark.vot_engine import VOTResetEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset

    engine = VOTResetEngine(failure_threshold=0.1, gap_frames=5)
    tracker = MOSSETracker()
    dataset = SyntheticDataset(num_sequences=5)

    result = engine.run(tracker, dataset, dataset_name="synthetic")
    print(result)
    print(f"EAO: {result.eao:.4f}  failures/seq: {result.mean_failures_per_sequence:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import MetricsEngine, iou as _iou_pair
from ..profiling.profiler import Profiler, ProfilingResult
from ..trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VOTSegment:
    """One continuous tracking segment between two resets (or start/end).

    A segment begins at :attr:`start_frame` (where the tracker was most
    recently initialised) and ends at :attr:`end_frame` (exclusive: the
    frame at which a failure was detected, or ``len(sequence)`` for the
    final segment).

    Args:
        start_frame: Index of the first tracked frame in this segment.
        end_frame:   Index of the first frame *after* the segment ends
                     (failure frame or sequence length).
        ious:        Per-frame IoU array for frames ``[start_frame, end_frame)``.
        predictions: Predicted bounding boxes, shape ``(L, 4)`` where
                     ``L = end_frame - start_frame``.
        reinit_gt:   Ground-truth box used to re-initialise the tracker at
                     ``start_frame``; ``None`` for the very first segment.
    """

    start_frame: int
    end_frame: int
    ious: np.ndarray
    predictions: np.ndarray
    reinit_gt: Optional[BBox] = None

    @property
    def length(self) -> int:
        """Number of frames in this segment."""
        return self.end_frame - self.start_frame

    @property
    def mean_iou(self) -> float:
        """Mean IoU over the segment frames."""
        return float(self.ious.mean()) if len(self.ious) else 0.0

    @property
    def failed(self) -> bool:
        """``True`` if this segment ended due to a tracking failure."""
        return bool(len(self.ious) > 0 and float(self.ious[-1]) == 0.0)


@dataclass
class VOTSequenceResult:
    """Full evaluation result for one sequence under the reset protocol.

    Attributes:
        sequence_name:   Name of the evaluated sequence.
        segments:        List of :class:`VOTSegment` objects (one per
                         continuous tracking phase between resets).
        failure_frames:  Frame indices at which tracking failures occurred.
        gap_frames:      Number of zero-IoU penalty frames per failure.
        frame_ious:      Complete per-frame IoU array for the *entire*
                         sequence length, with 0.0 inserted during gap
                         periods.  Shape ``(N,)``.
        profiling:       Timing and memory summary from :class:`Profiler`.
    """

    sequence_name: str
    segments: List[VOTSegment]
    failure_frames: List[int]
    gap_frames: int
    frame_ious: np.ndarray
    profiling: ProfilingResult

    @property
    def num_failures(self) -> int:
        return len(self.failure_frames)

    @property
    def eao(self) -> float:
        """Expected Average Overlap — mean IoU over the full sequence (incl. gap zeros)."""
        return float(self.frame_ious.mean()) if len(self.frame_ious) else 0.0

    @property
    def robustness(self) -> float:
        """Fraction of *non-gap* frames with IoU > 0 (i.e. tracker was alive)."""
        n_total = len(self.frame_ious)
        if n_total == 0:
            return 0.0
        n_gap = self.num_failures * self.gap_frames
        n_non_gap = max(n_total - n_gap, 0)
        if n_non_gap == 0:
            return 0.0
        non_gap_ious = [v for i, v in enumerate(self.frame_ious) if v > 0 or (
            not any(
                f < i <= f + self.gap_frames
                for f in self.failure_frames
            )
        )]
        alive = sum(1 for v in non_gap_ious if v > 0)
        return alive / len(non_gap_ious) if non_gap_ious else 0.0

    def summary(self) -> Dict:
        return {
            "sequence_name": self.sequence_name,
            "eao": round(self.eao, 4),
            "num_failures": self.num_failures,
            "num_segments": len(self.segments),
            "fps": round(self.profiling.fps, 2),
            "latency_mean_ms": round(self.profiling.latency_mean_ms, 3),
            "peak_memory_mb": round(self.profiling.peak_memory_mb, 2),
        }

    def __str__(self) -> str:
        return (
            f"VOTSequenceResult[{self.sequence_name}] "
            f"EAO={self.eao:.4f}  "
            f"failures={self.num_failures}  "
            f"segments={len(self.segments)}  "
            f"FPS={self.profiling.fps:.1f}"
        )


@dataclass
class VOTBenchmarkResult:
    """Aggregate result across all sequences under the VOT reset protocol.

    Attributes:
        tracker_name:     Name of the evaluated tracker.
        dataset_name:     Name of the dataset.
        failure_threshold: IoU threshold used to declare a failure.
        gap_frames:       Penalty gap between failure and re-initialization.
        sequence_results: Per-sequence :class:`VOTSequenceResult` objects.
    """

    tracker_name: str
    dataset_name: str
    failure_threshold: float
    gap_frames: int
    sequence_results: List[VOTSequenceResult] = field(default_factory=list)

    @property
    def eao(self) -> float:
        """Sequence-length-weighted Expected Average Overlap across all sequences."""
        total_frames = sum(len(r.frame_ious) for r in self.sequence_results)
        if total_frames == 0:
            return 0.0
        weighted_sum = sum(
            r.eao * len(r.frame_ious) for r in self.sequence_results
        )
        return float(weighted_sum / total_frames)

    @property
    def mean_failures_per_sequence(self) -> float:
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.num_failures for r in self.sequence_results]))

    @property
    def total_failures(self) -> int:
        return sum(r.num_failures for r in self.sequence_results)

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

    def eao_curve(self, seq_len_range: Tuple[int, int] = (100, 356)) -> float:
        """VOT-style EAO integrated over a typical sequence-length range.

        Computes the mean EAO restricted to sub-sequences of lengths within
        ``seq_len_range``.  Sequences shorter than ``seq_len_range[0]`` still
        contribute their full IoU array (they are not excluded).

        Args:
            seq_len_range: ``(min_len, max_len)`` frame range used in the
                official VOT challenge (default: ``(100, 356)`` per VOT2016).

        Returns:
            Scalar EAO in ``[0, 1]``.
        """
        min_len, max_len = seq_len_range
        all_frame_ious: List[np.ndarray] = []
        for r in self.sequence_results:
            arr = r.frame_ious
            if len(arr) == 0:
                continue
            # Restrict to the first max_len frames.
            truncated = arr[:max_len]
            all_frame_ious.append(truncated)

        if not all_frame_ious:
            return 0.0

        # For each position t (up to max_len), compute mean IoU across sequences
        # that have at least t frames (VOT expected overlap graph).
        max_t = max(len(a) for a in all_frame_ious)
        overlap_at_t = []
        for t in range(max_t):
            values = [a[t] for a in all_frame_ious if len(a) > t]
            if values:
                overlap_at_t.append(float(np.mean(values)))

        # Integrate over the typical length range [min_len, max_len].
        valid = overlap_at_t[min_len - 1 : max_len] if len(overlap_at_t) >= min_len else overlap_at_t
        return float(np.mean(valid)) if valid else float(np.mean(overlap_at_t))

    def summary(self) -> Dict:
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "failure_threshold": self.failure_threshold,
            "gap_frames": self.gap_frames,
            "num_sequences": len(self.sequence_results),
            "eao": round(self.eao, 4),
            "eao_vot_range": round(self.eao_curve(), 4),
            "total_failures": self.total_failures,
            "mean_failures_per_sequence": round(self.mean_failures_per_sequence, 2),
            "mean_fps": round(self.mean_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }

    def to_dict(self) -> Dict:
        """Return a full nested dict suitable for JSON export."""
        return {
            "summary": self.summary(),
            "sequences": [r.summary() for r in self.sequence_results],
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"VOTBenchmarkResult[{s['tracker']} on {s['dataset']}] "
            f"EAO={s['eao']}  "
            f"EAO(VOT-range)={s['eao_vot_range']}  "
            f"failures/seq={s['mean_failures_per_sequence']}  "
            f"FPS={s['mean_fps']}  "
            f"({s['num_sequences']} sequences)"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VOTResetEngine:
    """Evaluate a tracker under the VOT challenge reset protocol.

    Unlike the standard one-pass :class:`~eovot.benchmark.engine.BenchmarkEngine`,
    this engine re-initialises the tracker from the ground-truth whenever a
    failure is detected, inserting a ``gap_frames``-long penalty window of
    zero IoU.  This makes it possible to compute the **Expected Average
    Overlap (EAO)** — the primary metric of the VOT challenge — which jointly
    captures accuracy (IoU when tracking) and robustness (failure rate).

    Args:
        failure_threshold: IoU below which a frame counts as a failure.
            Standard VOT value: ``0.1``.
        gap_frames:         Number of frames to skip (zero-IoU penalty) after
            each failure before re-initialising.  Standard VOT value: ``5``.
        verbose:            Print per-sequence progress.  Default: ``True``.

    Example::

        engine = VOTResetEngine(failure_threshold=0.1, gap_frames=5)
        result = engine.run(tracker, dataset, "OTB100")
        print(result)
        # VOTBenchmarkResult[MOSSE on OTB100] EAO=0.2341 ...

        # Per-sequence breakdown
        for seq_result in result.sequence_results:
            print(seq_result)
    """

    def __init__(
        self,
        failure_threshold: float = 0.1,
        gap_frames: int = 5,
        verbose: bool = True,
    ) -> None:
        if not 0.0 <= failure_threshold <= 1.0:
            raise ValueError(f"failure_threshold must be in [0, 1], got {failure_threshold}")
        if gap_frames < 0:
            raise ValueError(f"gap_frames must be >= 0, got {gap_frames}")
        self.failure_threshold = failure_threshold
        self.gap_frames = gap_frames
        self.verbose = verbose
        self._metrics = MetricsEngine()

    def run(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> VOTBenchmarkResult:
        """Run the VOT reset protocol on every sequence in *dataset*.

        Args:
            tracker:       The tracker to evaluate.
            dataset:       Dataset providing :class:`~eovot.datasets.base.Sequence` objects.
            dataset_name:  Human-readable dataset identifier for the result.
            max_sequences: Cap the number of sequences evaluated.  Useful for
                           quick sanity-checks.  Default: ``None`` (all).

        Returns:
            :class:`VOTBenchmarkResult` with per-sequence and aggregate metrics.
        """
        result = VOTBenchmarkResult(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
            failure_threshold=self.failure_threshold,
            gap_frames=self.gap_frames,
        )
        n = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)

        if self.verbose:
            print(
                f"\n[VOT Reset] Evaluating {tracker.name} on {dataset_name} "
                f"({n} sequences, threshold={self.failure_threshold}, gap={self.gap_frames})"
            )
            print("-" * 70)

        for idx in range(n):
            seq = dataset[idx]
            seq_result = self._run_sequence(tracker, seq)
            result.sequence_results.append(seq_result)
            if self.verbose:
                print(
                    f"  [{idx + 1:>3}/{n}] {seq_result.sequence_name:<30s} "
                    f"EAO={seq_result.eao:.4f}  "
                    f"failures={seq_result.num_failures}  "
                    f"FPS={seq_result.profiling.fps:.1f}"
                )

        if self.verbose:
            print("-" * 70)
            print(result)

        return result

    # ------------------------------------------------------------------
    # Sequence-level evaluation
    # ------------------------------------------------------------------

    def _run_sequence(self, tracker: BaseTracker, seq: Sequence) -> VOTSequenceResult:
        """Evaluate *tracker* on one sequence under the reset protocol."""
        frames = list(seq)
        gt = seq.ground_truth
        n_frames = len(frames)

        profiler = Profiler()

        # Full per-frame IoU array (0.0 during gap periods, first frame = 1.0).
        frame_ious = np.zeros(n_frames, dtype=np.float64)
        frame_ious[0] = 1.0  # Initialisation frame: perfect overlap by convention.

        segments: List[VOTSegment] = []
        failure_frames: List[int] = []

        # State machine
        segment_start = 0
        seg_preds: List[BBox] = [tuple(gt[0])]  # type: ignore[misc]
        seg_ious: List[float] = [1.0]
        in_gap = False
        # gap_end stores the index of the *last* zero-IoU (penalty) frame.
        # Re-initialisation happens at gap_end + 1 (i > gap_end).
        gap_end = -1

        # Initialise tracker at frame 0.
        tracker.initialize(frames[0], tuple(gt[0]))  # type: ignore[arg-type]

        for i in range(1, n_frames):
            if in_gap:
                # Penalty period: zero IoU, no tracker update.
                frame_ious[i] = 0.0
                if i > gap_end:
                    # All gap frames consumed — re-initialise from GT.
                    reinit_gt = tuple(gt[i])  # type: ignore[misc]
                    tracker.initialize(frames[i], reinit_gt)
                    # Save the just-completed segment.
                    segments.append(
                        VOTSegment(
                            start_frame=segment_start,
                            end_frame=i,
                            ious=np.array(seg_ious, dtype=np.float64),
                            predictions=np.array(seg_preds, dtype=np.float64),
                        )
                    )
                    # Begin a new segment at the re-init frame.
                    segment_start = i
                    seg_preds = [reinit_gt]
                    seg_ious = [1.0]
                    frame_ious[i] = 1.0  # Re-init frame: perfect overlap by convention.
                    in_gap = False
                continue

            # Normal tracking.
            profiler.start_frame()
            pred = tracker.update(frames[i])
            profiler.end_frame()

            frame_iou = _iou_pair(pred, tuple(gt[i]))  # type: ignore[arg-type]
            frame_ious[i] = frame_iou
            seg_preds.append(pred)
            seg_ious.append(frame_iou)

            if frame_iou < self.failure_threshold:
                # Failure detected at frame i.
                failure_frames.append(i)
                # Zero the next gap_frames penalty frames in the pre-built array.
                # gap_end is the INDEX of the last zero-IoU frame; re-init is at gap_end+1.
                gap_end = i + self.gap_frames
                for g in range(i + 1, min(gap_end + 1, n_frames)):
                    frame_ious[g] = 0.0
                in_gap = True

        # Flush the final segment.
        if not in_gap or (in_gap and segment_start < n_frames):
            if seg_ious:
                segments.append(
                    VOTSegment(
                        start_frame=segment_start,
                        end_frame=n_frames,
                        ious=np.array(seg_ious, dtype=np.float64),
                        predictions=np.array(seg_preds, dtype=np.float64),
                    )
                )

        # Build profiling result (may have 0 profiled frames on very short seqs).
        try:
            prof_result = profiler.summary(tracker.name)
        except ValueError:
            # Fewer than 1 profiled frame (sequence too short or all gap frames).
            from ..profiling.profiler import ProfilingResult
            prof_result = ProfilingResult(
                tracker_name=tracker.name,
                frame_count=0,
                fps=0.0,
                latency_mean_ms=0.0,
                latency_std_ms=0.0,
                latency_p95_ms=0.0,
                peak_memory_mb=0.0,
            )

        return VOTSequenceResult(
            sequence_name=seq.name,
            segments=segments,
            failure_frames=failure_frames,
            gap_frames=self.gap_frames,
            frame_ious=frame_ious,
            profiling=prof_result,
        )
