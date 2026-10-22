"""VOT-protocol benchmark engine with re-initialization after tracking failures.

The standard benchmark engine (:class:`~eovot.benchmark.engine.BenchmarkEngine`)
runs a tracker straight through each sequence without intervening on failure.
This is fine for measuring raw accuracy but does not match the official VOT
challenge protocol, which requires:

1. Initialize the tracker on the first frame.
2. On each frame after burn-in, compare IoU to the ground-truth box.
3. When IoU drops below ``failure_threshold``, record a failure and begin
   a ``reinit_delay``-frame penalty window during which IoU is set to 0.
4. After the penalty window, re-initialize the tracker from the ground-truth
   box of the current frame.
5. Expected Average Overlap (EAO) is the mean IoU over all post-init frames,
   including penalty frames.

This makes EAO sensitive to both per-frame accuracy *and* the rate of
catastrophic failures, which is the correct measure for comparing trackers
in real deployment scenarios.

Typical usage::

    from eovot.benchmark.vot_engine import VOTBenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.kcf import KCFTracker

    dataset = SyntheticDataset(num_sequences=10, num_frames=150)
    tracker = KCFTracker()
    engine  = VOTBenchmarkEngine(verbose=True)
    result  = engine.run(tracker, dataset, dataset_name="Synthetic")

    print(result)
    # VOTBenchmarkResult[KCF on Synthetic] EAO=0.6124  ...
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import MetricsEngine
from ..profiling.profiler import Profiler, ProfilingResult


@dataclass
class ReinitEvent:
    """Records one tracker re-initialization event during VOT evaluation.

    Attributes:
        failure_frame: Index of the first frame with IoU < ``failure_threshold``.
        reinit_frame:  Frame index at which the tracker is re-initialized
            (``failure_frame + reinit_delay``, clamped to sequence length).
        penalty_frames: Number of frames assigned IoU 0 due to the penalty window.
    """

    failure_frame: int
    reinit_frame: int
    penalty_frames: int


@dataclass
class VOTSequenceResult:
    """Per-sequence result from a :class:`VOTBenchmarkEngine` run.

    Attributes:
        sequence_name:      Name of the evaluated sequence.
        frame_ious:         Per-frame IoU array ``(N,)``; penalty frames are 0.
        reinit_events:      List of re-initialization events detected.
        profiling:          Timing and memory summary for frames outside the
            penalty window.
        eao:                Expected Average Overlap — mean IoU over all
            post-initialization frames (including penalty frames).
        num_reinitializations: ``len(reinit_events)``.
    """

    sequence_name: str
    frame_ious: np.ndarray
    reinit_events: List[ReinitEvent]
    profiling: ProfilingResult
    eao: float
    num_reinitializations: int

    @property
    def mean_iou(self) -> float:
        """Mean IoU including penalty frames (equals ``eao``)."""
        return float(self.frame_ious.mean()) if len(self.frame_ious) else 0.0

    def __str__(self) -> str:
        return (
            f"VOTSequenceResult[{self.sequence_name}] "
            f"EAO={self.eao:.4f}  reinits={self.num_reinitializations}  "
            f"FPS={self.profiling.fps:.1f}"
        )


@dataclass
class VOTBenchmarkResult:
    """Aggregate result from a :class:`VOTBenchmarkEngine` run.

    Contains per-sequence :class:`VOTSequenceResult` objects and exposes
    aggregate scalars as properties.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Human-readable dataset label.
        sequence_results: Per-sequence results appended by :meth:`VOTBenchmarkEngine.run`.
    """

    tracker_name: str
    dataset_name: str
    sequence_results: List[VOTSequenceResult] = field(default_factory=list)

    @property
    def mean_eao(self) -> float:
        """Mean EAO across all sequences (the primary VOT scalar)."""
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.eao for r in self.sequence_results]))

    @property
    def mean_iou(self) -> float:
        """Alias for ``mean_eao`` — mean IoU including penalty frames."""
        return self.mean_eao

    @property
    def total_reinitializations(self) -> int:
        """Total number of re-initialization events across all sequences."""
        return sum(r.num_reinitializations for r in self.sequence_results)

    @property
    def mean_fps(self) -> float:
        """Mean FPS across sequences (penalty frames excluded from timing)."""
        fpss = [r.profiling.fps for r in self.sequence_results if r.profiling.fps > 0]
        return float(np.mean(fpss)) if fpss else 0.0

    @property
    def peak_memory_mb(self) -> float:
        """Peak memory footprint across all sequences (MB)."""
        if not self.sequence_results:
            return 0.0
        return float(np.max([r.profiling.peak_memory_mb for r in self.sequence_results]))

    def summary(self) -> dict:
        """Return a flat dict of aggregate scalars suitable for JSON export."""
        n = len(self.sequence_results)
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "num_sequences": n,
            "mean_eao": round(self.mean_eao, 4),
            "mean_iou": round(self.mean_iou, 4),
            "mean_fps": round(self.mean_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "total_reinitializations": self.total_reinitializations,
            "mean_reinits_per_sequence": round(
                self.total_reinitializations / max(n, 1), 2
            ),
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"VOTBenchmarkResult[{s['tracker']} on {s['dataset']}] "
            f"EAO={s['mean_eao']}  mIoU={s['mean_iou']}  "
            f"FPS={s['mean_fps']}  reinits={s['total_reinitializations']} "
            f"({s['num_sequences']} sequences)"
        )


class VOTBenchmarkEngine:
    """Benchmark engine implementing the VOT re-initialization evaluation protocol.

    Unlike :class:`~eovot.benchmark.engine.BenchmarkEngine`, which runs the
    tracker straight through each sequence, this engine detects failures and
    re-initializes the tracker from the ground-truth box after a penalty window.

    This produces EAO values that penalise catastrophic drift and are
    directly comparable to results reported in the VOT challenge papers.

    Args:
        failure_threshold: IoU below which a frame is considered a failure.
            Default: ``0.1`` (the standard VOT threshold).
        reinit_delay:      Number of penalty frames after a failure before the
            tracker is re-initialized.  Default: ``5`` (standard VOT).
        verbose:           Print per-sequence progress to stdout.
            Default: ``True``.

    Example::

        engine = VOTBenchmarkEngine(failure_threshold=0.1, reinit_delay=5)
        result = engine.run(KCFTracker(), dataset, dataset_name="GOT-10k-val")
        print(result.mean_eao)
    """

    def __init__(
        self,
        failure_threshold: float = 0.1,
        reinit_delay: int = 5,
        verbose: bool = True,
    ) -> None:
        if not (0.0 <= failure_threshold <= 1.0):
            raise ValueError(
                f"failure_threshold must be in [0, 1], got {failure_threshold}"
            )
        if reinit_delay < 0:
            raise ValueError(f"reinit_delay must be >= 0, got {reinit_delay}")
        self.failure_threshold = failure_threshold
        self.reinit_delay = reinit_delay
        self.verbose = verbose
        self._metrics = MetricsEngine()
        self._profiler = Profiler()

    def run(
        self,
        tracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> VOTBenchmarkResult:
        """Evaluate *tracker* using the VOT re-initialization protocol.

        Args:
            tracker:        Any :class:`~eovot.trackers.base.BaseTracker`.
            dataset:        Dataset to evaluate on.
            dataset_name:   Human-readable label used in the result summary.
            max_sequences:  Cap on the number of sequences to evaluate.
                ``None`` evaluates all sequences in *dataset*.

        Returns:
            :class:`VOTBenchmarkResult` with per-sequence and aggregate metrics.
        """
        result = VOTBenchmarkResult(
            tracker_name=tracker.name, dataset_name=dataset_name
        )
        n = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)

        if self.verbose:
            print(
                f"\nVOT Evaluation: {tracker.name} on {dataset_name} ({n} sequences)"
            )
            print(
                f"  failure_threshold={self.failure_threshold}  "
                f"reinit_delay={self.reinit_delay}"
            )
            print("-" * 60)

        for idx in range(n):
            seq = dataset[idx]
            seq_result = self._run_sequence(tracker, seq)
            result.sequence_results.append(seq_result)

            if self.verbose:
                print(
                    f"  [{idx + 1:>3}/{n}] {seq_result.sequence_name:<30s} "
                    f"EAO={seq_result.eao:.3f}  "
                    f"mIoU={seq_result.mean_iou:.3f}  "
                    f"reinits={seq_result.num_reinitializations}  "
                    f"FPS={seq_result.profiling.fps:.1f}"
                )

        if self.verbose:
            print("-" * 60)
            print(result)

        return result

    def _run_sequence(self, tracker, seq: Sequence) -> VOTSequenceResult:
        """Run a single sequence with VOT re-initialization."""
        self._profiler.reset()

        frames = list(seq)
        gt = seq.ground_truth
        n = len(frames)

        frame_ious = np.zeros(n, dtype=np.float64)
        reinit_events: List[ReinitEvent] = []

        # Penalty state
        penalty_countdown = 0
        in_penalty = False

        # Initialize on frame 0 — IoU is 1.0 by construction.
        tracker.initialize(frames[0], seq.init_bbox)
        frame_ious[0] = 1.0

        i = 1
        while i < n:
            if in_penalty:
                frame_ious[i] = 0.0
                penalty_countdown -= 1
                if penalty_countdown == 0:
                    in_penalty = False
                    reinit_gt = tuple(gt[i]) if i < len(gt) else seq.init_bbox  # type: ignore[arg-type]
                    tracker.initialize(frames[i], reinit_gt)
                    reinit_events[-1].reinit_frame = i
                i += 1
                continue

            # Normal tracking frame: time the update call.
            self._profiler.start_frame()
            pred = tracker.update(frames[i])
            self._profiler.end_frame()

            if i < len(gt):
                iou = self._metrics.batch_iou(
                    np.array([pred], dtype=np.float64),
                    np.array([gt[i]], dtype=np.float64),
                )[0]
                frame_ious[i] = float(iou)

                if float(iou) < self.failure_threshold:
                    in_penalty = True
                    penalty_countdown = self.reinit_delay
                    actual_penalty = min(self.reinit_delay, n - 1 - i)
                    reinit_events.append(
                        ReinitEvent(
                            failure_frame=i,
                            reinit_frame=min(i + self.reinit_delay, n - 1),
                            penalty_frames=actual_penalty,
                        )
                    )
            i += 1

        # EAO = mean IoU over all post-initialization frames (frames 1..n-1).
        eao = float(np.mean(frame_ious[1:])) if n > 1 else 0.0

        try:
            profiling = self._profiler.summary(tracker.name)
        except ValueError:
            profiling = _zero_profiling(tracker.name)

        return VOTSequenceResult(
            sequence_name=seq.name,
            frame_ious=frame_ious,
            reinit_events=reinit_events,
            profiling=profiling,
            eao=eao,
            num_reinitializations=len(reinit_events),
        )


def _zero_profiling(tracker_name: str) -> ProfilingResult:
    """Return a zeroed ProfilingResult for sequences too short to profile."""
    return ProfilingResult(
        tracker_name=tracker_name,
        frame_count=0,
        fps=0.0,
        latency_mean_ms=0.0,
        latency_std_ms=0.0,
        latency_p95_ms=0.0,
        latency_p99_ms=0.0,
        latency_cv=0.0,
        peak_memory_mb=0.0,
    )
