"""VOT Re-initialization Protocol for EOVOT.

Implements the VOT-challenge evaluation protocol where a tracker is
automatically re-initialized after each tracking failure.  This produces
metrics that are directly comparable to results published in the VOT
challenge and related literature.

Protocol Summary
----------------
1. Initialize tracker on frame 0 with the ground-truth bounding box.
2. Run the tracker frame-by-frame; record per-frame IoU.
3. When IoU drops below ``failure_threshold`` (default 0.1), mark a
   *failure* and begin a grace period of ``grace_frames`` frames (default 5)
   during which no predictions are made and the tracker is held idle.
4. After the grace period, re-initialize the tracker on the next available
   ground-truth box and continue from there.
5. Repeat until the sequence ends.
6. Compute *Expected Average Overlap (EAO)* by averaging the fragmented
   overlap sequence (padded with 0 for grace frames) over the full sequence.

This re-initialization scheme is identical to the one used in VOT2016–2022
and allows fair comparison with published baselines.

Key Classes
-----------
- :class:`VotSequenceResult` — per-sequence result including fragment list
  and re-initialization events.
- :class:`VotBenchmarkResult` — dataset-level aggregate with EAO and
  robustness counts.
- :class:`VotBenchmarkEngine` — runs the VOT protocol on a
  :class:`~eovot.datasets.base.BaseDataset`.

Example::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.vot_protocol import VotBenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    dataset = SyntheticDataset(num_sequences=5, num_frames=200, motion="random")
    engine  = VotBenchmarkEngine(failure_threshold=0.1, grace_frames=5)
    result  = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic-Random")

    print(result)
    print(f"EAO = {result.eao:.4f}")
    print(f"Failures = {result.total_failures}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import MetricsEngine
from ..profiling.profiler import Profiler, ProfilingResult
from ..trackers.base import BaseTracker


@dataclass
class ReinitEvent:
    """Records a single tracker re-initialization event within a sequence."""

    failure_frame: int
    """Frame index at which IoU dropped below the failure threshold."""

    reinit_frame: int
    """Frame index at which the tracker was re-initialized (after grace period)."""

    @property
    def grace_frames_used(self) -> int:
        """Number of frames skipped between failure and re-initialization."""
        return self.reinit_frame - self.failure_frame - 1


@dataclass
class VotSequenceResult:
    """VOT-protocol result for a single tracking sequence.

    Attributes:
        sequence_name: Name of the evaluated sequence.
        overlap_sequence: Per-frame IoU values with zeros during grace periods.
            Shape ``(N,)`` where ``N`` is the total sequence length.
        reinit_events: Ordered list of :class:`ReinitEvent` objects, one per
            tracked failure and re-initialization.
        profiling: Hardware profiling summary (FPS, latency, memory).
        eao: Expected Average Overlap — mean of ``overlap_sequence`` over all
            frames after the first initialization (not counting frame 0).
    """

    sequence_name: str
    overlap_sequence: np.ndarray
    reinit_events: List[ReinitEvent]
    profiling: ProfilingResult

    @property
    def num_failures(self) -> int:
        """Number of tracking failures (re-initializations) in this sequence."""
        return len(self.reinit_events)

    @property
    def eao(self) -> float:
        """Expected Average Overlap for this sequence.

        Mean of the full overlap sequence (frames 1 onward), where grace
        periods contribute 0, matching the VOT convention.
        """
        if len(self.overlap_sequence) <= 1:
            return 0.0
        return float(np.mean(self.overlap_sequence[1:]))

    def __str__(self) -> str:
        return (
            f"VotSequenceResult[{self.sequence_name}] "
            f"EAO={self.eao:.4f}  failures={self.num_failures}  "
            f"FPS={self.profiling.fps:.1f}"
        )


@dataclass
class VotBenchmarkResult:
    """Dataset-level aggregate of a VOT re-initialization benchmark run.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Name of the evaluated dataset.
        sequence_results: Per-sequence VOT results.
    """

    tracker_name: str
    dataset_name: str
    sequence_results: List[VotSequenceResult] = field(default_factory=list)

    @property
    def eao(self) -> float:
        """Dataset-level EAO — mean of per-sequence EAO values."""
        eaos = [r.eao for r in self.sequence_results]
        return float(np.mean(eaos)) if eaos else 0.0

    @property
    def total_failures(self) -> int:
        """Total number of re-initializations across all sequences."""
        return sum(r.num_failures for r in self.sequence_results)

    @property
    def mean_failures_per_sequence(self) -> float:
        """Average failures per sequence."""
        n = len(self.sequence_results)
        return self.total_failures / n if n else 0.0

    @property
    def mean_fps(self) -> float:
        """Mean throughput across all sequences (FPS)."""
        return float(np.mean([r.profiling.fps for r in self.sequence_results]))

    @property
    def mean_latency_ms(self) -> float:
        """Mean per-frame latency across all sequences (ms)."""
        return float(np.mean([r.profiling.latency_mean_ms for r in self.sequence_results]))

    def summary(self) -> dict:
        """Return a flat dict of aggregate metrics suitable for JSON export."""
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "protocol": "VOT-reinit",
            "num_sequences": len(self.sequence_results),
            "eao": round(self.eao, 4),
            "total_failures": self.total_failures,
            "mean_failures_per_sequence": round(self.mean_failures_per_sequence, 2),
            "mean_fps": round(self.mean_fps, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"VotBenchmarkResult[{s['tracker']} on {s['dataset']}] "
            f"EAO={s['eao']}  failures={s['total_failures']}  "
            f"FPS={s['mean_fps']}  ({s['num_sequences']} sequences)"
        )


class VotBenchmarkEngine:
    """Evaluate a tracker under the VOT re-initialization protocol.

    After each tracking failure the tracker is re-initialized on the
    ground-truth box ``grace_frames`` frames later.  The overlap sequence
    is padded with zeros during the grace period so that frequent failures
    are penalised in the EAO score.

    Args:
        failure_threshold: IoU below which a frame is considered a failure.
            Default: ``0.1`` (VOT standard).
        grace_frames: Frames to skip between failure and re-initialization.
            Default: ``5`` (VOT standard).
        verbose: Print per-sequence progress. Default: ``True``.

    Example::

        engine = VotBenchmarkEngine(failure_threshold=0.1, grace_frames=5)
        result = engine.run(tracker, dataset, dataset_name="OTB100")
        print(result)
        print(result.eao)
    """

    def __init__(
        self,
        failure_threshold: float = 0.1,
        grace_frames: int = 5,
        verbose: bool = True,
    ) -> None:
        if not 0.0 <= failure_threshold <= 1.0:
            raise ValueError(
                f"failure_threshold must be in [0, 1], got {failure_threshold}"
            )
        if grace_frames < 0:
            raise ValueError(f"grace_frames must be >= 0, got {grace_frames}")
        self.failure_threshold = failure_threshold
        self.grace_frames = grace_frames
        self.verbose = verbose
        self._metrics = MetricsEngine()
        self._profiler = Profiler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> VotBenchmarkResult:
        """Run the VOT protocol on every sequence in *dataset*.

        Args:
            tracker: Tracker implementing :class:`~eovot.trackers.base.BaseTracker`.
            dataset: Any :class:`~eovot.datasets.base.BaseDataset` instance.
            dataset_name: Label used in reports. Default: ``"unknown"``.
            max_sequences: Evaluate at most this many sequences. Default: all.

        Returns:
            :class:`VotBenchmarkResult` with per-sequence and aggregate metrics.
        """
        result = VotBenchmarkResult(
            tracker_name=tracker.name, dataset_name=dataset_name
        )
        n = (
            min(len(dataset), max_sequences)
            if max_sequences is not None
            else len(dataset)
        )

        if self.verbose:
            print(
                f"\nVOT Evaluation: {tracker.name} on {dataset_name} "
                f"({n} sequences, threshold={self.failure_threshold}, "
                f"grace={self.grace_frames})"
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
                    f"failures={seq_result.num_failures}  "
                    f"FPS={seq_result.profiling.fps:.1f}"
                )

        if self.verbose:
            print("-" * 60)
            print(result)

        return result

    # ------------------------------------------------------------------
    # Sequence-level protocol
    # ------------------------------------------------------------------

    def _run_sequence(
        self, tracker: BaseTracker, seq: Sequence
    ) -> VotSequenceResult:
        """Apply the VOT re-initialization protocol to a single sequence.

        Algorithm
        ---------
        Frames are processed one at a time.  Three tracker states drive the
        logic:

        * **INIT** — waiting to initialize (or re-initialize after grace).
        * **TRACKING** — tracker is running; measure IoU each frame.
        * **GRACE** — failure detected; skip this many frames, then go to INIT.

        Args:
            tracker: Any :class:`~eovot.trackers.base.BaseTracker`.
            seq: Sequence to evaluate.

        Returns:
            :class:`VotSequenceResult` with overlap sequence and event log.
        """
        self._profiler.reset()

        frames = list(seq)
        gt = seq.ground_truth
        n = len(frames)

        overlap_sequence = np.zeros(n, dtype=np.float64)
        reinit_events: List[ReinitEvent] = []

        state = "INIT"
        grace_countdown = 0
        failure_frame_idx = -1

        for i, frame in enumerate(frames):
            if state == "INIT":
                # Re-initialize tracker on current GT box.
                tracker.initialize(frame, tuple(gt[i]))  # type: ignore[arg-type]
                overlap_sequence[i] = 1.0  # initialization frame always gets IoU=1
                state = "TRACKING"
                if i > 0:
                    # Record the reinit (skip i==0 which is the mandatory first init)
                    reinit_events.append(
                        ReinitEvent(
                            failure_frame=failure_frame_idx,
                            reinit_frame=i,
                        )
                    )

            elif state == "GRACE":
                # Skip this frame; overlap stays 0.
                grace_countdown -= 1
                if grace_countdown <= 0:
                    state = "INIT"

            else:  # state == "TRACKING"
                self._profiler.start_frame()
                pred_bbox = tracker.update(frame)
                self._profiler.end_frame()

                pred = np.array(pred_bbox, dtype=np.float64).reshape(1, 4)
                gt_box = gt[i : i + 1]
                iou = float(self._metrics.batch_iou(pred, gt_box)[0])
                overlap_sequence[i] = iou

                if iou < self.failure_threshold:
                    # Failure detected — enter grace period.
                    failure_frame_idx = i
                    overlap_sequence[i] = 0.0
                    state = "GRACE"
                    grace_countdown = self.grace_frames

        # If no update frames were profiled (very short sequence), dummy profiling.
        try:
            profiling = self._profiler.summary(tracker.name)
        except ValueError:
            from ..profiling.profiler import ProfilingResult
            profiling = ProfilingResult(
                tracker_name=tracker.name,
                frame_count=0,
                fps=0.0,
                latency_mean_ms=0.0,
                latency_std_ms=0.0,
                latency_p95_ms=0.0,
                latency_p99_ms=0.0,
                peak_memory_mb=0.0,
            )

        return VotSequenceResult(
            sequence_name=seq.name,
            overlap_sequence=overlap_sequence,
            reinit_events=reinit_events,
            profiling=profiling,
        )
