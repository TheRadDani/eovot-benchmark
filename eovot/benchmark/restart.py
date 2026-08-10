"""VOT Restart evaluation protocol for robust tracker benchmarking.

Implements the restart-based robustness evaluation used in VOT challenges.
When a tracker completely loses the target (IoU == 0), it is penalized for
``grace_period`` frames then reinitialized from ground truth, allowing
uninterrupted evaluation on long sequences.

This protocol is essential for fair comparison of trackers on challenging
sequences: a tracker that "recovers" by luck after losing the target for
100 frames should not receive the same EAO as one that never loses it.

Typical usage::

    from eovot.benchmark.restart import RestartProtocolEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker

    engine = RestartProtocolEngine(failure_threshold=0.0, grace_period=5)
    dataset = SyntheticDataset(num_sequences=5, num_frames=200)
    tracker = MOSSETracker()
    result = engine.run(tracker, dataset, dataset_name="SyntheticDataset")
    print(result.to_markdown())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import MetricsEngine, iou as compute_iou
from ..trackers.base import BaseTracker


@dataclass
class RestartEvent:
    """A single tracking failure and its scheduled reinit frame.

    Attributes:
        failure_frame: Frame index at which IoU dropped to the failure threshold.
        restart_frame: Frame index at which the tracker was reinitialized.
        iou_at_failure: IoU value observed on the failure frame.
    """

    failure_frame: int
    restart_frame: int
    iou_at_failure: float


@dataclass
class RestartSequenceResult:
    """Per-sequence evaluation result from the restart protocol.

    ``ious`` covers all frames including grace-period frames (which are 0.0)
    and reinit frames (which are 1.0), giving a complete per-frame picture
    for EAO computation and visual inspection.

    Attributes:
        sequence_name: Name of the evaluated sequence.
        tracker_name: Name of the tracker.
        ious: Per-frame IoU values; shape ``(N,)``.
        predictions: Per-frame bounding boxes; shape ``(N, 4)``.
        ground_truths: Per-frame ground-truth boxes; shape ``(N, 4)``.
        restarts: List of restart events that occurred.
        fps: Mean frames per second (excluding initialization).
        mean_latency_ms: Mean per-frame latency in milliseconds.
    """

    sequence_name: str
    tracker_name: str
    ious: np.ndarray
    predictions: np.ndarray
    ground_truths: np.ndarray
    restarts: List[RestartEvent]
    fps: float
    mean_latency_ms: float

    @property
    def eao(self) -> float:
        """Expected Average Overlap: mean IoU across all frames including penalties."""
        if len(self.ious) == 0:
            return 0.0
        return float(np.mean(self.ious))

    @property
    def num_restarts(self) -> int:
        """Number of tracking failures on this sequence."""
        return len(self.restarts)

    @property
    def restart_frames(self) -> List[int]:
        """List of frame indices where failures were detected."""
        return [e.failure_frame for e in self.restarts]

    def to_dict(self) -> dict:
        return {
            "sequence_name": self.sequence_name,
            "tracker_name": self.tracker_name,
            "eao": round(self.eao, 4),
            "num_restarts": self.num_restarts,
            "fps": round(self.fps, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "restart_frames": self.restart_frames,
        }


@dataclass
class RestartBenchmarkResult:
    """Aggregated restart-protocol evaluation across an entire dataset.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Name of the dataset.
        sequence_results: Per-sequence results.
    """

    tracker_name: str
    dataset_name: str
    sequence_results: List[RestartSequenceResult] = field(default_factory=list)

    @property
    def mean_eao(self) -> float:
        """Mean Expected Average Overlap across all sequences."""
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.eao for r in self.sequence_results]))

    @property
    def mean_fps(self) -> float:
        """Mean FPS across all sequences."""
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.fps for r in self.sequence_results]))

    @property
    def total_restarts(self) -> int:
        """Total number of tracking failures across all sequences."""
        return sum(r.num_restarts for r in self.sequence_results)

    @property
    def mean_restarts_per_sequence(self) -> float:
        """Mean number of restarts per sequence."""
        if not self.sequence_results:
            return 0.0
        return self.total_restarts / len(self.sequence_results)

    def summary(self) -> dict:
        """Return a JSON-serializable summary dict."""
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "num_sequences": len(self.sequence_results),
            "mean_eao": round(self.mean_eao, 4),
            "mean_fps": round(self.mean_fps, 1),
            "total_restarts": self.total_restarts,
            "mean_restarts_per_sequence": round(self.mean_restarts_per_sequence, 2),
        }

    def to_markdown(self) -> str:
        """Format the summary as a Markdown table."""
        s = self.summary()
        lines = [
            f"## Restart Protocol: {self.tracker_name} on {self.dataset_name}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Sequences | {s['num_sequences']} |",
            f"| Mean EAO | {s['mean_eao']:.4f} |",
            f"| Mean FPS | {s['mean_fps']:.1f} |",
            f"| Total Restarts | {s['total_restarts']} |",
            f"| Restarts / Seq | {s['mean_restarts_per_sequence']:.2f} |",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"RestartBenchmarkResult[{s['tracker']} on {s['dataset']}] "
            f"EAO={s['mean_eao']}  FPS={s['mean_fps']}  "
            f"restarts={s['total_restarts']}  ({s['num_sequences']} sequences)"
        )


class RestartProtocolEngine:
    """Evaluate trackers using the VOT restart protocol.

    After a tracking failure (IoU ≤ ``failure_threshold``), the tracker is
    not run for the next ``grace_period`` frames (those frames receive IoU=0
    as a penalty).  The tracker is then reinitialized at frame
    ``failure + grace_period + 1`` using the ground-truth annotation, and
    evaluation continues normally.

    This mirrors the standard VOT challenge evaluation methodology, enabling
    direct comparison with published EAO results.

    Args:
        failure_threshold: IoU value at or below which a failure is declared.
            Default: ``0.0`` (strict failure only when boxes do not overlap).
        grace_period: Number of penalty frames inserted after each failure
            before reinitialization.  Default: ``5`` (VOT standard).
        verbose: Print per-sequence progress. Default: ``True``.

    Example::

        engine = RestartProtocolEngine(failure_threshold=0.0, grace_period=5)
        result  = engine.run(tracker, dataset, dataset_name="GOT-10k")
        print(result.to_markdown())
    """

    def __init__(
        self,
        failure_threshold: float = 0.0,
        grace_period: int = 5,
        verbose: bool = True,
    ) -> None:
        if not 0.0 <= failure_threshold < 1.0:
            raise ValueError(
                f"failure_threshold must be in [0, 1), got {failure_threshold}"
            )
        if grace_period < 0:
            raise ValueError(f"grace_period must be >= 0, got {grace_period}")
        self.failure_threshold = failure_threshold
        self.grace_period = grace_period
        self.verbose = verbose
        self._metrics = MetricsEngine()

    def run(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> RestartBenchmarkResult:
        """Run restart-protocol evaluation across all sequences in *dataset*.

        Args:
            tracker: Tracker to evaluate; must implement
                :class:`~eovot.trackers.base.BaseTracker`.
            dataset: Dataset to evaluate on.
            dataset_name: Human-readable label for the dataset.
            max_sequences: Evaluate at most this many sequences (for quick runs).

        Returns:
            :class:`RestartBenchmarkResult` with per-sequence and aggregate results.
        """
        n = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)

        if self.verbose:
            print(
                f"\n[Restart Protocol] {tracker.name} on {dataset_name} "
                f"({n} sequences, threshold={self.failure_threshold}, "
                f"grace={self.grace_period})"
            )
            print("-" * 60)

        result = RestartBenchmarkResult(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
        )

        for idx in range(n):
            seq = dataset[idx]
            seq_result = self._run_sequence(tracker, seq)
            result.sequence_results.append(seq_result)
            if self.verbose:
                print(
                    f"  [{idx + 1:>3}/{n}] {seq_result.sequence_name:<30s} "
                    f"EAO={seq_result.eao:.3f}  "
                    f"FPS={seq_result.fps:.1f}  "
                    f"restarts={seq_result.num_restarts}"
                )

        if self.verbose:
            print("-" * 60)
            print(result)

        return result

    def _run_sequence(
        self, tracker: BaseTracker, sequence: Sequence
    ) -> RestartSequenceResult:
        """Evaluate one sequence with the restart protocol.

        Protocol state machine:
          - TRACKING: run tracker.update(), detect failures.
          - GRACE: skip frames (penalize as 0), then reinitialize.
        """
        frames = list(sequence)
        gts = sequence.ground_truth
        n = len(frames)

        ious = np.zeros(n, dtype=np.float64)
        predictions = np.zeros((n, 4), dtype=np.float64)
        restarts: List[RestartEvent] = []
        latencies: List[float] = []

        # Frame 0: initialize tracker, count as perfect overlap.
        t0 = time.perf_counter()
        tracker.initialize(frames[0], tuple(gts[0]))  # type: ignore[arg-type]
        latencies.append(time.perf_counter() - t0)
        ious[0] = 1.0
        predictions[0] = gts[0].copy()

        # grace_period penalty frames followed by a reinit frame.
        # reinit_at == -1 means we are in normal tracking mode.
        reinit_at: int = -1

        for i in range(1, n):
            if reinit_at != -1:
                if i < reinit_at:
                    # Grace period: penalize this frame.
                    ious[i] = 0.0
                    predictions[i] = np.zeros(4)
                else:
                    # Reinit frame: initialize tracker with ground truth.
                    t0 = time.perf_counter()
                    tracker.initialize(frames[i], tuple(gts[i]))  # type: ignore[arg-type]
                    latencies.append(time.perf_counter() - t0)
                    ious[i] = 1.0
                    predictions[i] = gts[i].copy()
                    reinit_at = -1  # Return to normal tracking.
                continue

            # Normal tracking frame.
            t0 = time.perf_counter()
            pred = tracker.update(frames[i])
            latencies.append(time.perf_counter() - t0)

            frame_iou = compute_iou(pred, tuple(gts[i]))  # type: ignore[arg-type]
            ious[i] = frame_iou
            predictions[i] = np.array(pred, dtype=np.float64)

            if frame_iou <= self.failure_threshold:
                # Failure: enter grace period.
                # Reinit at frame i + grace_period + 1 (grace_period penalty frames
                # between i+1 and i+grace_period, then reinit on i+grace_period+1).
                reinit_at = i + self.grace_period + 1
                restarts.append(
                    RestartEvent(
                        failure_frame=i,
                        restart_frame=reinit_at,
                        iou_at_failure=frame_iou,
                    )
                )

        mean_latency_ms = float(np.mean(latencies)) * 1000.0 if latencies else 0.0
        fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

        return RestartSequenceResult(
            sequence_name=sequence.name,
            tracker_name=tracker.name,
            ious=ious,
            predictions=predictions,
            ground_truths=gts,
            restarts=restarts,
            fps=fps,
            mean_latency_ms=mean_latency_ms,
        )
