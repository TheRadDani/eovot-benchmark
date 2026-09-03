"""VOT-protocol compliant benchmark engine with tracker re-initialization.

The standard :class:`~eovot.benchmark.engine.BenchmarkEngine` evaluates
trackers continuously over a full sequence.  The Visual Object Tracking (VOT)
challenge protocol differs in one crucial respect: after a tracking failure
(IoU < threshold), the tracker is re-initialized with the ground-truth
bounding box following a short *grace period* (penalty window) of zero-IoU
frames.  This re-initialization protocol is necessary for computing a valid
Expected Average Overlap (EAO) that is comparable across published results.

This module implements the simplified VOT-protocol evaluation used by
VOT-2016 and later, minus the anchor-based sub-sequence sampling (which
requires a large enough sequence collection to be meaningful).  The scalar
EAO this engine produces is directly comparable to published VOT baselines.

Reference:
    Kristan et al., "The Visual Object Tracking VOT2016 Challenge Results."
    ECCV Workshops 2016. — defines the re-initialization protocol.

Typical usage::

    from eovot.benchmark.vot_engine import VotBenchmarkEngine
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=10, num_frames=200)
    engine = VotBenchmarkEngine(verbose=True)
    result = engine.run(KCFTracker(), dataset, dataset_name="Synthetic")
    print(result)
    print(f"EAO = {result.mean_eao:.4f}  Failures = {result.total_failures}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import iou as _iou, MetricsEngine
from ..profiling.profiler import Profiler, ProfilingResult
from ..trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VotSequenceResult:
    """Per-sequence result from VOT-protocol evaluation.

    Attributes:
        sequence_name: Identifier of the evaluated sequence.
        ious: Per-frame IoU array (shape ``(N,)``).  Frames within a grace
            period have ``iou = 0.0``; the first frame is always ``1.0``.
        profiling: Hardware profiling summary for this sequence.
        num_failures: Number of distinct tracking failures detected.
        failure_frames: Frame indices at which failures were declared.
        grace_frames: Set of frame indices that were inside a grace period.
        eao: Mean IoU over all frames, including grace-period penalty frames.
            This is the per-sequence EAO scalar used by the VOT protocol.
        predictions: ``(N, 4)`` array of tracker predictions; ``(0,0,0,0)``
            during grace frames and the first frame GT.
        ground_truths: ``(N, 4)`` ground-truth boxes aligned to predictions.
    """

    sequence_name: str
    ious: np.ndarray
    profiling: ProfilingResult
    num_failures: int
    failure_frames: List[int]
    grace_frames: set
    eao: float
    predictions: Optional[np.ndarray] = None
    ground_truths: Optional[np.ndarray] = None

    @property
    def mean_iou(self) -> float:
        return float(self.ious.mean()) if len(self.ious) else 0.0

    def __str__(self) -> str:
        return (
            f"VotSequenceResult[{self.sequence_name}] "
            f"EAO={self.eao:.4f}  mIoU={self.mean_iou:.4f}  "
            f"failures={self.num_failures}  "
            f"FPS={self.profiling.fps:.1f}"
        )


@dataclass
class VotBenchmarkResult:
    """Aggregate VOT-protocol result for a tracker on a full dataset.

    Attributes:
        tracker_name: Identifier of the evaluated tracker.
        dataset_name: Dataset on which evaluation was performed.
        sequence_results: Per-sequence ``VotSequenceResult`` objects.
        failure_threshold: IoU threshold used to declare a failure.
        grace_period_frames: Number of grace/penalty frames after each failure.
    """

    tracker_name: str
    dataset_name: str
    sequence_results: List[VotSequenceResult] = field(default_factory=list)
    failure_threshold: float = 0.1
    grace_period_frames: int = 5

    @property
    def total_failures(self) -> int:
        """Total failures across all sequences."""
        return sum(r.num_failures for r in self.sequence_results)

    @property
    def mean_eao(self) -> float:
        """Mean EAO (Expected Average Overlap) across all sequences."""
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.eao for r in self.sequence_results]))

    @property
    def mean_fps(self) -> float:
        """Mean FPS across all sequences (excludes grace-period frames)."""
        if not self.sequence_results:
            return 0.0
        return float(np.mean([r.profiling.fps for r in self.sequence_results]))

    @property
    def mean_failures_per_sequence(self) -> float:
        n = len(self.sequence_results)
        return self.total_failures / n if n else 0.0

    def summary(self) -> Dict:
        """Return a serialisable summary dict."""
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "protocol": "VOT",
            "failure_threshold": self.failure_threshold,
            "grace_period_frames": self.grace_period_frames,
            "num_sequences": len(self.sequence_results),
            "mean_eao": round(self.mean_eao, 4),
            "mean_fps": round(self.mean_fps, 2),
            "total_failures": self.total_failures,
            "mean_failures_per_sequence": round(self.mean_failures_per_sequence, 2),
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"VotBenchmarkResult[{s['tracker']} on {s['dataset']}] "
            f"EAO={s['mean_eao']}  "
            f"FPS={s['mean_fps']}  "
            f"failures={s['total_failures']}  "
            f"({s['num_sequences']} sequences)"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VotBenchmarkEngine:
    """VOT-protocol benchmark engine: evaluate a tracker with re-initialization.

    After a tracking failure (per-frame IoU < ``failure_threshold``), the
    tracker is penalised with ``grace_period_frames`` zero-IoU frames.  At
    the end of the grace period the tracker is re-initialized with the
    ground-truth box, and evaluation continues.  This matches the evaluation
    protocol used by the VOT challenge.

    Args:
        failure_threshold: IoU below which a frame is a failure.  Default: ``0.1``
            (the VOT standard).
        grace_period_frames: Number of frames to skip after a failure before
            re-initializing with GT.  Default: ``5`` (VOT standard).
        verbose: Print per-sequence progress to stdout.  Default: ``True``.

    Example::

        engine = VotBenchmarkEngine(failure_threshold=0.1, grace_period_frames=5)
        result = engine.run(tracker, dataset, dataset_name="OTB100")
        print(result)
    """

    def __init__(
        self,
        failure_threshold: float = 0.1,
        grace_period_frames: int = 5,
        verbose: bool = True,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.grace_period_frames = grace_period_frames
        self.verbose = verbose
        self._profiler = Profiler()

    def run(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> VotBenchmarkResult:
        """Evaluate *tracker* on every sequence in *dataset* using VOT protocol.

        Args:
            tracker: Any :class:`~eovot.trackers.base.BaseTracker` subclass.
            dataset: Dataset to evaluate on.
            dataset_name: Human-readable label for reports.
            max_sequences: Limit evaluation to first *N* sequences.  ``None``
                evaluates the full dataset.

        Returns:
            :class:`VotBenchmarkResult` with per-sequence and aggregate metrics.
        """
        result = VotBenchmarkResult(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
            failure_threshold=self.failure_threshold,
            grace_period_frames=self.grace_period_frames,
        )
        n = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)

        if self.verbose:
            print(
                f"\n[VOT] Evaluating {tracker.name} on {dataset_name} "
                f"({n} sequences)  "
                f"failure_thr={self.failure_threshold}  "
                f"grace={self.grace_period_frames} fr"
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
                    f"fail={seq_result.num_failures}  "
                    f"FPS={seq_result.profiling.fps:.1f}"
                )

        if self.verbose:
            print("-" * 60)
            print(result)

        return result

    def _run_sequence(self, tracker: BaseTracker, seq: Sequence) -> VotSequenceResult:
        """Run one sequence under the VOT re-initialization protocol."""
        self._profiler.reset()

        frames = list(seq)
        gt = seq.ground_truth
        n_frames = len(frames)

        ious: List[float] = []
        preds_list: List[BBox] = []
        failure_frames: List[int] = []
        grace_frames: set = set()

        # --- Frame 0: initialization ---
        tracker.initialize(frames[0], seq.init_bbox)
        preds_list.append(seq.init_bbox)
        ious.append(1.0)   # initialization frame always scores 1.0

        grace_countdown: int = 0

        for i in range(1, n_frames):
            frame = frames[i]

            if grace_countdown > 0:
                # Still in grace/penalty period.
                grace_frames.add(i)
                preds_list.append((0.0, 0.0, 0.0, 0.0))
                ious.append(0.0)
                grace_countdown -= 1

                if grace_countdown == 0 and i + 1 < n_frames:
                    # Re-initialize with GT at the frame AFTER the grace period.
                    # If we can't get GT for frame i, fall back to init_bbox.
                    gt_reinit: BBox = (
                        tuple(gt[i])  # type: ignore[arg-type]
                        if i < len(gt)
                        else seq.init_bbox
                    )
                    tracker.initialize(frames[i], gt_reinit)
                continue

            # --- Normal tracking step ---
            self._profiler.start_frame()
            bbox = tracker.update(frame)
            self._profiler.end_frame()

            preds_list.append(bbox)

            # Compute IoU against GT for this frame.
            if i < len(gt):
                frame_iou = _iou(bbox, tuple(gt[i]))  # type: ignore[arg-type]
            else:
                frame_iou = 0.0
            ious.append(frame_iou)

            # Failure detection.
            if frame_iou < self.failure_threshold:
                failure_frames.append(i)
                grace_countdown = self.grace_period_frames

        ious_arr = np.array(ious, dtype=np.float64)

        # EAO: mean IoU across all frames (including penalty zeros).
        eao = float(ious_arr.mean())

        # Build aligned arrays for downstream analysis.
        n_eval = min(len(preds_list), len(gt))
        preds_arr = np.array(preds_list[:n_eval], dtype=np.float64)
        gt_arr = gt[:n_eval].astype(np.float64) if hasattr(gt, "astype") else np.array(gt[:n_eval], dtype=np.float64)

        profiling = self._profiler.summary(tracker.name)

        return VotSequenceResult(
            sequence_name=seq.name,
            ious=ious_arr,
            profiling=profiling,
            num_failures=len(failure_frames),
            failure_frames=failure_frames,
            grace_frames=grace_frames,
            eao=eao,
            predictions=preds_arr,
            ground_truths=gt_arr,
        )
