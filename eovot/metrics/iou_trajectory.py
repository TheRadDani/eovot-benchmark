"""IoU trajectory analysis for visual object tracking evaluation.

Standard robustness analysis (RobustnessAnalyzer) answers *where* a tracker
fails and *how quickly* it recovers.  Temporal consistency analysis
(TemporalConsistencyAnalyzer) answers *how smoothly* the predicted boxes
move.  Neither characterises the *shape* of the IoU time series across the
whole sequence — whether accuracy builds up early, holds steady, drifts away
at the tail, or whether there is a distinct peak window of best performance.

This module fills that gap by decomposing per-frame IoU arrays into
interpretable trajectory statistics:

Segment IoUs
~~~~~~~~~~~~
The sequence is split into ``n_segments`` equal-duration bins.  The mean
and standard deviation of IoU inside each segment reveal whether accuracy
is front-loaded (good start, drift later), back-loaded (slow lock-on then
stable), or U-shaped (failure then recovery).

Convergence Frame
~~~~~~~~~~~~~~~~~
The first frame (after an optional burn-in) at which the rolling average IoU
exceeds a threshold — a scalar measure of how quickly the tracker achieves
reliable performance after initialization.  Returns ``-1`` if the threshold
is never met.

Peak Window
~~~~~~~~~~~
The sliding window of length ``window`` frames that achieves the highest
mean IoU.  This identifies the tracker's "best stretch" and its location
in the sequence.

Tail Drop
~~~~~~~~~
Difference between the mean IoU of the first ``(1 - tail_fraction)`` of the
sequence and the final ``tail_fraction``.  Positive values indicate degrading
performance at the end of long sequences — a key failure mode for correlation-
filter trackers that drift when the appearance model becomes stale.

IoU Stability
~~~~~~~~~~~~~
Coefficient of variation (CV = std / mean) of the IoU array, restricted to
frames where IoU is above the failure threshold.  Low CV implies the tracker
is consistently accurate in the frames it handles well; high CV signals
intermittent performance even in the "good" part of the sequence.

Typical usage::

    from eovot.metrics.iou_trajectory import IoUTrajectoryAnalyzer

    analyzer = IoUTrajectoryAnalyzer()

    # Per-sequence analysis
    result = analyzer.analyze(ious, tracker_name="MOSSE", sequence_name="car1")
    print(result)

    # Aggregate over a full benchmark run
    seq_ious = {r.sequence_name: r.ious for r in benchmark_result.sequence_results}
    agg = analyzer.analyze_benchmark(seq_ious, tracker_name="MOSSE")
    print(agg["aggregate"]["mean_tail_drop"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SegmentStats:
    """IoU statistics for one contiguous segment of a sequence.

    Attributes:
        index:      Zero-based segment index (0 = earliest).
        start_frame: First frame index of this segment.
        end_frame:   Exclusive end frame index.
        mean_iou:   Mean IoU within the segment.
        std_iou:    Standard deviation of IoU within the segment.
        n_frames:   Number of frames in this segment.
    """

    index: int
    start_frame: int
    end_frame: int
    mean_iou: float
    std_iou: float
    n_frames: int

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "mean_iou": round(self.mean_iou, 4),
            "std_iou": round(self.std_iou, 4),
            "n_frames": self.n_frames,
        }


@dataclass
class IoUTrajectoryResult:
    """Per-sequence IoU trajectory summary.

    Attributes:
        tracker_name:       Human-readable tracker identifier.
        sequence_name:      Sequence identifier.
        num_frames:         Total number of frames analysed.
        segments:           Per-segment IoU statistics (length = ``n_segments``).
        convergence_frame:  First frame ≥ convergence threshold, or ``-1`` if never.
        peak_window_start:  Start frame of the best sliding window, or ``-1`` if N/A.
        peak_window_mean_iou: Mean IoU inside the best window, or ``0.0``.
        tail_drop:          Mean IoU of early frames minus mean IoU of tail frames.
                            Positive → accuracy degrades toward the end.
        iou_stability:      Coefficient of variation of IoU on non-failure frames,
                            or ``0.0`` when all frames are below the failure threshold.
        overall_mean_iou:   Mean IoU across all frames.
    """

    tracker_name: str
    sequence_name: str
    num_frames: int
    segments: List[SegmentStats]
    convergence_frame: int
    peak_window_start: int
    peak_window_mean_iou: float
    tail_drop: float
    iou_stability: float
    overall_mean_iou: float

    def __str__(self) -> str:
        seg_means = "  ".join(f"s{s.index}={s.mean_iou:.3f}" for s in self.segments)
        return (
            f"IoUTrajectory[{self.tracker_name} on {self.sequence_name}] "
            f"mIoU={self.overall_mean_iou:.4f}  "
            f"convergence={self.convergence_frame}  "
            f"tail_drop={self.tail_drop:+.4f}  "
            f"stability={self.iou_stability:.4f}  "
            f"segments=[{seg_means}]"
        )

    def to_dict(self) -> Dict:
        return {
            "tracker_name": self.tracker_name,
            "sequence_name": self.sequence_name,
            "num_frames": self.num_frames,
            "overall_mean_iou": round(self.overall_mean_iou, 4),
            "convergence_frame": self.convergence_frame,
            "peak_window_start": self.peak_window_start,
            "peak_window_mean_iou": round(self.peak_window_mean_iou, 4),
            "tail_drop": round(self.tail_drop, 4),
            "iou_stability": round(self.iou_stability, 4),
            "segments": [s.to_dict() for s in self.segments],
        }


class IoUTrajectoryAnalyzer:
    """Analyse the temporal shape of per-frame IoU curves.

    Unlike :class:`~eovot.metrics.robustness.RobustnessAnalyzer`, which
    counts discrete failure events, this analyser characterises the
    *distribution* of IoU values across the sequence timeline: where
    accuracy peaks, how quickly the tracker converges, and whether
    performance degrades at the tail.

    Args:
        n_segments:         Number of equal-duration segments to split each
                            sequence into for segment-wise IoU statistics.
                            Default: ``3`` (early / middle / late thirds).
        convergence_threshold: IoU value that defines "reliable tracking".
                            Default: ``0.5``.
        peak_window:        Length (in frames) of the sliding window used to
                            find the best-performance stretch.  Default: ``10``.
        tail_fraction:      Fraction of the sequence that defines the "tail".
                            Default: ``0.2`` (last 20 % of frames).
        failure_threshold:  IoU below which a frame is considered a failure;
                            used when computing ``iou_stability``.
                            Default: ``0.1`` (standard VOT threshold).

    Example::

        analyzer = IoUTrajectoryAnalyzer()
        result = analyzer.analyze(seq_result.ious, "MOSSE", "car1")
        print(result.tail_drop, result.convergence_frame)
    """

    def __init__(
        self,
        n_segments: int = 3,
        convergence_threshold: float = 0.5,
        peak_window: int = 10,
        tail_fraction: float = 0.2,
        failure_threshold: float = 0.1,
    ) -> None:
        if n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {n_segments}.")
        if not 0.0 < convergence_threshold <= 1.0:
            raise ValueError(
                f"convergence_threshold must be in (0, 1], got {convergence_threshold}."
            )
        if peak_window < 1:
            raise ValueError(f"peak_window must be >= 1, got {peak_window}.")
        if not 0.0 < tail_fraction < 1.0:
            raise ValueError(
                f"tail_fraction must be in (0, 1), got {tail_fraction}."
            )
        if not 0.0 <= failure_threshold < 1.0:
            raise ValueError(
                f"failure_threshold must be in [0, 1), got {failure_threshold}."
            )

        self.n_segments = n_segments
        self.convergence_threshold = convergence_threshold
        self.peak_window = peak_window
        self.tail_fraction = tail_fraction
        self.failure_threshold = failure_threshold

    # ------------------------------------------------------------------
    # Individual metric computations
    # ------------------------------------------------------------------

    def compute_segment_ious(self, ious: np.ndarray) -> List[SegmentStats]:
        """Split IoU array into equal segments and compute per-segment statistics.

        Args:
            ious: Per-frame IoU array, shape ``(N,)``.

        Returns:
            List of :class:`SegmentStats` of length ``n_segments``.  Segments
            at the end of a non-divisible sequence are slightly longer by at
            most one frame (``np.array_split`` semantics).
        """
        n = len(ious)
        if n == 0:
            return []

        splits = np.array_split(np.arange(n), self.n_segments)
        stats: List[SegmentStats] = []
        for seg_idx, idx in enumerate(splits):
            if len(idx) == 0:
                continue
            seg = ious[idx]
            stats.append(
                SegmentStats(
                    index=seg_idx,
                    start_frame=int(idx[0]),
                    end_frame=int(idx[-1]) + 1,
                    mean_iou=float(seg.mean()),
                    std_iou=float(seg.std()),
                    n_frames=len(idx),
                )
            )
        return stats

    def compute_convergence_frame(self, ious: np.ndarray) -> int:
        """Find the first frame at which the rolling-mean IoU exceeds the threshold.

        Uses a rolling average of length ``peak_window`` to smooth transient
        spikes.  If the sequence is shorter than ``peak_window``, the raw IoU
        is used frame-by-frame.

        Args:
            ious: Per-frame IoU array.

        Returns:
            Zero-based frame index of the first qualifying frame, or ``-1``
            if the threshold is never reached.
        """
        n = len(ious)
        if n == 0:
            return -1

        w = min(self.peak_window, n)
        if w <= 1:
            smoothed = ious
        else:
            # Causal sliding mean: each element is the mean of the previous w frames
            kernel = np.ones(w, dtype=np.float64) / w
            smoothed = np.convolve(ious, kernel, mode="full")[:n]
            # The first (w-1) frames use partial windows — pad with raw value
            for i in range(min(w - 1, n)):
                smoothed[i] = float(ious[: i + 1].mean())

        for i, v in enumerate(smoothed):
            if float(v) >= self.convergence_threshold:
                return i
        return -1

    def compute_peak_window(self, ious: np.ndarray) -> Tuple[int, float]:
        """Find the sliding window with the highest mean IoU.

        Args:
            ious: Per-frame IoU array.

        Returns:
            ``(start_frame, mean_iou)`` of the best window.  Returns
            ``(-1, 0.0)`` for empty arrays.  If the sequence is shorter than
            ``peak_window``, the entire sequence is the single window.
        """
        n = len(ious)
        if n == 0:
            return -1, 0.0

        w = min(self.peak_window, n)
        if w == n:
            return 0, float(ious.mean())

        # Compute sliding window means efficiently using cumulative sum
        cs = np.concatenate([[0.0], np.cumsum(ious)])
        window_means = (cs[w:] - cs[:-w]) / w
        best_idx = int(np.argmax(window_means))
        return best_idx, float(window_means[best_idx])

    def compute_tail_drop(self, ious: np.ndarray) -> float:
        """Compute mean IoU of early frames minus mean IoU of tail frames.

        Args:
            ious: Per-frame IoU array.

        Returns:
            Tail drop scalar.  Positive → tracker loses accuracy near the end.
            Zero for sequences too short to split.
        """
        n = len(ious)
        tail_n = max(1, int(round(n * self.tail_fraction)))
        if n <= tail_n:
            return 0.0
        early_mean = float(ious[: n - tail_n].mean())
        tail_mean = float(ious[n - tail_n :].mean())
        return early_mean - tail_mean

    def compute_iou_stability(self, ious: np.ndarray) -> float:
        """Coefficient of variation of IoU on non-failure frames.

        Args:
            ious: Per-frame IoU array.

        Returns:
            CV = std / mean on frames with IoU > ``failure_threshold``.
            Returns ``0.0`` when no non-failure frames exist or mean is zero.
        """
        active = ious[ious > self.failure_threshold]
        if len(active) == 0:
            return 0.0
        mean_iou = float(active.mean())
        if mean_iou < 1e-9:
            return 0.0
        return float(active.std() / mean_iou)

    # ------------------------------------------------------------------
    # High-level analysis entry points
    # ------------------------------------------------------------------

    def analyze(
        self,
        ious: np.ndarray,
        tracker_name: str = "",
        sequence_name: str = "",
    ) -> IoUTrajectoryResult:
        """Run full IoU trajectory analysis on a single sequence.

        Args:
            ious:          Per-frame IoU array, shape ``(N,)``.  Use
                           ``SequenceResult.ious`` from
                           :class:`~eovot.benchmark.engine.BenchmarkEngine`.
            tracker_name:  Identifier stored in the result.
            sequence_name: Identifier stored in the result.

        Returns:
            :class:`IoUTrajectoryResult` with all trajectory metrics populated.
        """
        ious = np.asarray(ious, dtype=np.float64).ravel()
        n = len(ious)

        if n == 0:
            return IoUTrajectoryResult(
                tracker_name=tracker_name,
                sequence_name=sequence_name,
                num_frames=0,
                segments=[],
                convergence_frame=-1,
                peak_window_start=-1,
                peak_window_mean_iou=0.0,
                tail_drop=0.0,
                iou_stability=0.0,
                overall_mean_iou=0.0,
            )

        segments = self.compute_segment_ious(ious)
        convergence_frame = self.compute_convergence_frame(ious)
        peak_start, peak_mean = self.compute_peak_window(ious)
        tail_drop = self.compute_tail_drop(ious)
        stability = self.compute_iou_stability(ious)

        return IoUTrajectoryResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            num_frames=n,
            segments=segments,
            convergence_frame=convergence_frame,
            peak_window_start=peak_start,
            peak_window_mean_iou=peak_mean,
            tail_drop=tail_drop,
            iou_stability=stability,
            overall_mean_iou=float(ious.mean()),
        )

    def analyze_benchmark(
        self,
        sequence_ious: Dict[str, np.ndarray],
        tracker_name: str = "",
    ) -> Dict:
        """Aggregate IoU trajectory analysis across all sequences.

        Args:
            sequence_ious: Mapping ``{sequence_name: ious_array}``.  Build it
                with ``{r.sequence_name: r.ious for r in result.sequence_results}``.
            tracker_name:  Identifier for the tracker under evaluation.

        Returns:
            Dict with two keys:

            * ``"per_sequence"`` — ``{seq_name: IoUTrajectoryResult}``
            * ``"aggregate"`` — summary scalars averaged across all sequences
        """
        per_seq: Dict[str, IoUTrajectoryResult] = {}
        for seq_name, ious in sequence_ious.items():
            per_seq[seq_name] = self.analyze(
                ious, tracker_name=tracker_name, sequence_name=seq_name
            )

        if not per_seq:
            return {"per_sequence": {}, "aggregate": {}}

        results = list(per_seq.values())
        n = len(results)

        # Segment aggregation: mean IoU per segment slot across all sequences
        max_segs = max(len(r.segments) for r in results)
        seg_means: List[Optional[float]] = []
        for seg_idx in range(max_segs):
            vals = [
                r.segments[seg_idx].mean_iou
                for r in results
                if seg_idx < len(r.segments)
            ]
            seg_means.append(round(float(np.mean(vals)), 4) if vals else None)

        # Convergence: fraction of sequences that ever converge
        converged = [r for r in results if r.convergence_frame >= 0]
        convergence_rate = len(converged) / n
        mean_convergence_frame = (
            float(np.mean([r.convergence_frame for r in converged]))
            if converged
            else float("nan")
        )

        return {
            "per_sequence": per_seq,
            "aggregate": {
                "tracker_name": tracker_name,
                "num_sequences": n,
                "mean_overall_iou": round(
                    float(np.mean([r.overall_mean_iou for r in results])), 4
                ),
                "mean_tail_drop": round(
                    float(np.mean([r.tail_drop for r in results])), 4
                ),
                "mean_iou_stability": round(
                    float(np.mean([r.iou_stability for r in results])), 4
                ),
                "mean_peak_window_iou": round(
                    float(np.mean([r.peak_window_mean_iou for r in results])), 4
                ),
                "convergence_rate": round(convergence_rate, 4),
                "mean_convergence_frame": (
                    round(mean_convergence_frame, 1)
                    if not (
                        isinstance(mean_convergence_frame, float)
                        and mean_convergence_frame != mean_convergence_frame
                    )
                    else None
                ),
                "segment_mean_ious": seg_means,
            },
        }
