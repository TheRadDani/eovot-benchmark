"""Robustness metrics for visual object tracking evaluation.

Implements VOT-challenge-style robustness analysis on top of per-frame IoU
arrays produced by :class:`~eovot.benchmark.engine.BenchmarkEngine`:

- **Failure detection** — a frame is a "failure" when IoU drops below
  ``failure_threshold`` and the tracker was previously alive.
- **Recovery lag** — number of frames until IoU rises back above
  ``recovery_threshold`` after a failure.
- **Expected Average Overlap (EAO)** — mean IoU over all non-burn-in frames,
  a simplified scalar that combines accuracy and robustness into one number
  (full VOT-protocol EAO requires re-initialization; this is the raw version).
- **Survival curve** — probability that the tracker is "alive" (IoU ≥ threshold)
  at each normalized time step across a sequence collection.

Typical usage::

    from eovot.metrics.robustness import RobustnessAnalyzer

    analyzer = RobustnessAnalyzer()

    # Per-sequence analysis
    result = analyzer.analyze_sequence(ious, tracker_name="MOSSE", sequence_name="car1")
    print(result.num_failures, result.eao)

    # Aggregate over a whole benchmark run
    seq_ious = {r.sequence_name: r.ious for r in benchmark_result.sequence_results}
    agg = analyzer.analyze_benchmark(seq_ious, tracker_name="MOSSE")
    print(agg["aggregate"]["mean_eao"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RobustnessResult:
    """Per-sequence robustness summary."""

    tracker_name: str
    sequence_name: str
    num_failures: int
    """Number of distinct tracking failures detected."""

    failure_frames: List[int]
    """Frame indices at which failures begin."""

    recovery_lags: List[int]
    """Frames-to-recovery for each failure; equals ``len(ious) - failure_frame``
    when the tracker never recovers."""

    mean_recovery_lag: float
    """Mean of ``recovery_lags``, or 0.0 if no failures occurred."""

    eao: float
    """Expected Average Overlap — mean IoU over non-burn-in frames."""

    survival_rate: float
    """Fraction of non-burn-in frames where IoU ≥ failure_threshold."""

    def __str__(self) -> str:
        return (
            f"RobustnessResult[{self.tracker_name} on {self.sequence_name}] "
            f"failures={self.num_failures}  EAO={self.eao:.4f}  "
            f"survival={self.survival_rate:.3f}  "
            f"mean_recovery_lag={self.mean_recovery_lag:.1f} fr"
        )


class RobustnessAnalyzer:
    """Analyze tracker robustness from per-frame IoU sequences.

    Args:
        failure_threshold: IoU below which a frame is counted as a failure.
            Default: ``0.1`` (standard VOT threshold).
        recovery_threshold: IoU above which the tracker is considered recovered.
            Default: ``0.1`` (same as failure threshold, hysteresis-free).
        burn_in_frames: Frames at the start of each sequence to skip before
            looking for failures.  The first frame is initialization and its
            IoU is usually 1.0 by construction; a small burn-in avoids
            counting the stabilization period.  Default: ``5``.
    """

    def __init__(
        self,
        failure_threshold: float = 0.1,
        recovery_threshold: float = 0.1,
        burn_in_frames: int = 5,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        self.burn_in_frames = burn_in_frames

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def detect_failures(self, ious: np.ndarray) -> List[int]:
        """Return the frame indices where tracking failures begin.

        A failure starts on the first frame (after burn-in) where IoU drops
        below ``failure_threshold``.  Subsequent below-threshold frames are
        part of the same failure and are not double-counted.  The tracker is
        considered recovered when IoU rises back to ``recovery_threshold``.

        Args:
            ious: Per-frame IoU array, shape ``(N,)``.

        Returns:
            List of frame indices (int) at which new failures begin.
        """
        failures: List[int] = []
        in_failure = False

        for i in range(self.burn_in_frames, len(ious)):
            iou = float(ious[i])
            if not in_failure and iou < self.failure_threshold:
                failures.append(i)
                in_failure = True
            elif in_failure and iou >= self.recovery_threshold:
                in_failure = False

        return failures

    def compute_recovery_lags(
        self, ious: np.ndarray, failure_frames: List[int]
    ) -> List[int]:
        """Compute frames-to-recovery for each detected failure.

        Args:
            ious:           Per-frame IoU array.
            failure_frames: Output of :meth:`detect_failures`.

        Returns:
            List of integers, one per failure.  If the tracker never recovers
            from a failure at frame ``f``, the lag is ``len(ious) - f``.
        """
        lags: List[int] = []
        for f in failure_frames:
            lag = 0
            for i in range(f + 1, len(ious)):
                lag += 1
                if float(ious[i]) >= self.recovery_threshold:
                    break
            else:
                # Loop completed without recovery.
                lag = len(ious) - f
            lags.append(lag)
        return lags

    def compute_eao(self, ious: np.ndarray) -> float:
        """Compute a simplified Expected Average Overlap.

        EAO is the mean IoU over all frames after the burn-in period.
        This is the "raw" EAO without the VOT re-initialization protocol.
        For fully VOT-compliant EAO, sequences must be sub-sampled and
        trackers re-initialized on failure.

        Args:
            ious: Per-frame IoU array.

        Returns:
            Mean IoU in ``[0, 1]``, or ``0.0`` for very short sequences.
        """
        if len(ious) <= self.burn_in_frames:
            return 0.0
        return float(np.mean(ious[self.burn_in_frames:]))

    def survival_curve(
        self, ious_list: List[np.ndarray], n_points: int = 100
    ) -> np.ndarray:
        """Estimate the tracker survival curve across a collection of sequences.

        The survival curve gives the probability that the tracker is "alive"
        (IoU ≥ ``failure_threshold``) at normalized time ``t ∈ [0, 1]``.

        Args:
            ious_list: List of per-frame IoU arrays (one per sequence).
            n_points:  Number of evenly-spaced time points.  Default: ``100``.

        Returns:
            ``(n_points,)`` float array with values in ``[0, 1]``.
        """
        curve = np.zeros(n_points, dtype=np.float64)
        valid = [s for s in ious_list if len(s) > 0]
        if not valid:
            return curve

        ts = np.linspace(0.0, 1.0, n_points)
        for ious in valid:
            n = len(ious)
            for j, t in enumerate(ts):
                frame = min(int(t * (n - 1)), n - 1)
                if float(ious[frame]) >= self.failure_threshold:
                    curve[j] += 1.0

        curve /= len(valid)
        return curve

    # ------------------------------------------------------------------
    # High-level analysis entry points
    # ------------------------------------------------------------------

    def analyze_sequence(
        self,
        ious: np.ndarray,
        tracker_name: str = "",
        sequence_name: str = "",
    ) -> RobustnessResult:
        """Run full robustness analysis on a single sequence.

        Args:
            ious:          Per-frame IoU array, shape ``(N,)``.
            tracker_name:  Identifier stored in the result.
            sequence_name: Identifier stored in the result.

        Returns:
            :class:`RobustnessResult` populated with all statistics.
        """
        ious = np.asarray(ious, dtype=np.float64)

        failure_frames = self.detect_failures(ious)
        recovery_lags = self.compute_recovery_lags(ious, failure_frames)
        eao = self.compute_eao(ious)

        n_active = int(np.sum(ious[self.burn_in_frames:] >= self.failure_threshold))
        denom = max(len(ious) - self.burn_in_frames, 1)
        survival_rate = n_active / denom

        return RobustnessResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            num_failures=len(failure_frames),
            failure_frames=failure_frames,
            recovery_lags=recovery_lags,
            mean_recovery_lag=float(np.mean(recovery_lags)) if recovery_lags else 0.0,
            eao=eao,
            survival_rate=survival_rate,
        )

    def analyze_benchmark(
        self,
        sequence_ious: Dict[str, np.ndarray],
        tracker_name: str = "",
    ) -> Dict:
        """Aggregate robustness analysis across all sequences in a benchmark run.

        Args:
            sequence_ious: Mapping ``{sequence_name: ious_array}``.
            tracker_name:  Identifier for the tracker under evaluation.

        Returns:
            Dict with two keys:

            * ``"per_sequence"`` — ``{seq_name: RobustnessResult}``
            * ``"aggregate"`` — summary scalars across all sequences
        """
        per_seq: Dict[str, RobustnessResult] = {}
        for seq_name, ious in sequence_ious.items():
            per_seq[seq_name] = self.analyze_sequence(
                ious, tracker_name=tracker_name, sequence_name=seq_name
            )

        n = len(per_seq)
        total_failures = sum(r.num_failures for r in per_seq.values())
        mean_eao = float(np.mean([r.eao for r in per_seq.values()])) if n else 0.0
        mean_survival = float(np.mean([r.survival_rate for r in per_seq.values()])) if n else 0.0
        mean_lag = float(
            np.mean([r.mean_recovery_lag for r in per_seq.values() if r.num_failures > 0])
        ) if any(r.num_failures > 0 for r in per_seq.values()) else 0.0

        return {
            "per_sequence": per_seq,
            "aggregate": {
                "tracker_name": tracker_name,
                "num_sequences": n,
                "total_failures": total_failures,
                "mean_failures_per_sequence": total_failures / n if n else 0.0,
                "mean_eao": round(mean_eao, 4),
                "mean_survival_rate": round(mean_survival, 4),
                "mean_recovery_lag_frames": round(mean_lag, 2),
            },
        }
