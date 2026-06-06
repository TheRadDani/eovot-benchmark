"""Robustness metrics for visual object tracking evaluation.

.. note::
    This module also provides :class:`ConfidenceAwareRobustnessAnalyzer`, which
    supplements IoU-based failure detection with PSR confidence scores produced
    by correlation-filter trackers (MOSSE, KCF).  Using confidence enables
    *early* failure warnings before IoU collapses to zero, giving edge systems
    time to trigger re-initialization or fallback strategies.

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
from typing import Dict, List, Optional, Tuple

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


@dataclass
class ConfidenceRobustnessResult:
    """Extended robustness result that includes confidence-signal statistics."""

    base: RobustnessResult
    """Standard IoU-based robustness result."""

    mean_confidence: float
    """Mean PSR-derived confidence over the sequence (excluding init frame)."""

    min_confidence: float
    """Minimum confidence observed — indicates worst-case response quality."""

    confidence_failure_rate: float
    """Fraction of frames where confidence fell below ``confidence_threshold``."""

    early_warnings: List[int]
    """Frame indices where confidence dropped below threshold *before* a
    corresponding IoU failure was detected — true early warnings."""

    def __str__(self) -> str:
        return (
            f"ConfidenceRobustnessResult[{self.base.tracker_name} on "
            f"{self.base.sequence_name}] "
            f"failures={self.base.num_failures}  EAO={self.base.eao:.4f}  "
            f"mean_conf={self.mean_confidence:.3f}  "
            f"early_warnings={len(self.early_warnings)}"
        )


class ConfidenceAwareRobustnessAnalyzer:
    """Extends :class:`RobustnessAnalyzer` with PSR-confidence-based analysis.

    Combines per-frame IoU arrays (from ground truth) with per-frame confidence
    scores (from the tracker's internal state, e.g. PSR) to produce richer
    robustness diagnostics:

    * **Early warning detection** — identifies frames where confidence falls
      before IoU collapses, enabling proactive re-initialization.
    * **Confidence-failure rate** — fraction of frames below
      ``confidence_threshold``.
    * **Confidence–IoU correlation** — Pearson r between confidence and IoU,
      quantifying how predictive PSR is for this tracker/dataset combination.

    Args:
        confidence_threshold: Normalised confidence below which a frame is
            flagged as uncertain.  Default: ``0.3``.
        failure_threshold:    IoU threshold forwarded to the underlying
            :class:`RobustnessAnalyzer`.  Default: ``0.1``.
        burn_in_frames:       Frames to skip at sequence start.  Default: ``5``.
        early_warning_lead:   How many frames *before* an IoU failure a
            confidence drop must occur to count as a true early warning.
            Default: ``3``.

    Example::

        from eovot.metrics.robustness import ConfidenceAwareRobustnessAnalyzer

        analyzer = ConfidenceAwareRobustnessAnalyzer()
        result = analyzer.analyze(ious, confidence_scores,
                                  tracker_name="MOSSE", sequence_name="car1")
        print(result.early_warnings)
        print(result.base.eao)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        failure_threshold: float = 0.1,
        burn_in_frames: int = 5,
        early_warning_lead: int = 3,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.early_warning_lead = early_warning_lead
        self._base_analyzer = RobustnessAnalyzer(
            failure_threshold=failure_threshold,
            recovery_threshold=failure_threshold,
            burn_in_frames=burn_in_frames,
        )

    def analyze(
        self,
        ious: np.ndarray,
        confidence_scores: np.ndarray,
        tracker_name: str = "",
        sequence_name: str = "",
    ) -> ConfidenceRobustnessResult:
        """Run combined IoU + confidence analysis on a single sequence.

        Args:
            ious:             Per-frame IoU array, shape ``(N,)``.  Includes
                              the initialisation frame (index 0, IoU ≈ 1).
            confidence_scores: Per-update confidence scores, shape ``(N-1,)``.
                              Index ``i`` corresponds to frame ``i+1`` in *ious*.
            tracker_name:     Stored in the result for reporting.
            sequence_name:    Stored in the result for reporting.

        Returns:
            :class:`ConfidenceRobustnessResult` with both IoU-based and
            confidence-based statistics.
        """
        ious = np.asarray(ious, dtype=np.float64)
        confidence_scores = np.asarray(confidence_scores, dtype=np.float64)

        base = self._base_analyzer.analyze_sequence(ious, tracker_name, sequence_name)

        burn = self._base_analyzer.burn_in_frames
        # confidence_scores are 0-indexed to update frames (frame 1 onward).
        conf_analysis = confidence_scores[max(0, burn - 1):]

        mean_conf = float(conf_analysis.mean()) if len(conf_analysis) else 0.0
        min_conf = float(conf_analysis.min()) if len(conf_analysis) else 0.0
        conf_failure_rate = float(
            np.mean(conf_analysis < self.confidence_threshold)
        ) if len(conf_analysis) else 0.0

        early_warnings = self._find_early_warnings(
            ious, confidence_scores, base.failure_frames
        )

        return ConfidenceRobustnessResult(
            base=base,
            mean_confidence=mean_conf,
            min_confidence=min_conf,
            confidence_failure_rate=conf_failure_rate,
            early_warnings=early_warnings,
        )

    def confidence_iou_correlation(
        self,
        ious: np.ndarray,
        confidence_scores: np.ndarray,
    ) -> float:
        """Compute Pearson correlation between confidence scores and IoU.

        A high positive correlation (r > 0.5) means PSR is a reliable proxy
        for tracking quality on this sequence/dataset, validating its use for
        autonomous failure detection.

        Args:
            ious:             Per-frame IoU array, shape ``(N,)``.
            confidence_scores: Per-update confidence scores, shape ``(N-1,)``.

        Returns:
            Pearson r in ``[-1, 1]``, or ``0.0`` if inputs are too short or
            have zero variance.
        """
        ious = np.asarray(ious, dtype=np.float64)
        confidence_scores = np.asarray(confidence_scores, dtype=np.float64)

        n = min(len(confidence_scores), len(ious) - 1)
        if n < 2:
            return 0.0

        iou_aligned = ious[1 : n + 1]
        conf_aligned = confidence_scores[:n]

        if iou_aligned.std() < 1e-10 or conf_aligned.std() < 1e-10:
            return 0.0

        return float(np.corrcoef(iou_aligned, conf_aligned)[0, 1])

    def _find_early_warnings(
        self,
        ious: np.ndarray,
        confidence_scores: np.ndarray,
        failure_frames: List[int],
    ) -> List[int]:
        """Identify frames where confidence dropped before an IoU failure.

        A frame ``f_conf`` is an early warning if:
        - confidence_scores[f_conf - 1] < confidence_threshold
        - There exists an IoU failure at frame ``f_iou`` with
          0 < f_iou - f_conf ≤ early_warning_lead

        Args:
            ious:           Per-frame IoU array.
            confidence_scores: Per-update confidence scores (len = len(ious)-1).
            failure_frames: Output of :meth:`RobustnessAnalyzer.detect_failures`.

        Returns:
            Sorted list of frame indices that qualify as early warnings.
        """
        if not failure_frames or len(confidence_scores) == 0:
            return []

        failure_set = set(failure_frames)
        warnings: List[int] = []

        for conf_frame in range(1, len(confidence_scores) + 1):
            if conf_frame not in failure_set:
                conf_val = confidence_scores[conf_frame - 1]
                if conf_val < self.confidence_threshold:
                    # Check if a failure follows within the lead window.
                    for lead in range(1, self.early_warning_lead + 1):
                        if (conf_frame + lead) in failure_set:
                            warnings.append(conf_frame)
                            break

        return sorted(set(warnings))
