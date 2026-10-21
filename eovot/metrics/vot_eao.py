"""VOT-protocol Expected Average Overlap (EAO) calculator.

Implements the failure-penalty simulation approach to EAO that closely
approximates the full VOT EAO without requiring tracker re-initialization
at runtime.  All computation works on pre-recorded per-frame IoU arrays.

The proper VOT EAO protocol (Kristan et al., VOT2015) re-initializes the
tracker 5 frames after each failure.  Since re-running the tracker requires
the raw frames and tracker state, we simulate the protocol by zeroing out
IoU for the penalty window after each failure, then computing the expected
overlap curve over those adjusted sequences.

This approximation is accurate when the tracker's post-reinitialization
behaviour is representative of its initial behaviour — a reasonable
assumption for stationary-background sequences.

References
----------
Kristan, M. et al. (2015). The Visual Object Tracking VOT2015 Challenge
Results. ICCV 2015 Workshops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class VotEaoResult:
    """EAO analysis for a single tracker over a full benchmark run.

    Attributes:
        tracker_name:         Human-readable identifier.
        eao:                  Expected Average Overlap scalar (VOT-protocol
                              approximation).  Higher is better.
        sequence_count:       Number of sequences in the benchmark run.
        total_failures:       Total failure events across all sequences.
        mean_failures_per_seq: Average failure count per sequence.
        overlap_curve:        Mean per-frame overlap at each sequence length,
                              shape ``(eao_high,)``.
        eao_low:              Lower bound of the temporal interval used for
                              EAO averaging (1-indexed).
        eao_high:             Upper bound of the temporal interval.
    """

    tracker_name: str
    eao: float
    sequence_count: int
    total_failures: int
    mean_failures_per_seq: float
    overlap_curve: np.ndarray
    eao_low: int
    eao_high: int

    def __str__(self) -> str:
        return (
            f"VotEaoResult[{self.tracker_name}] "
            f"EAO={self.eao:.4f}  "
            f"failures={self.total_failures}  "
            f"sequences={self.sequence_count}"
        )


class VotEaoCalculator:
    """Compute VOT-protocol EAO by simulating failure penalty windows.

    Unlike the simplified EAO in
    :class:`~eovot.metrics.robustness.RobustnessAnalyzer` — which averages
    IoU over the raw sequence — this calculator applies the VOT
    re-initialization protocol:

    1. Detect failure frames (IoU < ``failure_threshold``).
    2. Zero out IoU for the ``penalty_frames`` frames immediately following
       each failure (representing the mandatory pause before re-initialization).
    3. Build an "expected overlap" curve: for each sequence length N, compute
       the mean IoU over the first N frames across all sequences that are at
       least N frames long.
    4. EAO = mean of the overlap curve over the temporal interval
       ``[eao_low, eao_high]`` (default 100–356, matching VOT2015).

    Typical usage::

        from eovot.metrics.vot_eao import VotEaoCalculator

        calc = VotEaoCalculator()
        ious_per_seq = {r.sequence_name: r.ious
                        for r in benchmark_result.sequence_results}
        result = calc.compute(ious_per_seq, tracker_name="KCF")
        print(f"EAO = {result.eao:.4f}")

    Compare multiple trackers::

        tracker_ious = {
            "KCF":  {"seq1": ious_kcf_1, "seq2": ious_kcf_2},
            "CSRT": {"seq1": ious_csrt_1, "seq2": ious_csrt_2},
        }
        ranking = calc.compare(tracker_ious)
        for r in ranking:
            print(r)
    """

    def __init__(
        self,
        failure_threshold: float = 0.1,
        penalty_frames: int = 5,
        eao_low: int = 100,
        eao_high: int = 356,
    ) -> None:
        """
        Args:
            failure_threshold: IoU below which a frame counts as a failure.
                Default ``0.1`` matches the VOT challenge criterion.
            penalty_frames:    Frames zeroed out immediately after each
                failure.  VOT uses 5.
            eao_low:           Start of temporal interval for EAO averaging
                (1-indexed frame index).  VOT2015 default: 100.
            eao_high:          End of temporal interval (inclusive, 1-indexed).
                VOT2015 default: 356.
        """
        self.failure_threshold = failure_threshold
        self.penalty_frames = penalty_frames
        self.eao_low = eao_low
        self.eao_high = eao_high

    # ------------------------------------------------------------------ #
    # Core per-sequence operations                                         #
    # ------------------------------------------------------------------ #

    def apply_penalty(self, ious: np.ndarray) -> np.ndarray:
        """Apply VOT failure-penalty windows to a per-frame IoU array.

        After each failure frame, the next ``penalty_frames`` IoU values are
        set to 0.0.  This represents the mandatory pause before the tracker
        is re-initialized.  The failure frame itself retains its original
        value (it triggered the penalty, but its IoU still counts toward the
        overlap curve).

        Args:
            ious: Per-frame IoU array, shape ``(N,)``.

        Returns:
            Modified copy, dtype ``float64``, same shape as input.
        """
        arr = np.asarray(ious, dtype=np.float64).copy()
        n = len(arr)
        i = 0
        while i < n:
            if arr[i] < self.failure_threshold:
                end = min(i + 1 + self.penalty_frames, n)
                arr[i + 1 : end] = 0.0
                i = end
            else:
                i += 1
        return arr

    def _count_failures(self, ious: np.ndarray) -> int:
        """Count the number of distinct failure events in a raw IoU array."""
        arr = np.asarray(ious, dtype=np.float64)
        if len(arr) == 0:
            return 0
        below = arr < self.failure_threshold
        transitions = np.diff(below.astype(np.int8), prepend=0)
        return int((transitions == 1).sum())

    # ------------------------------------------------------------------ #
    # Overlap curve construction                                           #
    # ------------------------------------------------------------------ #

    def overlap_curve_for_sequences(
        self, sequences: List[np.ndarray]
    ) -> np.ndarray:
        """Build the expected-overlap curve from penalized IoU sequences.

        For each frame index k (0-indexed, up to ``eao_high - 1``), the
        expected overlap at that index is the mean IoU over all sequences
        that have at least ``k + 1`` frames.  Shorter sequences do not
        contribute to positions beyond their length.

        Args:
            sequences: Penalized IoU arrays, typically the output of
                :meth:`apply_penalty` applied to each sequence.

        Returns:
            Float array of shape ``(eao_high,)``.  Positions with no
            contributing sequences remain 0.0.
        """
        curve = np.zeros(self.eao_high, dtype=np.float64)
        counts = np.zeros(self.eao_high, dtype=np.int64)

        for ious in sequences:
            n = min(len(ious), self.eao_high)
            curve[:n] += ious[:n]
            counts[:n] += 1

        valid = counts > 0
        curve[valid] /= counts[valid]
        return curve

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def compute(
        self,
        sequence_ious: Dict[str, np.ndarray],
        tracker_name: str = "",
    ) -> VotEaoResult:
        """Compute VOT-style EAO for one tracker over a benchmark run.

        Args:
            sequence_ious: ``{sequence_name: per_frame_ious}`` mapping.
                Each value is a float array of shape ``(N,)`` where N is
                the number of evaluated frames.
            tracker_name:  Identifier stored in the result.

        Returns:
            :class:`VotEaoResult` with the EAO scalar, overlap curve, and
            failure statistics.
        """
        penalized: List[np.ndarray] = []
        total_failures = 0

        for ious in sequence_ious.values():
            arr = np.asarray(ious, dtype=np.float64)
            total_failures += self._count_failures(arr)
            penalized.append(self.apply_penalty(arr))

        curve = self.overlap_curve_for_sequences(penalized)

        # Convert 1-indexed [eao_low, eao_high] to 0-indexed slice
        lo = max(0, self.eao_low - 1)
        hi = min(self.eao_high, len(curve))
        eao_window = curve[lo:hi]
        eao = float(eao_window.mean()) if len(eao_window) > 0 else 0.0

        n = len(sequence_ious)
        return VotEaoResult(
            tracker_name=tracker_name,
            eao=eao,
            sequence_count=n,
            total_failures=total_failures,
            mean_failures_per_seq=total_failures / n if n > 0 else 0.0,
            overlap_curve=curve,
            eao_low=self.eao_low,
            eao_high=self.eao_high,
        )

    def compare(
        self,
        tracker_ious: Dict[str, Dict[str, np.ndarray]],
    ) -> List[VotEaoResult]:
        """Compute and rank EAO for multiple trackers simultaneously.

        Args:
            tracker_ious: ``{tracker_name: {seq_name: ious}}`` nested dict.

        Returns:
            List of :class:`VotEaoResult`, sorted by EAO descending
            (best tracker first).
        """
        results = [
            self.compute(seq_ious, tracker_name=name)
            for name, seq_ious in tracker_ious.items()
        ]
        results.sort(key=lambda r: r.eao, reverse=True)
        return results
