"""VOT challenge evaluation protocol: EAO, Robustness, and Accuracy.

Implements the standard Visual Object Tracking (VOT) challenge evaluation
methodology used by VOT2016–VOT2022.  This is the de-facto research standard
for comparing single-object trackers and is required for publishable results.

Protocol summary
----------------
1. A sequence is evaluated from the first frame.
2. When the tracker's predicted IoU drops to zero (a *failure*), it is
   re-initialised ``reinit_delay`` frames later.
3. **Accuracy** — mean overlap during successfully tracked frames
   (failure + reinit frames excluded).
4. **Robustness** — mean failure rate normalised by sequence length.
5. **EAO** (Expected Average Overlap) — the expected no-reset overlap
   averaged over a canonical range of sequence lengths, combining both
   accuracy and robustness into a single comparable scalar.

Reference
---------
Kristan et al., "The Visual Object Tracking VOT2016 Challenge Results",
ECCV Workshops, 2016.  https://doi.org/10.1007/978-3-319-48881-3_54
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Overlap threshold below which a tracking failure is declared
FAILURE_THRESHOLD: float = 0.0

# Frames to skip (and zero-out) after a failure before re-initialisation
REINIT_DELAY: int = 5

# Canonical sequence-length range used by the VOT challenge for EAO
EAO_LOW: int = 100
EAO_HIGH: int = 356


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class VOTSequenceResult:
    """VOT evaluation result for a single tracking sequence.

    Attributes:
        sequence_name: Identifier of the evaluated sequence.
        overlaps: Per-frame IoU after applying the re-initialisation protocol.
            Frames inside a re-init window are set to 0.
        failures: Number of tracking failures detected in this sequence.
        accuracy: Mean overlap during tracked frames (excl. failure windows).
    """

    sequence_name: str
    overlaps: List[float]
    failures: int
    accuracy: float


@dataclass
class VOTMetrics:
    """Aggregate VOT metrics across a set of sequences.

    Attributes:
        accuracy: Mean per-sequence accuracy (overlap during tracked frames).
        robustness: Mean failure rate (failures / sequence_length per seq).
        eao: Expected Average Overlap — canonical single-number comparison.
        sequence_results: Per-sequence breakdown used to compute the aggregate.
    """

    accuracy: float
    robustness: float
    eao: float
    sequence_results: List[VOTSequenceResult] = field(default_factory=list)

    def summary(self) -> dict:
        """Return a plain-dict summary suitable for JSON export."""
        return {
            "accuracy": round(self.accuracy, 4),
            "robustness": round(self.robustness, 4),
            "eao": round(self.eao, 4),
            "n_sequences": len(self.sequence_results),
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"VOTMetrics  EAO={s['eao']}  "
            f"Accuracy={s['accuracy']}  "
            f"Robustness={s['robustness']}  "
            f"({s['n_sequences']} sequences)"
        )


# ---------------------------------------------------------------------------
# Core protocol functions
# ---------------------------------------------------------------------------


def simulate_reinit_overlaps(
    raw_overlaps: List[float],
    failure_threshold: float = FAILURE_THRESHOLD,
    reinit_delay: int = REINIT_DELAY,
) -> Tuple[List[float], int]:
    """Apply the VOT re-initialisation protocol to a raw overlap sequence.

    When a failure is detected (IoU <= ``failure_threshold``) the tracker is
    conceptually re-initialised after ``reinit_delay`` frames.  The failure
    frame and the delay frames are set to 0 in the returned sequence.

    Args:
        raw_overlaps: Raw per-frame IoU values ``[0, 1]``.
        failure_threshold: IoU at or below which a failure is declared.
        reinit_delay: Number of frames (including the failure frame itself)
            zeroed out before re-initialisation.

    Returns:
        ``(modified_overlaps, n_failures)`` where *modified_overlaps* has
        failure windows zeroed and *n_failures* is the count of detected
        failures.
    """
    overlaps = list(raw_overlaps)
    n_failures = 0
    i = 0
    while i < len(overlaps):
        if overlaps[i] <= failure_threshold:
            n_failures += 1
            end = min(i + reinit_delay + 1, len(overlaps))
            for j in range(i, end):
                overlaps[j] = 0.0
            i = end
        else:
            i += 1
    return overlaps, n_failures


def compute_sequence_accuracy(
    overlaps: List[float],
    failure_threshold: float = FAILURE_THRESHOLD,
) -> float:
    """Mean overlap over frames that are not inside a failure/reinit window.

    Frames with overlap == ``failure_threshold`` (i.e. the zeroed-out windows
    produced by :func:`simulate_reinit_overlaps`) are excluded from the mean.

    Args:
        overlaps: Per-frame overlap after the re-init protocol has been applied.
        failure_threshold: Value used to identify excluded frames.

    Returns:
        Mean overlap in ``[0, 1]``, or 0 if all frames are excluded.
    """
    tracked = [v for v in overlaps if v > failure_threshold]
    return float(np.mean(tracked)) if tracked else 0.0


def compute_eao(
    sequence_overlaps: List[List[float]],
    low: int = EAO_LOW,
    high: int = EAO_HIGH,
) -> float:
    """Compute Expected Average Overlap (EAO).

    EAO measures the expected no-reset overlap averaged over a range of
    sequence lengths ``[low, high]``.  Short sequences are padded by repeating
    their last overlap value.

    Args:
        sequence_overlaps: Per-sequence overlap arrays *after* the re-init
            protocol has been applied (output of :func:`simulate_reinit_overlaps`).
        low: Lower bound of the sequence-length range.  Default: 100.
        high: Upper bound of the sequence-length range.  Default: 356.

    Returns:
        EAO score in ``[0, 1]`` — higher is better.  Returns 0 if the input
        is empty or all sequences are shorter than *low*.
    """
    if not sequence_overlaps:
        return 0.0

    max_len = max((len(s) for s in sequence_overlaps), default=0)
    if max_len == 0:
        return 0.0

    # Clamp range to available data
    effective_high = min(high, max_len)
    effective_low = min(low, effective_high)

    if effective_low >= effective_high:
        effective_low = max(1, effective_high // 2)

    # Pad each sequence to length `effective_high` by repeating the last value
    padded: List[List[float]] = []
    for seq in sequence_overlaps:
        if not seq:
            padded.append([0.0] * effective_high)
        else:
            pad_value = seq[-1]
            extension = [pad_value] * max(0, effective_high - len(seq))
            padded.append(list(seq) + extension)

    # Average over sequence lengths in [low, high]
    eao_curve = [
        float(np.mean([np.mean(seq[:n]) for seq in padded]))
        for n in range(effective_low, effective_high + 1)
    ]

    return float(np.mean(eao_curve)) if eao_curve else 0.0


# ---------------------------------------------------------------------------
# High-level evaluator
# ---------------------------------------------------------------------------


class VOTEvaluator:
    """Evaluate tracker results under the VOT re-initialisation protocol.

    Computes per-sequence accuracy and failure count, then aggregates to
    tracker-level Accuracy, Robustness, and EAO metrics.

    Args:
        failure_threshold: IoU at or below which a failure is declared.
            Default: ``0.0`` (only complete misses counted, as in VOT).
        reinit_delay: Frames zeroed after a failure before re-init.
            Default: ``5`` (VOT standard).
        eao_low: Lower sequence-length bound for EAO computation.
            Default: ``100``.
        eao_high: Upper sequence-length bound for EAO computation.
            Default: ``356``.

    Example::

        evaluator = VOTEvaluator()
        vot = evaluator.evaluate(
            sequence_names=["seq1", "seq2"],
            sequence_overlaps=[[0.8, 0.7, 0.0, ...], [0.6, 0.5, ...]],
        )
        print(vot)
        # VOTMetrics  EAO=0.3120  Accuracy=0.6840  Robustness=0.0123  (2 sequences)
    """

    def __init__(
        self,
        failure_threshold: float = FAILURE_THRESHOLD,
        reinit_delay: int = REINIT_DELAY,
        eao_low: int = EAO_LOW,
        eao_high: int = EAO_HIGH,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reinit_delay = reinit_delay
        self.eao_low = eao_low
        self.eao_high = eao_high

    def evaluate_sequence(
        self,
        sequence_name: str,
        raw_overlaps: List[float],
    ) -> VOTSequenceResult:
        """Evaluate a single sequence under the VOT protocol.

        Args:
            sequence_name: Identifier embedded in the result.
            raw_overlaps: Raw per-frame IoU values from the tracker.

        Returns:
            :class:`VOTSequenceResult` with protocol-adjusted overlaps,
            failure count, and accuracy.
        """
        sim_overlaps, n_failures = simulate_reinit_overlaps(
            raw_overlaps,
            self.failure_threshold,
            self.reinit_delay,
        )
        accuracy = compute_sequence_accuracy(sim_overlaps, self.failure_threshold)
        return VOTSequenceResult(
            sequence_name=sequence_name,
            overlaps=sim_overlaps,
            failures=n_failures,
            accuracy=accuracy,
        )

    def evaluate(
        self,
        sequence_names: List[str],
        sequence_overlaps: List[List[float]],
    ) -> VOTMetrics:
        """Evaluate all sequences and compute aggregate VOT metrics.

        Args:
            sequence_names: List of sequence identifiers (one per sequence).
            sequence_overlaps: Per-sequence raw IoU arrays (one per sequence).

        Returns:
            :class:`VOTMetrics` with accuracy, robustness, and EAO populated.

        Raises:
            ValueError: If *sequence_names* and *sequence_overlaps* differ in
                length, or if either is empty.
        """
        if len(sequence_names) != len(sequence_overlaps):
            raise ValueError(
                f"sequence_names length ({len(sequence_names)}) must match "
                f"sequence_overlaps length ({len(sequence_overlaps)})"
            )
        if not sequence_names:
            raise ValueError("At least one sequence is required for evaluation.")

        seq_results: List[VOTSequenceResult] = []
        all_sim_overlaps: List[List[float]] = []

        for name, raw in zip(sequence_names, sequence_overlaps):
            result = self.evaluate_sequence(name, raw)
            seq_results.append(result)
            all_sim_overlaps.append(result.overlaps)

        mean_accuracy = float(np.mean([r.accuracy for r in seq_results]))

        # Robustness: failures normalised by sequence length
        failure_rates = [
            r.failures / max(len(raw), 1)
            for r, raw in zip(seq_results, sequence_overlaps)
        ]
        mean_robustness = float(np.mean(failure_rates))

        eao = compute_eao(all_sim_overlaps, self.eao_low, self.eao_high)

        return VOTMetrics(
            accuracy=mean_accuracy,
            robustness=mean_robustness,
            eao=eao,
            sequence_results=seq_results,
        )
