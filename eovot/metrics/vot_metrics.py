"""VOT-standard evaluation metrics for visual object tracking.

Implements the Expected Average Overlap (EAO), Robustness, and
Accuracy-Robustness (AR) scores as defined in the VOT Challenge
evaluation protocol.

These metrics go beyond simple mean-IoU and success-AUC by explicitly
modelling tracker *failures* and *re-initializations*, which is critical
for edge deployment where a single dropped target can be catastrophic.

References:
    Kristan et al., "A Novel Performance Evaluation Methodology for
    Single-Target Trackers", IEEE TPAMI 2016.

    Kristan et al., "The Visual Object Tracking VOT2018 Challenge Results",
    ECCV Workshops 2018.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class VOTResult:
    """Container for a full VOT-standard evaluation result.

    Attributes:
        accuracy:        Mean IoU over successfully tracked frames (higher is better).
        robustness:      Failure rate expressed as failures per 100 frames (lower is better).
        eao:             Expected Average Overlap scalar (higher is better).
        failure_count:   Total number of tracking failures across all sequences.
        sequence_length: Total number of frames evaluated.
        eao_curve:       Per-length EAO values, shape ``(eao_max_length,)``.
    """

    accuracy: float
    robustness: float
    eao: float
    failure_count: int
    sequence_length: int
    eao_curve: np.ndarray = field(repr=False)

    def __str__(self) -> str:
        return (
            f"VOTResult("
            f"EAO={self.eao:.4f}, "
            f"accuracy={self.accuracy:.4f}, "
            f"robustness={self.robustness:.2f}/100fr, "
            f"failures={self.failure_count})"
        )


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def detect_failures(
    iou_sequence: np.ndarray,
    threshold: float = 0.0,
) -> List[int]:
    """Return the indices of frames where the tracker fails.

    A frame is considered a failure when its IoU falls at or below
    *threshold*.  The VOT protocol uses ``threshold=0`` (complete target
    loss), which is the default here.

    Args:
        iou_sequence: Per-frame IoU values, shape ``(N,)``.
        threshold:    IoU threshold; frames at or below this value are failures.

    Returns:
        List of zero-based frame indices where failures occur.
    """
    return [int(i) for i, v in enumerate(iou_sequence) if v <= threshold]


def extract_subsequences(
    iou_sequence: np.ndarray,
    burn_in: int = 10,
    failure_threshold: float = 0.0,
) -> List[np.ndarray]:
    """Split an IoU sequence into subsequences separated by re-initializations.

    Following the VOT re-initialization protocol: when a failure occurs at
    frame *f*, the tracker is re-initialized after ``burn_in`` frames (i.e.,
    frames ``f`` through ``f + burn_in`` are skipped), and a new subsequence
    begins at ``f + burn_in + 1``.

    Args:
        iou_sequence:      Per-frame IoU values, shape ``(N,)``.
        burn_in:           Frames to skip after each failure (VOT default: 10).
        failure_threshold: IoU threshold for failure detection.

    Returns:
        List of numpy arrays, each containing the IoU values of one
        uninterrupted tracking run.
    """
    arr = np.asarray(iou_sequence, dtype=float)
    subsequences: List[np.ndarray] = []
    current_start = 0
    n = len(arr)

    while current_start < n:
        subseq = arr[current_start:]
        # Find first failure inside this sub-view
        failures = [i for i, v in enumerate(subseq) if v <= failure_threshold]

        if not failures:
            if len(subseq) > 0:
                subsequences.append(subseq)
            break

        failure_local = failures[0]
        if failure_local > 0:
            subsequences.append(subseq[:failure_local])

        # Jump past the failure frame and the burn-in window
        current_start += failure_local + 1 + burn_in

    return subsequences


# ---------------------------------------------------------------------------
# Scalar metric functions
# ---------------------------------------------------------------------------


def compute_accuracy(
    iou_sequence: np.ndarray,
    failure_threshold: float = 0.0,
) -> float:
    """Compute VOT accuracy: mean IoU over non-failure frames.

    Failure frames (IoU ≤ threshold) are excluded from the average because
    they represent gaps between re-initializations, not genuine tracking.

    Args:
        iou_sequence:      Per-frame IoU values.
        failure_threshold: Frames at or below this value are excluded.

    Returns:
        Mean IoU over valid frames, or ``0.0`` if no valid frames exist.
    """
    arr = np.asarray(iou_sequence, dtype=float)
    valid = arr[arr > failure_threshold]
    return float(valid.mean()) if len(valid) > 0 else 0.0


def compute_robustness(
    iou_sequence: np.ndarray,
    burn_in: int = 10,
    failure_threshold: float = 0.0,
) -> Tuple[float, int]:
    """Compute VOT robustness: normalized failure rate.

    Robustness is expressed as the number of failures per 100 frames,
    allowing fair comparison across sequences of different lengths.

    Failures are counted directly by simulating the VOT re-initialization
    protocol: each time a failure frame is encountered during a tracking
    run, the counter increments and ``burn_in`` frames are skipped.

    Args:
        iou_sequence:      Per-frame IoU values.
        burn_in:           Frames burned in after each failure.
        failure_threshold: IoU threshold for failure detection.

    Returns:
        Tuple ``(robustness_score, failure_count)`` where *robustness_score*
        is in units of failures per 100 frames (lower is better).
    """
    arr = np.asarray(iou_sequence, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0

    failure_count = 0
    current_start = 0

    while current_start < n:
        subseq = arr[current_start:]
        failures = [i for i, v in enumerate(subseq) if v <= failure_threshold]
        if not failures:
            break
        # A failure occurred — count it and jump past the burn-in window
        failure_count += 1
        failure_local = failures[0]
        current_start += failure_local + 1 + burn_in

    robustness = (failure_count / n) * 100.0
    return float(robustness), int(failure_count)


def compute_eao(
    iou_sequences: List[np.ndarray],
    min_length: int = 100,
    max_length: int = 356,
    burn_in: int = 10,
    failure_threshold: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """Compute Expected Average Overlap (EAO) — the primary VOT metric.

    EAO captures both accuracy and robustness in a single scalar by
    computing the expected overlap over the distribution of sequence
    lengths typical of the benchmark.

    The EAO *curve* ``C`` is defined as::

        C[l] = mean overlap at position l, averaged over all subsequences
               of length ≥ l (after re-initialization).

    The scalar EAO is the mean of ``C`` over the integration window
    ``[min_length, max_length)``, which should match the "typical" length
    range of the benchmark (VOT2018 default: 100–356).

    Args:
        iou_sequences:    List of per-sequence IoU arrays.
        min_length:       Start of the EAO integration window (inclusive).
        max_length:       End of the EAO integration window (exclusive).
                          Also the length of the returned curve.
        burn_in:          Frames to skip after failures.
        failure_threshold: IoU threshold for failure detection.

    Returns:
        Tuple ``(eao_scalar, eao_curve)`` where *eao_curve* has shape
        ``(max_length,)``.
    """
    if not iou_sequences:
        return 0.0, np.zeros(max_length)

    # Collect all subsequences from all sequences
    all_subsequences: List[np.ndarray] = []
    for seq_iou in iou_sequences:
        subs = extract_subsequences(
            np.asarray(seq_iou, dtype=float),
            burn_in=burn_in,
            failure_threshold=failure_threshold,
        )
        all_subsequences.extend(subs)

    if not all_subsequences:
        return 0.0, np.zeros(max_length)

    # Accumulate overlap values and counts per position
    overlap_sum = np.zeros(max_length, dtype=float)
    counts = np.zeros(max_length, dtype=int)

    for subseq in all_subsequences:
        seq_len = min(len(subseq), max_length)
        overlap_sum[:seq_len] += subseq[:seq_len]
        counts[:seq_len] += 1

    # Normalize — positions with no data stay at 0 (no extrapolation)
    eao_curve = np.zeros(max_length, dtype=float)
    valid_mask = counts > 0
    eao_curve[valid_mask] = overlap_sum[valid_mask] / counts[valid_mask]

    # Scalar: mean over the integration window
    window = eao_curve[min_length:max_length]
    eao_scalar = float(window.mean()) if len(window) > 0 else 0.0

    return eao_scalar, eao_curve


# ---------------------------------------------------------------------------
# High-level engine
# ---------------------------------------------------------------------------


class VOTMetricsEngine:
    """VOT-standard metrics engine for single-target tracker evaluation.

    Computes Accuracy, Robustness, and Expected Average Overlap (EAO)
    following the VOT Challenge evaluation protocol.  Input is a list of
    per-sequence IoU arrays (as produced by :class:`~eovot.benchmark.engine.BenchmarkEngine`).

    Args:
        burn_in:           Frames to skip after each failure re-initialization
                           (VOT default: 10).
        failure_threshold: IoU at or below which a frame is classified as a
                           failure (VOT default: 0.0 — complete target loss).
        eao_min_length:    Start of EAO integration window (inclusive).
                           VOT2018 default: 100.
        eao_max_length:    End of EAO integration window (exclusive) and
                           length of the returned EAO curve.
                           VOT2018 default: 356.

    Example::

        engine = VOTMetricsEngine()
        result = engine.evaluate([seq.ious for seq in benchmark_result.sequence_results])
        print(result)  # EAO=0.2341, accuracy=0.5812, robustness=3.21/100fr
        ar = engine.ar_score(result)
    """

    def __init__(
        self,
        burn_in: int = 10,
        failure_threshold: float = 0.0,
        eao_min_length: int = 100,
        eao_max_length: int = 356,
    ) -> None:
        self.burn_in = burn_in
        self.failure_threshold = failure_threshold
        self.eao_min_length = eao_min_length
        self.eao_max_length = eao_max_length

    def evaluate(self, iou_sequences: List[np.ndarray]) -> VOTResult:
        """Run the full VOT evaluation on a collection of per-sequence IoU arrays.

        Args:
            iou_sequences: List of numpy arrays, each containing per-frame
                           IoU values for one tracking sequence.

        Returns:
            :class:`VOTResult` with accuracy, robustness, EAO, and EAO curve.
        """
        if not iou_sequences:
            return VOTResult(
                accuracy=0.0,
                robustness=0.0,
                eao=0.0,
                failure_count=0,
                sequence_length=0,
                eao_curve=np.zeros(self.eao_max_length),
            )

        # Accuracy: per-sequence, then averaged
        accuracies = [
            compute_accuracy(np.asarray(s, dtype=float), self.failure_threshold)
            for s in iou_sequences
        ]
        mean_accuracy = float(np.mean(accuracies))

        # Robustness: aggregate across all sequences
        total_failures = 0
        total_frames = 0
        for s in iou_sequences:
            arr = np.asarray(s, dtype=float)
            _, n_fail = compute_robustness(arr, self.burn_in, self.failure_threshold)
            total_failures += n_fail
            total_frames += len(arr)

        mean_robustness = (total_failures / max(total_frames, 1)) * 100.0

        # EAO
        eao_scalar, eao_curve = compute_eao(
            iou_sequences,
            min_length=self.eao_min_length,
            max_length=self.eao_max_length,
            burn_in=self.burn_in,
            failure_threshold=self.failure_threshold,
        )

        return VOTResult(
            accuracy=mean_accuracy,
            robustness=mean_robustness,
            eao=eao_scalar,
            failure_count=total_failures,
            sequence_length=total_frames,
            eao_curve=eao_curve,
        )

    def ar_score(self, vot_result: VOTResult) -> float:
        """Compute a combined Accuracy-Robustness (AR) score.

        The AR score is the harmonic mean of *accuracy* and *reliability*,
        where reliability = ``1 − robustness / 100``.  It penalises both
        low accuracy and high failure rates equally, making it a useful
        single-number summary for ranking trackers.

        Args:
            vot_result: Result from :meth:`evaluate`.

        Returns:
            AR score in ``[0, 1]``, higher is better.
        """
        # Clamp robustness (failures/100fr) to [0, 1] for combination
        robustness_norm = min(vot_result.robustness / 100.0, 1.0)
        reliability = 1.0 - robustness_norm
        denom = vot_result.accuracy + reliability

        if denom == 0.0:
            return 0.0

        return float(2.0 * vot_result.accuracy * reliability / denom)

    def compare(
        self,
        tracker_results: dict,
    ) -> dict:
        """Evaluate and compare multiple trackers.

        Args:
            tracker_results: Mapping of tracker name → list of per-sequence
                             IoU arrays.

        Returns:
            Mapping of tracker name → :class:`VOTResult`, sorted by EAO
            descending.
        """
        scores = {
            name: self.evaluate(seqs)
            for name, seqs in tracker_results.items()
        }
        return dict(sorted(scores.items(), key=lambda kv: kv[1].eao, reverse=True))
