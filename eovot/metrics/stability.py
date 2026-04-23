"""IoU stability metrics for edge-deployment tracking evaluation.

Standard VOT metrics (success AUC, precision AUC) summarise *average*
accuracy but hide temporal behaviour that is critical for edge deployment:

- A tracker that oscillates wildly between IoU 0.9 and 0.1 has the same
  mean IoU as one that holds steady at 0.5, yet the first is far less
  reliable in a real system.
- On resource-constrained devices, tracker *recovery* after a drift event
  matters as much as peak accuracy.

This module provides lightweight, numpy-only metrics that characterise the
*temporal stability* of per-frame IoU sequences.

Typical usage::

    from eovot.metrics.stability import compute_stability

    stability = compute_stability(sr.ious)
    print(stability)            # IoUStabilityMetrics(...)
    print(stability.to_dict())  # ready for JSON export
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IoUStabilityMetrics:
    """Temporal stability summary for a single tracking sequence.

    Attributes:
        iou_std: Standard deviation of per-frame IoU.  Lower is more stable.
        iou_range: Max IoU minus min IoU across the sequence.
        failure_rate: Fraction of frames with IoU below *failure_threshold*
            (default 0.1, matching the VOT re-init criterion).
        longest_failure_run: Length of the longest consecutive failure streak.
        recovery_rate: Fraction of failure streaks that end with the tracker
            recovering (next frame IoU >= *recovery_threshold*, default 0.5).
            ``nan`` when there are no failures.
        temporal_autocorr: Lag-1 autocorrelation of the IoU series — values
            near 1 indicate smooth, consistent tracking; values near 0 indicate
            frame-to-frame jitter.
    """

    iou_std: float
    iou_range: float
    failure_rate: float
    longest_failure_run: int
    recovery_rate: float
    temporal_autocorr: float

    def __str__(self) -> str:
        rec = f"{self.recovery_rate:.3f}" if not np.isnan(self.recovery_rate) else "n/a"
        return (
            f"IoUStabilityMetrics("
            f"std={self.iou_std:.4f}, "
            f"range={self.iou_range:.4f}, "
            f"failure_rate={self.failure_rate:.4f}, "
            f"longest_run={self.longest_failure_run}, "
            f"recovery={rec}, "
            f"autocorr={self.temporal_autocorr:.4f})"
        )

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        rec = self.recovery_rate if not np.isnan(self.recovery_rate) else None
        return {
            "iou_std": round(self.iou_std, 6),
            "iou_range": round(self.iou_range, 6),
            "failure_rate": round(self.failure_rate, 6),
            "longest_failure_run": self.longest_failure_run,
            "recovery_rate": round(rec, 4) if rec is not None else None,
            "temporal_autocorr": round(self.temporal_autocorr, 6),
        }


def compute_stability(
    ious: np.ndarray,
    failure_threshold: float = 0.1,
    recovery_threshold: float = 0.5,
) -> IoUStabilityMetrics:
    """Compute temporal stability metrics from a per-frame IoU sequence.

    Args:
        ious: ``(N,)`` array of per-frame IoU values in ``[0, 1]``.
        failure_threshold: IoU below this value is counted as a tracking
            failure.  Matches the VOT Challenge re-initialisation criterion
            (default ``0.1``).
        recovery_threshold: Minimum IoU for the frame immediately following
            a failure run to count as a successful recovery (default ``0.5``).

    Returns:
        :class:`IoUStabilityMetrics` with all stability scalars populated.

    Raises:
        ValueError: If *ious* is empty or contains values outside ``[0, 1]``.
    """
    ious = np.asarray(ious, dtype=np.float64)
    if len(ious) == 0:
        raise ValueError("ious array must not be empty.")
    if ious.min() < 0.0 or ious.max() > 1.0 + 1e-9:
        raise ValueError(f"IoU values must be in [0, 1], got range [{ious.min():.4f}, {ious.max():.4f}].")

    iou_std = float(ious.std())
    iou_range = float(ious.max() - ious.min())

    failures = ious < failure_threshold
    failure_rate = float(failures.mean())

    longest_run, recovery_rate = _failure_run_stats(ious, failures, recovery_threshold)

    temporal_autocorr = _lag1_autocorr(ious)

    return IoUStabilityMetrics(
        iou_std=iou_std,
        iou_range=iou_range,
        failure_rate=failure_rate,
        longest_failure_run=longest_run,
        recovery_rate=recovery_rate,
        temporal_autocorr=temporal_autocorr,
    )


def _failure_run_stats(
    ious: np.ndarray,
    failures: np.ndarray,
    recovery_threshold: float,
) -> tuple[int, float]:
    """Return (longest_failure_run, recovery_rate) from a failure mask."""
    n = len(failures)
    longest = 0
    current = 0
    streaks_total = 0
    streaks_recovered = 0
    in_failure = False

    for i in range(n):
        if failures[i]:
            current += 1
            if not in_failure:
                in_failure = True
        else:
            if in_failure:
                longest = max(longest, current)
                streaks_total += 1
                if ious[i] >= recovery_threshold:
                    streaks_recovered += 1
                current = 0
                in_failure = False

    if in_failure:
        longest = max(longest, current)

    recovery_rate = (
        float(streaks_recovered) / streaks_total if streaks_total > 0 else float("nan")
    )
    return longest, recovery_rate


def _lag1_autocorr(x: np.ndarray) -> float:
    """Pearson lag-1 autocorrelation of array *x*."""
    if len(x) < 2:
        return float("nan")
    mu = x.mean()
    demeaned = x - mu
    var = float((demeaned ** 2).mean())
    if var < 1e-12:
        return 1.0  # constant series is perfectly autocorrelated
    cov = float((demeaned[:-1] * demeaned[1:]).mean())
    return float(cov / var)
