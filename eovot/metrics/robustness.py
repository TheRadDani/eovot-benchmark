"""VOT robustness metrics for EOVOT.

Implements the robustness evaluation protocol used by the VOT challenge:

- **Failure Rate** — proportion of frames where the tracker is considered
  "lost" (IoU < threshold).  Lower is better.
- **Robustness Score** — 1 − failure_rate, bounded in [0, 1].  Higher is
  better.
- **Expected Average Overlap (EAO)** — the standard VOT scalar that
  measures average overlap on fixed-length sub-sequences, averaging across
  all sequence lengths in [min_len, max_len].  Unlike raw mean-IoU, EAO
  weights short and long sequences equally, penalising trackers that fail
  early and never recover.

These metrics complement the accuracy metrics (mIoU, success AUC,
precision AUC) defined in :mod:`eovot.metrics.accuracy` and are
especially important for edge deployment where brief failure modes may
be safety-critical.

References
----------
Kristan et al., "The Visual Object Tracking VOT2016 Challenge Results."
ECCV Workshop, 2016. — EAO definition and evaluation protocol.

Example
-------
::

    from eovot.metrics.robustness import compute_robustness_metrics
    import numpy as np

    ious = np.array([0.8, 0.75, 0.05, 0.0, 0.0, 0.72, 0.68])
    metrics = compute_robustness_metrics(ious)
    print(metrics)
    # RobustnessMetrics(failure_rate=0.286, robustness=0.714, eao=0.414, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RobustnessMetrics:
    """Scalar robustness summary for a tracker on a sequence or dataset.

    Attributes
    ----------
    failure_rate:
        Fraction of frames where IoU < ``threshold`` (tracker considered
        lost).  Range [0, 1]; lower is better.
    robustness:
        ``1 - failure_rate``.  Range [0, 1]; higher is better.
    eao:
        Expected Average Overlap — mean overlap over sub-sequences of
        varying lengths in ``[min_len, max_len]``.  Range [0, 1]; higher
        is better.
    mean_iou_active:
        Mean IoU restricted to frames where the tracker is *not* lost
        (IoU ≥ ``threshold``).  ``None`` when every frame is a failure.
    threshold:
        The IoU threshold used to define tracker failure.
    """

    failure_rate: float
    robustness: float
    eao: float
    mean_iou_active: Optional[float]
    threshold: float

    def __str__(self) -> str:
        active = f"{self.mean_iou_active:.4f}" if self.mean_iou_active is not None else "N/A"
        return (
            f"RobustnessMetrics("
            f"failure_rate={self.failure_rate:.4f}, "
            f"robustness={self.robustness:.4f}, "
            f"eao={self.eao:.4f}, "
            f"mean_iou_active={active}, "
            f"threshold={self.threshold})"
        )

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        return {
            "failure_rate": round(self.failure_rate, 4),
            "robustness": round(self.robustness, 4),
            "eao": round(self.eao, 4),
            "mean_iou_active": round(self.mean_iou_active, 4) if self.mean_iou_active is not None else None,
            "threshold": self.threshold,
        }


def compute_failure_rate(ious: np.ndarray, threshold: float = 0.1) -> float:
    """Fraction of frames where IoU < *threshold* (tracker lost).

    Args:
        ious:      Per-frame IoU values, shape ``(N,)``.
        threshold: IoU below which the tracker is considered to have
                   failed.  VOT default is ``0.1``.

    Returns:
        Failure rate in ``[0, 1]``.  Returns ``0.0`` for empty arrays.
    """
    if len(ious) == 0:
        return 0.0
    return float((ious < threshold).mean())


def compute_eao(
    ious: np.ndarray,
    min_len: int = 10,
    max_len: int = 100,
) -> float:
    """Expected Average Overlap (simplified VOT protocol).

    Computes the mean IoU over all contiguous sub-sequences of lengths in
    ``[min_len, max_len]`` and averages across lengths.  This gives equal
    weight to performance at different temporal scales and penalises
    trackers that fail early.

    When the sequence is shorter than ``min_len``, the full sequence mean
    is returned as a fallback.

    Args:
        ious:    Per-frame IoU values, shape ``(N,)``.
        min_len: Minimum sub-sequence length (inclusive).
        max_len: Maximum sub-sequence length (inclusive).

    Returns:
        EAO score in ``[0, 1]``.  Returns ``0.0`` for empty arrays.
    """
    n = len(ious)
    if n == 0:
        return 0.0

    # Fallback: sequence shorter than min_len
    if n < min_len:
        return float(ious.mean())

    actual_max = min(max_len, n)
    per_length_means: list[float] = []

    for length in range(min_len, actual_max + 1):
        # Sliding window: collect mean IoU for every sub-sequence of this length.
        windows = np.lib.stride_tricks.sliding_window_view(ious, length)
        per_length_means.append(float(windows.mean(axis=1).mean()))

    return float(np.mean(per_length_means))


def compute_robustness_metrics(
    ious: np.ndarray,
    threshold: float = 0.1,
    min_len: int = 10,
    max_len: int = 100,
) -> RobustnessMetrics:
    """Compute all VOT robustness metrics from a per-frame IoU array.

    Args:
        ious:      Per-frame IoU values, shape ``(N,)``.
        threshold: IoU below which the tracker is considered lost.
        min_len:   Minimum sub-sequence length for EAO computation.
        max_len:   Maximum sub-sequence length for EAO computation.

    Returns:
        :class:`RobustnessMetrics` with all scalar summaries populated.

    Example::

        import numpy as np
        from eovot.metrics.robustness import compute_robustness_metrics

        ious = np.array([0.8, 0.75, 0.05, 0.0, 0.72, 0.68])
        m = compute_robustness_metrics(ious)
        print(m.failure_rate)   # 0.333...
        print(m.eao)            # ~0.5
    """
    failure_rate = compute_failure_rate(ious, threshold)
    eao = compute_eao(ious, min_len=min_len, max_len=max_len)

    active_mask = ious >= threshold
    mean_iou_active: Optional[float] = (
        float(ious[active_mask].mean()) if active_mask.any() else None
    )

    return RobustnessMetrics(
        failure_rate=failure_rate,
        robustness=1.0 - failure_rate,
        eao=eao,
        mean_iou_active=mean_iou_active,
        threshold=threshold,
    )
