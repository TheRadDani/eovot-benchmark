"""Extended tracking metrics aligned with the GOT-10k and VOT evaluation protocols.

Adds metrics beyond the standard OTB success/precision curves:

- :func:`success_at_threshold` — success rate (SR) at a single IoU threshold.
  GOT-10k reports SR_0.5 and SR_0.75 alongside the mean Average Overlap (AO).
- :func:`robustness_rate` — fraction of frames where the tracker has not lost
  the target (IoU ≥ failure_threshold), as used by the VOT challenge.
- :func:`eao_score` — simplified Expected Average Overlap (EAO), a single
  scalar that combines accuracy and robustness in one measure.
- :class:`ExtendedMetrics` — dataclass bundling all scalar summaries.
- :class:`ExtendedMetricsEngine` — drop-in replacement for
  :class:`~eovot.metrics.accuracy.MetricsEngine` that computes all metrics
  in one pass.

References:
    Kristan et al., "The Seventh Visual Object Tracking VOT2019 Challenge Results."
    ICCV Workshops, 2019.

    Huang et al., "GOT-10k: A Large High-Diversity Benchmark for Generic
    Object Tracking in the Wild." IEEE TPAMI, 2021.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .accuracy import MetricsEngine, AccuracyMetrics

# Re-export the BBox type alias for convenience.
from .accuracy import BBox  # noqa: F401


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------


def success_at_threshold(ious: np.ndarray, threshold: float) -> float:
    """Compute the success rate at a single IoU threshold.

    Args:
        ious: Per-frame IoU values, shape ``(N,)``.
        threshold: IoU threshold in ``[0, 1]``.  Frames with IoU
            **strictly greater** than this value count as successes,
            matching the OTB/GOT-10k convention.

    Returns:
        Fraction of frames with ``IoU > threshold``, in ``[0, 1]``.
        Returns ``0.0`` for an empty array.
    """
    if len(ious) == 0:
        return 0.0
    return float((ious > threshold).mean())


def robustness_rate(ious: np.ndarray, failure_threshold: float = 0.1) -> float:
    """Compute the robustness rate — fraction of non-failure frames.

    A frame is considered a *failure* when the IoU falls below
    *failure_threshold*.  This metric captures the tracker's long-sequence
    stability independently of its localisation accuracy.

    Args:
        ious: Per-frame IoU values, shape ``(N,)``.
        failure_threshold: IoU value at or above which a frame is considered
            a *success*.  Default: ``0.1`` (standard VOT failure definition).

    Returns:
        Fraction of frames with ``IoU ≥ failure_threshold``, in ``[0, 1]``.
        Returns ``0.0`` for an empty array.
    """
    if len(ious) == 0:
        return 0.0
    return float((ious >= failure_threshold).mean())


def eao_score(
    ious: np.ndarray,
    min_len: int = 10,
    max_len: int = 100,
) -> float:
    """Compute a simplified Expected Average Overlap (EAO).

    EAO estimates the average performance of a tracker on sequences of
    lengths in ``[min_len, max_len]`` by computing the cumulative temporal
    mean overlap at each frame position and averaging over the evaluation
    window.

    This approximation replaces the full VOT bootstrapped Monte Carlo
    estimate with a deterministic temporal profile, making it cheap to
    compute online and suitable for per-sequence summaries.

    Algorithm:

    1. Build the temporal success profile:
       ``profile[t] = mean(ious[0 : t+1])`` for ``t = 0 … N-1``.
    2. Average ``profile[min_len-1 : max_len]`` to get the EAO scalar.

    Args:
        ious: Per-frame IoU values for the sequence(s), shape ``(N,)``.
        min_len: Start of the evaluation window (inclusive). Default: ``10``.
        max_len: End of the evaluation window (inclusive). Default: ``100``.

    Returns:
        EAO scalar in ``[0, 1]``.  Higher is better.
        Returns ``0.0`` for an empty array or when the window is invalid.

    References:
        Kristan et al., "The VOT2015 Challenge Results." ICCV Workshops, 2015.
    """
    n = len(ious)
    if n == 0:
        return 0.0

    # Clamp window to available frames.
    actual_max = min(max_len, n)
    actual_min = min(min_len, actual_max)
    if actual_min >= actual_max:
        # Degenerate window — fall back to mean over available frames.
        return float(ious[:actual_max].mean()) if actual_max > 0 else 0.0

    # Cumulative mean: profile[t] = mean IoU over frames 0 … t.
    temporal_profile = np.cumsum(ious) / np.arange(1, n + 1)
    window = temporal_profile[actual_min - 1 : actual_max]
    return float(window.mean()) if len(window) > 0 else 0.0


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExtendedMetrics:
    """Scalar accuracy summary including GOT-10k and VOT-style metrics.

    Attributes:
        mean_iou: Mean IoU (Average Overlap, AO) across all evaluated frames.
        success_auc: Area under the standard success curve (IoU thresholds 0→1).
        precision_auc: Normalised AUC of the precision curve (0→50 px).
        sr_0_5: Success rate at IoU threshold 0.5 (GOT-10k SR₀.₅).
        sr_0_75: Success rate at IoU threshold 0.75 (GOT-10k SR₀.₇₅).
        robustness: Fraction of frames with IoU ≥ 0.1 (VOT robustness rate).
        eao: Simplified Expected Average Overlap over the default window.
    """

    mean_iou: float
    success_auc: float
    precision_auc: float
    sr_0_5: float
    sr_0_75: float
    robustness: float
    eao: float

    def __str__(self) -> str:
        return (
            f"ExtendedMetrics("
            f"AO={self.mean_iou:.4f}, "
            f"AUC={self.success_auc:.4f}, "
            f"SR_0.5={self.sr_0_5:.4f}, "
            f"SR_0.75={self.sr_0_75:.4f}, "
            f"robustness={self.robustness:.4f}, "
            f"EAO={self.eao:.4f})"
        )

    def to_dict(self) -> Dict[str, float]:
        """Return a plain dict suitable for JSON serialisation."""
        return {
            "mean_iou": round(self.mean_iou, 4),
            "success_auc": round(self.success_auc, 4),
            "precision_auc": round(self.precision_auc, 4),
            "sr_0_5": round(self.sr_0_5, 4),
            "sr_0_75": round(self.sr_0_75, 4),
            "robustness": round(self.robustness, 4),
            "eao": round(self.eao, 4),
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ExtendedMetricsEngine(MetricsEngine):
    """Metrics engine with GOT-10k and VOT-style extended evaluation.

    Inherits all standard methods from
    :class:`~eovot.metrics.accuracy.MetricsEngine` (IoU, success curve,
    precision curve, :meth:`compute_all`) and adds one new method:
    :meth:`compute_extended`.

    Example::

        engine = ExtendedMetricsEngine()
        result = engine.compute_extended(preds, gts)
        print(f"AO={result.mean_iou:.3f}  SR_0.5={result.sr_0_5:.3f}  EAO={result.eao:.3f}")
    """

    def compute_extended(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
        failure_threshold: float = 0.1,
        eao_min_len: int = 10,
        eao_max_len: int = 100,
    ) -> ExtendedMetrics:
        """Compute all extended metrics in a single pass.

        Args:
            preds: ``(N, 4)`` predicted boxes in ``(x, y, w, h)`` format.
            gts:   ``(N, 4)`` ground-truth boxes in ``(x, y, w, h)`` format.
            failure_threshold: IoU below which a frame counts as a failure.
                Default: ``0.1``.
            eao_min_len: Minimum sequence length for the EAO evaluation
                window. Default: ``10``.
            eao_max_len: Maximum sequence length for the EAO evaluation
                window. Default: ``100``.

        Returns:
            :class:`ExtendedMetrics` with all scalar summaries populated.
        """
        base: AccuracyMetrics = self.compute_all(preds, gts)
        ious = self.batch_iou(preds, gts)

        return ExtendedMetrics(
            mean_iou=base.mean_iou,
            success_auc=base.success_auc,
            precision_auc=base.precision_auc,
            sr_0_5=success_at_threshold(ious, 0.5),
            sr_0_75=success_at_threshold(ious, 0.75),
            robustness=robustness_rate(ious, failure_threshold),
            eao=eao_score(ious, eao_min_len, eao_max_len),
        )
