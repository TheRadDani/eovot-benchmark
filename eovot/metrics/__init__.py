"""Metrics sub-package — accuracy, VOT protocol, and efficiency evaluation."""

from .accuracy import (
    iou,
    center_distance,
    AccuracyMetrics,
    MetricsEngine,
)
from .vot_metrics import (
    VOTEvaluator,
    VOTMetrics,
    VOTSequenceResult,
    simulate_reinit_overlaps,
    compute_eao,
)

__all__ = [
    # Accuracy
    "iou",
    "center_distance",
    "AccuracyMetrics",
    "MetricsEngine",
    # VOT protocol
    "VOTEvaluator",
    "VOTMetrics",
    "VOTSequenceResult",
    "simulate_reinit_overlaps",
    "compute_eao",
]
