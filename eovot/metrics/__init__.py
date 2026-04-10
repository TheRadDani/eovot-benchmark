"""Metrics sub-package — accuracy and efficiency evaluation."""

from .accuracy import (
    iou,
    center_distance,
    AccuracyMetrics,
    MetricsEngine,
)
from .robustness import (
    RobustnessMetrics,
    compute_robustness,
    normalized_precision_curve,
    success_rate_at_threshold,
)

__all__ = [
    # Accuracy
    "iou",
    "center_distance",
    "AccuracyMetrics",
    "MetricsEngine",
    # Robustness
    "RobustnessMetrics",
    "compute_robustness",
    "normalized_precision_curve",
    "success_rate_at_threshold",
]
