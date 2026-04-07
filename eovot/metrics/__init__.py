"""Metrics sub-package — accuracy, VOT protocol, and efficiency evaluation."""

from .accuracy import (
    iou,
    center_distance,
    AccuracyMetrics,
    MetricsEngine,
)
from .efficiency import (
    EdgeEfficiencyScorer,
    EfficiencyResult,
    score_from_summary,
)

__all__ = [
    # Accuracy
    "iou",
    "center_distance",
    "AccuracyMetrics",
    "MetricsEngine",
    # Edge efficiency
    "EdgeEfficiencyScorer",
    "EfficiencyResult",
    "score_from_summary",
]
