"""Metrics sub-package — accuracy and efficiency evaluation."""

from .accuracy import (
    AccuracyMetrics,
    MetricsEngine,
    center_distance,
    iou,
)
from .edge_score import (
    HARDWARE_PROFILES,
    EdgeDeploymentScorer,
    EdgeScore,
    ParetoPoint,
    TrackerMetrics,
)

__all__ = [
    "iou",
    "center_distance",
    "AccuracyMetrics",
    "MetricsEngine",
    "EdgeDeploymentScorer",
    "EdgeScore",
    "ParetoPoint",
    "TrackerMetrics",
    "HARDWARE_PROFILES",
]
