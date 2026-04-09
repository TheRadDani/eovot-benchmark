"""Metrics sub-package — accuracy and efficiency evaluation."""

from .accuracy import (
    iou,
    center_distance,
    AccuracyMetrics,
    MetricsEngine,
)
from .vot_metrics import (
    VOTResult,
    VOTMetricsEngine,
    compute_accuracy,
    compute_robustness,
    compute_eao,
    detect_failures,
    extract_subsequences,
)

__all__ = [
    # accuracy.py
    "iou",
    "center_distance",
    "AccuracyMetrics",
    "MetricsEngine",
    # vot_metrics.py
    "VOTResult",
    "VOTMetricsEngine",
    "compute_accuracy",
    "compute_robustness",
    "compute_eao",
    "detect_failures",
    "extract_subsequences",
]
