"""Metrics sub-package — accuracy and efficiency evaluation."""

from .accuracy import (
    iou,
    center_distance,
    AccuracyMetrics,
    MetricsEngine,
)
from .statistical import (
    BootstrapCI,
    PairwiseComparison,
    TrackerRanking,
    bootstrap_ci,
    cohens_d,
    vargha_delaney_a12,
    wilcoxon_test,
    compare_trackers,
    rank_trackers,
)

__all__ = [
    "iou",
    "center_distance",
    "AccuracyMetrics",
    "MetricsEngine",
    "BootstrapCI",
    "PairwiseComparison",
    "TrackerRanking",
    "bootstrap_ci",
    "cohens_d",
    "vargha_delaney_a12",
    "wilcoxon_test",
    "compare_trackers",
    "rank_trackers",
]
