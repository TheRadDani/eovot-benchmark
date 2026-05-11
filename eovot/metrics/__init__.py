"""Metrics sub-package — accuracy and efficiency evaluation."""

from .accuracy import (
    iou,
    center_distance,
    AccuracyMetrics,
    MetricsEngine,
)
from .robustness import RobustnessAnalyzer, RobustnessResult
from .statistical import (
    BootstrapCI,
    WilcoxonResult,
    StatisticalSummary,
    StatisticalComparison,
    bootstrap_ci,
    wilcoxon_signed_rank,
)

__all__ = [
    "iou",
    "center_distance",
    "AccuracyMetrics",
    "MetricsEngine",
    "RobustnessAnalyzer",
    "RobustnessResult",
    "BootstrapCI",
    "WilcoxonResult",
    "StatisticalSummary",
    "StatisticalComparison",
    "bootstrap_ci",
    "wilcoxon_signed_rank",
]
