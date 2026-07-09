"""Metrics sub-package — accuracy, robustness, efficiency, temporal consistency, and statistical testing."""

from .accuracy import (
    iou,
    center_distance,
    normalized_center_distance,
    AccuracyMetrics,
    MetricsEngine,
)
from .robustness import RobustnessAnalyzer, RobustnessResult
from .efficiency import EfficiencyEntry, EfficiencyMetricsEngine
from .temporal import TemporalConsistencyAnalyzer, TemporalConsistencyResult
from .statistical import (
    BootstrapCI,
    WilcoxonResult,
    PairwiseSummary,
    StatisticalTestEngine,
)

__all__ = [
    "iou",
    "center_distance",
    "normalized_center_distance",
    "AccuracyMetrics",
    "MetricsEngine",
    "RobustnessAnalyzer",
    "RobustnessResult",
    "EfficiencyEntry",
    "EfficiencyMetricsEngine",
    "TemporalConsistencyAnalyzer",
    "TemporalConsistencyResult",
    "BootstrapCI",
    "WilcoxonResult",
    "PairwiseSummary",
    "StatisticalTestEngine",
]
