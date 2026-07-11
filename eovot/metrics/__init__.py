"""Metrics sub-package — accuracy, robustness, efficiency, temporal consistency, statistical testing, and attribute analysis."""

from .accuracy import (
    iou,
    center_distance,
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
from .attribute_analysis import (
    AttributeAnalyzer,
    AttributeMetrics,
    AttributeReport,
    OTB_ATTRIBUTES,
    assign_synthetic_attributes,
)

__all__ = [
    "iou",
    "center_distance",
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
    "AttributeAnalyzer",
    "AttributeMetrics",
    "AttributeReport",
    "OTB_ATTRIBUTES",
    "assign_synthetic_attributes",
]
