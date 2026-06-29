"""Metrics sub-package — accuracy, robustness, efficiency, temporal consistency, statistical testing, and attribute analysis."""

from .accuracy import (
    iou,
    center_distance,
    AccuracyMetrics,
    MetricsEngine,
)
from .attributes import (
    STANDARD_ATTRIBUTES,
    AttributeAnnotations,
    AttributeResult,
    AttributeAnalyzer,
    auto_annotate_from_gt,
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
    "AccuracyMetrics",
    "MetricsEngine",
    "STANDARD_ATTRIBUTES",
    "AttributeAnnotations",
    "AttributeResult",
    "AttributeAnalyzer",
    "auto_annotate_from_gt",
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
