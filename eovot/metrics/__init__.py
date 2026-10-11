"""Metrics sub-package — accuracy, robustness, efficiency, temporal consistency, statistical testing, attribute analysis, and composite deployability scoring."""

from .composite import CDSEntry, CompositeDeployabilityScore
from .accuracy import (
    iou,
    center_distance,
    AccuracyMetrics,
    MetricsEngine,
)
from .attributes import (
    ALL_ATTRIBUTES,
    ATTRIBUTE_DESCRIPTIONS,
    AttributeAnalyzer,
    AttributeDetector,
    AttributePerformanceTable,
    SequenceAttributes,
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
    "CDSEntry",
    "CompositeDeployabilityScore",
    "iou",
    "center_distance",
    "AccuracyMetrics",
    "MetricsEngine",
    "ALL_ATTRIBUTES",
    "ATTRIBUTE_DESCRIPTIONS",
    "AttributeAnalyzer",
    "AttributeDetector",
    "AttributePerformanceTable",
    "SequenceAttributes",
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
