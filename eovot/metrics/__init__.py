"""Metrics sub-package — accuracy, robustness, efficiency, temporal consistency, statistical testing, and challenge attributes."""

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
from .challenge import (
    ATTRIBUTES as CHALLENGE_ATTRIBUTES,
    FM,
    SV,
    LR,
    ARC,
    OV,
    SequenceAttributeLabels,
    AttributeAccuracy,
    ChallengeDetector,
    ChallengeAnalyzer,
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
    "CHALLENGE_ATTRIBUTES",
    "FM",
    "SV",
    "LR",
    "ARC",
    "OV",
    "SequenceAttributeLabels",
    "AttributeAccuracy",
    "ChallengeDetector",
    "ChallengeAnalyzer",
]
