"""Metrics sub-package — accuracy, robustness, efficiency, temporal
consistency, statistical testing, attribute analysis, and difficulty scoring."""

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
from .difficulty import (
    DifficultyFactors,
    SequenceDifficultyScorer,
    SequenceDifficultyEntry,
    DifficultyReport,
    score_dataset,
    TierStats,
    StratifiedBenchmarkReport,
    stratify_benchmark_result,
    TIER_EASY_THRESHOLD,
    TIER_HARD_THRESHOLD,
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
    "ALL_ATTRIBUTES",
    "ATTRIBUTE_DESCRIPTIONS",
    "AttributeAnalyzer",
    "AttributeDetector",
    "AttributePerformanceTable",
    "SequenceAttributes",
    "DifficultyFactors",
    "SequenceDifficultyScorer",
    "SequenceDifficultyEntry",
    "DifficultyReport",
    "score_dataset",
    "TierStats",
    "StratifiedBenchmarkReport",
    "stratify_benchmark_result",
    "TIER_EASY_THRESHOLD",
    "TIER_HARD_THRESHOLD",
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
