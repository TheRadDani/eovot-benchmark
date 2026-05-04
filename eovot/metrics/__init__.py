"""Metrics sub-package — accuracy, robustness, and multi-objective scoring."""

from .accuracy import (
    iou,
    center_distance,
    AccuracyMetrics,
    MetricsEngine,
)
from .robustness import RobustnessAnalyzer, RobustnessResult
from .scoring import (
    ScoringWeights,
    RESEARCH_WEIGHTS,
    EDGE_WEIGHTS,
    BALANCED_WEIGHTS,
    ENERGY_WEIGHTS,
    PRESET_WEIGHTS,
    compute_composite_scores,
    pareto_frontier,
)

__all__ = [
    "iou",
    "center_distance",
    "AccuracyMetrics",
    "MetricsEngine",
    "RobustnessAnalyzer",
    "RobustnessResult",
    "ScoringWeights",
    "RESEARCH_WEIGHTS",
    "EDGE_WEIGHTS",
    "BALANCED_WEIGHTS",
    "ENERGY_WEIGHTS",
    "PRESET_WEIGHTS",
    "compute_composite_scores",
    "pareto_frontier",
]
