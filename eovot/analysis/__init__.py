"""Analysis utilities for EOVOT benchmark experiments."""

from .skip_analysis import FrameSkipAnalyzer, SkipRateResult
from .resolution_analysis import ResolutionScaleAnalyzer, ScaleResult, ScaleEntry
from .pareto import ParetoAnalyzer, ParetoEntry, TrackerPoint, compute_pareto_front

__all__ = [
    "FrameSkipAnalyzer",
    "SkipRateResult",
    "ResolutionScaleAnalyzer",
    "ScaleResult",
    "ScaleEntry",
    "ParetoAnalyzer",
    "ParetoEntry",
    "TrackerPoint",
    "compute_pareto_front",
]
