"""Analysis utilities for EOVOT benchmark experiments."""

from .adaptive_analysis import AdaptiveBudgetAnalyzer, AdaptiveBudgetEntry, AdaptiveBudgetReport
from .resolution_analysis import ResolutionScaleAnalyzer, ScaleEntry, ScaleResult
from .skip_analysis import FrameSkipAnalyzer, SkipRateResult

__all__ = [
    "AdaptiveBudgetAnalyzer",
    "AdaptiveBudgetEntry",
    "AdaptiveBudgetReport",
    "FrameSkipAnalyzer",
    "SkipRateResult",
    "ResolutionScaleAnalyzer",
    "ScaleResult",
    "ScaleEntry",
]
