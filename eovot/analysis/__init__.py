"""Analysis utilities for EOVOT benchmark experiments."""

from .skip_analysis import FrameSkipAnalyzer, SkipRateResult
from .resolution_analysis import ResolutionScaleAnalyzer, ScaleResult, ScaleEntry
from .latency_budget import (
    LatencyBudgetAnalyzer,
    LatencyBudgetEntry,
    LatencyBudgetReport,
)

__all__ = [
    "FrameSkipAnalyzer",
    "SkipRateResult",
    "ResolutionScaleAnalyzer",
    "ScaleResult",
    "ScaleEntry",
    "LatencyBudgetAnalyzer",
    "LatencyBudgetEntry",
    "LatencyBudgetReport",
]
