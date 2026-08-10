"""Analysis utilities for EOVOT benchmark experiments."""

from .skip_analysis import FrameSkipAnalyzer, SkipRateResult
from .latency_budget import (
    LatencyBudgetAnalyzer,
    LatencyBudgetEntry,
    LatencyBudgetReport,
)

__all__ = [
    "FrameSkipAnalyzer",
    "SkipRateResult",
    "LatencyBudgetAnalyzer",
    "LatencyBudgetEntry",
    "LatencyBudgetReport",
]
