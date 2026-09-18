"""Analysis utilities for EOVOT benchmark experiments."""

from .skip_analysis import FrameSkipAnalyzer, SkipRateResult
from .resolution_analysis import ResolutionScaleAnalyzer, ScaleResult, ScaleEntry
from .efficiency_budget import BudgetAnalyzer, BudgetReport, BudgetEntry, HardwareBudget

__all__ = [
    "FrameSkipAnalyzer",
    "SkipRateResult",
    "ResolutionScaleAnalyzer",
    "ScaleResult",
    "ScaleEntry",
    "BudgetAnalyzer",
    "BudgetReport",
    "BudgetEntry",
    "HardwareBudget",
]
