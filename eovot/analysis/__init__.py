"""Analysis utilities for EOVOT benchmark experiments."""

from .skip_analysis import FrameSkipAnalyzer, SkipRateResult
from .resolution_analysis import ResolutionScaleAnalyzer, ScaleResult, ScaleEntry
from .iou_distribution import (
    IoUDistributionAnalyzer,
    IoUDistributionResult,
    BIMODALITY_THRESHOLD,
)

__all__ = [
    "FrameSkipAnalyzer",
    "SkipRateResult",
    "ResolutionScaleAnalyzer",
    "ScaleResult",
    "ScaleEntry",
    "IoUDistributionAnalyzer",
    "IoUDistributionResult",
    "BIMODALITY_THRESHOLD",
]
