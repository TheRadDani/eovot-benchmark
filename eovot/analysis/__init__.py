"""Analysis utilities for EOVOT benchmark experiments."""

from .skip_analysis import FrameSkipAnalyzer, SkipRateResult
from .resolution_analysis import ResolutionScaleAnalyzer, ScaleResult, ScaleEntry
from .cross_dataset import (
    CrossDatasetAnalyzer,
    GeneralizationReport,
    GeneralizationEntry,
    MetricBundle,
)

__all__ = [
    "FrameSkipAnalyzer",
    "SkipRateResult",
    "ResolutionScaleAnalyzer",
    "ScaleResult",
    "ScaleEntry",
    "CrossDatasetAnalyzer",
    "GeneralizationReport",
    "GeneralizationEntry",
    "MetricBundle",
]
