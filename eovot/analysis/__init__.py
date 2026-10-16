"""Analysis utilities for EOVOT benchmark experiments."""

from .skip_analysis import FrameSkipAnalyzer, SkipRateResult
from .resolution_analysis import ResolutionScaleAnalyzer, ScaleResult, ScaleEntry
from .multi_dataset import MultiDatasetAggregator, AggregatedEntry, AGGREGATION_METRICS

__all__ = [
    "FrameSkipAnalyzer",
    "SkipRateResult",
    "ResolutionScaleAnalyzer",
    "ScaleResult",
    "ScaleEntry",
    "MultiDatasetAggregator",
    "AggregatedEntry",
    "AGGREGATION_METRICS",
]
