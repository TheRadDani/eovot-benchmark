"""Analysis utilities for EOVOT benchmark experiments."""

from .regression import (
    RegressionDetector,
    RegressionReport,
    MetricDelta,
    MetricStatus,
    load_scalars_from_file,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    ALL_TRACKED_METRICS,
)
from .skip_analysis import FrameSkipAnalyzer, SkipRateResult

__all__ = [
    "FrameSkipAnalyzer",
    "SkipRateResult",
    "RegressionDetector",
    "RegressionReport",
    "MetricDelta",
    "MetricStatus",
    "load_scalars_from_file",
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
    "ALL_TRACKED_METRICS",
]
