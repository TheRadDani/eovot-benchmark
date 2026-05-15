"""Metrics sub-package — accuracy and efficiency evaluation."""

from .accuracy import (
    iou,
    center_distance,
    AccuracyMetrics,
    MetricsEngine,
)
from .robustness import RobustnessAnalyzer, RobustnessResult
from .attributes import (
    AttributeAnalyzer,
    AttributeAnalysis,
    AttributeResult,
    LASOT_ATTRIBUTES,
    OTB_ATTRIBUTES,
    AUTO_DERIVABLE_ATTRIBUTES,
    derive_fast_motion_mask,
    derive_scale_variation_mask,
    derive_low_resolution_mask,
)

__all__ = [
    "iou",
    "center_distance",
    "AccuracyMetrics",
    "MetricsEngine",
    "RobustnessAnalyzer",
    "RobustnessResult",
    "AttributeAnalyzer",
    "AttributeAnalysis",
    "AttributeResult",
    "LASOT_ATTRIBUTES",
    "OTB_ATTRIBUTES",
    "AUTO_DERIVABLE_ATTRIBUTES",
    "derive_fast_motion_mask",
    "derive_scale_variation_mask",
    "derive_low_resolution_mask",
]
