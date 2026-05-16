"""Metrics sub-package — accuracy and efficiency evaluation."""

from .accuracy import (
    iou,
    center_distance,
    AccuracyMetrics,
    MetricsEngine,
)
from .robustness import RobustnessAnalyzer, RobustnessResult
from .got10k_eval import (
    compute_ao,
    compute_sr,
    GOT10kSequenceResult,
    GOT10kReport,
    GOT10kEvaluator,
)

__all__ = [
    "iou",
    "center_distance",
    "AccuracyMetrics",
    "MetricsEngine",
    "RobustnessAnalyzer",
    "RobustnessResult",
    "compute_ao",
    "compute_sr",
    "GOT10kSequenceResult",
    "GOT10kReport",
    "GOT10kEvaluator",
]
