"""Experiment sub-package — reproducible multi-tracker experiment management."""

from .cross_eval import CrossDatasetEvaluator, CrossEvalTable, TrackerCrossResult
from .runner import ExperimentRunner
from .snapshot import ReproducibilitySnapshot

__all__ = [
    "CrossDatasetEvaluator",
    "CrossEvalTable",
    "ExperimentRunner",
    "ReproducibilitySnapshot",
    "TrackerCrossResult",
]
