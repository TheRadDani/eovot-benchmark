"""Experiment sub-package — reproducible multi-tracker experiment management."""

from .cross_dataset import CrossDatasetEvaluator, CrossDatasetReport
from .runner import ExperimentRunner
from .snapshot import ReproducibilitySnapshot

__all__ = [
    "CrossDatasetEvaluator",
    "CrossDatasetReport",
    "ExperimentRunner",
    "ReproducibilitySnapshot",
]
