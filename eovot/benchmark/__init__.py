"""Benchmark sub-package — core evaluation engine."""

from .engine import BenchmarkEngine, BenchmarkResult, SequenceResult
from .multi_dataset import CrossDatasetEntry, MultiDatasetEvaluator, MultiDatasetResult

__all__ = [
    "BenchmarkEngine",
    "BenchmarkResult",
    "SequenceResult",
    "CrossDatasetEntry",
    "MultiDatasetEvaluator",
    "MultiDatasetResult",
]
