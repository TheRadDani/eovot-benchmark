"""Benchmark sub-package — core evaluation engine."""

from .engine import BenchmarkEngine, BenchmarkResult, SequenceResult
from .multirun import ConfidenceInterval, MultiRunBenchmark, MultiRunResult

__all__ = [
    "BenchmarkEngine",
    "BenchmarkResult",
    "SequenceResult",
    "ConfidenceInterval",
    "MultiRunBenchmark",
    "MultiRunResult",
]
