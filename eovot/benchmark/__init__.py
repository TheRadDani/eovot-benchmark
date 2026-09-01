"""Benchmark sub-package — core evaluation engine."""

from .engine import BenchmarkEngine, BenchmarkResult, SequenceResult
from .multi_run import MultiRunBenchmark, MultiRunEntry

__all__ = [
    "BenchmarkEngine",
    "BenchmarkResult",
    "SequenceResult",
    "MultiRunBenchmark",
    "MultiRunEntry",
]
