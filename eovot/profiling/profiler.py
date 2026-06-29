"""Hardware-aware profiler for EOVOT tracker evaluation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import psutil


@dataclass
class ProfilingResult:
    """Hardware profiling summary for one tracker run.

    Latency percentiles expose the full tail distribution — essential for
    hard real-time edge systems where mean latency underestimates the
    worst-case frame budget:

    * ``latency_p50_ms`` — median; robust centre of the distribution.
    * ``latency_p95_ms`` — 1 in 20 frames exceed this; soft real-time bound.
    * ``latency_p99_ms`` — 1 in 100 frames exceed this; hard real-time bound.
    """

    tracker_name: str
    frame_count: int
    fps: float
    latency_mean_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    peak_memory_mb: float

    def __str__(self) -> str:
        return (
            f"ProfilingResult[{self.tracker_name}] "
            f"FPS={self.fps:.1f}  "
            f"latency={self.latency_mean_ms:.2f}±{self.latency_std_ms:.2f} ms  "
            f"p50={self.latency_p50_ms:.2f} ms  "
            f"p95={self.latency_p95_ms:.2f} ms  "
            f"p99={self.latency_p99_ms:.2f} ms  "
            f"mem={self.peak_memory_mb:.1f} MiB  "
            f"frames={self.frame_count}"
        )


class Profiler:
    """Collect per-frame timing and memory statistics."""

    def __init__(self) -> None:
        self._process = psutil.Process(os.getpid())
        self._latencies: List[float] = []
        self._peak_memory_mb: float = 0.0
        self._t0: Optional[float] = None

    def start_frame(self) -> None:
        """Mark the start of a tracker update call."""
        self._t0 = time.perf_counter()

    def end_frame(self) -> float:
        """Mark the end of a tracker update call and return elapsed ms."""
        if self._t0 is None:
            raise RuntimeError("end_frame() called before start_frame()")
        elapsed_ms = (time.perf_counter() - self._t0) * 1_000.0
        self._t0 = None
        self._latencies.append(elapsed_ms)
        mem_mb = self._process.memory_info().rss / (1024 ** 2)
        self._peak_memory_mb = max(self._peak_memory_mb, mem_mb)
        return elapsed_ms

    def summary(self, tracker_name: str = "unknown") -> ProfilingResult:
        """Return aggregated :class:`ProfilingResult`."""
        if not self._latencies:
            raise ValueError("No frames profiled.")
        arr = np.array(self._latencies)
        mean_ms = float(arr.mean())
        return ProfilingResult(
            tracker_name=tracker_name,
            frame_count=len(arr),
            fps=1_000.0 / mean_ms if mean_ms > 0 else float("inf"),
            latency_mean_ms=mean_ms,
            latency_std_ms=float(arr.std()),
            latency_p50_ms=float(np.percentile(arr, 50)),
            latency_p95_ms=float(np.percentile(arr, 95)),
            latency_p99_ms=float(np.percentile(arr, 99)),
            peak_memory_mb=self._peak_memory_mb,
        )

    def reset(self) -> None:
        """Clear accumulated statistics."""
        self._latencies.clear()
        self._peak_memory_mb = 0.0
        self._t0 = None
