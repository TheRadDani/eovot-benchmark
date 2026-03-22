"""Hardware-aware profiler for EOVOT tracker evaluation.

Measures per-frame wall-clock latency, derived FPS, and RSS memory usage
using only the Python standard library and ``psutil`` — no GPU-specific
dependencies required.

Design goals
------------
* **Minimal overhead** — timing wraps a single ``time.perf_counter`` pair;
  memory is sampled lazily after each frame.
* **Portable** — works on any OS / device (laptop, Raspberry Pi, Jetson).
* **Extensible** — GPU / power measurement can be added in a sub-class.

Example::

    profiler = Profiler()
    for frame in sequence:
        profiler.start_frame()
        bbox = tracker.update(frame)
        profiler.end_frame()
    result = profiler.summary(tracker_name="MOSSE")
    print(result)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import psutil


@dataclass
class ProfilingResult:
    """Hardware profiling summary for one tracker run."""

    tracker_name: str
    frame_count: int
    fps: float
    """Mean frames-per-second (1000 / mean_latency_ms)."""
    latency_mean_ms: float
    """Mean per-frame latency in milliseconds."""
    latency_std_ms: float
    """Standard deviation of per-frame latency."""
    latency_p95_ms: float
    """95th-percentile latency — proxy for worst-case frame time."""
    peak_memory_mb: float
    """Peak RSS memory of the process during the run (MiB)."""

    def __str__(self) -> str:
        return (
            f"ProfilingResult[{self.tracker_name}] "
            f"FPS={self.fps:.1f}  "
            f"latency={self.latency_mean_ms:.2f}±{self.latency_std_ms:.2f} ms  "
            f"p95={self.latency_p95_ms:.2f} ms  "
            f"mem={self.peak_memory_mb:.1f} MiB  "
            f"frames={self.frame_count}"
        )


class Profiler:
    """Collect per-frame timing and memory statistics.

    Usage is a simple start/end pair around each tracker update call.
    Call :meth:`reset` between sequences so statistics don't bleed across.
    """

    def __init__(self) -> None:
        self._process = psutil.Process(os.getpid())
        self._latencies: List[float] = []
        self._peak_memory_mb: float = 0.0
        self._t0: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Per-frame interface                                                  #
    # ------------------------------------------------------------------ #

    def start_frame(self) -> None:
        """Mark the start of a tracker update call."""
        self._t0 = time.perf_counter()

    def end_frame(self) -> float:
        """Mark the end of a tracker update call.

        Returns:
            Elapsed time for this frame in **milliseconds**.

        Raises:
            RuntimeError: If called without a preceding :meth:`start_frame`.
        """
        if self._t0 is None:
            raise RuntimeError("end_frame() called before start_frame()")
        elapsed_ms = (time.perf_counter() - self._t0) * 1_000.0
        self._t0 = None
        self._latencies.append(elapsed_ms)
        # Sample RSS memory after each frame (cheap syscall on Linux)
        mem_mb = self._process.memory_info().rss / (1024 ** 2)
        self._peak_memory_mb = max(self._peak_memory_mb, mem_mb)
        return elapsed_ms

    # ------------------------------------------------------------------ #
    # Aggregate reporting                                                  #
    # ------------------------------------------------------------------ #

    def summary(self, tracker_name: str = "unknown") -> ProfilingResult:
        """Return an aggregated :class:`ProfilingResult`.

        Args:
            tracker_name: Tracker identifier for the report.

        Returns:
            :class:`ProfilingResult` with FPS, latency stats, and memory.

        Raises:
            ValueError: If no frames have been profiled yet.
        """
        if not self._latencies:
            raise ValueError(
                "No frames profiled. Call start_frame()/end_frame() first."
            )
        arr = np.array(self._latencies)
        mean_ms = float(arr.mean())
        return ProfilingResult(
            tracker_name=tracker_name,
            frame_count=len(arr),
            fps=1_000.0 / mean_ms if mean_ms > 0 else float("inf"),
            latency_mean_ms=mean_ms,
            latency_std_ms=float(arr.std()),
            latency_p95_ms=float(np.percentile(arr, 95)),
            peak_memory_mb=self._peak_memory_mb,
        )

    def reset(self) -> None:
        """Clear all accumulated statistics (call between sequences)."""
        self._latencies.clear()
        self._peak_memory_mb = 0.0
        self._t0 = None
