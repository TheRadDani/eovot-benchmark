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
    """Hardware profiling summary for one tracker run."""

    tracker_name: str
    frame_count: int
    fps: float
    latency_mean_ms: float
    latency_std_ms: float
    latency_p95_ms: float
    peak_memory_mb: float
    latency_p99_ms: float = 0.0
    """99th-percentile per-frame latency (ms) — tail-risk indicator for SLA analysis."""
    latency_cv: float = 0.0
    """Coefficient of variation (std / mean) of per-frame latency.
    High CV (> 0.3) signals jitter-heavy execution that hurts real-time guarantees."""
    init_latency_ms: float = 0.0
    """Latency of the first-frame initialize() call in milliseconds.

    Separates cold-start cost (model loading, memory allocation, JIT warm-up)
    from steady-state update latency.  On edge hardware the first-frame cost can
    be orders of magnitude higher than update cost — a critical deployment metric
    that is invisible when only measuring mean update FPS.
    """

    @property
    def init_overhead_ratio(self) -> float:
        """Ratio of init latency to mean update latency (dimensionless).

        Values >> 1 indicate a heavy cold-start cost.  Returns 0.0 when
        ``latency_mean_ms`` is zero (no update frames profiled).
        """
        if self.latency_mean_ms <= 0.0:
            return 0.0
        return self.init_latency_ms / self.latency_mean_ms

    def __str__(self) -> str:
        return (
            f"ProfilingResult[{self.tracker_name}] "
            f"init={self.init_latency_ms:.2f} ms  "
            f"FPS={self.fps:.1f}  "
            f"latency={self.latency_mean_ms:.2f}±{self.latency_std_ms:.2f} ms  "
            f"p95={self.latency_p95_ms:.2f} ms  p99={self.latency_p99_ms:.2f} ms  "
            f"CV={self.latency_cv:.3f}  "
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
        self._init_t0: Optional[float] = None
        self._init_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # Initialization profiling
    # ------------------------------------------------------------------

    def start_init(self) -> None:
        """Mark the start of the tracker's first-frame initialize() call."""
        self._init_t0 = time.perf_counter()

    def end_init(self) -> float:
        """Mark the end of initialize() and return elapsed milliseconds.

        Returns:
            Time spent in initialize() in milliseconds.

        Raises:
            RuntimeError: If called before :meth:`start_init`.
        """
        if self._init_t0 is None:
            raise RuntimeError("end_init() called before start_init()")
        elapsed_ms = (time.perf_counter() - self._init_t0) * 1_000.0
        self._init_t0 = None
        self._init_latency_ms = elapsed_ms
        return elapsed_ms

    # ------------------------------------------------------------------
    # Update-frame profiling
    # ------------------------------------------------------------------

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
        """Return aggregated :class:`ProfilingResult`.

        The returned result includes :attr:`~ProfilingResult.init_latency_ms`
        when :meth:`start_init` / :meth:`end_init` were called; otherwise it
        defaults to ``0.0``.
        """
        if not self._latencies:
            raise ValueError("No frames profiled.")
        arr = np.array(self._latencies)
        mean_ms = float(arr.mean())
        cv = float(arr.std() / mean_ms) if mean_ms > 0 else 0.0
        return ProfilingResult(
            tracker_name=tracker_name,
            frame_count=len(arr),
            fps=1_000.0 / mean_ms if mean_ms > 0 else float("inf"),
            latency_mean_ms=mean_ms,
            latency_std_ms=float(arr.std()),
            latency_p95_ms=float(np.percentile(arr, 95)),
            latency_p99_ms=float(np.percentile(arr, 99)),
            latency_cv=cv,
            peak_memory_mb=self._peak_memory_mb,
            init_latency_ms=self._init_latency_ms,
        )

    def reset(self) -> None:
        """Clear accumulated statistics."""
        self._latencies.clear()
        self._peak_memory_mb = 0.0
        self._t0 = None
        self._init_t0 = None
        self._init_latency_ms = 0.0
