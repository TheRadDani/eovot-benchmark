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
    """Time spent in :meth:`~eovot.trackers.base.BaseTracker.initialize` (ms).

    Captures the cold-start cost of a tracker: model loading, template
    extraction, histogram initialisation, etc.  On edge devices this can
    be several hundred milliseconds for deep-learning trackers and is a
    hard constraint when trackers must re-initialise after target loss.
    Defaults to 0.0 for backward-compatibility with results produced by
    older versions that did not measure initialisation time.
    """

    @property
    def cold_start_ratio(self) -> float:
        """Ratio of init time to mean per-frame update time.

        A value of 1.0 means the tracker spends as long initialising as
        it does on a single update frame.  Values >> 1 indicate that
        re-initialisation on target loss would dominate latency.
        """
        return self.init_latency_ms / self.latency_mean_ms if self.latency_mean_ms > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"ProfilingResult[{self.tracker_name}] "
            f"FPS={self.fps:.1f}  "
            f"latency={self.latency_mean_ms:.2f}±{self.latency_std_ms:.2f} ms  "
            f"p95={self.latency_p95_ms:.2f} ms  p99={self.latency_p99_ms:.2f} ms  "
            f"CV={self.latency_cv:.3f}  "
            f"init={self.init_latency_ms:.2f} ms  "
            f"mem={self.peak_memory_mb:.1f} MiB  "
            f"frames={self.frame_count}"
        )


class Profiler:
    """Collect per-frame timing and memory statistics.

    Measures both per-frame *update* latency and the one-off *init* latency
    of :meth:`~eovot.trackers.base.BaseTracker.initialize`.  Init latency is
    important for edge deployments where trackers must re-initialise on
    target-loss events; it is exposed as
    :attr:`~ProfilingResult.init_latency_ms` in the result.
    """

    def __init__(self) -> None:
        self._process = psutil.Process(os.getpid())
        self._latencies: List[float] = []
        self._peak_memory_mb: float = 0.0
        self._t0: Optional[float] = None
        self._init_latency_ms: float = 0.0
        self._t_init: Optional[float] = None

    # ------------------------------------------------------------------
    # Init-phase timing
    # ------------------------------------------------------------------

    def start_init(self) -> None:
        """Mark the start of a :meth:`~eovot.trackers.base.BaseTracker.initialize` call."""
        self._t_init = time.perf_counter()

    def end_init(self) -> float:
        """Mark the end of the init call.

        Returns:
            Elapsed initialisation time in milliseconds.

        Raises:
            RuntimeError: If called without a preceding :meth:`start_init`.
        """
        if self._t_init is None:
            raise RuntimeError("end_init() called before start_init()")
        elapsed_ms = (time.perf_counter() - self._t_init) * 1_000.0
        self._t_init = None
        self._init_latency_ms = elapsed_ms
        mem_mb = self._process.memory_info().rss / (1024 ** 2)
        self._peak_memory_mb = max(self._peak_memory_mb, mem_mb)
        return elapsed_ms

    # ------------------------------------------------------------------
    # Per-frame update timing
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
        """Return aggregated :class:`ProfilingResult`."""
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
        self._init_latency_ms = 0.0
        self._t_init = None
