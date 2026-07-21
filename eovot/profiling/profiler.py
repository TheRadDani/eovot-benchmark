"""Hardware-aware profiler for EOVOT tracker evaluation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import psutil


@dataclass
class ProfilingResult:
    """Hardware profiling summary for one tracker run.

    Latency percentiles provide a richer picture than mean alone:
    - ``latency_p50_ms`` (median) is robust to the long tail of spike frames.
    - ``latency_p95_ms`` captures the cost of occasional heavy frames.
    - ``latency_p99_ms`` is the critical metric for real-time deadline analysis:
      a real-time system with a 33 ms budget must keep p99 below that ceiling.

    ``latency_p50_ms`` and ``latency_p99_ms`` default to ``0.0`` so that
    code that constructs :class:`ProfilingResult` directly (e.g. in tests or
    device simulation) does not need to be updated.  :meth:`Profiler.summary`
    always populates them from real measurements.
    """

    tracker_name: str
    frame_count: int
    fps: float
    latency_mean_ms: float
    latency_std_ms: float
    latency_p95_ms: float
    peak_memory_mb: float
    latency_p50_ms: float = field(default=0.0)
    latency_p99_ms: float = field(default=0.0)
    warmup_frames_excluded: int = field(default=0)

    def __str__(self) -> str:
        wu = f"  warmup_excl={self.warmup_frames_excluded}" if self.warmup_frames_excluded else ""
        return (
            f"ProfilingResult[{self.tracker_name}] "
            f"FPS={self.fps:.1f}  "
            f"latency={self.latency_mean_ms:.2f}±{self.latency_std_ms:.2f} ms  "
            f"p50={self.latency_p50_ms:.2f} ms  "
            f"p95={self.latency_p95_ms:.2f} ms  "
            f"p99={self.latency_p99_ms:.2f} ms  "
            f"mem={self.peak_memory_mb:.1f} MiB  "
            f"frames={self.frame_count}"
            f"{wu}"
        )


class Profiler:
    """Collect per-frame timing and memory statistics.

    Args:
        warmup_frames: Number of initial frames to discard from latency
            statistics.  These frames are still *profiled* (start/end_frame
            must be called), but their timings are excluded from
            :meth:`summary`.  Warm-up eliminates JIT-compilation and
            CPU-cache cold-start bias from benchmark measurements.
            Default: ``0`` (no warm-up, backward compatible).

    Example::

        profiler = Profiler(warmup_frames=5)
        for i, frame in enumerate(sequence):
            if i == 0:
                tracker.initialize(frame, bbox)
            else:
                profiler.start_frame()
                bbox = tracker.update(frame)
                profiler.end_frame()   # first 5 calls excluded from summary
        result = profiler.summary("KCF")
        print(result.latency_p99_ms)
    """

    def __init__(self, warmup_frames: int = 0) -> None:
        if warmup_frames < 0:
            raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames}")
        self.warmup_frames = warmup_frames
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
        """Return aggregated :class:`ProfilingResult`.

        Latency statistics are computed over post-warmup frames only.
        The ``frame_count`` field reflects the number of *measured* frames
        (total minus warm-up), so FPS is also computed on steady-state data.

        Raises:
            ValueError: If no post-warmup frames have been profiled.
        """
        if not self._latencies:
            raise ValueError("No frames profiled.")
        all_arr = np.array(self._latencies)
        n_warmup = min(self.warmup_frames, len(all_arr))
        arr = all_arr[n_warmup:]
        if len(arr) == 0:
            raise ValueError(
                f"All {len(all_arr)} profiled frame(s) were consumed by warm-up "
                f"(warmup_frames={self.warmup_frames}).  "
                f"Run more frames than warmup_frames."
            )
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
            warmup_frames_excluded=n_warmup,
        )


    def reset(self) -> None:
        """Clear accumulated statistics."""
        self._latencies.clear()
        self._peak_memory_mb = 0.0
        self._t0 = None
