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

    # Warm-up-aware steady-state fields (populated when warmup_frames > 0)
    warmup_frame_count: int = 0
    """Number of frames excluded from latency statistics as warm-up."""
    steady_state_latency_mean_ms: float = 0.0
    """Mean latency excluding warm-up frames; equals ``latency_mean_ms`` when
    ``warmup_frame_count == 0``."""
    steady_state_fps: float = 0.0
    """FPS computed from ``steady_state_latency_mean_ms``; equals ``fps`` when
    ``warmup_frame_count == 0``."""

    # Real-time budget tracking
    budget_ms: Optional[float] = None
    """Target per-frame latency budget in ms (e.g. 33.3 ms for 30 FPS real-time).
    ``None`` when budget tracking was not requested."""
    budget_violation_rate: float = 0.0
    """Fraction of *steady-state* frames whose latency exceeded ``budget_ms``.
    ``0.0`` when ``budget_ms`` is ``None`` (no budget configured)."""

    def __str__(self) -> str:
        parts = [
            f"ProfilingResult[{self.tracker_name}]",
            f"FPS={self.fps:.1f}",
            f"latency={self.latency_mean_ms:.2f}±{self.latency_std_ms:.2f} ms",
            f"p95={self.latency_p95_ms:.2f} ms",
            f"p99={self.latency_p99_ms:.2f} ms",
            f"CV={self.latency_cv:.3f}",
            f"mem={self.peak_memory_mb:.1f} MiB",
            f"frames={self.frame_count}",
        ]
        if self.warmup_frame_count > 0:
            parts.append(
                f"steady_FPS={self.steady_state_fps:.1f}"
                f"(warmup={self.warmup_frame_count}fr)"
            )
        if self.budget_ms is not None:
            parts.append(
                f"budget={self.budget_ms:.1f}ms"
                f" violation={self.budget_violation_rate:.1%}"
            )
        return "  ".join(parts)


class Profiler:
    """Collect per-frame timing and memory statistics.

    Args:
        warmup_frames: Number of initial frames to exclude from steady-state
            latency statistics.  The excluded frames are still counted in
            ``frame_count`` but their latencies are isolated into a separate
            warm-up bucket.  Default: ``0`` (all frames included — backward
            compatible behaviour).
        budget_ms: Optional real-time latency budget in milliseconds.  When
            set, :meth:`summary` computes ``budget_violation_rate`` — the
            fraction of *steady-state* frames that exceeded this threshold.
            Useful for SLA-style edge deployment decisions.
            Example: ``budget_ms=33.3`` for a 30 FPS real-time requirement.
    """

    def __init__(
        self,
        warmup_frames: int = 0,
        budget_ms: Optional[float] = None,
    ) -> None:
        if warmup_frames < 0:
            raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames}")
        if budget_ms is not None and budget_ms <= 0:
            raise ValueError(f"budget_ms must be positive, got {budget_ms}")
        self._warmup_frames = warmup_frames
        self._budget_ms = budget_ms
        self._process = psutil.Process(os.getpid())
        self._warmup_latencies: List[float] = []
        self._steady_latencies: List[float] = []
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

        # Route to warm-up or steady-state bucket
        recorded = len(self._warmup_latencies) + len(self._steady_latencies)
        if recorded < self._warmup_frames:
            self._warmup_latencies.append(elapsed_ms)
        else:
            self._steady_latencies.append(elapsed_ms)

        mem_mb = self._process.memory_info().rss / (1024 ** 2)
        self._peak_memory_mb = max(self._peak_memory_mb, mem_mb)
        return elapsed_ms

    def summary(self, tracker_name: str = "unknown") -> ProfilingResult:
        """Return aggregated :class:`ProfilingResult`.

        When ``warmup_frames > 0``, the statistics in ``latency_mean_ms``,
        ``fps``, etc. reflect **all** recorded frames (warm-up + steady-state)
        for backward compatibility.  The new ``steady_state_latency_mean_ms``
        and ``steady_state_fps`` fields report pure steady-state performance.

        Args:
            tracker_name: Human-readable name embedded in the result.

        Returns:
            :class:`ProfilingResult` with all scalar summaries populated.

        Raises:
            ValueError: If no frames have been profiled yet.
        """
        all_latencies = self._warmup_latencies + self._steady_latencies
        if not all_latencies:
            raise ValueError("No frames profiled.")

        arr = np.array(all_latencies, dtype=np.float64)
        mean_ms = float(arr.mean())
        cv = float(arr.std() / mean_ms) if mean_ms > 0 else 0.0

        # Steady-state statistics
        steady = self._steady_latencies
        if steady:
            ss_arr = np.array(steady, dtype=np.float64)
            ss_mean = float(ss_arr.mean())
            ss_fps = 1_000.0 / ss_mean if ss_mean > 0 else float("inf")
        else:
            # Fewer frames than warmup_frames — no steady-state data
            ss_mean = mean_ms
            ss_fps = 1_000.0 / mean_ms if mean_ms > 0 else float("inf")

        # Budget violation rate (computed on steady-state frames only)
        violation_rate = 0.0
        if self._budget_ms is not None and steady:
            ss_arr_b = np.array(steady, dtype=np.float64)
            violation_rate = float((ss_arr_b > self._budget_ms).mean())

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
            warmup_frame_count=len(self._warmup_latencies),
            steady_state_latency_mean_ms=ss_mean,
            steady_state_fps=ss_fps,
            budget_ms=self._budget_ms,
            budget_violation_rate=violation_rate,
        )

    def reset(self) -> None:
        """Clear accumulated statistics while preserving configuration."""
        self._warmup_latencies.clear()
        self._steady_latencies.clear()
        self._peak_memory_mb = 0.0
        self._t0 = None
