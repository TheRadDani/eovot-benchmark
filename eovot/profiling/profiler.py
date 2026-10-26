"""Hardware-aware profiler for EOVOT tracker evaluation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    latency_ci_95_ms: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    """Bootstrap 95 % confidence interval (lower, upper) on mean latency in ms.

    Computed from 2 000 bootstrap resamples of the per-frame latency
    distribution using the percentile method.  A narrow interval indicates
    that the mean estimate is stable; a wide interval suggests high variance
    or too few measured frames.
    """
    warm_up_frames_skipped: int = 0
    """Number of leading frames discarded before latency recording began.

    These frames are still executed (keeping tracker state valid) but their
    wall-clock times are excluded so that CPU/cache warm-up effects do not
    bias the reported latency statistics.
    """

    def __str__(self) -> str:
        ci_lo, ci_hi = self.latency_ci_95_ms
        return (
            f"ProfilingResult[{self.tracker_name}] "
            f"FPS={self.fps:.1f}  "
            f"latency={self.latency_mean_ms:.2f}±{self.latency_std_ms:.2f} ms  "
            f"95%CI=[{ci_lo:.2f},{ci_hi:.2f}] ms  "
            f"p95={self.latency_p95_ms:.2f} ms  p99={self.latency_p99_ms:.2f} ms  "
            f"CV={self.latency_cv:.3f}  "
            f"mem={self.peak_memory_mb:.1f} MiB  "
            f"frames={self.frame_count}  "
            f"warm_up={self.warm_up_frames_skipped}"
        )


def _bootstrap_ci_mean(
    data: np.ndarray,
    n_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of *data*.

    Args:
        data:        1-D array of observations.
        n_resamples: Number of bootstrap resamples.  2 000 is sufficient for
                     a 95 % interval; increase for tighter bounds on 99 %.
        confidence:  Desired confidence level (default 0.95).
        seed:        RNG seed for reproducibility.

    Returns:
        ``(lower, upper)`` in the same units as *data*.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    # Vectorised resampling: draw all indices at once
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = data[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


class Profiler:
    """Collect per-frame timing and memory statistics.

    Args:
        warm_up_frames: Number of leading frames to execute but *not* record.
            These frames allow the CPU instruction cache, branch predictor, and
            any JIT-compiled paths to reach steady state before measurement
            begins.  Typical values: 5 for CPU-bound OpenCV trackers, 10–20 for
            deep models with first-inference overhead.  Default: ``5``.
        n_bootstrap: Number of resamples used to compute the 95 % confidence
            interval on mean latency.  Lower values are faster; 2 000 is the
            recommended minimum for a reliable 95 % CI.  Default: ``2_000``.

    Example::

        profiler = Profiler(warm_up_frames=10)
        for i, frame in enumerate(sequence):
            profiler.start_frame()
            bbox = tracker.update(frame)
            profiler.end_frame()

        result = profiler.summary("MyTracker")
        print(result.latency_ci_95_ms)   # (lo_ms, hi_ms)
    """

    def __init__(self, warm_up_frames: int = 0, n_bootstrap: int = 2_000) -> None:
        if warm_up_frames < 0:
            raise ValueError("warm_up_frames must be >= 0")
        self._process = psutil.Process(os.getpid())
        self._warm_up_frames = warm_up_frames
        self._n_bootstrap = n_bootstrap
        self._latencies: List[float] = []
        self._warmup_latencies: List[float] = []  # fallback when seq < warm_up_frames
        self._peak_memory_mb: float = 0.0
        self._t0: Optional[float] = None
        self._frames_seen: int = 0  # counts both warm-up and recorded frames

    # ------------------------------------------------------------------
    # Per-frame API
    # ------------------------------------------------------------------

    def start_frame(self) -> None:
        """Mark the start of a tracker update call."""
        self._t0 = time.perf_counter()

    def end_frame(self) -> float:
        """Mark the end of a tracker update call and return elapsed ms.

        During the warm-up period (first ``warm_up_frames`` calls) the elapsed
        time is returned for diagnostics but is **not** stored in the latency
        distribution used by :meth:`summary`.

        Returns:
            Elapsed time in milliseconds (always, including warm-up frames).
        """
        if self._t0 is None:
            raise RuntimeError("end_frame() called before start_frame()")
        elapsed_ms = (time.perf_counter() - self._t0) * 1_000.0
        self._t0 = None

        # Always track peak memory — warm-up frames can cause real allocations.
        mem_mb = self._process.memory_info().rss / (1024 ** 2)
        self._peak_memory_mb = max(self._peak_memory_mb, mem_mb)

        # Record latency after warm-up is complete; keep warm-up times as fallback.
        if self._frames_seen >= self._warm_up_frames:
            self._latencies.append(elapsed_ms)
        else:
            self._warmup_latencies.append(elapsed_ms)

        self._frames_seen += 1
        return elapsed_ms

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summary(self, tracker_name: str = "unknown") -> ProfilingResult:
        """Return aggregated :class:`ProfilingResult`.

        When the sequence is shorter than ``warm_up_frames`` (all frames fell
        inside the warm-up window), the profiler falls back to the warm-up
        frames rather than raising so that short sequences in tests and edge
        cases remain functional.  ``warm_up_frames_skipped`` is set to ``0``
        in that case to signal that no frames were actually discarded.

        Raises:
            ValueError: If no frames at all have been profiled.
        """
        if self._latencies:
            arr = np.array(self._latencies, dtype=np.float64)
            frames_skipped = min(self._warm_up_frames, self._frames_seen)
        elif self._warmup_latencies:
            # Sequence shorter than warm_up_frames — use all available frames.
            arr = np.array(self._warmup_latencies, dtype=np.float64)
            frames_skipped = 0
        else:
            raise ValueError("No frames profiled.")
        mean_ms = float(arr.mean())
        cv = float(arr.std() / mean_ms) if mean_ms > 0 else 0.0
        ci = _bootstrap_ci_mean(arr, n_resamples=self._n_bootstrap) if len(arr) >= 2 else (mean_ms, mean_ms)
        return ProfilingResult(
            tracker_name=tracker_name,
            frame_count=len(arr),
            fps=1_000.0 / mean_ms if mean_ms > 0 else float("inf"),
            latency_mean_ms=mean_ms,
            latency_std_ms=float(arr.std()),
            latency_p95_ms=float(np.percentile(arr, 95)),
            latency_p99_ms=float(np.percentile(arr, 99)),
            latency_cv=cv,
            latency_ci_95_ms=ci,
            warm_up_frames_skipped=frames_skipped,
            peak_memory_mb=self._peak_memory_mb,
        )

    def reset(self) -> None:
        """Clear accumulated statistics, including the warm-up counter."""
        self._latencies.clear()
        self._warmup_latencies.clear()
        self._peak_memory_mb = 0.0
        self._t0 = None
        self._frames_seen = 0
