"""GPU-aware profiler for EOVOT tracker evaluation.

Extends the CPU-based :class:`~eovot.profiling.profiler.Profiler` with CUDA
event timing, GPU memory tracking, and optional device utilization sampling.

When CUDA is not available the profiler degrades gracefully: GPU fields in
:class:`GpuProfilingResult` are set to ``0.0`` / ``None``, and the wall-clock
latency from :class:`~eovot.profiling.profiler.Profiler` is used as the
primary timing source.  This means the same code path works on both desktop
GPUs and CPU-only edge devices.

Typical usage::

    from eovot.profiling.gpu_profiler import GpuProfiler

    gpu = GpuProfiler()
    for frame in sequence:
        gpu.start_frame()
        bbox = tracker.update(frame)  # may dispatch CUDA kernels
        gpu.end_frame()

    result = gpu.summary("SiamRPN")
    print(result)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Optional CUDA support — import lazily so the module loads on CPU systems.
# ---------------------------------------------------------------------------
try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore[assignment]
    _CUDA_AVAILABLE = False


def cuda_available() -> bool:
    """Return ``True`` when PyTorch + CUDA are present and a device is found."""
    return _CUDA_AVAILABLE


@dataclass
class GpuProfilingResult:
    """Hardware profiling summary for one tracker run, including GPU metrics.

    All GPU fields default to ``0.0`` / ``None`` when CUDA is unavailable;
    callers should check :attr:`gpu_available` before using them.
    """

    tracker_name: str
    frame_count: int

    # --- Timing (wall-clock, always available) ---
    fps: float
    latency_mean_ms: float
    latency_std_ms: float
    latency_p95_ms: float

    # --- CPU memory ---
    peak_cpu_memory_mb: float

    # --- GPU availability ---
    gpu_available: bool
    gpu_device_name: Optional[str]

    # --- CUDA event timing (GPU only) ---
    cuda_latency_mean_ms: float
    cuda_latency_std_ms: float
    cuda_latency_p95_ms: float

    # --- GPU memory (snapshot at end_frame) ---
    peak_gpu_memory_allocated_mb: float
    """Peak GPU memory occupied by live tensors (MiB)."""
    peak_gpu_memory_reserved_mb: float
    """Peak GPU memory held by the PyTorch caching allocator (MiB)."""

    def __str__(self) -> str:
        base = (
            f"GpuProfilingResult[{self.tracker_name}] "
            f"FPS={self.fps:.1f}  "
            f"wall={self.latency_mean_ms:.2f}±{self.latency_std_ms:.2f} ms  "
            f"p95={self.latency_p95_ms:.2f} ms  "
            f"cpu_mem={self.peak_cpu_memory_mb:.1f} MiB  "
            f"frames={self.frame_count}"
        )
        if self.gpu_available:
            base += (
                f"\n  GPU[{self.gpu_device_name}]  "
                f"cuda={self.cuda_latency_mean_ms:.2f}±{self.cuda_latency_std_ms:.2f} ms  "
                f"gpu_alloc={self.peak_gpu_memory_allocated_mb:.1f} MiB  "
                f"gpu_reserved={self.peak_gpu_memory_reserved_mb:.1f} MiB"
            )
        return base

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all metrics."""
        d: dict = {
            "tracker_name": self.tracker_name,
            "frame_count": self.frame_count,
            "fps": round(self.fps, 2),
            "latency_mean_ms": round(self.latency_mean_ms, 3),
            "latency_std_ms": round(self.latency_std_ms, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 3),
            "peak_cpu_memory_mb": round(self.peak_cpu_memory_mb, 2),
            "gpu_available": self.gpu_available,
            "gpu_device_name": self.gpu_device_name,
        }
        if self.gpu_available:
            d.update({
                "cuda_latency_mean_ms": round(self.cuda_latency_mean_ms, 3),
                "cuda_latency_std_ms": round(self.cuda_latency_std_ms, 3),
                "cuda_latency_p95_ms": round(self.cuda_latency_p95_ms, 3),
                "peak_gpu_memory_allocated_mb": round(self.peak_gpu_memory_allocated_mb, 2),
                "peak_gpu_memory_reserved_mb": round(self.peak_gpu_memory_reserved_mb, 2),
            })
        return d


class GpuProfiler:
    """Collect per-frame wall-clock timing, CPU memory, and (when available) CUDA metrics.

    The profiler is a drop-in enhancement for the existing :class:`~eovot.profiling.profiler.Profiler`.
    It exposes the same ``start_frame`` / ``end_frame`` / ``summary`` interface so it
    can be swapped into :class:`~eovot.benchmark.engine.BenchmarkEngine` with no other
    code changes.

    Args:
        device: CUDA device index or string (e.g. ``0`` or ``"cuda:1"``).
            Ignored when CUDA is unavailable.  Default: ``0`` (first GPU).
        sync_before_start: If ``True``, call ``torch.cuda.synchronize()`` at
            the start of each frame to flush any pending GPU work from the
            *previous* tracker call before recording the start event.
            Recommended for benchmarking; adds a small overhead (~0.05 ms).
            Default: ``True``.

    Example::

        gpu = GpuProfiler()
        for frame in sequence:
            gpu.start_frame()
            bbox = tracker.update(frame)
            gpu.end_frame()
        result = gpu.summary("MyTracker")
        print(result)
    """

    def __init__(
        self,
        device: int = 0,
        sync_before_start: bool = True,
    ) -> None:
        self._process = psutil.Process(os.getpid())
        self._device = device
        self._sync_before_start = sync_before_start

        self._wall_latencies: List[float] = []
        self._cuda_latencies: List[float] = []
        self._peak_cpu_mb: float = 0.0
        self._peak_gpu_alloc_mb: float = 0.0
        self._peak_gpu_reserved_mb: float = 0.0

        # Wall-clock reference
        self._t0: Optional[float] = None

        # CUDA events for precise GPU timing
        self._ev_start = None
        self._ev_end = None
        if _CUDA_AVAILABLE:
            self._ev_start = torch.cuda.Event(enable_timing=True)
            self._ev_end = torch.cuda.Event(enable_timing=True)

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def start_frame(self) -> None:
        """Mark the beginning of a tracker update call.

        Records a CUDA start event (when available) and a wall-clock timestamp.
        """
        if _CUDA_AVAILABLE and self._sync_before_start:
            torch.cuda.synchronize(self._device)
        if _CUDA_AVAILABLE:
            self._ev_start.record(
                torch.cuda.current_stream(self._device)
            )
        self._t0 = time.perf_counter()

    def end_frame(self) -> Tuple[float, float]:
        """Mark the end of a tracker update call.

        Returns:
            ``(wall_ms, cuda_ms)`` where ``cuda_ms`` is ``0.0`` when CUDA is
            unavailable.

        Raises:
            RuntimeError: If called before :meth:`start_frame`.
        """
        if self._t0 is None:
            raise RuntimeError("end_frame() called before start_frame()")

        if _CUDA_AVAILABLE:
            self._ev_end.record(
                torch.cuda.current_stream(self._device)
            )
            torch.cuda.synchronize(self._device)
            cuda_ms = self._ev_start.elapsed_time(self._ev_end)
        else:
            cuda_ms = 0.0

        wall_ms = (time.perf_counter() - self._t0) * 1_000.0
        self._t0 = None

        self._wall_latencies.append(wall_ms)
        self._cuda_latencies.append(cuda_ms)

        # CPU memory snapshot
        cpu_mb = self._process.memory_info().rss / (1024 ** 2)
        self._peak_cpu_mb = max(self._peak_cpu_mb, cpu_mb)

        # GPU memory snapshot
        if _CUDA_AVAILABLE:
            alloc_mb = torch.cuda.memory_allocated(self._device) / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved(self._device) / (1024 ** 2)
            self._peak_gpu_alloc_mb = max(self._peak_gpu_alloc_mb, alloc_mb)
            self._peak_gpu_reserved_mb = max(self._peak_gpu_reserved_mb, reserved_mb)

        return wall_ms, cuda_ms

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, tracker_name: str = "unknown") -> GpuProfilingResult:
        """Return an aggregated :class:`GpuProfilingResult`.

        Args:
            tracker_name: Human-readable tracker identifier.

        Raises:
            ValueError: If no frames have been profiled.
        """
        if not self._wall_latencies:
            raise ValueError("No frames profiled — call start_frame/end_frame at least once.")

        wall = np.array(self._wall_latencies, dtype=np.float64)
        mean_wall = float(wall.mean())

        device_name: Optional[str] = None
        if _CUDA_AVAILABLE:
            device_name = torch.cuda.get_device_name(self._device)

        cuda_arr = np.array(self._cuda_latencies, dtype=np.float64)

        return GpuProfilingResult(
            tracker_name=tracker_name,
            frame_count=len(wall),
            fps=1_000.0 / mean_wall if mean_wall > 0 else float("inf"),
            latency_mean_ms=mean_wall,
            latency_std_ms=float(wall.std()),
            latency_p95_ms=float(np.percentile(wall, 95)),
            peak_cpu_memory_mb=self._peak_cpu_mb,
            gpu_available=_CUDA_AVAILABLE,
            gpu_device_name=device_name,
            cuda_latency_mean_ms=float(cuda_arr.mean()) if _CUDA_AVAILABLE else 0.0,
            cuda_latency_std_ms=float(cuda_arr.std()) if _CUDA_AVAILABLE else 0.0,
            cuda_latency_p95_ms=float(np.percentile(cuda_arr, 95)) if _CUDA_AVAILABLE else 0.0,
            peak_gpu_memory_allocated_mb=self._peak_gpu_alloc_mb,
            peak_gpu_memory_reserved_mb=self._peak_gpu_reserved_mb,
        )

    def reset(self) -> None:
        """Clear accumulated statistics for reuse across sequences."""
        self._wall_latencies.clear()
        self._cuda_latencies.clear()
        self._peak_cpu_mb = 0.0
        self._peak_gpu_alloc_mb = 0.0
        self._peak_gpu_reserved_mb = 0.0
        self._t0 = None
