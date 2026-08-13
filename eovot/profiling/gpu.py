"""GPU memory and power profiler for EOVOT — NVML-backed, CPU-safe fallback.

This module adds optional GPU profiling alongside the existing CPU profiler so
that deep-learning trackers (e.g. DaSiamRPN, NanoTracker, future PyTorch models)
can be evaluated on their true hardware footprint — not just CPU wall-clock time.

Design goals
------------
* **Zero-dependency on non-GPU machines** — when ``pynvml`` is unavailable or
  no CUDA GPU is present, :class:`GpuProfiler` returns a valid
  :class:`GpuProfilingResult` with ``available=False`` and all numeric fields
  set to ``0.0``.  The rest of the benchmark pipeline sees the same dataclass
  regardless of whether a GPU is present.
* **Frame-level granularity** — memory and power are sampled at the start and
  end of every ``tracker.update()`` call, matching the sampling strategy of the
  existing :class:`~eovot.profiling.profiler.Profiler`.
* **BenchmarkEngine integration** — pass ``gpu_device_index=0`` (or any valid
  device index) to :class:`~eovot.benchmark.engine.BenchmarkEngine` to activate
  GPU profiling; results are stored in each
  :class:`~eovot.benchmark.engine.SequenceResult`.

Requires (optional):
    pip install pynvml>=11.0

Example::

    from eovot.profiling.gpu import GpuProfiler

    profiler = GpuProfiler(device_index=0)
    if profiler.available:
        profiler.start_frame()
        bbox = tracker.update(frame)
        profiler.end_frame()
        result = profiler.summary("MyTracker")
        print(result)
    else:
        print("No NVML-capable GPU detected — GPU profiling skipped.")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class GpuProfilingResult:
    """Per-run GPU memory and power summary.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        device_name: NVML device name string, e.g. ``"NVIDIA GeForce RTX 3080"``.
        device_index: CUDA device index (0-based).
        frame_count: Number of ``update()`` frames profiled.
        available: ``True`` when NVML was successfully initialised and data
            was collected; ``False`` when the profiler fell back to no-op mode.
        peak_gpu_memory_mb: Peak GPU memory used (MiB) across all profiled frames.
        mean_gpu_memory_mb: Mean GPU memory used (MiB) across all profiled frames.
        mean_gpu_power_w: Mean GPU power draw (Watts) across all profiled frames.
        peak_gpu_power_w: Peak GPU power draw (Watts) across all profiled frames.
        mean_gpu_util_pct: Mean GPU compute utilisation (%) across all frames.
        energy_per_frame_mj: Estimated GPU energy per frame (milli-Joules),
            computed as ``mean_gpu_power_w × mean_latency_s × 1000``.
            This field is ``0.0`` unless set by the caller (the profiler does not
            know per-frame latency internally; use it together with
            :class:`~eovot.profiling.profiler.ProfilingResult`).
    """

    tracker_name: str
    device_name: str
    device_index: int
    frame_count: int
    available: bool
    peak_gpu_memory_mb: float
    mean_gpu_memory_mb: float
    mean_gpu_power_w: float
    peak_gpu_power_w: float
    mean_gpu_util_pct: float = 0.0
    energy_per_frame_mj: float = 0.0

    def __str__(self) -> str:
        if not self.available:
            return (
                f"GpuProfilingResult[{self.tracker_name}]  "
                f"GPU profiling unavailable (pynvml not installed or no GPU)"
            )
        return (
            f"GpuProfilingResult[{self.tracker_name} on {self.device_name}]  "
            f"peak_mem={self.peak_gpu_memory_mb:.1f} MiB  "
            f"mean_mem={self.mean_gpu_memory_mb:.1f} MiB  "
            f"mean_power={self.mean_gpu_power_w:.1f} W  "
            f"peak_power={self.peak_gpu_power_w:.1f} W  "
            f"util={self.mean_gpu_util_pct:.1f}%  "
            f"frames={self.frame_count}"
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable plain dict of all fields."""
        return {
            "tracker_name": self.tracker_name,
            "device_name": self.device_name,
            "device_index": self.device_index,
            "frame_count": self.frame_count,
            "available": self.available,
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 2),
            "mean_gpu_memory_mb": round(self.mean_gpu_memory_mb, 2),
            "mean_gpu_power_w": round(self.mean_gpu_power_w, 3),
            "peak_gpu_power_w": round(self.peak_gpu_power_w, 3),
            "mean_gpu_util_pct": round(self.mean_gpu_util_pct, 2),
            "energy_per_frame_mj": round(self.energy_per_frame_mj, 4),
        }


class GpuProfiler:
    """Collect per-frame GPU memory, power, and utilisation via NVML.

    Uses ``pynvml`` to query the NVIDIA Management Library (NVML) at the start
    and end of every profiled frame.  Gracefully degrades to a no-op when
    ``pynvml`` is not installed or no NVIDIA GPU is available — callers never
    need to guard with ``try/except``.

    Args:
        device_index: CUDA device ordinal (0-based).  Defaults to ``0``
            (the first GPU).  Ignored when NVML is unavailable.

    Example::

        profiler = GpuProfiler(device_index=0)
        for i, frame in enumerate(sequence):
            if i == 0:
                tracker.initialize(frame, init_bbox)
            else:
                profiler.start_frame()
                bbox = tracker.update(frame)
                profiler.end_frame()
        result = profiler.summary("MOSSE")
        print(result)
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._available = False
        self._handle = None
        self._device_name = "N/A"

        # Memory samples (MiB) collected via NVML
        self._memory_samples: List[float] = []
        # Power samples (Watts) collected via NVML
        self._power_samples: List[float] = []
        # Utilisation samples (%) collected via NVML
        self._util_samples: List[float] = []

        self._try_init_nvml()

    # ------------------------------------------------------------------
    # NVML initialisation (best-effort)
    # ------------------------------------------------------------------

    def _try_init_nvml(self) -> None:
        """Attempt to initialise NVML; set ``_available = False`` on any failure."""
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0 or self._device_index >= device_count:
                return  # No GPU or invalid index — remain in no-op mode

            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            raw_name = pynvml.nvmlDeviceGetName(self._handle)
            self._device_name = (
                raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            )
            self._available = True
        except Exception:
            # pynvml not installed, no GPU, driver issue — silently degrade.
            self._available = False
            self._handle = None

    # ------------------------------------------------------------------
    # Frame-level API
    # ------------------------------------------------------------------

    def start_frame(self) -> None:
        """Sample GPU state at the beginning of a tracker update call."""
        if not self._available:
            return
        self._sample()

    def end_frame(self) -> None:
        """Sample GPU state at the end of a tracker update call."""
        if not self._available:
            return
        self._sample()

    def _sample(self) -> None:
        """Read current GPU memory, power, and utilisation from NVML."""
        try:
            import pynvml  # type: ignore[import-untyped]

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._memory_samples.append(mem_info.used / (1024 ** 2))  # bytes → MiB

            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self._power_samples.append(power_mw / 1_000.0)  # mW → W
            except Exception:
                pass  # Some GPUs don't support power queries

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._util_samples.append(float(util.gpu))
            except Exception:
                pass

        except Exception:
            pass  # NVML errors during profiling are non-fatal

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(
        self,
        tracker_name: str = "unknown",
        mean_latency_ms: Optional[float] = None,
    ) -> GpuProfilingResult:
        """Return aggregated :class:`GpuProfilingResult`.

        Args:
            tracker_name: Identifier stored in the result.
            mean_latency_ms: Optional mean per-frame latency from
                :class:`~eovot.profiling.profiler.ProfilingResult`; when
                provided, ``energy_per_frame_mj`` is populated using
                ``mean_gpu_power_w × mean_latency_s × 1000``.

        Returns:
            A fully populated :class:`GpuProfilingResult`.  When NVML is
            unavailable all numeric fields are ``0.0`` and ``available=False``.
        """
        if not self._available or not self._memory_samples:
            return GpuProfilingResult(
                tracker_name=tracker_name,
                device_name=self._device_name,
                device_index=self._device_index,
                frame_count=len(self._memory_samples),
                available=False,
                peak_gpu_memory_mb=0.0,
                mean_gpu_memory_mb=0.0,
                mean_gpu_power_w=0.0,
                peak_gpu_power_w=0.0,
                mean_gpu_util_pct=0.0,
                energy_per_frame_mj=0.0,
            )

        mem_arr = np.array(self._memory_samples, dtype=np.float64)
        peak_mem = float(mem_arr.max())
        mean_mem = float(mem_arr.mean())

        mean_power = 0.0
        peak_power = 0.0
        if self._power_samples:
            pw_arr = np.array(self._power_samples, dtype=np.float64)
            mean_power = float(pw_arr.mean())
            peak_power = float(pw_arr.max())

        mean_util = 0.0
        if self._util_samples:
            mean_util = float(np.mean(self._util_samples))

        energy_mj = 0.0
        if mean_latency_ms is not None and mean_latency_ms > 0 and mean_power > 0:
            energy_mj = mean_power * (mean_latency_ms / 1_000.0) * 1_000.0

        # Each frame contributes two samples (start + end); frame count is half.
        frame_count = max(1, len(self._memory_samples) // 2)

        return GpuProfilingResult(
            tracker_name=tracker_name,
            device_name=self._device_name,
            device_index=self._device_index,
            frame_count=frame_count,
            available=True,
            peak_gpu_memory_mb=peak_mem,
            mean_gpu_memory_mb=mean_mem,
            mean_gpu_power_w=mean_power,
            peak_gpu_power_w=peak_power,
            mean_gpu_util_pct=mean_util,
            energy_per_frame_mj=energy_mj,
        )

    def reset(self) -> None:
        """Clear all accumulated samples for re-use across sequences."""
        self._memory_samples.clear()
        self._power_samples.clear()
        self._util_samples.clear()

    @property
    def available(self) -> bool:
        """``True`` when NVML was successfully initialised and a GPU is present."""
        return self._available

    @property
    def device_name(self) -> str:
        """Human-readable GPU device name, or ``"N/A"`` when unavailable."""
        return self._device_name

    def __repr__(self) -> str:
        status = f"device={self._device_name}" if self._available else "unavailable"
        return f"GpuProfiler({status})"
