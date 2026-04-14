"""GPU profiling module for EOVOT — NVIDIA NVML power and memory monitor.

Measures per-frame GPU power draw, memory usage, and utilisation via the
NVIDIA Management Library (NVML).  On systems without an NVIDIA GPU or the
``pynvml`` package, all measurements return zero and :meth:`GPUProfiler.is_available`
returns ``False``, so the module is safe to import anywhere.

Energy model (GPU)::

    E_frame (J) = power_mW(t) / 1000 * elapsed_s(t)

This is more accurate than the TDP-based CPU estimate because NVML reports
*actual* GPU board power, not a theoretical maximum.

Requirements:
    pip install pynvml       # NVIDIA driver must also be present

Typical usage::

    from eovot.profiling.gpu import GPUProfiler

    profiler = GPUProfiler(device_index=0)
    if profiler.is_available():
        print(f"Profiling on: {profiler.device_name}")

    for i, frame in enumerate(sequence):
        if i == 0:
            tracker.initialize(frame, bbox)
        else:
            profiler.start_frame()
            bbox = tracker.update(frame)
            profiler.end_frame()

    result = profiler.summary("my_gpu_tracker")
    print(result)

References:
    NVIDIA Management Library (NVML) Developer Guide:
    https://docs.nvidia.com/deploy/nvml-api/index.html
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import pynvml  # type: ignore[import-untyped]
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


@dataclass
class GPUProfilingResult:
    """Per-run GPU profiling summary.

    All energy and power values are derived from real NVML board-power
    measurements (mW) combined with precise wall-clock timing.  When NVML
    is unavailable, all numeric fields are zero and ``nvml_available`` is
    ``False``.

    Attributes:
        tracker_name: Identifier of the profiled tracker.
        frame_count: Number of update frames measured.
        device_name: GPU device name string (e.g. ``"Tesla T4"``), or ``"N/A"``.
        total_energy_j: Total GPU energy consumed (Joules).
        mean_power_w: Mean GPU board power draw (Watts).
        energy_per_frame_mj: Mean per-frame GPU energy (milli-Joules).
        peak_memory_mb: Peak GPU memory used (MiB).
        mean_gpu_util_pct: Mean GPU core utilisation (%).
        peak_gpu_util_pct: Peak GPU core utilisation (%).
        nvml_available: ``True`` when measurements came from NVML.
    """

    tracker_name: str
    frame_count: int
    device_name: str
    total_energy_j: float
    mean_power_w: float
    energy_per_frame_mj: float
    peak_memory_mb: float
    mean_gpu_util_pct: float
    peak_gpu_util_pct: float
    nvml_available: bool

    def __str__(self) -> str:
        if not self.nvml_available:
            return (
                f"GPUProfilingResult[{self.tracker_name}] "
                f"NVML unavailable — GPU metrics not collected"
            )
        return (
            f"GPUProfilingResult[{self.tracker_name}] "
            f"device={self.device_name}  "
            f"total={self.total_energy_j:.4f} J  "
            f"mean_power={self.mean_power_w:.2f} W  "
            f"per_frame={self.energy_per_frame_mj:.3f} mJ  "
            f"mem_peak={self.peak_memory_mb:.1f} MiB  "
            f"gpu_util={self.mean_gpu_util_pct:.1f}%"
            f" (peak {self.peak_gpu_util_pct:.1f}%)  "
            f"frames={self.frame_count}"
        )

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        return {
            "tracker_name": self.tracker_name,
            "frame_count": self.frame_count,
            "device_name": self.device_name,
            "total_energy_j": round(self.total_energy_j, 6),
            "mean_power_w": round(self.mean_power_w, 4),
            "energy_per_frame_mj": round(self.energy_per_frame_mj, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "mean_gpu_util_pct": round(self.mean_gpu_util_pct, 2),
            "peak_gpu_util_pct": round(self.peak_gpu_util_pct, 2),
            "nvml_available": self.nvml_available,
        }


class GPUProfiler:
    """Measure per-frame GPU power, memory, and utilisation via NVML.

    On systems without an NVIDIA GPU or ``pynvml``, the profiler silently
    operates in **stub mode** — all power/memory readings are zero and
    :meth:`is_available` returns ``False``.  The timing pipeline still runs
    correctly, so the benchmark engine remains device-agnostic.

    Args:
        device_index: NVML device index (0 for the first GPU, 1 for the
            second, etc.).  Ignored when NVML is unavailable.

    Example::

        profiler = GPUProfiler(device_index=0)
        if profiler.is_available():
            print(f"Profiling GPU: {profiler.device_name}")

        for i, frame in enumerate(frames):
            if i == 0:
                tracker.initialize(frame, bbox)
            else:
                profiler.start_frame()
                pred_bbox = tracker.update(frame)
                profiler.end_frame()

        result = profiler.summary("SiamFC-Lite")
        print(result)
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._handle = None
        self._device_name: str = "N/A"
        self._nvml_active: bool = False

        # Per-frame measurement buffers
        self._t0: Optional[float] = None
        self._latencies_s: List[float] = []
        self._power_mw_samples: List[float] = []
        self._mem_mb_samples: List[float] = []
        self._util_pct_samples: List[float] = []

        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                raw_name = pynvml.nvmlDeviceGetName(self._handle)
                self._device_name = (
                    raw_name.decode("utf-8")
                    if isinstance(raw_name, bytes)
                    else raw_name
                )
                self._nvml_active = True
            except Exception:
                self._nvml_active = False

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if pynvml is importable and at least one NVML GPU is present."""
        if not _HAS_NVML:
            return False
        try:
            pynvml.nvmlInit()
            pynvml.nvmlDeviceGetHandleByIndex(0)
            return True
        except Exception:
            return False

    @property
    def device_name(self) -> str:
        """Human-readable GPU model string, or ``"N/A"`` when NVML is unavailable."""
        return self._device_name

    def start_frame(self) -> None:
        """Mark the start of a tracker update call."""
        self._t0 = time.perf_counter()

    def end_frame(self) -> float:
        """Mark the end of a tracker update call.

        Samples GPU power, memory, and utilisation, then records elapsed time.

        Returns:
            Estimated GPU energy consumed during this frame (milli-Joules),
            or ``0.0`` when NVML is unavailable.

        Raises:
            RuntimeError: If called without a preceding :meth:`start_frame`.
        """
        if self._t0 is None:
            raise RuntimeError("end_frame() called before start_frame()")
        elapsed_s = time.perf_counter() - self._t0
        self._t0 = None
        self._latencies_s.append(elapsed_s)

        power_mw: float = 0.0
        mem_mb: float = 0.0
        gpu_util_pct: float = 0.0

        if self._nvml_active and self._handle is not None:
            try:
                # nvmlDeviceGetPowerUsage returns milliwatts
                power_mw = float(pynvml.nvmlDeviceGetPowerUsage(self._handle))
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                mem_mb = mem_info.used / (1024.0 ** 2)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                gpu_util_pct = float(util.gpu)
            except Exception:
                pass  # transient NVML errors should not abort the benchmark

        self._power_mw_samples.append(power_mw)
        self._mem_mb_samples.append(mem_mb)
        self._util_pct_samples.append(gpu_util_pct)

        # E (mJ) = power (mW) * time (s)  — units: mW·s = mJ
        energy_mj = power_mw * elapsed_s
        return energy_mj

    def summary(self, tracker_name: str = "unknown") -> GPUProfilingResult:
        """Return a :class:`GPUProfilingResult` aggregating all frame measurements.

        Args:
            tracker_name: Identifier embedded in the returned result object.

        Raises:
            ValueError: If no frames have been profiled yet.
        """
        if not self._latencies_s:
            raise ValueError("No frames profiled — call start_frame/end_frame first.")

        lat_arr = np.array(self._latencies_s)
        power_arr = np.array(self._power_mw_samples)   # mW
        mem_arr = np.array(self._mem_mb_samples)        # MiB
        util_arr = np.array(self._util_pct_samples)     # %

        # Energy in Joules: (mW / 1000) * s = W·s = J
        energies_j = (power_arr / 1000.0) * lat_arr
        total_j = float(energies_j.sum())
        elapsed_total = float(lat_arr.sum())
        mean_power_w = total_j / elapsed_total if elapsed_total > 0.0 else 0.0
        # mean energy per frame converted from J to mJ
        energy_per_frame_mj = float(energies_j.mean()) * 1_000.0

        return GPUProfilingResult(
            tracker_name=tracker_name,
            frame_count=len(lat_arr),
            device_name=self._device_name,
            total_energy_j=total_j,
            mean_power_w=mean_power_w,
            energy_per_frame_mj=energy_per_frame_mj,
            peak_memory_mb=float(mem_arr.max()) if len(mem_arr) else 0.0,
            mean_gpu_util_pct=float(util_arr.mean()) if len(util_arr) else 0.0,
            peak_gpu_util_pct=float(util_arr.max()) if len(util_arr) else 0.0,
            nvml_available=self._nvml_active,
        )

    def reset(self) -> None:
        """Clear all accumulated measurements for re-use across sequences."""
        self._latencies_s.clear()
        self._power_mw_samples.clear()
        self._mem_mb_samples.clear()
        self._util_pct_samples.clear()
        self._t0 = None

    def __del__(self) -> None:
        """Shut down NVML gracefully when the profiler is garbage-collected."""
        if self._nvml_active and _HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
