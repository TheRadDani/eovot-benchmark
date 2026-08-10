"""GPU-aware profiler using NVIDIA NVML via pynvml.

Collects per-frame GPU utilisation, memory usage, and power draw during
tracker evaluation.  Provides the same start_frame / end_frame / summary
interface as the CPU :class:`~eovot.profiling.profiler.Profiler` so both
can be used side-by-side in the benchmark engine.

Falls back gracefully when ``pynvml`` is not installed **or** when no CUDA
device is present, so the same calling code runs on CPU-only edge hardware
(Raspberry Pi, Coral Dev Board) without raising import errors.  All numeric
fields in the result are ``NaN`` when the GPU is unavailable, and
:attr:`GPUProfilingResult.gpu_available` is ``False``.

Usage::

    from eovot.profiling.gpu_profiler import GPUProfiler

    gpu = GPUProfiler(device_id=0)   # falls back silently if no GPU
    for i, frame in enumerate(sequence):
        if i == 0:
            tracker.initialize(frame, bbox)
        else:
            gpu.start_frame()
            bbox = tracker.update(frame)
            gpu.end_frame()

    result = gpu.summary("MyTracker")
    print(result)
    print(result.to_dict())   # JSON-serialisable dict

Install the optional dependency::

    pip install pynvml

Reference:
    NVIDIA Management Library (NVML):
    https://docs.nvidia.com/deploy/nvml-api/index.html

    pynvml Python bindings:
    https://pypi.org/project/pynvml/
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import pynvml  # type: ignore

    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


# Sentinel for missing measurements
_NAN: float = math.nan


@dataclass
class GPUProfilingResult:
    """GPU hardware metrics collected during one tracker run.

    When no GPU is available, all numeric fields are ``NaN`` and
    :attr:`gpu_available` is ``False``.

    Attributes:
        tracker_name: Identifier of the profiled tracker.
        frame_count: Number of frames sampled.
        device_name: GPU model string (e.g. ``"NVIDIA GeForce RTX 3080"``).
        gpu_utilization_mean_pct: Mean GPU core utilisation (%).
        gpu_utilization_peak_pct: Peak GPU core utilisation (%).
        gpu_memory_used_mean_mb: Mean GPU device-memory in use (MiB).
        gpu_memory_peak_mb: Peak GPU device-memory in use (MiB).
        gpu_power_mean_w: Mean GPU board power draw (W).  ``NaN`` when the
            GPU does not support power reporting (e.g. consumer GPUs with
            restricted driver access).
        gpu_power_peak_w: Peak GPU board power draw (W).
        gpu_available: Whether NVML initialised and a GPU was found.
    """

    tracker_name: str
    frame_count: int
    device_name: str
    gpu_utilization_mean_pct: float
    gpu_utilization_peak_pct: float
    gpu_memory_used_mean_mb: float
    gpu_memory_peak_mb: float
    gpu_power_mean_w: float
    gpu_power_peak_w: float
    gpu_available: bool

    def __str__(self) -> str:
        if not self.gpu_available:
            return (
                f"GPUProfilingResult[{self.tracker_name}] "
                f"GPU not available (pynvml missing or no CUDA device)"
            )
        pow_str = (
            f"  power={self.gpu_power_mean_w:.1f} W (peak {self.gpu_power_peak_w:.1f} W)"
            if not math.isnan(self.gpu_power_mean_w)
            else "  power=N/A"
        )
        return (
            f"GPUProfilingResult[{self.tracker_name}] "
            f"device={self.device_name}  "
            f"util={self.gpu_utilization_mean_pct:.1f}% "
            f"(peak {self.gpu_utilization_peak_pct:.1f}%)  "
            f"mem={self.gpu_memory_used_mean_mb:.1f} MiB "
            f"(peak {self.gpu_memory_peak_mb:.1f} MiB)"
            f"{pow_str}  "
            f"frames={self.frame_count}"
        )

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation.

        ``NaN`` values are converted to ``None`` so the output is valid JSON.
        """

        def _fmt(v: float, decimals: int = 2) -> Optional[float]:
            return round(v, decimals) if not math.isnan(v) else None

        return {
            "tracker_name": self.tracker_name,
            "frame_count": self.frame_count,
            "device_name": self.device_name,
            "gpu_utilization_mean_pct": _fmt(self.gpu_utilization_mean_pct),
            "gpu_utilization_peak_pct": _fmt(self.gpu_utilization_peak_pct),
            "gpu_memory_used_mean_mb": _fmt(self.gpu_memory_used_mean_mb),
            "gpu_memory_peak_mb": _fmt(self.gpu_memory_peak_mb),
            "gpu_power_mean_w": _fmt(self.gpu_power_mean_w, 3),
            "gpu_power_peak_w": _fmt(self.gpu_power_peak_w, 3),
            "gpu_available": self.gpu_available,
        }


class GPUProfiler:
    """Per-frame GPU metrics collector using NVIDIA NVML.

    Mirrors the start_frame / end_frame / summary / reset interface of
    :class:`~eovot.profiling.profiler.Profiler` for drop-in use alongside
    the CPU profiler.

    When NVML is unavailable or no GPU is found, all methods are no-ops
    and :meth:`summary` returns a result with ``gpu_available=False``.

    Args:
        device_id: CUDA device index (0-based).  Default: ``0``.

    Raises:
        ValueError: If ``device_id`` is negative.

    Example::

        gpu = GPUProfiler(device_id=0)
        if gpu.gpu_available:
            print(f"Profiling on: {gpu.device_name}")
        else:
            print("GPU unavailable — metrics will be NaN")
    """

    def __init__(self, device_id: int = 0) -> None:
        if device_id < 0:
            raise ValueError(f"device_id must be >= 0, got {device_id}")

        self._device_id = device_id
        self._handle: object = None
        self._device_name: str = "N/A"
        self._gpu_available: bool = False
        self._supports_power: bool = False

        self._util_pcts: List[float] = []
        self._mem_used_mb: List[float] = []
        self._power_w: List[float] = []

        self._try_init_nvml()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def gpu_available(self) -> bool:
        """``True`` if NVML initialised successfully and a GPU was found."""
        return self._gpu_available

    @property
    def device_name(self) -> str:
        """GPU model string, or ``"N/A"`` when no GPU is present."""
        return self._device_name

    # ------------------------------------------------------------------
    # Profiling interface
    # ------------------------------------------------------------------

    def start_frame(self) -> None:
        """Mark the start of a tracker update call.

        No-op when no GPU is available.  Provided for API symmetry with
        :class:`~eovot.profiling.profiler.Profiler`.
        """
        # GPU sampling is snapshot-based (end_frame only) — no clock needed.

    def end_frame(self) -> None:
        """Sample GPU metrics at the end of a tracker update call.

        Reads GPU utilisation, memory used, and (if supported) power draw
        via NVML.  Transient NVML errors are swallowed so a single bad
        read does not abort the benchmark.
        """
        if not self._gpu_available or self._handle is None:
            return
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._util_pcts.append(float(util.gpu))
            self._mem_used_mb.append(float(mem_info.used) / (1024.0 ** 2))

            if self._supports_power:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self._power_w.append(float(power_mw) / 1_000.0)
        except Exception:
            pass

    def summary(self, tracker_name: str = "unknown") -> GPUProfilingResult:
        """Return an aggregated :class:`GPUProfilingResult`.

        Args:
            tracker_name: Identifier embedded in the result object.

        Returns:
            :class:`GPUProfilingResult`.  Numeric fields are ``NaN`` when
            the GPU was unavailable or no frames were profiled.
        """
        if not self._gpu_available or not self._util_pcts:
            return GPUProfilingResult(
                tracker_name=tracker_name,
                frame_count=len(self._util_pcts),
                device_name=self._device_name,
                gpu_utilization_mean_pct=_NAN,
                gpu_utilization_peak_pct=_NAN,
                gpu_memory_used_mean_mb=_NAN,
                gpu_memory_peak_mb=_NAN,
                gpu_power_mean_w=_NAN,
                gpu_power_peak_w=_NAN,
                gpu_available=False,
            )

        util_arr = np.asarray(self._util_pcts, dtype=np.float64)
        mem_arr = np.asarray(self._mem_used_mb, dtype=np.float64)
        power_arr = (
            np.asarray(self._power_w, dtype=np.float64)
            if self._power_w
            else np.array([_NAN])
        )

        return GPUProfilingResult(
            tracker_name=tracker_name,
            frame_count=len(util_arr),
            device_name=self._device_name,
            gpu_utilization_mean_pct=float(util_arr.mean()),
            gpu_utilization_peak_pct=float(util_arr.max()),
            gpu_memory_used_mean_mb=float(mem_arr.mean()),
            gpu_memory_peak_mb=float(mem_arr.max()),
            gpu_power_mean_w=float(np.nanmean(power_arr)),
            gpu_power_peak_w=float(np.nanmax(power_arr)),
            gpu_available=True,
        )

    def reset(self) -> None:
        """Clear all accumulated frame-level samples."""
        self._util_pcts.clear()
        self._mem_used_mb.clear()
        self._power_w.clear()

    def __del__(self) -> None:
        if _NVML_AVAILABLE and self._gpu_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_init_nvml(self) -> None:
        """Attempt to initialise NVML; silently disable on any failure."""
        if not _NVML_AVAILABLE:
            return
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if self._device_id >= device_count:
                return
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_id)
            raw_name = pynvml.nvmlDeviceGetName(self._handle)
            self._device_name = (
                raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            )
            self._gpu_available = True

            # Probe power support — not all GPUs / driver versions expose it.
            try:
                pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self._supports_power = True
            except Exception:
                self._supports_power = False

        except Exception:
            self._gpu_available = False
