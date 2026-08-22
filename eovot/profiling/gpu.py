"""GPU-aware profiler for EOVOT tracker evaluation.

Measures GPU utilisation, VRAM occupancy, and power draw during tracker
inference by polling ``nvidia-smi`` in a background thread.  Falls back
gracefully to CPU-only mode when no NVIDIA GPU or ``nvidia-smi`` binary is
available, so the profiler is usable on any hardware.

Energy model (GPU path)::

    P_gpu(t)  = gpu_power_w(t)          # direct nvidia-smi reading
    E_frame   = mean(P_gpu) * latency_s

This complements :class:`~eovot.profiling.energy.EnergyProfiler`, which
models CPU power from TDP × utilisation.  The two profilers may be used
simultaneously to capture total system power on GPU-accelerated trackers.

Typical usage::

    profiler = GPUProfiler(device_index=0, poll_interval_ms=10)
    for i, frame in enumerate(sequence):
        if i == 0:
            tracker.initialize(frame, bbox)
        else:
            profiler.start_frame()
            bbox = tracker.update(frame)
            result_mj = profiler.end_frame()
    gpu_result = profiler.summary("my_tracker")
    print(gpu_result)

Reference:
    NVIDIA Management Library (NVML) / nvidia-smi documentation.
    https://developer.nvidia.com/management-library-nvml
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class GPUProfilingResult:
    """Hardware profiling summary including GPU metrics for one tracker run.

    Attributes:
        tracker_name: Identifier of the profiled tracker.
        frame_count: Number of tracker-update frames profiled.
        device_index: CUDA device index that was monitored.
        gpu_available: ``True`` when an NVIDIA GPU and ``nvidia-smi`` were
            found; ``False`` means all GPU fields are ``0.0`` / ``None``.
        peak_gpu_memory_mb: Highest VRAM usage observed (MiB).
        mean_gpu_utilization_pct: Mean GPU core utilisation across all frames.
        peak_gpu_utilization_pct: Highest GPU core utilisation observed.
        mean_gpu_power_w: Mean GPU board power draw (Watts).
        peak_gpu_power_w: Highest GPU board power draw observed (Watts).
        total_gpu_energy_j: Total estimated GPU energy (Joules).
        mean_gpu_energy_per_frame_mj: Mean GPU energy per tracker-update
            frame (milli-Joules).
        gpu_name: Device name string returned by ``nvidia-smi``, or ``""``
            when no GPU is available.
    """

    tracker_name: str
    frame_count: int
    device_index: int
    gpu_available: bool
    peak_gpu_memory_mb: float
    mean_gpu_utilization_pct: float
    peak_gpu_utilization_pct: float
    mean_gpu_power_w: float
    peak_gpu_power_w: float
    total_gpu_energy_j: float
    mean_gpu_energy_per_frame_mj: float
    gpu_name: str = ""

    def __str__(self) -> str:
        if not self.gpu_available:
            return (
                f"GPUProfilingResult[{self.tracker_name}] "
                f"gpu_available=False  frames={self.frame_count}"
            )
        return (
            f"GPUProfilingResult[{self.tracker_name}] "
            f"gpu={self.gpu_name}  "
            f"vram_peak={self.peak_gpu_memory_mb:.1f} MiB  "
            f"util={self.mean_gpu_utilization_pct:.1f}% "
            f"(peak {self.peak_gpu_utilization_pct:.1f}%)  "
            f"power={self.mean_gpu_power_w:.1f} W "
            f"(peak {self.peak_gpu_power_w:.1f} W)  "
            f"energy={self.total_gpu_energy_j:.4f} J  "
            f"per_frame={self.mean_gpu_energy_per_frame_mj:.3f} mJ  "
            f"frames={self.frame_count}"
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict representation."""
        return {
            "tracker_name": self.tracker_name,
            "frame_count": self.frame_count,
            "device_index": self.device_index,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 2),
            "mean_gpu_utilization_pct": round(self.mean_gpu_utilization_pct, 2),
            "peak_gpu_utilization_pct": round(self.peak_gpu_utilization_pct, 2),
            "mean_gpu_power_w": round(self.mean_gpu_power_w, 3),
            "peak_gpu_power_w": round(self.peak_gpu_power_w, 3),
            "total_gpu_energy_j": round(self.total_gpu_energy_j, 6),
            "mean_gpu_energy_per_frame_mj": round(self.mean_gpu_energy_per_frame_mj, 4),
        }


# ---------------------------------------------------------------------------
# Internal nvidia-smi polling helper
# ---------------------------------------------------------------------------


class _NvidiaSMIPoller:
    """Background thread that continuously samples ``nvidia-smi`` output.

    Polls ``nvidia-smi`` at a configurable interval and stores raw samples
    so that :class:`GPUProfiler` can slice per-frame windows from the stream.

    Args:
        device_index: CUDA device index to monitor.
        poll_interval_ms: Approximate sampling interval in milliseconds.
            Lower values give finer-grained per-frame averages at the cost
            of extra subprocess overhead.  Default: ``10`` ms.
    """

    # Fields queried from nvidia-smi for each sample.
    _FIELDS = "utilization.gpu,memory.used,power.draw"
    _FIELD_COUNT = 3

    def __init__(self, device_index: int = 0, poll_interval_ms: float = 10.0) -> None:
        self._device_index = device_index
        self._interval_s = poll_interval_ms / 1_000.0
        self._lock = threading.Lock()
        # Parallel lists of per-sample readings.
        self._timestamps: List[float] = []
        self._util_pcts: List[float] = []
        self._memory_mbs: List[float] = []
        self._power_ws: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.gpu_name: str = ""

    def start(self) -> None:
        """Launch the background polling thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def samples_since(self, t_start: float) -> dict:
        """Return all raw samples collected at or after *t_start*.

        Returns a dict with keys ``util_pcts``, ``memory_mbs``, ``power_ws``,
        and ``timestamps`` — each a list of floats.  Returns empty lists if
        no samples fall in the window.
        """
        with self._lock:
            idx = _bisect_left(self._timestamps, t_start)
            return {
                "util_pcts": list(self._util_pcts[idx:]),
                "memory_mbs": list(self._memory_mbs[idx:]),
                "power_ws": list(self._power_ws[idx:]),
                "timestamps": list(self._timestamps[idx:]),
            }

    def _poll_loop(self) -> None:
        """Blocking loop; runs in a daemon thread."""
        # Fetch GPU name once.
        self.gpu_name = self._query_gpu_name()
        while not self._stop_event.is_set():
            sample = self._query_sample()
            if sample is not None:
                ts = time.monotonic()
                with self._lock:
                    self._timestamps.append(ts)
                    self._util_pcts.append(sample[0])
                    self._memory_mbs.append(sample[1])
                    self._power_ws.append(sample[2])
            self._stop_event.wait(self._interval_s)

    def _query_sample(self) -> Optional[tuple]:
        """Run a single ``nvidia-smi`` query.  Returns ``None`` on failure."""
        cmd = [
            "nvidia-smi",
            f"--id={self._device_index}",
            f"--query-gpu={self._FIELDS}",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=1.0)
            parts = out.decode().strip().split(",")
            if len(parts) < self._FIELD_COUNT:
                return None
            util = float(parts[0].strip())
            mem = float(parts[1].strip())
            # Power draw may read as "N/A" on some boards (e.g. display GPUs).
            power_raw = parts[2].strip()
            power = float(power_raw) if power_raw not in ("N/A", "[N/A]") else 0.0
            return (util, mem, power)
        except Exception:
            return None

    def _query_gpu_name(self) -> str:
        cmd = [
            "nvidia-smi",
            f"--id={self._device_index}",
            "--query-gpu=gpu_name",
            "--format=csv,noheader",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=2.0)
            return out.decode().strip()
        except Exception:
            return ""


def _bisect_left(lst: List[float], value: float) -> int:
    """Return the leftmost index in *lst* where ``lst[i] >= value``."""
    lo, hi = 0, len(lst)
    while lo < hi:
        mid = (lo + hi) // 2
        if lst[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Public profiler
# ---------------------------------------------------------------------------


class GPUProfiler:
    """Collect per-frame GPU utilisation, VRAM, and power during tracker inference.

    Starts a daemon thread that continuously samples ``nvidia-smi`` and slices
    the samples into per-frame windows when :meth:`start_frame` /
    :meth:`end_frame` are called.  Falls back to CPU-only mode silently when
    ``nvidia-smi`` is not available (e.g., no NVIDIA GPU, CUDA not installed,
    running on ARM-only board).

    Args:
        device_index: CUDA device index to monitor.  Default: ``0``.
        poll_interval_ms: Background sampling interval in milliseconds.
            Values between 5 ms and 50 ms offer a good accuracy/overhead
            trade-off for typical tracker frame times (10–200 ms).
            Default: ``10`` ms.

    Example::

        profiler = GPUProfiler(device_index=0)
        for i, frame in enumerate(sequence):
            if i == 0:
                tracker.initialize(frame, bbox)
            else:
                profiler.start_frame()
                bbox = tracker.update(frame)
                profiler.end_frame()
        result = profiler.summary("my_tracker")
        print(result)
    """

    def __init__(self, device_index: int = 0, poll_interval_ms: float = 10.0) -> None:
        self._device_index = device_index
        self._poll_interval_ms = poll_interval_ms
        self._available = self._check_nvidiasmi()
        self._poller: Optional[_NvidiaSMIPoller] = None

        # Per-frame accumulated statistics.
        self._frame_util_pcts: List[float] = []
        self._frame_memory_mbs: List[float] = []
        self._frame_power_ws: List[float] = []
        self._frame_latencies_s: List[float] = []
        self._t_frame_start: Optional[float] = None

        if self._available:
            self._poller = _NvidiaSMIPoller(
                device_index=device_index,
                poll_interval_ms=poll_interval_ms,
            )
            self._poller.start()

    # ------------------------------------------------------------------
    # Frame-level interface
    # ------------------------------------------------------------------

    def start_frame(self) -> None:
        """Mark the beginning of one tracker-update call."""
        self._t_frame_start = time.monotonic()

    def end_frame(self) -> float:
        """Mark the end of one tracker-update call.

        Slices per-frame GPU samples from the background poller and appends
        aggregated stats.

        Returns:
            Estimated GPU energy for this frame in milli-Joules, or ``0.0``
            when no GPU is available.

        Raises:
            RuntimeError: If called without a preceding :meth:`start_frame`.
        """
        if self._t_frame_start is None:
            raise RuntimeError("end_frame() called before start_frame()")
        t_end = time.monotonic()
        elapsed_s = t_end - self._t_frame_start
        self._frame_latencies_s.append(elapsed_s)
        self._t_frame_start = None

        if not self._available or self._poller is None:
            self._frame_util_pcts.append(0.0)
            self._frame_memory_mbs.append(0.0)
            self._frame_power_ws.append(0.0)
            return 0.0

        samples = self._poller.samples_since(t_end - elapsed_s)
        util_pcts = samples["util_pcts"] or [0.0]
        memory_mbs = samples["memory_mbs"] or [0.0]
        power_ws = samples["power_ws"] or [0.0]

        mean_util = float(np.mean(util_pcts))
        peak_mem = float(np.max(memory_mbs))
        mean_power = float(np.mean(power_ws))

        self._frame_util_pcts.append(mean_util)
        self._frame_memory_mbs.append(peak_mem)
        self._frame_power_ws.append(mean_power)

        energy_mj = mean_power * elapsed_s * 1_000.0
        return energy_mj

    def summary(self, tracker_name: str = "unknown") -> GPUProfilingResult:
        """Return aggregated :class:`GPUProfilingResult` for the profiled run.

        Args:
            tracker_name: Identifier embedded in the result object.

        Raises:
            ValueError: If no frames have been profiled yet.
        """
        if not self._frame_latencies_s:
            raise ValueError("No frames profiled — call start_frame/end_frame first.")

        lat_arr = np.array(self._frame_latencies_s)
        util_arr = np.array(self._frame_util_pcts)
        mem_arr = np.array(self._frame_memory_mbs)
        power_arr = np.array(self._frame_power_ws)

        # Energy per frame: power_w × latency_s × 1000 → mJ
        energy_mj_arr = power_arr * lat_arr * 1_000.0
        total_energy_j = float(energy_mj_arr.sum()) / 1_000.0

        gpu_name = self._poller.gpu_name if self._poller is not None else ""

        return GPUProfilingResult(
            tracker_name=tracker_name,
            frame_count=len(lat_arr),
            device_index=self._device_index,
            gpu_available=self._available,
            gpu_name=gpu_name,
            peak_gpu_memory_mb=float(mem_arr.max()),
            mean_gpu_utilization_pct=float(util_arr.mean()),
            peak_gpu_utilization_pct=float(util_arr.max()),
            mean_gpu_power_w=float(power_arr.mean()),
            peak_gpu_power_w=float(power_arr.max()),
            total_gpu_energy_j=total_energy_j,
            mean_gpu_energy_per_frame_mj=float(energy_mj_arr.mean()),
        )

    def reset(self) -> None:
        """Clear all accumulated per-frame statistics.

        The background polling thread continues running; only the per-frame
        aggregation buffers are reset.  Call this between sequences to get
        per-sequence GPU summaries from a single profiler instance.
        """
        self._frame_util_pcts.clear()
        self._frame_memory_mbs.clear()
        self._frame_power_ws.clear()
        self._frame_latencies_s.clear()
        self._t_frame_start = None

    def close(self) -> None:
        """Stop the background polling thread.

        Call this when the profiler is no longer needed.  Accessing
        :meth:`start_frame` or :meth:`end_frame` after :meth:`close` will
        still work (returning 0.0 for GPU energy), but no new samples will be
        collected.
        """
        if self._poller is not None:
            self._poller.stop()
            self._poller = None

    def __enter__(self) -> "GPUProfiler":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_nvidiasmi() -> bool:
        """Return ``True`` if ``nvidia-smi`` is available and reports a GPU."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            count = int(out.decode().strip().split("\n")[0])
            return count > 0
        except Exception:
            return False
