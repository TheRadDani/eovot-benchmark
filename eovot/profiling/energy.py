"""Hardware-aware energy profiling for EOVOT.

Estimates energy consumption of tracker inference across different hardware
platforms.  Supports three measurement backends, selected automatically:

1. **NVIDIA GPU** (discrete GPU / Jetson GPU) — reads instantaneous power via
   ``nvidia-smi --query-gpu=power.draw`` at a configurable sampling interval.
2. **RAPL** (Running Average Power Limit, Intel/AMD x86) — reads the
   ``/sys/class/powercap/intel-rapl/`` energy counters directly.  Requires
   either root privileges or the ``energy_perf_policy`` capability.
3. **CPU estimation** (portable fallback) — combines psutil CPU frequency and
   utilisation readings with a TDP-based linear model to produce a rough
   power estimate.  Accuracy ±30–50 %; suitable for relative comparisons.

Usage::

    from eovot.profiling.energy import EnergyProfiler, HardwareBackend

    profiler = EnergyProfiler()          # auto-detects backend
    print(profiler.backend)              # e.g. HardwareBackend.NVIDIA_GPU

    profiler.start()
    for frame in sequence:
        tracker.update(frame)
    result = profiler.stop()

    print(f"Energy: {result.total_energy_j:.3f} J")
    print(f"Mean power: {result.mean_power_w:.2f} W")
    print(f"Backend: {result.backend}")
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import psutil


class HardwareBackend(str, Enum):
    """Available energy measurement backends."""

    NVIDIA_GPU = "nvidia_gpu"
    """NVIDIA discrete GPU or Jetson GPU via nvidia-smi."""

    RAPL = "rapl"
    """Intel/AMD RAPL energy counters (x86 systems)."""

    CPU_ESTIMATE = "cpu_estimate"
    """Portable CPU utilisation + TDP estimate (fallback)."""


@dataclass
class EnergyResult:
    """Energy consumption summary for a single tracker run.

    All power values are in **Watts**; energy values are in **Joules**.
    Wall-clock elapsed time is in **seconds**.
    """

    backend: HardwareBackend
    """Which measurement backend produced these readings."""

    elapsed_s: float
    """Total wall-clock time between :meth:`EnergyProfiler.start` and
    :meth:`EnergyProfiler.stop`."""

    power_samples_w: List[float] = field(default_factory=list)
    """Raw instantaneous power readings in Watts."""

    total_energy_j: float = 0.0
    """Estimated total energy consumed in Joules (= ∫ P dt)."""

    mean_power_w: float = 0.0
    """Mean power draw in Watts."""

    peak_power_w: float = 0.0
    """Peak instantaneous power draw in Watts."""

    device_name: str = "unknown"
    """Human-readable device label (e.g. GPU model or CPU brand string)."""

    def __str__(self) -> str:
        return (
            f"EnergyResult[{self.backend.value}] "
            f"energy={self.total_energy_j:.3f} J  "
            f"mean_power={self.mean_power_w:.2f} W  "
            f"peak_power={self.peak_power_w:.2f} W  "
            f"elapsed={self.elapsed_s:.2f} s  "
            f"device={self.device_name}"
        )

    def to_dict(self) -> dict:
        return {
            "backend": self.backend.value,
            "elapsed_s": round(self.elapsed_s, 3),
            "total_energy_j": round(self.total_energy_j, 4),
            "mean_power_w": round(self.mean_power_w, 4),
            "peak_power_w": round(self.peak_power_w, 4),
            "device_name": self.device_name,
            "num_samples": len(self.power_samples_w),
        }


class EnergyProfiler:
    """Measure energy consumption of inference workloads.

    Args:
        backend: Force a specific backend.  When ``None`` (default), the best
            available backend is selected automatically via
            :meth:`detect_backend`.
        sample_interval_s: Polling interval in seconds between power readings.
            Lower values give finer resolution but add overhead.  Default: 0.1.
        cpu_tdp_w: Assumed CPU Thermal Design Power (TDP) in Watts, used by
            the ``CPU_ESTIMATE`` backend.  Defaults to a conservative 15 W
            (suitable for laptop/embedded CPUs).  Set to your CPU's actual TDP
            for better accuracy.

    Example::

        profiler = EnergyProfiler(sample_interval_s=0.05)
        profiler.start()
        run_tracker(frames)
        result = profiler.stop()
        print(result)
    """

    def __init__(
        self,
        backend: Optional[HardwareBackend] = None,
        sample_interval_s: float = 0.1,
        cpu_tdp_w: float = 15.0,
    ) -> None:
        self.sample_interval_s = sample_interval_s
        self.cpu_tdp_w = cpu_tdp_w
        self.backend = backend or self.detect_backend()

        self._t_start: Optional[float] = None
        self._samples: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # RAPL: cache the energy counter file paths at init time.
        self._rapl_paths: List[Path] = _discover_rapl_paths()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin power sampling in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("EnergyProfiler.start() called while already running")
        self._samples.clear()
        self._stop_event.clear()
        self._t_start = time.perf_counter()

        if self.backend == HardwareBackend.RAPL:
            # For RAPL, read the initial counter and let the background thread
            # record deltas.
            self._rapl_start_j = _read_rapl_energy_j(self._rapl_paths)
        elif self.backend == HardwareBackend.NVIDIA_GPU:
            # Prime the nvidia-smi query (first call is slower).
            _nvidia_smi_power_w()

        target = {
            HardwareBackend.NVIDIA_GPU: self._sample_nvidia,
            HardwareBackend.RAPL: self._sample_rapl,
            HardwareBackend.CPU_ESTIMATE: self._sample_cpu,
        }[self.backend]

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> EnergyResult:
        """Stop sampling and return an :class:`EnergyResult`.

        Raises:
            RuntimeError: If :meth:`start` has not been called.
        """
        if self._t_start is None:
            raise RuntimeError("EnergyProfiler.stop() called before start()")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.sample_interval_s * 10)

        elapsed_s = time.perf_counter() - self._t_start

        if not self._samples:
            # Fallback: use a single CPU estimate for the whole window.
            self._samples = [_cpu_power_estimate(self.cpu_tdp_w)]

        mean_w = float(sum(self._samples) / len(self._samples))
        peak_w = float(max(self._samples))

        # Energy ≈ mean power × elapsed time (trapezoidal approximation)
        if len(self._samples) > 1:
            dt = elapsed_s / (len(self._samples) - 1)
            total_j = float(sum(
                (self._samples[i] + self._samples[i + 1]) / 2 * dt
                for i in range(len(self._samples) - 1)
            ))
        else:
            total_j = mean_w * elapsed_s

        result = EnergyResult(
            backend=self.backend,
            elapsed_s=elapsed_s,
            power_samples_w=list(self._samples),
            total_energy_j=total_j,
            mean_power_w=mean_w,
            peak_power_w=peak_w,
            device_name=self._device_name(),
        )

        # Reset for next run.
        self._t_start = None
        self._thread = None
        return result

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_backend() -> HardwareBackend:
        """Auto-detect the best available energy measurement backend.

        Priority: NVIDIA GPU > RAPL > CPU estimate.

        Returns:
            The most capable :class:`HardwareBackend` available on this system.
        """
        if _nvidia_smi_available():
            return HardwareBackend.NVIDIA_GPU
        if _rapl_available():
            return HardwareBackend.RAPL
        return HardwareBackend.CPU_ESTIMATE

    # ------------------------------------------------------------------
    # Background sampling threads
    # ------------------------------------------------------------------

    def _sample_nvidia(self) -> None:
        while not self._stop_event.is_set():
            w = _nvidia_smi_power_w()
            if w is not None:
                self._samples.append(w)
            self._stop_event.wait(timeout=self.sample_interval_s)

    def _sample_rapl(self) -> None:
        prev_j = self._rapl_start_j
        prev_t = time.perf_counter()
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.sample_interval_s)
            curr_j = _read_rapl_energy_j(self._rapl_paths)
            curr_t = time.perf_counter()
            dt = curr_t - prev_t
            if dt > 0 and curr_j >= prev_j:
                power_w = (curr_j - prev_j) / dt
                self._samples.append(power_w)
            prev_j, prev_t = curr_j, curr_t

    def _sample_cpu(self) -> None:
        while not self._stop_event.is_set():
            self._samples.append(_cpu_power_estimate(self.cpu_tdp_w))
            self._stop_event.wait(timeout=self.sample_interval_s)

    # ------------------------------------------------------------------
    # Device name helpers
    # ------------------------------------------------------------------

    def _device_name(self) -> str:
        if self.backend == HardwareBackend.NVIDIA_GPU:
            return _nvidia_device_name()
        if self.backend == HardwareBackend.RAPL:
            return _cpu_brand_string()
        return _cpu_brand_string()


# ---------------------------------------------------------------------------
# NVIDIA-smi helpers
# ---------------------------------------------------------------------------

def _nvidia_smi_available() -> bool:
    """Return True if nvidia-smi is present and returns valid power readings."""
    return _nvidia_smi_power_w() is not None


def _nvidia_smi_power_w() -> Optional[float]:
    """Query instantaneous GPU power draw via nvidia-smi.

    Returns:
        Power in Watts, or ``None`` if nvidia-smi is unavailable or fails.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            timeout=2.0,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        lines = [ln.strip() for ln in out.splitlines() if ln.strip() and ln.strip() != "N/A"]
        if lines:
            # Sum across all GPUs if multi-GPU.
            return sum(float(ln) for ln in lines)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            ValueError):
        pass
    return None


def _nvidia_device_name() -> str:
    """Return a human-readable GPU name via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=2.0,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out.splitlines()[0].strip() if out else "NVIDIA GPU"
    except Exception:
        return "NVIDIA GPU"


# ---------------------------------------------------------------------------
# RAPL helpers (Intel/AMD x86 Linux)
# ---------------------------------------------------------------------------

_RAPL_ROOT = Path("/sys/class/powercap/intel-rapl")


def _rapl_available() -> bool:
    """Return True if RAPL energy counters are accessible."""
    paths = _discover_rapl_paths()
    return bool(paths)


def _discover_rapl_paths() -> List[Path]:
    """Find all readable RAPL energy_uj counter files."""
    paths: List[Path] = []
    if not _RAPL_ROOT.is_dir():
        return paths
    for domain_dir in sorted(_RAPL_ROOT.iterdir()):
        counter = domain_dir / "energy_uj"
        if counter.exists() and counter.is_file():
            try:
                counter.read_text()
                paths.append(counter)
            except PermissionError:
                pass
    return paths


def _read_rapl_energy_j(paths: List[Path]) -> float:
    """Read the sum of RAPL energy counters in Joules."""
    total_uj = 0.0
    for path in paths:
        try:
            total_uj += float(path.read_text().strip())
        except (ValueError, OSError):
            pass
    return total_uj / 1e6  # µJ → J


# ---------------------------------------------------------------------------
# CPU estimation helpers (portable fallback)
# ---------------------------------------------------------------------------

def _cpu_power_estimate(tdp_w: float) -> float:
    """Estimate instantaneous CPU power from utilisation and frequency.

    Model: ``P ≈ TDP × cpu_util × (freq / max_freq)^3``

    The cubic frequency scaling reflects the approximate relationship between
    dynamic power and clock frequency under voltage scaling (DVFS).

    Args:
        tdp_w: Assumed CPU TDP in Watts.

    Returns:
        Estimated instantaneous power in Watts.
    """
    util = psutil.cpu_percent(interval=None) / 100.0
    freq_info = psutil.cpu_freq()
    if freq_info is not None and freq_info.max > 0:
        freq_ratio = min(freq_info.current / freq_info.max, 1.0)
    else:
        freq_ratio = 1.0

    # Idle power floor: assume ~10% of TDP at 0% utilisation.
    idle_fraction = 0.10
    dynamic_fraction = (1.0 - idle_fraction) * util * (freq_ratio ** 3)
    return tdp_w * (idle_fraction + dynamic_fraction)


def _cpu_brand_string() -> str:
    """Return the CPU brand string from /proc/cpuinfo if available."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "CPU"
