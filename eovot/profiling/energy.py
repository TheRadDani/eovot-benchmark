"""CPU-based energy consumption estimator for EOVOT tracker profiling.

Estimates per-frame and per-sequence energy consumption using CPU utilization
and a configurable Thermal Design Power (TDP) value.  This is a practical
approximation suitable for comparing tracker efficiency across devices without
requiring external power-measurement hardware.

Energy model::

    P_cpu(t) = tdp_watts * (cpu_util_pct(t) / 100.0)
    E_frame   = P_cpu * latency_seconds

Caveats:
- TDP is a manufacturer-specified *maximum* power envelope, so estimates are
  an upper bound on actual CPU power draw.
- GPU power is not included; use NVIDIA NVML or tegrastats for GPU-enabled
  trackers.
- For battery-constrained edge devices, replace ``tdp_watts`` with a measured
  device-level idle/load power from the device datasheet.

Reference:
    Patterson et al., "Carbon Emissions and Large Neural Network Training."
    arXiv 2104.10350 (2021) — motivates energy-aware ML benchmarking.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import psutil


@dataclass
class EnergyResult:
    """Energy consumption summary for one tracker run.

    All energy values are derived from CPU utilisation × TDP × elapsed time.

    Attributes:
        tracker_name: Identifier of the profiled tracker.
        frame_count: Number of frames included in the measurement.
        tdp_watts: TDP value (W) used for the estimate.
        total_energy_j: Total estimated energy consumed (Joules).
        mean_power_w: Mean estimated power draw (Watts).
        energy_per_frame_mj: Mean energy per frame (milli-Joules).
        peak_cpu_pct: Highest recorded CPU utilisation (%).
        mean_cpu_pct: Mean CPU utilisation across all frames (%).
    """

    tracker_name: str
    frame_count: int
    tdp_watts: float
    total_energy_j: float
    mean_power_w: float
    energy_per_frame_mj: float
    peak_cpu_pct: float
    mean_cpu_pct: float

    def __str__(self) -> str:
        return (
            f"EnergyResult[{self.tracker_name}] "
            f"total={self.total_energy_j:.4f} J  "
            f"mean_power={self.mean_power_w:.2f} W  "
            f"per_frame={self.energy_per_frame_mj:.3f} mJ  "
            f"cpu={self.mean_cpu_pct:.1f}% (peak {self.peak_cpu_pct:.1f}%)  "
            f"frames={self.frame_count}"
        )

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        return {
            "tracker_name": self.tracker_name,
            "frame_count": self.frame_count,
            "tdp_watts": self.tdp_watts,
            "total_energy_j": round(self.total_energy_j, 6),
            "mean_power_w": round(self.mean_power_w, 4),
            "energy_per_frame_mj": round(self.energy_per_frame_mj, 4),
            "peak_cpu_pct": round(self.peak_cpu_pct, 2),
            "mean_cpu_pct": round(self.mean_cpu_pct, 2),
        }


class EnergyProfiler:
    """Measure per-frame CPU utilisation and estimate energy consumption.

    Wraps ``psutil.cpu_percent`` to sample CPU load at each frame boundary
    and combines it with precise wall-clock timing to estimate energy.

    Args:
        tdp_watts: Thermal Design Power in Watts.  Use the CPU TDP from the
            device datasheet.  Common values:
            - Raspberry Pi 4: ~6 W
            - Jetson Nano:    ~10 W
            - Laptop CPU:     ~15–28 W
            - Desktop CPU:    ~65–125 W
            Defaults to ``15.0`` (conservative laptop CPU estimate).

    Example::

        profiler = EnergyProfiler(tdp_watts=10.0)
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

    def __init__(self, tdp_watts: float = 15.0) -> None:
        if tdp_watts <= 0:
            raise ValueError(f"tdp_watts must be positive, got {tdp_watts}")
        self.tdp_watts = tdp_watts
        self._process = psutil.Process(os.getpid())
        self._latencies_s: List[float] = []
        self._cpu_pcts: List[float] = []
        self._t0: Optional[float] = None
        # Prime psutil's CPU percent baseline (first call always returns 0.0).
        psutil.cpu_percent(interval=None)

    def start_frame(self) -> None:
        """Mark the start of a tracker update call."""
        self._t0 = time.perf_counter()

    def end_frame(self) -> float:
        """Mark the end of a tracker update call.

        Samples CPU utilisation and records elapsed time.

        Returns:
            Estimated energy consumed during this frame (milli-Joules).

        Raises:
            RuntimeError: If called without a preceding :meth:`start_frame`.
        """
        if self._t0 is None:
            raise RuntimeError("end_frame() called before start_frame()")
        elapsed_s = time.perf_counter() - self._t0
        self._t0 = None

        # Non-blocking sample: reflects CPU usage since the last call.
        cpu_pct = psutil.cpu_percent(interval=None)
        self._latencies_s.append(elapsed_s)
        self._cpu_pcts.append(cpu_pct)

        energy_mj = self._frame_energy_mj(cpu_pct, elapsed_s)
        return energy_mj

    def summary(self, tracker_name: str = "unknown") -> EnergyResult:
        """Return aggregated :class:`EnergyResult` for the profiled run.

        Args:
            tracker_name: Identifier embedded in the result object.

        Raises:
            ValueError: If no frames have been profiled yet.
        """
        if not self._latencies_s:
            raise ValueError("No frames profiled — call start_frame/end_frame first.")

        lat_arr = np.array(self._latencies_s)
        cpu_arr = np.array(self._cpu_pcts)

        # Per-frame energy (J) = TDP × cpu_fraction × elapsed_s
        energies_j = self.tdp_watts * (cpu_arr / 100.0) * lat_arr
        total_j = float(energies_j.sum())
        mean_power = total_j / float(lat_arr.sum()) if lat_arr.sum() > 0 else 0.0

        return EnergyResult(
            tracker_name=tracker_name,
            frame_count=len(lat_arr),
            tdp_watts=self.tdp_watts,
            total_energy_j=total_j,
            mean_power_w=mean_power,
            energy_per_frame_mj=float(energies_j.mean()) * 1_000.0,
            peak_cpu_pct=float(cpu_arr.max()),
            mean_cpu_pct=float(cpu_arr.mean()),
        )

    def reset(self) -> None:
        """Clear all accumulated measurements."""
        self._latencies_s.clear()
        self._cpu_pcts.clear()
        self._t0 = None
        psutil.cpu_percent(interval=None)  # re-prime baseline

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _frame_energy_mj(self, cpu_pct: float, elapsed_s: float) -> float:
        """Compute single-frame energy estimate in milli-Joules."""
        return self.tdp_watts * (cpu_pct / 100.0) * elapsed_s * 1_000.0
