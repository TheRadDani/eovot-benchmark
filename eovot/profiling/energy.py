"""CPU-based energy estimation for edge-aware tracker evaluation.

On embedded and edge hardware (Raspberry Pi, Jetson Nano, mobile SoCs),
energy consumption is a first-class deployment constraint alongside latency
and memory.  This module provides a lightweight, cross-platform energy
estimator based on CPU utilisation and a configurable Thermal Design Power
(TDP) budget — no hardware power meters required.

Estimation model
----------------
At each frame boundary the process's CPU utilisation *u* (0–100 %) is
sampled via ``psutil``.  The instantaneous power estimate is::

    P_est  =  (u / 100) * tdp_watts          [W]

The energy consumed during frame processing of duration *Δt* seconds is::

    E_frame  =  P_est * Δt * 1000            [mJ]

Typical TDP values for reference platforms:

============================================  ======
Platform                                        TDP
============================================  ======
Raspberry Pi 4 (SoC, not full board)           6 W
NVIDIA Jetson Nano (10 W mode)                10 W
Intel Core i5-1135G7 (laptop, 2021)           28 W
Apple M1 (efficiency cores only)               3 W
Generic x86-64 desktop                        65 W
============================================  ======

Usage
-----
Standalone (wraps an existing :class:`~eovot.profiling.profiler.Profiler`)::

    from eovot.profiling.energy import EnergyEstimator

    estimator = EnergyEstimator(tdp_watts=10.0)   # Jetson Nano
    for i, frame in enumerate(sequence):
        if i == 0:
            tracker.initialize(frame, bbox)
        else:
            estimator.start_frame()
            bbox = tracker.update(frame)
            estimator.end_frame()

    result = estimator.summary(tracker_name="MOSSE")
    print(result)
    # EnergyResult[MOSSE] 142.3 mJ total | 0.84 mJ/frame | avg 18.7 mW

Notes
-----
- CPU utilisation is measured *per process* via ``psutil.Process.cpu_percent``.
  The first call after a ``reset()`` always returns 0.0 (psutil initialisation
  artefact); this sample is excluded from statistics.
- The estimate is approximate.  For accurate measurements use a hardware power
  analyser (e.g. INA219 breakout, Monsoon AAA) and integrate over time.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import psutil


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EnergyResult:
    """Energy profiling summary for one tracker run.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        frame_count: Number of frames profiled (excludes first-frame init).
        tdp_watts: Assumed TDP used for estimation.
        total_energy_mj: Estimated total energy consumed over the sequence (mJ).
        mean_energy_per_frame_mj: Mean energy per frame (mJ).
        mean_power_mw: Mean instantaneous power draw during inference (mW).
        peak_power_mw: Peak estimated power sample observed (mW).
    """

    tracker_name: str
    frame_count: int
    tdp_watts: float
    total_energy_mj: float
    mean_energy_per_frame_mj: float
    mean_power_mw: float
    peak_power_mw: float

    def __str__(self) -> str:
        return (
            f"EnergyResult[{self.tracker_name}] "
            f"{self.total_energy_mj:.1f} mJ total | "
            f"{self.mean_energy_per_frame_mj:.3f} mJ/frame | "
            f"avg {self.mean_power_mw:.1f} mW | "
            f"peak {self.peak_power_mw:.1f} mW  "
            f"(TDP={self.tdp_watts:.0f} W, {self.frame_count} frames)"
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all fields."""
        return {
            "tracker_name": self.tracker_name,
            "frame_count": self.frame_count,
            "tdp_watts": self.tdp_watts,
            "total_energy_mj": round(self.total_energy_mj, 4),
            "mean_energy_per_frame_mj": round(self.mean_energy_per_frame_mj, 4),
            "mean_power_mw": round(self.mean_power_mw, 2),
            "peak_power_mw": round(self.peak_power_mw, 2),
        }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class EnergyEstimator:
    """Estimate per-frame energy consumption from CPU utilisation and TDP.

    Args:
        tdp_watts: Assumed Thermal Design Power of the target device in
            Watts.  Defaults to 15 W (a reasonable mid-point for an edge
            CPU such as a Raspberry Pi CM4 or Jetson Nano in 10 W mode).
            See module docstring for common platform values.

    Example::

        from eovot.profiling.energy import EnergyEstimator

        estimator = EnergyEstimator(tdp_watts=10.0)   # Jetson Nano
        estimator.reset()

        for i, frame in enumerate(sequence):
            if i == 0:
                tracker.initialize(frame, bbox)
            else:
                estimator.start_frame()
                bbox = tracker.update(frame)
                estimator.end_frame()

        result = estimator.summary("MyTracker")
        print(result.total_energy_mj)
    """

    def __init__(self, tdp_watts: float = 15.0) -> None:
        if tdp_watts <= 0:
            raise ValueError(f"tdp_watts must be positive, got {tdp_watts}")
        self.tdp_watts = tdp_watts
        self._process = psutil.Process(os.getpid())
        # Warm up the cpu_percent counter so first real sample is meaningful.
        self._process.cpu_percent(interval=None)

        self._frame_energies_mj: List[float] = []
        self._frame_powers_mw: List[float] = []
        self._t0: Optional[float] = None

    def start_frame(self) -> None:
        """Mark the start of a tracker update step.

        Must be called immediately before the tracker's ``update()`` call so
        that elapsed time measurement is as tight as possible.
        """
        # Sample CPU utilisation *before* the frame starts so the delta window
        # covers the upcoming processing interval.
        self._process.cpu_percent(interval=None)  # reset the window
        self._t0 = time.perf_counter()

    def end_frame(self) -> float:
        """Mark the end of a tracker update step.

        Returns:
            Estimated energy consumed during this frame in millijoules.

        Raises:
            RuntimeError: If called without a preceding :meth:`start_frame`.
        """
        if self._t0 is None:
            raise RuntimeError("end_frame() called before start_frame()")

        elapsed_s = time.perf_counter() - self._t0
        self._t0 = None

        # cpu_percent returns utilisation as 0–100 over the interval since the
        # last call.  For a single-core TDP model we normalise by logical CPUs.
        raw_util = self._process.cpu_percent(interval=None)
        n_cpu = psutil.cpu_count(logical=True) or 1
        # Clamp to [0, 100] regardless of multi-threaded over-counting.
        util_fraction = min(raw_util / n_cpu, 100.0) / 100.0

        power_w = util_fraction * self.tdp_watts
        energy_mj = power_w * elapsed_s * 1_000.0

        self._frame_powers_mw.append(power_w * 1_000.0)
        self._frame_energies_mj.append(energy_mj)
        return energy_mj

    def summary(self, tracker_name: str = "unknown") -> EnergyResult:
        """Return aggregated :class:`EnergyResult` over all profiled frames.

        Args:
            tracker_name: Name of the tracker for labelling.

        Raises:
            ValueError: If no frames have been profiled yet.
        """
        if not self._frame_energies_mj:
            raise ValueError(
                "No frames profiled — call start_frame()/end_frame() at least once."
            )

        total_mj = sum(self._frame_energies_mj)
        n = len(self._frame_energies_mj)
        mean_mj = total_mj / n
        mean_power = sum(self._frame_powers_mw) / n
        peak_power = max(self._frame_powers_mw)

        return EnergyResult(
            tracker_name=tracker_name,
            frame_count=n,
            tdp_watts=self.tdp_watts,
            total_energy_mj=total_mj,
            mean_energy_per_frame_mj=mean_mj,
            mean_power_mw=mean_power,
            peak_power_mw=peak_power,
        )

    def reset(self) -> None:
        """Clear accumulated statistics and re-prime the CPU counter."""
        self._frame_energies_mj.clear()
        self._frame_powers_mw.clear()
        self._t0 = None
        # Re-prime so the next end_frame() delta window starts fresh.
        self._process.cpu_percent(interval=None)
