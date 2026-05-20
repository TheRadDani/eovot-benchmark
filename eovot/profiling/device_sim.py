"""Edge device simulation for EOVOT benchmark results.

Projects host-machine ProfilingResults onto known edge hardware profiles using
a CPU speed-factor model, memory limit checks, device-specific TDP energy
estimates, and a thermal throttling curve for sustained-load scenarios.

This module lets researchers answer "how would tracker X perform on a
Raspberry Pi 4?" without physical access to that hardware, enabling large-scale
edge deployment analysis from a single benchmark run.

Built-in profiles
-----------------
    rpi4          – Raspberry Pi 4B (Cortex-A72 @ 1.5 GHz, 4 GB, 7.5 W)
    rpi5          – Raspberry Pi 5  (Cortex-A76 @ 2.4 GHz, 8 GB, 12 W)
    jetson_nano   – NVIDIA Jetson Nano (Cortex-A57 @ 1.43 GHz, 4 GB, 10 W)
    jetson_xnx    – NVIDIA Jetson Xavier NX (Carmel @ 1.9 GHz, 8 GB, 15 W)
    coral_board   – Google Coral Dev Board (i.MX 8M @ 1.5 GHz, 1 GB, 4 W)
    snapdragon888 – Qualcomm Snapdragon 888 mobile (Kryo 680 @ 2.84 GHz, 6 GB, 15 W)

Speed factors are calibrated against a reference Intel Core i7 class host
(~2 500 Cinebench R23 single-thread points). If your benchmark machine is
significantly faster or slower, supply ``host_calibration_factor`` accordingly.

Example::

    from eovot.profiling.device_sim import DeviceSimulator

    sim = DeviceSimulator()

    # Single device
    result = sim.simulate(profiling_result, "rpi4", sustained_seconds=120.0)
    print(result)

    # All built-in devices — great for a paper's edge-fleet table
    all_results = sim.simulate_all(profiling_result, sustained_seconds=60.0)
    print(sim.to_markdown_table(all_results))

    # Register a custom board
    from eovot.profiling.device_sim import DeviceProfile
    sim.register_device("my_board", DeviceProfile(
        name="my_board",
        display_name="Custom SBC",
        cpu_speed_factor=0.20,
        memory_limit_mb=2048.0,
        tdp_watts=8.0,
    ))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Device profile
# ---------------------------------------------------------------------------

@dataclass
class DeviceProfile:
    """Hardware specification for one edge device.

    Attributes:
        name: Short identifier used as a lookup key (e.g. ``"rpi4"``).
        display_name: Human-readable label for reports.
        cpu_speed_factor: Ratio of device single-thread throughput to the
            reference host (Intel Core i7 class). A value of ``0.12`` means
            the device runs at 12 % of host speed.
        memory_limit_mb: Total device RAM in megabytes.
        tdp_watts: Thermal design power in watts, used for energy estimation.
        throttle_onset_seconds: Seconds of sustained load before thermal
            throttling begins. ``0`` means throttling is never modelled.
        throttle_factor: Speed multiplier when fully throttled, in ``(0, 1]``.
            Represents the fraction of nominal performance available after the
            device thermal governor has reduced CPU clock.
        throttle_ramp_seconds: Seconds over which performance linearly
            degrades from nominal to ``throttle_factor`` after onset.
        notes: Free-text description of the device / assumptions.
    """

    name: str
    display_name: str
    cpu_speed_factor: float
    memory_limit_mb: float
    tdp_watts: float
    throttle_onset_seconds: float = 45.0
    throttle_factor: float = 0.65
    throttle_ramp_seconds: float = 30.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Built-in device profiles
# ---------------------------------------------------------------------------

#: Pre-configured profiles for common edge deployment targets.
#: Speed factors calibrated against an Intel Core i7-10750H reference host.
KNOWN_DEVICES: Dict[str, DeviceProfile] = {
    "rpi4": DeviceProfile(
        name="rpi4",
        display_name="Raspberry Pi 4B (4 GB)",
        cpu_speed_factor=0.12,
        memory_limit_mb=3_800.0,
        tdp_watts=7.5,
        throttle_onset_seconds=40.0,
        throttle_factor=0.55,
        throttle_ramp_seconds=25.0,
        notes="Cortex-A72 @ 1.5 GHz; throttles aggressively without active cooling",
    ),
    "rpi5": DeviceProfile(
        name="rpi5",
        display_name="Raspberry Pi 5 (8 GB)",
        cpu_speed_factor=0.28,
        memory_limit_mb=7_800.0,
        tdp_watts=12.0,
        throttle_onset_seconds=60.0,
        throttle_factor=0.70,
        throttle_ramp_seconds=30.0,
        notes="Cortex-A76 @ 2.4 GHz; significantly faster than RPi 4",
    ),
    "jetson_nano": DeviceProfile(
        name="jetson_nano",
        display_name="NVIDIA Jetson Nano (4 GB)",
        cpu_speed_factor=0.10,
        memory_limit_mb=3_800.0,
        tdp_watts=10.0,
        throttle_onset_seconds=50.0,
        throttle_factor=0.60,
        throttle_ramp_seconds=20.0,
        notes="Cortex-A57 @ 1.43 GHz; GPU not modelled — CPU-only estimate",
    ),
    "jetson_xnx": DeviceProfile(
        name="jetson_xnx",
        display_name="NVIDIA Jetson Xavier NX (8 GB)",
        cpu_speed_factor=0.36,
        memory_limit_mb=7_800.0,
        tdp_watts=15.0,
        throttle_onset_seconds=90.0,
        throttle_factor=0.75,
        throttle_ramp_seconds=30.0,
        notes="Carmel ARM @ 1.9 GHz 6-core; GPU not modelled — CPU-only estimate",
    ),
    "coral_board": DeviceProfile(
        name="coral_board",
        display_name="Google Coral Dev Board (1 GB)",
        cpu_speed_factor=0.07,
        memory_limit_mb=900.0,
        tdp_watts=4.0,
        throttle_onset_seconds=30.0,
        throttle_factor=0.50,
        throttle_ramp_seconds=20.0,
        notes="NXP i.MX 8M @ 1.5 GHz; Edge TPU not modelled — CPU-only estimate; 1 GB RAM is a hard constraint",
    ),
    "snapdragon888": DeviceProfile(
        name="snapdragon888",
        display_name="Snapdragon 888 Mobile SoC (6 GB)",
        cpu_speed_factor=0.34,
        memory_limit_mb=5_800.0,
        tdp_watts=15.0,
        throttle_onset_seconds=55.0,
        throttle_factor=0.65,
        throttle_ramp_seconds=25.0,
        notes="Kryo 680 Prime @ 2.84 GHz; mobile thermal envelope causes early throttling",
    ),
}


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------

@dataclass
class DeviceSimResult:
    """Estimated tracker performance on one edge device.

    All latency, FPS, and energy values are projected from the host-machine
    ProfilingResult using the device's CPU speed factor and TDP.

    Attributes:
        device_name: Short device key (e.g. ``"rpi4"``).
        display_name: Human-readable device label.
        tracker_name: Name of the tracker that was profiled.
        host_fps: Measured FPS on the host benchmark machine.
        estimated_fps: Projected FPS on the target device.
        host_latency_ms: Measured mean per-frame latency on the host.
        estimated_latency_ms: Projected mean per-frame latency on the device.
        host_memory_mb: Peak RSS memory footprint measured on the host.
        estimated_memory_mb: Expected peak memory on the device (same as host;
            memory is algorithm-determined, not clock-dependent).
        memory_limit_mb: Device RAM ceiling.
        fits_in_memory: Whether the tracker is expected to fit in device RAM.
        thermal_state: ``"nominal"``, ``"transitioning"``, or ``"throttled"``
            based on the ``sustained_seconds`` argument to :meth:`DeviceSimulator.simulate`.
        effective_speed_factor: Actual CPU speed factor after throttling is applied.
        estimated_energy_mj_per_frame: Estimated energy per frame in millijoules,
            using the device TDP and projected latency.
        notes: Device-specific notes from the :class:`DeviceProfile`.
    """

    device_name: str
    display_name: str
    tracker_name: str
    host_fps: float
    estimated_fps: float
    host_latency_ms: float
    estimated_latency_ms: float
    host_memory_mb: float
    estimated_memory_mb: float
    memory_limit_mb: float
    fits_in_memory: bool
    thermal_state: str
    effective_speed_factor: float
    estimated_energy_mj_per_frame: float
    notes: str = ""

    def __str__(self) -> str:
        mem_flag = "OK" if self.fits_in_memory else "OOM!"
        return (
            f"DeviceSimResult[{self.device_name}]  "
            f"FPS={self.estimated_fps:.1f}  "
            f"latency={self.estimated_latency_ms:.1f} ms  "
            f"mem={self.estimated_memory_mb:.0f}/{self.memory_limit_mb:.0f} MB ({mem_flag})  "
            f"energy={self.estimated_energy_mj_per_frame:.3f} mJ/frame  "
            f"thermal={self.thermal_state}"
        )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class DeviceSimulator:
    """Project host-machine ProfilingResults onto edge device hardware profiles.

    Args:
        host_calibration_factor: Multiplier applied before projection to correct
            for host machines that are faster (> 1.0) or slower (< 1.0) than
            the reference Intel Core i7 class. Default ``1.0`` assumes the
            reference host.

    Example::

        sim = DeviceSimulator()
        result = sim.simulate(prof_result, "rpi4", sustained_seconds=60.0)
        print(sim.to_markdown_table(sim.simulate_all(prof_result)))
    """

    #: CPU utilisation fraction assumed during tracker inference for energy calc.
    CPU_UTIL_FRACTION: float = 0.70

    def __init__(self, host_calibration_factor: float = 1.0) -> None:
        if host_calibration_factor <= 0:
            raise ValueError("host_calibration_factor must be positive.")
        self._host_cal = host_calibration_factor
        self._profiles: Dict[str, DeviceProfile] = {
            k: v for k, v in KNOWN_DEVICES.items()
        }

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def register_device(self, name: str, profile: DeviceProfile) -> None:
        """Add or replace a device profile.

        Args:
            name: Lookup key (used in :meth:`simulate`).
            profile: :class:`DeviceProfile` with hardware specifications.
        """
        self._profiles[name] = profile

    def list_devices(self) -> List[str]:
        """Return sorted list of registered device names."""
        return sorted(self._profiles)

    def get_profile(self, name: str) -> DeviceProfile:
        """Return the profile for *name*, raising ``KeyError`` if unknown."""
        if name not in self._profiles:
            available = ", ".join(self.list_devices())
            raise KeyError(f"Unknown device '{name}'. Available: {available}")
        return self._profiles[name]

    # ------------------------------------------------------------------
    # Thermal model
    # ------------------------------------------------------------------

    def _thermal_speed_factor(
        self,
        profile: DeviceProfile,
        sustained_seconds: float,
    ) -> Tuple[float, str]:
        """Compute effective CPU speed factor after thermal throttling.

        Args:
            profile: Device profile with throttling parameters.
            sustained_seconds: Duration of continuous tracking in seconds.
                ``0.0`` means a cold-start / burst scenario (no throttling).

        Returns:
            ``(effective_factor, thermal_state)`` where thermal_state is one
            of ``"nominal"``, ``"transitioning"``, or ``"throttled"``.
        """
        onset = profile.throttle_onset_seconds
        ramp = profile.throttle_ramp_seconds

        if sustained_seconds <= 0.0 or onset <= 0.0:
            return profile.cpu_speed_factor, "nominal"

        if sustained_seconds < onset:
            return profile.cpu_speed_factor, "nominal"

        elapsed_into_ramp = sustained_seconds - onset
        if elapsed_into_ramp >= ramp:
            effective = profile.cpu_speed_factor * profile.throttle_factor
            return effective, "throttled"

        # Linear interpolation through the ramp
        alpha = elapsed_into_ramp / ramp
        degraded = 1.0 - alpha * (1.0 - profile.throttle_factor)
        effective = profile.cpu_speed_factor * degraded
        return effective, "transitioning"

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        result: ProfilingResult,
        device_name: str,
        sustained_seconds: float = 0.0,
    ) -> DeviceSimResult:
        """Project one ProfilingResult onto a target device.

        Args:
            result: Host-machine profiling data for a single tracker run.
            device_name: Key of the target device (see :func:`list_devices`).
            sustained_seconds: How long the tracker will run continuously on
                the device. Controls thermal throttling intensity.
                ``0`` (default) models a cold-start burst scenario.

        Returns:
            :class:`DeviceSimResult` with all projected metrics.

        Raises:
            KeyError: If *device_name* is not registered.
        """
        profile = self.get_profile(device_name)
        effective_factor, thermal_state = self._thermal_speed_factor(
            profile, sustained_seconds
        )

        # Adjust for host calibration then apply device factor
        projection = effective_factor / self._host_cal

        host_lat = result.latency_mean_ms
        est_latency = host_lat / projection if projection > 0 else float("inf")
        est_fps = 1_000.0 / est_latency if est_latency > 0 else 0.0

        # Memory footprint is determined by the algorithm, not the CPU clock
        est_memory = result.peak_memory_mb

        # Energy per frame: TDP × latency × CPU utilisation fraction
        est_energy_mj = (
            profile.tdp_watts
            * (est_latency / 1_000.0)
            * self.CPU_UTIL_FRACTION
            * 1_000.0  # W·s → mJ
        )

        return DeviceSimResult(
            device_name=profile.name,
            display_name=profile.display_name,
            tracker_name=result.tracker_name,
            host_fps=result.fps,
            estimated_fps=est_fps,
            host_latency_ms=host_lat,
            estimated_latency_ms=est_latency,
            host_memory_mb=result.peak_memory_mb,
            estimated_memory_mb=est_memory,
            memory_limit_mb=profile.memory_limit_mb,
            fits_in_memory=est_memory <= profile.memory_limit_mb,
            thermal_state=thermal_state,
            effective_speed_factor=effective_factor,
            estimated_energy_mj_per_frame=est_energy_mj,
            notes=profile.notes,
        )

    def simulate_all(
        self,
        result: ProfilingResult,
        sustained_seconds: float = 0.0,
        device_names: Optional[List[str]] = None,
    ) -> List[DeviceSimResult]:
        """Simulate across all (or a subset of) registered devices.

        Args:
            result: Host-machine profiling data.
            sustained_seconds: Sustained-load duration for thermal modelling.
            device_names: Subset of device keys to simulate. If ``None``,
                all registered devices are used (alphabetical order).

        Returns:
            List of :class:`DeviceSimResult`, one per device, sorted by
            estimated FPS (highest first).
        """
        targets = device_names if device_names is not None else self.list_devices()
        results = [
            self.simulate(result, name, sustained_seconds) for name in targets
        ]
        results.sort(key=lambda r: r.estimated_fps, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_table(self, results: List[DeviceSimResult]) -> str:
        """Format simulation results as a Markdown table.

        Args:
            results: Output of :meth:`simulate` or :meth:`simulate_all`.

        Returns:
            Multi-line Markdown table string ready to embed in reports.

        Example output::

            | Rank | Device               | FPS  | Latency(ms) | Mem(MB) | Fits? | Thermal     | mJ/frame |
            |------|----------------------|-----:|------------:|--------:|:-----:|-------------|----------|
            | 1    | Jetson Xavier NX     | 28.3 |        35.3 |   248.0 |  ✓   | nominal     |   0.371  |
        """
        lines = [
            "| Rank | Device | FPS | Latency (ms) | Mem (MB) | Fits? | Thermal | mJ/frame |",
            "|------|--------|----:|-------------:|---------:|:-----:|---------|----------|",
        ]
        for rank, r in enumerate(results, start=1):
            fits = "✓" if r.fits_in_memory else "✗ OOM"
            lines.append(
                f"| {rank} | {r.display_name} "
                f"| {r.estimated_fps:.1f} "
                f"| {r.estimated_latency_ms:.1f} "
                f"| {r.estimated_memory_mb:.0f} / {r.memory_limit_mb:.0f} "
                f"| {fits} "
                f"| {r.thermal_state} "
                f"| {r.estimated_energy_mj_per_frame:.3f} |"
            )
        return "\n".join(lines)

    def to_summary_dict(self, results: List[DeviceSimResult]) -> List[dict]:
        """Convert simulation results to a list of plain dicts (for JSON export).

        Args:
            results: Output of :meth:`simulate_all`.

        Returns:
            List of dicts with all :class:`DeviceSimResult` fields.
        """
        return [
            {
                "device": r.device_name,
                "display_name": r.display_name,
                "tracker": r.tracker_name,
                "host_fps": round(r.host_fps, 2),
                "estimated_fps": round(r.estimated_fps, 2),
                "host_latency_ms": round(r.host_latency_ms, 3),
                "estimated_latency_ms": round(r.estimated_latency_ms, 3),
                "host_memory_mb": round(r.host_memory_mb, 1),
                "estimated_memory_mb": round(r.estimated_memory_mb, 1),
                "memory_limit_mb": r.memory_limit_mb,
                "fits_in_memory": r.fits_in_memory,
                "thermal_state": r.thermal_state,
                "effective_speed_factor": round(r.effective_speed_factor, 4),
                "estimated_energy_mj_per_frame": round(r.estimated_energy_mj_per_frame, 4),
            }
            for r in results
        ]
