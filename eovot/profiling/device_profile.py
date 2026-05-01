"""Device profiles and deployability checking for edge-aware tracker evaluation.

This module provides:

- :class:`DeviceProfile` — dataclass encoding the hardware constraints of a
  target deployment platform (FPS budget, memory ceiling, TDP).
- :data:`DEVICE_PRESETS` — a registry of common edge and desktop devices.
- :class:`DeployabilityChecker` — evaluates whether a benchmark result meets
  a device's constraints and returns a structured :class:`DeployabilityReport`.

Motivation
----------
Academic trackers are typically evaluated on desktop-class hardware where FPS
and memory are not limiting factors.  EOVOT bridges this gap by letting
researchers quantify *which trackers can actually run* on a target device
before investing in deployment work.

Usage::

    from eovot.profiling.device_profile import DEVICE_PRESETS, DeployabilityChecker

    checker = DeployabilityChecker()
    report = checker.check(benchmark_result, DEVICE_PRESETS["raspberry_pi4"])
    print(report)

    if report.deployable:
        print("Tracker meets all constraints — ready for edge deployment.")
    else:
        for violation in report.violations:
            print(f"  FAIL: {violation}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from eovot.benchmark.engine import BenchmarkResult


@dataclass
class DeviceProfile:
    """Hardware constraints for a target deployment device.

    All limits are *soft* constraints used to generate the deployability
    report; exceeding one marks the tracker as non-deployable on that device
    but does not raise an error.

    Attributes:
        name: Human-readable device name (e.g. ``"Raspberry Pi 4"``).
        min_fps: Minimum required throughput in frames per second.
            Set to ``None`` to skip the FPS check.
        max_memory_mb: Maximum peak RSS memory in megabytes.
            Set to ``None`` to skip the memory check.
        tdp_watts: CPU Thermal Design Power in Watts, used as the TDP value
            for energy estimation.  Set to ``None`` if no energy budget is
            defined.
        max_energy_per_frame_mj: Maximum allowed per-frame energy in
            milli-Joules.  Only evaluated when the benchmark result includes
            energy profiling data.  Set to ``None`` to skip.
        max_latency_ms: Maximum acceptable mean per-frame latency in
            milliseconds (= 1000 / min_fps when not set explicitly).
            Overrides the FPS-derived latency threshold when provided.
        description: Free-text description of the device for report output.
    """

    name: str
    min_fps: Optional[float] = None
    max_memory_mb: Optional[float] = None
    tdp_watts: Optional[float] = None
    max_energy_per_frame_mj: Optional[float] = None
    max_latency_ms: Optional[float] = None
    description: str = ""

    @property
    def derived_max_latency_ms(self) -> Optional[float]:
        """Latency ceiling derived from min_fps when max_latency_ms is not set."""
        if self.max_latency_ms is not None:
            return self.max_latency_ms
        if self.min_fps is not None and self.min_fps > 0:
            return 1_000.0 / self.min_fps
        return None

    def __str__(self) -> str:
        parts = [f"DeviceProfile({self.name!r}"]
        if self.min_fps is not None:
            parts.append(f"min_fps={self.min_fps}")
        if self.max_memory_mb is not None:
            parts.append(f"max_mem={self.max_memory_mb} MB")
        if self.tdp_watts is not None:
            parts.append(f"TDP={self.tdp_watts} W")
        return ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Built-in device presets
# ---------------------------------------------------------------------------

DEVICE_PRESETS: Dict[str, DeviceProfile] = {
    "raspberry_pi4": DeviceProfile(
        name="Raspberry Pi 4",
        min_fps=15.0,
        max_memory_mb=256.0,
        tdp_watts=6.0,
        max_energy_per_frame_mj=10.0,
        description=(
            "Raspberry Pi 4 Model B (4 GB variant). "
            "ARM Cortex-A72 @ 1.5 GHz, 4 GB LPDDR4. "
            "Typical tracking use-case: 15 FPS real-time with 6 W TDP."
        ),
    ),
    "jetson_nano": DeviceProfile(
        name="NVIDIA Jetson Nano",
        min_fps=25.0,
        max_memory_mb=512.0,
        tdp_watts=10.0,
        max_energy_per_frame_mj=8.0,
        description=(
            "NVIDIA Jetson Nano Developer Kit. "
            "Quad-core ARM Cortex-A57 @ 1.43 GHz, 4 GB LPDDR4, Maxwell GPU. "
            "5–10 W power mode; GPU-accelerated trackers can achieve 30+ FPS."
        ),
    ),
    "jetson_orin_nano": DeviceProfile(
        name="NVIDIA Jetson Orin Nano",
        min_fps=30.0,
        max_memory_mb=1024.0,
        tdp_watts=15.0,
        max_energy_per_frame_mj=6.0,
        description=(
            "NVIDIA Jetson Orin Nano 8 GB. "
            "6-core Arm Cortex-A78AE, 1024-core Ampere GPU. "
            "10–15 W power mode; suitable for real-time deep tracking."
        ),
    ),
    "laptop_cpu": DeviceProfile(
        name="Laptop CPU",
        min_fps=30.0,
        max_memory_mb=2048.0,
        tdp_watts=15.0,
        max_energy_per_frame_mj=50.0,
        description=(
            "Generic laptop with a 15 W TDP CPU (e.g. Intel Core i5/i7 U-series). "
            "Baseline for development and non-edge evaluation."
        ),
    ),
    "desktop_cpu": DeviceProfile(
        name="Desktop CPU",
        min_fps=60.0,
        max_memory_mb=8192.0,
        tdp_watts=65.0,
        max_energy_per_frame_mj=200.0,
        description=(
            "Desktop workstation with a 65 W TDP CPU. "
            "Used as an unconstrained upper-bound reference."
        ),
    ),
    "coral_usb": DeviceProfile(
        name="Google Coral USB Accelerator",
        min_fps=15.0,
        max_memory_mb=128.0,
        tdp_watts=2.5,
        max_energy_per_frame_mj=5.0,
        description=(
            "Google Coral USB Accelerator attached to a host CPU. "
            "Edge TPU @ 2 TOPS, ~2.5 W peak; suited for quantised models."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Deployability report
# ---------------------------------------------------------------------------

@dataclass
class ConstraintResult:
    """Outcome of checking a single hardware constraint."""

    name: str
    passed: bool
    measured: float
    limit: float
    unit: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name}: "
            f"measured={self.measured:.3f} {self.unit}  "
            f"limit={self.limit:.3f} {self.unit}"
        )


@dataclass
class DeployabilityReport:
    """Full deployability assessment of one tracker on one device.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        device: :class:`DeviceProfile` the tracker was checked against.
        constraint_results: Ordered list of individual constraint outcomes.
        deployable: ``True`` when all checked constraints passed.
        notes: Optional list of informational strings (warnings, tips).
    """

    tracker_name: str
    device: DeviceProfile
    constraint_results: List[ConstraintResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def deployable(self) -> bool:
        return all(c.passed for c in self.constraint_results)

    @property
    def violations(self) -> List[str]:
        """Human-readable descriptions of all failed constraints."""
        return [str(c) for c in self.constraint_results if not c.passed]

    def summary_dict(self) -> Dict:
        """Return a JSON-serialisable summary dict."""
        return {
            "tracker": self.tracker_name,
            "device": self.device.name,
            "deployable": self.deployable,
            "constraints": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "measured": round(c.measured, 4),
                    "limit": round(c.limit, 4),
                    "unit": c.unit,
                }
                for c in self.constraint_results
            ],
            "notes": self.notes,
        }

    def __str__(self) -> str:
        status = "DEPLOYABLE" if self.deployable else "NOT DEPLOYABLE"
        lines = [
            f"DeployabilityReport — {self.tracker_name} on {self.device.name}",
            f"  Verdict: {status}",
            "  Constraints:",
        ]
        for c in self.constraint_results:
            lines.append(f"    {c}")
        if self.notes:
            lines.append("  Notes:")
            for n in self.notes:
                lines.append(f"    • {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class DeployabilityChecker:
    """Evaluate whether a :class:`~eovot.benchmark.engine.BenchmarkResult`
    meets the hardware constraints of a :class:`DeviceProfile`.

    Example::

        from eovot.profiling.device_profile import DeployabilityChecker, DEVICE_PRESETS

        checker = DeployabilityChecker()
        report = checker.check(result, DEVICE_PRESETS["raspberry_pi4"])
        print(report)
    """

    def check(
        self,
        result: "BenchmarkResult",
        device: DeviceProfile,
    ) -> DeployabilityReport:
        """Evaluate *result* against *device* constraints.

        Checks (in order):
        1. Mean FPS ≥ ``device.min_fps`` (if set).
        2. Mean latency ≤ derived latency ceiling (if FPS or latency limit set).
        3. Peak memory ≤ ``device.max_memory_mb`` (if set).
        4. Mean energy per frame ≤ ``device.max_energy_per_frame_mj`` (if set
           and energy data is present in the result).

        Args:
            result: Output of :class:`~eovot.benchmark.engine.BenchmarkEngine.run`.
            device: Target device profile to evaluate against.

        Returns:
            :class:`DeployabilityReport` with per-constraint outcomes.
        """
        report = DeployabilityReport(tracker_name=result.tracker_name, device=device)

        # 1. FPS check
        if device.min_fps is not None:
            measured_fps = result.mean_fps
            passed = measured_fps >= device.min_fps
            report.constraint_results.append(
                ConstraintResult(
                    name="Mean FPS",
                    passed=passed,
                    measured=measured_fps,
                    limit=device.min_fps,
                    unit="fps",
                )
            )

        # 2. Latency check
        lat_limit = device.derived_max_latency_ms
        if lat_limit is not None:
            mean_lat = float(
                sum(sr.profiling.latency_mean_ms for sr in result.sequence_results)
                / len(result.sequence_results)
            )
            passed = mean_lat <= lat_limit
            report.constraint_results.append(
                ConstraintResult(
                    name="Mean Latency",
                    passed=passed,
                    measured=mean_lat,
                    limit=lat_limit,
                    unit="ms",
                )
            )

        # 3. Memory check
        if device.max_memory_mb is not None:
            measured_mem = result.peak_memory_mb
            passed = measured_mem <= device.max_memory_mb
            report.constraint_results.append(
                ConstraintResult(
                    name="Peak Memory",
                    passed=passed,
                    measured=measured_mem,
                    limit=device.max_memory_mb,
                    unit="MB",
                )
            )

        # 4. Energy check (only when both limit and data are available)
        energy_available = any(
            sr.energy is not None for sr in result.sequence_results
        )
        if device.max_energy_per_frame_mj is not None:
            if energy_available:
                mef = result.mean_energy_per_frame_mj
                if mef is not None:
                    passed = mef <= device.max_energy_per_frame_mj
                    report.constraint_results.append(
                        ConstraintResult(
                            name="Energy per Frame",
                            passed=passed,
                            measured=mef,
                            limit=device.max_energy_per_frame_mj,
                            unit="mJ",
                        )
                    )
            else:
                report.notes.append(
                    f"Energy limit is {device.max_energy_per_frame_mj} mJ/frame but "
                    "no energy data was collected. Re-run BenchmarkEngine with "
                    f"tdp_watts={device.tdp_watts} to enable energy profiling."
                )

        # Informational notes
        if not report.deployable:
            report.notes.append(
                f"Consider a lighter tracker or reduce resolution to meet "
                f"{device.name} constraints."
            )
        else:
            report.notes.append(
                f"All constraints satisfied — tracker is compatible with {device.name}."
            )

        return report

    def check_all(
        self,
        result: "BenchmarkResult",
        devices: Optional[Dict[str, DeviceProfile]] = None,
    ) -> Dict[str, DeployabilityReport]:
        """Check *result* against multiple devices at once.

        Args:
            result: Benchmark result to evaluate.
            devices: Dict mapping device key → :class:`DeviceProfile`.
                Defaults to :data:`DEVICE_PRESETS`.

        Returns:
            Dict mapping device key → :class:`DeployabilityReport`.
        """
        if devices is None:
            devices = DEVICE_PRESETS
        return {key: self.check(result, profile) for key, profile in devices.items()}
