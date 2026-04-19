"""Hardware device profiles for edge-aware benchmarking.

Pre-configured profiles for common edge and embedded computing platforms.
Use :func:`assess_edge_compliance` to evaluate whether a tracker's measured
performance fits within the resource envelope of a target device.

Example::

    from eovot.profiling.device_profiles import JETSON_NANO, assess_edge_compliance

    report = assess_edge_compliance(benchmark_result, JETSON_NANO)
    print(report)
    if report.compliant:
        print(f"{benchmark_result.tracker_name} is deployable on {JETSON_NANO.name}")

Reference:
    Redmon & Farhadi, "YOLO9000: Better, Faster, Stronger." CVPR 2017 —
    motivates FPS-first design for real-time embedded vision systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DeviceProfile:
    """Hardware specification for an edge or embedded computing device.

    Attributes:
        name: Human-readable device name (e.g. ``"Raspberry Pi 4B"``).
        tdp_watts: CPU/SoC Thermal Design Power (W).
        ram_mb: Total available system RAM (MB).
        target_fps: Minimum acceptable tracking throughput (frames/sec).
        max_memory_mb: Maximum allowable peak RSS memory for the tracker (MB).
        max_energy_per_frame_mj: Per-frame energy budget (mJ).
            ``None`` means this device imposes no energy constraint.
        description: Optional free-form notes (ISA, OS, typical use-case).
    """

    name: str
    tdp_watts: float
    ram_mb: int
    target_fps: float
    max_memory_mb: float
    max_energy_per_frame_mj: Optional[float] = None
    description: str = ""

    def __str__(self) -> str:
        return (
            f"DeviceProfile[{self.name}] "
            f"TDP={self.tdp_watts}W  RAM={self.ram_mb}MB  "
            f"target_fps≥{self.target_fps}  mem≤{self.max_memory_mb}MB"
        )


# ---------------------------------------------------------------------------
# Pre-configured device profiles (ordered most → least constrained)
# ---------------------------------------------------------------------------

RASPBERRY_PI_4B = DeviceProfile(
    name="Raspberry Pi 4B",
    tdp_watts=6.4,
    ram_mb=4096,
    target_fps=15.0,
    max_memory_mb=512.0,
    max_energy_per_frame_mj=5.0,
    description="ARM Cortex-A72 quad-core @ 1.5 GHz, 4 GB LPDDR4",
)

JETSON_NANO = DeviceProfile(
    name="NVIDIA Jetson Nano",
    tdp_watts=10.0,
    ram_mb=4096,
    target_fps=25.0,
    max_memory_mb=768.0,
    max_energy_per_frame_mj=8.0,
    description="ARM Cortex-A57 quad-core + 128-core Maxwell GPU, 4 GB LPDDR4",
)

JETSON_ORIN_NANO = DeviceProfile(
    name="NVIDIA Jetson Orin Nano",
    tdp_watts=15.0,
    ram_mb=8192,
    target_fps=60.0,
    max_memory_mb=1536.0,
    max_energy_per_frame_mj=4.0,
    description="6-core Cortex-A78AE + 1024-core Ampere GPU, 8 GB LPDDR5",
)

INTEL_NUC = DeviceProfile(
    name="Intel NUC (Core i5)",
    tdp_watts=28.0,
    ram_mb=16384,
    target_fps=60.0,
    max_memory_mb=2048.0,
    max_energy_per_frame_mj=20.0,
    description="Intel Core i5-1235U, 16 GB DDR4, compact embedded PC",
)

LAPTOP_MID = DeviceProfile(
    name="Mid-range Laptop",
    tdp_watts=45.0,
    ram_mb=16384,
    target_fps=60.0,
    max_memory_mb=4096.0,
    max_energy_per_frame_mj=40.0,
    description="Intel Core i7 / Ryzen 7 class, typical research workstation",
)

DESKTOP_SERVER = DeviceProfile(
    name="Desktop / Server",
    tdp_watts=125.0,
    ram_mb=65536,
    target_fps=120.0,
    max_memory_mb=16384.0,
    max_energy_per_frame_mj=None,
    description="High-end desktop or cloud instance — no energy budget constraint",
)

ALL_PROFILES: List[DeviceProfile] = [
    RASPBERRY_PI_4B,
    JETSON_NANO,
    JETSON_ORIN_NANO,
    INTEL_NUC,
    LAPTOP_MID,
    DESKTOP_SERVER,
]

PROFILE_REGISTRY: Dict[str, DeviceProfile] = {p.name: p for p in ALL_PROFILES}


def get_profile(name: str) -> DeviceProfile:
    """Look up a device profile by name (case-insensitive substring match).

    Args:
        name: Device name or unambiguous substring (e.g. ``"jetson nano"``).

    Returns:
        Matching :class:`DeviceProfile`.

    Raises:
        KeyError: If no profile matches.
    """
    key = name.strip().lower()
    for profile_name, profile in PROFILE_REGISTRY.items():
        if key == profile_name.lower() or key in profile_name.lower():
            return profile
    raise KeyError(
        f"No device profile matches '{name}'. "
        f"Available: {list(PROFILE_REGISTRY.keys())}"
    )


# ---------------------------------------------------------------------------
# Compliance assessment
# ---------------------------------------------------------------------------

@dataclass
class Criterion:
    """Result of a single edge-compliance check."""

    name: str
    passed: bool
    measured: float
    threshold: float
    unit: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        op = "≥" if "fps" in self.name.lower() else "≤"
        return (
            f"[{status}] {self.name}: "
            f"{self.measured:.2f} {self.unit} "
            f"(required {op} {self.threshold:.2f} {self.unit})"
        )


@dataclass
class EdgeComplianceReport:
    """Summary of whether a tracker satisfies a device's resource constraints.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        device: Target :class:`DeviceProfile`.
        criteria: Individual pass/fail results for each constraint.
    """

    tracker_name: str
    device: DeviceProfile
    criteria: List[Criterion] = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        """``True`` if every criterion passed."""
        return all(c.passed for c in self.criteria)

    @property
    def n_passed(self) -> int:
        return sum(c.passed for c in self.criteria)

    def __str__(self) -> str:
        status = "COMPLIANT ✓" if self.compliant else "NON-COMPLIANT ✗"
        lines = [
            f"EdgeComplianceReport — {self.tracker_name} on {self.device.name}",
            f"Overall: {status} ({self.n_passed}/{len(self.criteria)} criteria passed)",
            "-" * 60,
        ]
        for c in self.criteria:
            lines.append(f"  {c}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "tracker_name": self.tracker_name,
            "device": self.device.name,
            "compliant": self.compliant,
            "n_passed": self.n_passed,
            "n_criteria": len(self.criteria),
            "criteria": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "measured": round(c.measured, 4),
                    "threshold": c.threshold,
                    "unit": c.unit,
                }
                for c in self.criteria
            ],
        }


def assess_edge_compliance(benchmark_result, device: DeviceProfile) -> EdgeComplianceReport:
    """Assess whether a benchmark result satisfies *device*'s constraints.

    Checks the following criteria:

    1. **Throughput** — mean FPS ≥ ``device.target_fps``
    2. **Peak Memory** — peak RSS ≤ ``device.max_memory_mb``
    3. **Energy/Frame** — mean energy per frame ≤ ``device.max_energy_per_frame_mj``
       (only when the result contains energy data *and* the device specifies a budget)

    Args:
        benchmark_result: A :class:`~eovot.benchmark.engine.BenchmarkResult`
            returned by :class:`~eovot.benchmark.engine.BenchmarkEngine`.
        device: Target :class:`DeviceProfile`.

    Returns:
        :class:`EdgeComplianceReport` with per-criterion verdicts.
    """
    report = EdgeComplianceReport(tracker_name=benchmark_result.tracker_name, device=device)

    fps = benchmark_result.mean_fps
    report.criteria.append(Criterion(
        name="Throughput (FPS)",
        passed=fps >= device.target_fps,
        measured=fps,
        threshold=device.target_fps,
        unit="fps",
    ))

    mem = benchmark_result.peak_memory_mb
    report.criteria.append(Criterion(
        name="Peak Memory",
        passed=mem <= device.max_memory_mb,
        measured=mem,
        threshold=device.max_memory_mb,
        unit="MB",
    ))

    energy_mj = benchmark_result.mean_energy_per_frame_mj
    if energy_mj is not None and device.max_energy_per_frame_mj is not None:
        report.criteria.append(Criterion(
            name="Energy/Frame",
            passed=energy_mj <= device.max_energy_per_frame_mj,
            measured=energy_mj,
            threshold=device.max_energy_per_frame_mj,
            unit="mJ",
        ))

    return report


def compliance_matrix(
    benchmark_results: list,
    devices: Optional[List[DeviceProfile]] = None,
) -> List[List[EdgeComplianceReport]]:
    """Compute a compliance matrix for multiple trackers × multiple devices.

    Args:
        benchmark_results: List of
            :class:`~eovot.benchmark.engine.BenchmarkResult` objects.
        devices: Target :class:`DeviceProfile` list.
            Defaults to :data:`ALL_PROFILES`.

    Returns:
        2-D list ``matrix[tracker_idx][device_idx]`` of
        :class:`EdgeComplianceReport` objects.
    """
    if devices is None:
        devices = ALL_PROFILES
    return [
        [assess_edge_compliance(br, dev) for dev in devices]
        for br in benchmark_results
    ]
