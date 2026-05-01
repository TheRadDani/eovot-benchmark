"""Hardware device profile registry for EOVOT edge-aware benchmarking.

Provides a structured, named catalogue of common edge and desktop hardware
platforms with their energy and compute characteristics.  Profiles are used
by :class:`~eovot.profiling.energy.EnergyProfiler` to set a realistic TDP
estimate without requiring the user to look up datasheet values.

Usage::

    from eovot.profiling.device_profiles import get_profile, list_profiles

    profile = get_profile("jetson-nano")
    print(profile.tdp_watts)       # 10.0
    print(profile.device_class)    # "edge-gpu"

    for name, p in list_profiles():
        print(f"{name}: {p.description}")

Device classes
--------------
- ``"edge-cpu"``   — microcontrollers / low-power SBCs (RPi, etc.)
- ``"edge-gpu"``   — GPU-equipped edge boards (Jetson family)
- ``"laptop"``     — typical consumer / research laptop CPUs
- ``"workstation"``— desktop-class CPUs and GPU workstations
- ``"server"``     — data-centre / cloud inference nodes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DeviceProfile:
    """Hardware characteristics for one target platform.

    Attributes:
        name: Short identifier used as registry key (e.g. ``"jetson-nano"``).
        display_name: Human-readable name for reports.
        device_class: Category string — one of ``"edge-cpu"``, ``"edge-gpu"``,
            ``"laptop"``, ``"workstation"``, or ``"server"``.
        tdp_watts: CPU Thermal Design Power in Watts.  Used as the upper bound
            for :class:`~eovot.profiling.energy.EnergyProfiler`.
        peak_power_w: Whole-board peak power draw (optional).  Includes GPU,
            memory, I/O — useful for battery-budget calculations.
        ram_gb: Total RAM in GiB (useful for memory-constrained profiling).
        cpu_cores: Number of physical CPU cores.
        has_gpu: Whether the device includes an integrated or discrete GPU.
        typical_fps_target: Rough real-time tracking FPS threshold for this
            class of device (informational — used in reporting).
        description: One-line human description for CLI help text.
    """

    name: str
    display_name: str
    device_class: str
    tdp_watts: float
    peak_power_w: Optional[float] = None
    ram_gb: float = 0.0
    cpu_cores: int = 1
    has_gpu: bool = False
    typical_fps_target: float = 30.0
    description: str = ""

    def __post_init__(self) -> None:
        valid_classes = {"edge-cpu", "edge-gpu", "laptop", "workstation", "server"}
        if self.device_class not in valid_classes:
            raise ValueError(
                f"device_class must be one of {valid_classes}, got {self.device_class!r}"
            )
        if self.tdp_watts <= 0:
            raise ValueError(f"tdp_watts must be positive, got {self.tdp_watts}")

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "device_class": self.device_class,
            "tdp_watts": self.tdp_watts,
            "peak_power_w": self.peak_power_w,
            "ram_gb": self.ram_gb,
            "cpu_cores": self.cpu_cores,
            "has_gpu": self.has_gpu,
            "typical_fps_target": self.typical_fps_target,
            "description": self.description,
        }

    def __str__(self) -> str:
        gpu_tag = " +GPU" if self.has_gpu else ""
        return (
            f"DeviceProfile[{self.name}] "
            f"class={self.device_class}{gpu_tag}  "
            f"TDP={self.tdp_watts}W  "
            f"RAM={self.ram_gb}GB  "
            f"cores={self.cpu_cores}"
        )


# ---------------------------------------------------------------------------
# Built-in device registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, DeviceProfile] = {}


def _register(profile: DeviceProfile) -> DeviceProfile:
    _REGISTRY[profile.name] = profile
    return profile


# ---- Raspberry Pi family --------------------------------------------------

_register(DeviceProfile(
    name="rpi3b",
    display_name="Raspberry Pi 3B",
    device_class="edge-cpu",
    tdp_watts=4.0,
    peak_power_w=6.5,
    ram_gb=1.0,
    cpu_cores=4,
    has_gpu=False,
    typical_fps_target=10.0,
    description="Raspberry Pi 3B (1 GB RAM, Cortex-A53 quad-core @ 1.2 GHz)",
))

_register(DeviceProfile(
    name="rpi4",
    display_name="Raspberry Pi 4 (4 GB)",
    device_class="edge-cpu",
    tdp_watts=6.0,
    peak_power_w=8.5,
    ram_gb=4.0,
    cpu_cores=4,
    has_gpu=False,
    typical_fps_target=20.0,
    description="Raspberry Pi 4 (4 GB RAM, Cortex-A72 quad-core @ 1.8 GHz)",
))

_register(DeviceProfile(
    name="rpi5",
    display_name="Raspberry Pi 5 (8 GB)",
    device_class="edge-cpu",
    tdp_watts=12.0,
    peak_power_w=15.0,
    ram_gb=8.0,
    cpu_cores=4,
    has_gpu=False,
    typical_fps_target=40.0,
    description="Raspberry Pi 5 (8 GB RAM, Cortex-A76 quad-core @ 2.4 GHz)",
))

# ---- NVIDIA Jetson family -------------------------------------------------

_register(DeviceProfile(
    name="jetson-nano",
    display_name="NVIDIA Jetson Nano",
    device_class="edge-gpu",
    tdp_watts=10.0,
    peak_power_w=10.0,
    ram_gb=4.0,
    cpu_cores=4,
    has_gpu=True,
    typical_fps_target=30.0,
    description="NVIDIA Jetson Nano (4 GB, 128-core Maxwell GPU, 10 W TDP)",
))

_register(DeviceProfile(
    name="jetson-xavier-nx",
    display_name="NVIDIA Jetson Xavier NX",
    device_class="edge-gpu",
    tdp_watts=15.0,
    peak_power_w=20.0,
    ram_gb=8.0,
    cpu_cores=6,
    has_gpu=True,
    typical_fps_target=60.0,
    description="NVIDIA Jetson Xavier NX (8 GB, 384-core Volta GPU, 15 W mode)",
))

_register(DeviceProfile(
    name="jetson-agx-orin",
    display_name="NVIDIA Jetson AGX Orin",
    device_class="edge-gpu",
    tdp_watts=60.0,
    peak_power_w=75.0,
    ram_gb=32.0,
    cpu_cores=12,
    has_gpu=True,
    typical_fps_target=120.0,
    description="NVIDIA Jetson AGX Orin (32 GB, 2048-core Ampere GPU, 60 W max)",
))

# ---- Coral / NPU devices -------------------------------------------------

_register(DeviceProfile(
    name="coral-dev-board",
    display_name="Google Coral Dev Board",
    device_class="edge-cpu",
    tdp_watts=2.0,
    peak_power_w=4.0,
    ram_gb=1.0,
    cpu_cores=4,
    has_gpu=False,
    typical_fps_target=15.0,
    description="Google Coral Dev Board (1 GB RAM, i.MX 8M + Edge TPU, ~2 W CPU TDP)",
))

# ---- Laptop / Consumer ---------------------------------------------------

_register(DeviceProfile(
    name="laptop-low",
    display_name="Laptop (low-power, U-series)",
    device_class="laptop",
    tdp_watts=15.0,
    peak_power_w=25.0,
    ram_gb=16.0,
    cpu_cores=4,
    has_gpu=False,
    typical_fps_target=100.0,
    description="Typical ultrabook CPU (Intel Core U-series / AMD Ryzen 5 U, 15 W TDP)",
))

_register(DeviceProfile(
    name="laptop-mid",
    display_name="Laptop (mid-range, H-series)",
    device_class="laptop",
    tdp_watts=28.0,
    peak_power_w=45.0,
    ram_gb=16.0,
    cpu_cores=8,
    has_gpu=True,
    typical_fps_target=150.0,
    description="Mid-range laptop CPU with dGPU (Intel Core H / Ryzen 7 H, 28 W TDP)",
))

# ---- Workstation / Desktop -----------------------------------------------

_register(DeviceProfile(
    name="desktop-cpu",
    display_name="Desktop CPU (mid-range)",
    device_class="workstation",
    tdp_watts=65.0,
    peak_power_w=120.0,
    ram_gb=32.0,
    cpu_cores=8,
    has_gpu=False,
    typical_fps_target=300.0,
    description="Mid-range desktop CPU (Intel Core i7 / AMD Ryzen 7, 65 W TDP)",
))

_register(DeviceProfile(
    name="workstation-gpu",
    display_name="GPU Workstation (RTX-class)",
    device_class="workstation",
    tdp_watts=125.0,
    peak_power_w=350.0,
    ram_gb=64.0,
    cpu_cores=16,
    has_gpu=True,
    typical_fps_target=500.0,
    description="High-end CPU + NVIDIA RTX GPU workstation (125 W CPU TDP)",
))

# ---- Server / Cloud ------------------------------------------------------

_register(DeviceProfile(
    name="server-cpu",
    display_name="Server CPU (Xeon / EPYC)",
    device_class="server",
    tdp_watts=200.0,
    peak_power_w=400.0,
    ram_gb=128.0,
    cpu_cores=32,
    has_gpu=False,
    typical_fps_target=1000.0,
    description="Cloud server CPU (Intel Xeon / AMD EPYC, 200 W TDP)",
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_profile(name: str) -> DeviceProfile:
    """Return a :class:`DeviceProfile` by its registry key.

    Args:
        name: Case-insensitive profile identifier (e.g. ``"jetson-nano"``,
              ``"rpi4"``, ``"laptop-mid"``).

    Raises:
        KeyError: If no profile with *name* is registered.

    Example::

        profile = get_profile("rpi4")
        energy_profiler = EnergyProfiler(tdp_watts=profile.tdp_watts)
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Unknown device profile {name!r}. Available profiles: {available}"
        )
    return _REGISTRY[key]


def list_profiles() -> List[Tuple[str, DeviceProfile]]:
    """Return all registered profiles as ``(name, DeviceProfile)`` pairs.

    Profiles are returned sorted alphabetically by name.

    Example::

        for name, profile in list_profiles():
            print(f"{name:20s} — {profile.description}")
    """
    return sorted(_REGISTRY.items())


def register_profile(profile: DeviceProfile) -> None:
    """Add a custom :class:`DeviceProfile` to the global registry.

    Raises:
        ValueError: If a profile with the same name is already registered.

    Example::

        custom = DeviceProfile(
            name="my-board",
            display_name="My Custom Board",
            device_class="edge-cpu",
            tdp_watts=5.0,
            description="Custom SBC with 5 W TDP",
        )
        register_profile(custom)
    """
    if profile.name in _REGISTRY:
        raise ValueError(
            f"A profile named {profile.name!r} is already registered. "
            "Use a unique name or modify _REGISTRY directly."
        )
    _REGISTRY[profile.name] = profile
