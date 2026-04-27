"""Predefined edge hardware profiles for EOVOT deployment feasibility analysis.

Each :class:`HardwareProfile` encodes the key constraints of a real edge
deployment target: power budget, available RAM, minimum acceptable FPS, and
CPU core count.  These profiles let researchers ask "can tracker X run on
device Y?" without owning the physical hardware.

Usage::

    from eovot.profiling.hardware_profiles import get_profile, PROFILES

    profile = get_profile("jetson-nano")
    print(profile.target_fps)   # 30.0
    print(profile.memory_limit_mb)  # 4096

    for slug, p in PROFILES.items():
        print(slug, p.tdp_watts)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class HardwareProfile:
    """Specification for an edge deployment target.

    Args:
        name: Human-readable device name (e.g., ``"Jetson Nano"``).
        tdp_watts: Thermal design power in Watts.  Used by
            :class:`~eovot.profiling.energy.EnergyProfiler` as the TDP
            argument when profiling on this device.
        memory_limit_mb: Maximum RAM available to the tracker process (MiB).
            Includes both model weights and working buffers.
        target_fps: Minimum acceptable frame rate for real-time deployment.
        cpu_cores: Number of physical CPU cores available to the process.
        description: Free-text hardware notes for report headers.
    """

    name: str
    tdp_watts: float
    memory_limit_mb: int
    target_fps: float
    cpu_cores: int
    description: str = ""

    def __str__(self) -> str:
        return (
            f"HardwareProfile({self.name!r}  "
            f"TDP={self.tdp_watts}W  "
            f"mem≤{self.memory_limit_mb}MiB  "
            f"target≥{self.target_fps}fps  "
            f"cores={self.cpu_cores})"
        )


# ---------------------------------------------------------------------------
# Built-in device profiles
# ---------------------------------------------------------------------------

JETSON_NANO = HardwareProfile(
    name="Jetson Nano",
    tdp_watts=10.0,
    memory_limit_mb=4096,
    target_fps=30.0,
    cpu_cores=4,
    description="NVIDIA Jetson Nano 4GB — quad-core Cortex-A57 @ 1.43 GHz, 128-core Maxwell GPU",
)

RASPBERRY_PI_4 = HardwareProfile(
    name="Raspberry Pi 4",
    tdp_watts=6.0,
    memory_limit_mb=4096,
    target_fps=15.0,
    cpu_cores=4,
    description="Raspberry Pi 4 Model B 4GB — quad-core Cortex-A72 @ 1.5 GHz",
)

RASPBERRY_PI_ZERO_2W = HardwareProfile(
    name="Raspberry Pi Zero 2W",
    tdp_watts=2.5,
    memory_limit_mb=512,
    target_fps=5.0,
    cpu_cores=4,
    description="Raspberry Pi Zero 2W — quad-core Cortex-A53 @ 1 GHz, 512 MB RAM",
)

INTEL_NCS2 = HardwareProfile(
    name="Intel NCS2",
    tdp_watts=1.0,
    memory_limit_mb=512,
    target_fps=60.0,
    cpu_cores=1,
    description="Intel Neural Compute Stick 2 — Myriad X VPU, inference via OpenVINO",
)

CORAL_EDGE_TPU = HardwareProfile(
    name="Coral Edge TPU",
    tdp_watts=2.0,
    memory_limit_mb=8192,
    target_fps=100.0,
    cpu_cores=4,
    description="Google Coral Dev Board — NXP i.MX 8M + Edge TPU @ 4 TOPS",
)

LAPTOP_CPU = HardwareProfile(
    name="Laptop CPU",
    tdp_watts=28.0,
    memory_limit_mb=16384,
    target_fps=30.0,
    cpu_cores=8,
    description="Generic laptop — 8-core x86 CPU @ ~3 GHz, 16 GB RAM",
)

DESKTOP_GPU = HardwareProfile(
    name="Desktop GPU",
    tdp_watts=250.0,
    memory_limit_mb=32768,
    target_fps=120.0,
    cpu_cores=16,
    description="Desktop workstation — 16-core CPU + NVIDIA RTX GPU, 32 GB RAM",
)


#: Registry mapping short slug → :class:`HardwareProfile`.
PROFILES: Dict[str, HardwareProfile] = {
    "jetson-nano": JETSON_NANO,
    "rpi4": RASPBERRY_PI_4,
    "rpi-zero-2w": RASPBERRY_PI_ZERO_2W,
    "intel-ncs2": INTEL_NCS2,
    "coral-tpu": CORAL_EDGE_TPU,
    "laptop": LAPTOP_CPU,
    "desktop": DESKTOP_GPU,
}


def get_profile(name: str) -> HardwareProfile:
    """Return a :class:`HardwareProfile` by slug key.

    Args:
        name: Case-sensitive slug from :data:`PROFILES`
            (e.g., ``"jetson-nano"``, ``"rpi4"``).

    Returns:
        The corresponding :class:`HardwareProfile`.

    Raises:
        KeyError: If *name* is not in :data:`PROFILES`.

    Example::

        profile = get_profile("rpi4")
        print(profile.target_fps)  # 15.0
    """
    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise KeyError(
            f"Unknown hardware profile {name!r}. "
            f"Available profiles: {available}"
        )
    return PROFILES[name]
