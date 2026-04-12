"""Hardware device profiles for edge-aware benchmarking in EOVOT.

Provides TDP estimates and memory constraints for common edge, mobile, and
server platforms.  Use these profiles to parameterise
:class:`~eovot.profiling.energy.EnergyProfiler` without hard-coding device
figures in experiment scripts.

Example::

    from eovot.profiling.hardware_profiles import get_profile
    from eovot.profiling.energy import EnergyProfiler

    profile = get_profile("jetson_nano")
    profiler = EnergyProfiler(tdp_watts=profile.total_tdp_w)

All TDP values are sourced from official manufacturer datasheets and
AnandTech/WikiChip reviews.  Values represent thermal *design* power
(upper envelope) rather than measured average consumption; treat estimates
as conservative upper bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class HardwareProfile:
    """Static hardware specification for an edge or server device.

    Attributes:
        name:         Human-readable device identifier.
        cpu_tdp_w:    CPU Thermal Design Power in Watts.
        gpu_tdp_w:    GPU TDP in Watts, or ``None`` if no discrete/integrated GPU.
        total_tdp_w:  Full SoC / board TDP (CPU + GPU + memory + interconnect).
        ram_gb:       Total DRAM capacity in gigabytes.
        category:     Device tier: ``'edge'``, ``'mobile'``, ``'laptop'``,
                      or ``'server'``.
        notes:        Optional free-text description of key specs.
    """

    name: str
    cpu_tdp_w: float
    gpu_tdp_w: Optional[float]
    total_tdp_w: float
    ram_gb: float
    category: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Device library
# ---------------------------------------------------------------------------
# Keys are lowercase identifiers used by get_profile().

HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    # ---- Raspberry Pi family -----------------------------------------------
    "rpi4": HardwareProfile(
        name="Raspberry Pi 4 Model B",
        cpu_tdp_w=4.0,
        gpu_tdp_w=None,
        total_tdp_w=6.0,
        ram_gb=4.0,
        category="edge",
        notes="ARM Cortex-A72 quad-core @ 1.8 GHz; VideoCore VI GPU not used for tracking",
    ),
    "rpi5": HardwareProfile(
        name="Raspberry Pi 5",
        cpu_tdp_w=5.0,
        gpu_tdp_w=None,
        total_tdp_w=9.0,
        ram_gb=4.0,
        category="edge",
        notes="ARM Cortex-A76 quad-core @ 2.4 GHz; recommended PSU 5 V / 5 A",
    ),
    # ---- NVIDIA Jetson family ----------------------------------------------
    "jetson_nano": HardwareProfile(
        name="NVIDIA Jetson Nano (4 GB)",
        cpu_tdp_w=5.0,
        gpu_tdp_w=5.0,
        total_tdp_w=10.0,
        ram_gb=4.0,
        category="edge",
        notes="ARM Cortex-A57 quad-core + Maxwell GPU (128 CUDA cores); 5 W / 10 W mode",
    ),
    "jetson_nx": HardwareProfile(
        name="NVIDIA Jetson Xavier NX",
        cpu_tdp_w=10.0,
        gpu_tdp_w=10.0,
        total_tdp_w=20.0,
        ram_gb=8.0,
        category="edge",
        notes="6-core Carmel ARM + Volta GPU (384 CUDA cores); configurable 10/15/20 W modes",
    ),
    "jetson_agx": HardwareProfile(
        name="NVIDIA Jetson AGX Xavier",
        cpu_tdp_w=20.0,
        gpu_tdp_w=30.0,
        total_tdp_w=60.0,
        ram_gb=16.0,
        category="edge",
        notes="8-core Carmel ARM + Volta GPU (512 CUDA cores); 10/15/30 W modes available",
    ),
    "jetson_orin_nano": HardwareProfile(
        name="NVIDIA Jetson Orin Nano (8 GB)",
        cpu_tdp_w=7.0,
        gpu_tdp_w=8.0,
        total_tdp_w=15.0,
        ram_gb=8.0,
        category="edge",
        notes="6-core Cortex-A78AE + Ampere GPU (1024 CUDA cores); next-gen edge platform",
    ),
    "jetson_orin_agx": HardwareProfile(
        name="NVIDIA Jetson AGX Orin (64 GB)",
        cpu_tdp_w=30.0,
        gpu_tdp_w=30.0,
        total_tdp_w=60.0,
        ram_gb=64.0,
        category="edge",
        notes="12-core Cortex-A78AE + Ampere GPU (2048 CUDA cores); up to 275 TOPS",
    ),
    # ---- Mobile / laptop ---------------------------------------------------
    "intel_core_u15w": HardwareProfile(
        name="Intel Core Ultra (15 W TDP-up)",
        cpu_tdp_w=15.0,
        gpu_tdp_w=None,
        total_tdp_w=15.0,
        ram_gb=16.0,
        category="laptop",
        notes="Typical ultrabook PBP; configurable 8–28 W range",
    ),
    "apple_m1": HardwareProfile(
        name="Apple M1",
        cpu_tdp_w=10.0,
        gpu_tdp_w=5.0,
        total_tdp_w=20.0,
        ram_gb=8.0,
        category="laptop",
        notes="Apple Silicon unified memory; measured ~15–25 W peak chip power",
    ),
    "apple_m2": HardwareProfile(
        name="Apple M2",
        cpu_tdp_w=10.0,
        gpu_tdp_w=8.0,
        total_tdp_w=22.0,
        ram_gb=8.0,
        category="laptop",
        notes="Apple Silicon second-gen; measured ~18–25 W peak chip power",
    ),
    "apple_m3": HardwareProfile(
        name="Apple M3",
        cpu_tdp_w=11.0,
        gpu_tdp_w=8.0,
        total_tdp_w=22.0,
        ram_gb=8.0,
        category="laptop",
        notes="3nm process; hardware ray-tracing GPU; ~18–26 W peak",
    ),
    # ---- Desktop / workstation / server ------------------------------------
    "intel_i7_125w": HardwareProfile(
        name="Intel Core i7 (125 W TDP)",
        cpu_tdp_w=125.0,
        gpu_tdp_w=None,
        total_tdp_w=125.0,
        ram_gb=32.0,
        category="server",
        notes="High-end desktop CPU; use as upper-bound energy reference for classical trackers",
    ),
    "amd_ryzen9_105w": HardwareProfile(
        name="AMD Ryzen 9 (105 W TDP)",
        cpu_tdp_w=105.0,
        gpu_tdp_w=None,
        total_tdp_w=105.0,
        ram_gb=32.0,
        category="server",
        notes="High-core-count desktop CPU; strong multi-threaded tracking baseline",
    ),
}


def get_profile(name: str) -> HardwareProfile:
    """Return the :class:`HardwareProfile` for *name*.

    Lookup is case- and whitespace-insensitive.

    Args:
        name: Profile key (e.g. ``"rpi4"``, ``"jetson_nano"``).
              See :data:`HARDWARE_PROFILES` for all available keys.

    Returns:
        The matching :class:`HardwareProfile`.

    Raises:
        KeyError: If *name* is not in :data:`HARDWARE_PROFILES`.

    Example::

        from eovot.profiling.hardware_profiles import get_profile
        profile = get_profile("jetson_nx")
        print(profile.total_tdp_w)  # 20.0
    """
    key = name.lower().strip().replace(" ", "_")
    if key not in HARDWARE_PROFILES:
        available = ", ".join(sorted(HARDWARE_PROFILES))
        raise KeyError(
            f"Unknown hardware profile {name!r}. "
            f"Available profiles: {available}"
        )
    return HARDWARE_PROFILES[key]


def list_profiles() -> Dict[str, str]:
    """Return a ``{key: full_name}`` mapping of all registered device profiles.

    Useful for CLI help text or documentation generation.

    Returns:
        Dict mapping short profile key to the device's full display name.

    Example::

        from eovot.profiling.hardware_profiles import list_profiles
        for key, name in list_profiles().items():
            print(f"{key:25s}  {name}")
    """
    return {k: v.name for k, v in HARDWARE_PROFILES.items()}
