"""Hardware profiles for edge-device deployment targeting.

Defines :class:`HardwareProfile` — a dataclass that captures the key
deployment constraints of a target device — and a curated registry of common
edge, embedded, and desktop platforms.

These profiles are used by :class:`~eovot.profiling.deployment_advisor.DeploymentAdvisor`
to score tracker benchmark results against real-world constraints.

Typical usage::

    from eovot.profiling.hardware_profiles import PROFILES, get_profile

    profile = get_profile("jetson_nano")
    print(profile.target_fps)   # 30.0
    print(profile.memory_mb)    # 4096

Custom profiles can be created directly::

    custom = HardwareProfile(
        name="my_device",
        display_name="Custom MCU",
        tdp_watts=3.0,
        memory_mb=512,
        target_fps=15.0,
        latency_budget_ms=66.6,
        power_budget_w=2.5,
        description="Custom microcontroller board",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HardwareProfile:
    """Deployment constraints for a target hardware platform.

    Attributes:
        name:               Canonical identifier (used as dict key).
        display_name:       Human-readable label for reports.
        tdp_watts:          CPU Thermal Design Power in Watts.
        memory_mb:          Available RAM in megabytes.
        target_fps:         Minimum frames per second for real-time operation.
        latency_budget_ms:  Maximum per-frame processing time in milliseconds.
        power_budget_w:     Maximum sustained power draw in Watts.
        description:        Free-text device specification summary.
        category:           Device tier: ``"ultra_low"``, ``"edge"``,
                            ``"embedded"``, ``"laptop"``, or ``"workstation"``.
    """

    name: str
    display_name: str
    tdp_watts: float
    memory_mb: float
    target_fps: float
    latency_budget_ms: float
    power_budget_w: float
    description: str = ""
    category: str = "edge"

    def __post_init__(self) -> None:
        if self.tdp_watts <= 0:
            raise ValueError(f"tdp_watts must be positive, got {self.tdp_watts}")
        if self.memory_mb <= 0:
            raise ValueError(f"memory_mb must be positive, got {self.memory_mb}")
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {self.target_fps}")
        if self.latency_budget_ms <= 0:
            raise ValueError(f"latency_budget_ms must be positive")
        if self.power_budget_w <= 0:
            raise ValueError(f"power_budget_w must be positive")

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "tdp_watts": self.tdp_watts,
            "memory_mb": self.memory_mb,
            "target_fps": self.target_fps,
            "latency_budget_ms": self.latency_budget_ms,
            "power_budget_w": self.power_budget_w,
            "description": self.description,
            "category": self.category,
        }


# ---------------------------------------------------------------------------
# Built-in device registry
# ---------------------------------------------------------------------------

PROFILES: Dict[str, HardwareProfile] = {
    "raspberry_pi_4": HardwareProfile(
        name="raspberry_pi_4",
        display_name="Raspberry Pi 4 (4 GB)",
        tdp_watts=6.0,
        memory_mb=4096,
        target_fps=25.0,
        latency_budget_ms=40.0,
        power_budget_w=5.0,
        description="Quad-core ARM Cortex-A72 @ 1.5 GHz, 4 GB LPDDR4, no GPU",
        category="ultra_low",
    ),
    "raspberry_pi_5": HardwareProfile(
        name="raspberry_pi_5",
        display_name="Raspberry Pi 5 (8 GB)",
        tdp_watts=12.0,
        memory_mb=8192,
        target_fps=30.0,
        latency_budget_ms=33.0,
        power_budget_w=10.0,
        description="Quad-core ARM Cortex-A76 @ 2.4 GHz, 8 GB LPDDR4X, no GPU",
        category="ultra_low",
    ),
    "jetson_nano": HardwareProfile(
        name="jetson_nano",
        display_name="NVIDIA Jetson Nano (4 GB)",
        tdp_watts=10.0,
        memory_mb=4096,
        target_fps=30.0,
        latency_budget_ms=33.0,
        power_budget_w=10.0,
        description=(
            "Quad-core ARM Cortex-A57 @ 1.43 GHz, "
            "128-core Maxwell GPU, 4 GB LPDDR4"
        ),
        category="edge",
    ),
    "jetson_xavier_nx": HardwareProfile(
        name="jetson_xavier_nx",
        display_name="NVIDIA Jetson Xavier NX (8 GB)",
        tdp_watts=15.0,
        memory_mb=8192,
        target_fps=60.0,
        latency_budget_ms=16.0,
        power_budget_w=15.0,
        description=(
            "6-core NVIDIA Carmel ARM @ 1.4 GHz, "
            "384-core Volta GPU, 8 GB LPDDR4x"
        ),
        category="edge",
    ),
    "jetson_orin_nano": HardwareProfile(
        name="jetson_orin_nano",
        display_name="NVIDIA Jetson Orin Nano (8 GB)",
        tdp_watts=15.0,
        memory_mb=8192,
        target_fps=60.0,
        latency_budget_ms=16.0,
        power_budget_w=15.0,
        description=(
            "6-core Arm Cortex-A78AE, "
            "1024-core Ampere GPU, 8 GB LPDDR5"
        ),
        category="edge",
    ),
    "intel_nuc_i5": HardwareProfile(
        name="intel_nuc_i5",
        display_name="Intel NUC (i5-1135G7)",
        tdp_watts=28.0,
        memory_mb=16384,
        target_fps=60.0,
        latency_budget_ms=16.0,
        power_budget_w=20.0,
        description=(
            "4-core/8-thread Tiger Lake @ 2.4–4.2 GHz, "
            "Intel Iris Xe, 16 GB DDR4"
        ),
        category="embedded",
    ),
    "laptop_cpu": HardwareProfile(
        name="laptop_cpu",
        display_name="Generic Laptop CPU (15 W TDP)",
        tdp_watts=15.0,
        memory_mb=8192,
        target_fps=30.0,
        latency_budget_ms=33.0,
        power_budget_w=15.0,
        description="Representative mobile CPU, ~15 W TDP class",
        category="laptop",
    ),
    "workstation": HardwareProfile(
        name="workstation",
        display_name="Workstation CPU (65 W TDP)",
        tdp_watts=65.0,
        memory_mb=32768,
        target_fps=120.0,
        latency_budget_ms=8.0,
        power_budget_w=65.0,
        description="Desktop-class CPU, no strict power constraint",
        category="workstation",
    ),
}


def get_profile(name: str) -> HardwareProfile:
    """Retrieve a profile by canonical name.

    Args:
        name: One of the keys in :data:`PROFILES`.

    Returns:
        The corresponding :class:`HardwareProfile`.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise KeyError(
            f"Unknown hardware profile {name!r}. Available: {available}"
        )
    return PROFILES[name]


def list_profiles() -> List[HardwareProfile]:
    """Return all built-in profiles sorted by TDP (ascending)."""
    return sorted(PROFILES.values(), key=lambda p: p.tdp_watts)
