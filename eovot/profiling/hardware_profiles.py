"""Hardware device profiles for edge-aware benchmark evaluation.

Defines :class:`HardwareProfile` — a device descriptor that captures TDP,
memory limit, and target FPS — together with factory presets for common
edge platforms.  Profiles can be loaded from YAML for experiment
reproducibility.

Usage::

    from eovot.profiling.hardware_profiles import HardwareProfile, PROFILES

    profile = PROFILES["jetson_nano"]
    print(profile.tdp_watts)          # 10.0
    print(profile.memory_limit_mb)    # 4096.0

    # Load from a custom YAML file:
    profile = HardwareProfile.from_yaml("configs/hardware/my_device.yaml")

Reference device TDPs (approximate):
    - Raspberry Pi 4:      ~6 W (BCM2711 SoC)
    - NVIDIA Jetson Nano:  ~10 W (5W / 10W mode)
    - Jetson Orin Nano:    ~15 W (15W mode)
    - Mobile x86 laptop:  ~28 W (configurable TDP)
    - Desktop x86 CPU:    ~95 W (typical 12th-gen i9)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml  # pyyaml — already in requirements.txt


@dataclass
class HardwareProfile:
    """Describes the deployment constraints of an edge target device.

    Attributes:
        name: Human-readable device identifier.
        tdp_watts: Thermal Design Power in Watts (used for energy estimates).
        memory_limit_mb: Available DRAM in MB (RSS cap for the tracker process).
        target_fps: Minimum acceptable frame rate for real-time operation.
        description: Optional free-text note about the device.
    """

    name: str
    tdp_watts: float
    memory_limit_mb: float
    target_fps: float
    description: str = ""

    def __post_init__(self) -> None:
        if self.tdp_watts <= 0:
            raise ValueError(f"tdp_watts must be positive, got {self.tdp_watts}")
        if self.memory_limit_mb <= 0:
            raise ValueError(f"memory_limit_mb must be positive, got {self.memory_limit_mb}")
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {self.target_fps}")

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "HardwareProfile":
        """Create a :class:`HardwareProfile` from a plain dict (e.g. parsed YAML).

        Args:
            d: Dict with keys ``name``, ``tdp_watts``, ``memory_limit_mb``,
               ``target_fps``, and optionally ``description``.
        """
        return cls(
            name=d["name"],
            tdp_watts=float(d["tdp_watts"]),
            memory_limit_mb=float(d["memory_limit_mb"]),
            target_fps=float(d["target_fps"]),
            description=str(d.get("description", "")),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "HardwareProfile":
        """Load a :class:`HardwareProfile` from a YAML config file.

        Args:
            path: Path to a YAML file whose top-level keys match the
                  dataclass fields (see ``configs/hardware/`` for examples).
        """
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain dict suitable for YAML / JSON serialisation."""
        return {
            "name": self.name,
            "tdp_watts": self.tdp_watts,
            "memory_limit_mb": self.memory_limit_mb,
            "target_fps": self.target_fps,
            "description": self.description,
        }

    def __str__(self) -> str:
        return (
            f"HardwareProfile[{self.name}] "
            f"TDP={self.tdp_watts}W  "
            f"mem={self.memory_limit_mb}MB  "
            f"target_fps={self.target_fps}"
        )


# ---------------------------------------------------------------------------
# Built-in presets for common edge platforms
# ---------------------------------------------------------------------------

PROFILES: Dict[str, HardwareProfile] = {
    "raspberry_pi4": HardwareProfile(
        name="Raspberry Pi 4",
        tdp_watts=6.0,
        memory_limit_mb=4096.0,
        target_fps=10.0,
        description="BCM2711 quad-core Cortex-A72 @ 1.8 GHz, 4 GB LPDDR4",
    ),
    "jetson_nano": HardwareProfile(
        name="NVIDIA Jetson Nano",
        tdp_watts=10.0,
        memory_limit_mb=4096.0,
        target_fps=20.0,
        description="Quad-core Cortex-A57 + 128-core Maxwell GPU, 4 GB LPDDR4",
    ),
    "jetson_orin_nano": HardwareProfile(
        name="NVIDIA Jetson Orin Nano",
        tdp_watts=15.0,
        memory_limit_mb=8192.0,
        target_fps=30.0,
        description="6-core Cortex-A78AE + 1024-core Ampere GPU, 8 GB LPDDR5",
    ),
    "x86_laptop": HardwareProfile(
        name="x86 Laptop CPU",
        tdp_watts=28.0,
        memory_limit_mb=16384.0,
        target_fps=30.0,
        description="Typical mobile x86-64 processor (e.g. Intel Core i7 / AMD Ryzen 7)",
    ),
    "x86_desktop": HardwareProfile(
        name="x86 Desktop CPU",
        tdp_watts=95.0,
        memory_limit_mb=32768.0,
        target_fps=60.0,
        description="Typical desktop x86-64 processor (e.g. Intel Core i9 / AMD Ryzen 9)",
    ),
}
