"""Hardware profiling and device classification for EOVOT.

Auto-detects CPU count, available memory, and classifies the current device
into a :class:`DeviceClass` category.  The classification drives the default
TDP estimate used in energy profiling and the tracker performance expectations
in :class:`~eovot.selection.tracker_selector.TrackerSelector`.

Device classification heuristic
--------------------------------
The heuristic is intentionally simple so that it works without any
external tooling (e.g. no ``dmidecode``, no NVIDIA SMI):

1. CPU count ≤ 4 AND total RAM ≤ 4 GB  → ``MICROCONTROLLER``
2. CPU count ≤ 4 AND total RAM ≤ 8 GB  → ``EMBEDDED``
3. CPU count ≤ 8 AND total RAM ≤ 16 GB → ``LAPTOP``
4. Otherwise                            → ``DESKTOP``

Users can always override by constructing :class:`HardwareProfile` directly
or loading from a YAML preset (see ``configs/hardware_profiles/``).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import psutil


class DeviceClass(str, Enum):
    """Coarse device category used to set TDP defaults and performance targets."""

    MICROCONTROLLER = "microcontroller"  # e.g. Raspberry Pi Zero
    EMBEDDED = "embedded"               # e.g. Raspberry Pi 4, Jetson Nano
    LAPTOP = "laptop"                   # e.g. i5/i7 ultrabook
    DESKTOP = "desktop"                 # e.g. workstation / server

    # Canonical TDP estimates (Watts) per device class.
    @property
    def typical_tdp_watts(self) -> float:
        return {
            DeviceClass.MICROCONTROLLER: 3.0,
            DeviceClass.EMBEDDED: 8.0,
            DeviceClass.LAPTOP: 15.0,
            DeviceClass.DESKTOP: 65.0,
        }[self]

    # Expected CPU performance scaling factor (relative to a baseline laptop).
    @property
    def performance_factor(self) -> float:
        return {
            DeviceClass.MICROCONTROLLER: 0.15,
            DeviceClass.EMBEDDED: 0.35,
            DeviceClass.LAPTOP: 1.0,
            DeviceClass.DESKTOP: 2.5,
        }[self]


@dataclass
class HardwareProfile:
    """Snapshot of relevant hardware characteristics for one device.

    Attributes:
        cpu_count: Logical CPU core count.
        total_ram_gb: Total installed RAM in GiB.
        available_ram_gb: Currently available RAM in GiB (at profile creation time).
        device_class: Coarse device category.
        tdp_watts: Estimated CPU TDP in Watts (defaults to the class typical).
        cpu_freq_mhz: Current CPU frequency in MHz, if available.
        platform_tag: Short OS/arch string for provenance (e.g. ``"Linux-aarch64"``).
    """

    cpu_count: int
    total_ram_gb: float
    available_ram_gb: float
    device_class: DeviceClass
    tdp_watts: float
    cpu_freq_mhz: Optional[float] = None
    platform_tag: str = field(default_factory=lambda: f"{platform.system()}-{platform.machine()}")

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls, tdp_watts: Optional[float] = None) -> "HardwareProfile":
        """Auto-detect hardware and return a :class:`HardwareProfile`.

        Args:
            tdp_watts: Override the TDP estimate.  If ``None``, the typical
                value for the detected :class:`DeviceClass` is used.

        Returns:
            A populated :class:`HardwareProfile` for the current machine.
        """
        cpu_count = psutil.cpu_count(logical=True) or 1
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024 ** 3)
        available_ram_gb = mem.available / (1024 ** 3)

        freq_info = psutil.cpu_freq()
        cpu_freq_mhz = float(freq_info.current) if freq_info else None

        device_class = cls._classify(cpu_count, total_ram_gb)
        effective_tdp = tdp_watts if tdp_watts is not None else device_class.typical_tdp_watts

        return cls(
            cpu_count=cpu_count,
            total_ram_gb=round(total_ram_gb, 2),
            available_ram_gb=round(available_ram_gb, 2),
            device_class=device_class,
            tdp_watts=effective_tdp,
            cpu_freq_mhz=round(cpu_freq_mhz, 1) if cpu_freq_mhz else None,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "HardwareProfile":
        """Deserialise from a plain dict (e.g. loaded from a YAML preset).

        Args:
            d: Dict with keys matching :class:`HardwareProfile` field names.
               ``device_class`` may be a string value of :class:`DeviceClass`.

        Returns:
            Populated :class:`HardwareProfile`.
        """
        d = dict(d)
        if isinstance(d.get("device_class"), str):
            d["device_class"] = DeviceClass(d["device_class"])
        return cls(**d)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(cpu_count: int, total_ram_gb: float) -> DeviceClass:
        if cpu_count <= 4 and total_ram_gb <= 4.0:
            return DeviceClass.MICROCONTROLLER
        if cpu_count <= 4 and total_ram_gb <= 8.0:
            return DeviceClass.EMBEDDED
        if cpu_count <= 8 and total_ram_gb <= 16.0:
            return DeviceClass.LAPTOP
        return DeviceClass.DESKTOP

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for YAML/JSON export."""
        return {
            "cpu_count": self.cpu_count,
            "total_ram_gb": self.total_ram_gb,
            "available_ram_gb": self.available_ram_gb,
            "device_class": self.device_class.value,
            "tdp_watts": self.tdp_watts,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "platform_tag": self.platform_tag,
        }

    def __str__(self) -> str:
        freq = f"  freq={self.cpu_freq_mhz:.0f}MHz" if self.cpu_freq_mhz else ""
        return (
            f"HardwareProfile[{self.device_class.value}] "
            f"cpus={self.cpu_count}  "
            f"ram={self.total_ram_gb:.1f}GB (avail={self.available_ram_gb:.1f}GB)  "
            f"tdp={self.tdp_watts}W{freq}  "
            f"platform={self.platform_tag}"
        )
