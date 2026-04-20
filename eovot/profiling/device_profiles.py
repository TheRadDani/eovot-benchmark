"""Hardware device profiles for edge-aware benchmarking.

Provides pre-defined ``DeviceProfile`` objects covering common edge and cloud
hardware targets.  Each profile bundles the key parameters needed for
energy estimation (TDP) and deployment feasibility assessment (RAM, cores).

Usage::

    from eovot.profiling.device_profiles import get_profile

    profile = get_profile("jetson_nano")
    engine = BenchmarkEngine(tdp_watts=profile.tdp_watts)

Available profiles (use the string key with :func:`get_profile`):

======================  ==========  ========  ======  =========
Key                     TDP (W)     RAM (GB)  Cores   GPU
======================  ==========  ========  ======  =========
raspberry_pi_4          6.4         4.0       4       No
jetson_nano             10.0        4.0       4       Yes
jetson_xavier_nx        15.0        8.0       6       Yes
coral_dev_board         2.0         1.0       4       No (TPU)
intel_nuc_i5            28.0        16.0      4       No
laptop_cpu              15.0        16.0      8       No
desktop_gpu_workstation 250.0       64.0      16      Yes
======================  ==========  ========  ======  =========
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DeviceProfile:
    """Hardware specification for a target deployment device.

    Attributes:
        key: Short identifier (snake_case) used as lookup key.
        name: Human-readable device name.
        tdp_watts: Thermal Design Power in Watts — used as the upper-bound
            CPU power envelope in energy estimates.
        ram_gb: Total system RAM in GB — used to assess memory feasibility.
        cpu_cores: Number of available CPU cores.
        has_gpu: Whether the device has a programmable GPU.
        gpu_vram_gb: GPU video RAM in GB (``None`` if no GPU).
        description: Optional free-text description with context notes.
    """

    key: str
    name: str
    tdp_watts: float
    ram_gb: float
    cpu_cores: int
    has_gpu: bool = False
    gpu_vram_gb: Optional[float] = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.tdp_watts <= 0:
            raise ValueError(f"tdp_watts must be positive, got {self.tdp_watts}")
        if self.ram_gb <= 0:
            raise ValueError(f"ram_gb must be positive, got {self.ram_gb}")
        if self.cpu_cores < 1:
            raise ValueError(f"cpu_cores must be >= 1, got {self.cpu_cores}")

    def to_dict(self) -> Dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "key": self.key,
            "name": self.name,
            "tdp_watts": self.tdp_watts,
            "ram_gb": self.ram_gb,
            "cpu_cores": self.cpu_cores,
            "has_gpu": self.has_gpu,
            "gpu_vram_gb": self.gpu_vram_gb,
            "description": self.description,
        }

    def __str__(self) -> str:
        gpu_str = f"  GPU {self.gpu_vram_gb} GB VRAM" if self.has_gpu else ""
        return (
            f"DeviceProfile[{self.key}] {self.name} | "
            f"TDP={self.tdp_watts}W  RAM={self.ram_gb}GB  "
            f"cores={self.cpu_cores}{gpu_str}"
        )


# ---------------------------------------------------------------------------
# Pre-defined device profiles
# ---------------------------------------------------------------------------

DEVICE_PROFILES: Dict[str, DeviceProfile] = {
    "raspberry_pi_4": DeviceProfile(
        key="raspberry_pi_4",
        name="Raspberry Pi 4 Model B (4 GB)",
        tdp_watts=6.4,
        ram_gb=4.0,
        cpu_cores=4,
        description="ARM Cortex-A72 @ 1.8 GHz. Typical constrained IoT edge device.",
    ),
    "jetson_nano": DeviceProfile(
        key="jetson_nano",
        name="NVIDIA Jetson Nano (4 GB)",
        tdp_watts=10.0,
        ram_gb=4.0,
        cpu_cores=4,
        has_gpu=True,
        gpu_vram_gb=4.0,
        description="ARM Cortex-A57 + 128-core Maxwell GPU. Entry-level Jetson platform.",
    ),
    "jetson_xavier_nx": DeviceProfile(
        key="jetson_xavier_nx",
        name="NVIDIA Jetson Xavier NX (8 GB)",
        tdp_watts=15.0,
        ram_gb=8.0,
        cpu_cores=6,
        has_gpu=True,
        gpu_vram_gb=8.0,
        description="6-core Carmel + 384-core Volta GPU. Suitable for real-time DL tracking.",
    ),
    "coral_dev_board": DeviceProfile(
        key="coral_dev_board",
        name="Google Coral Dev Board",
        tdp_watts=2.0,
        ram_gb=1.0,
        cpu_cores=4,
        description=(
            "ARM Cortex-A53 + Edge TPU (8 TOPS int8). Very low power; "
            "classical trackers or quantised DL models recommended."
        ),
    ),
    "intel_nuc_i5": DeviceProfile(
        key="intel_nuc_i5",
        name="Intel NUC (Core i5, 16 GB)",
        tdp_watts=28.0,
        ram_gb=16.0,
        cpu_cores=4,
        description="Compact x86 desktop. Good balance of performance and power for edge servers.",
    ),
    "laptop_cpu": DeviceProfile(
        key="laptop_cpu",
        name="Laptop CPU (15 W TDP)",
        tdp_watts=15.0,
        ram_gb=16.0,
        cpu_cores=8,
        description="Generic laptop CPU baseline (e.g. Intel Core i7-U series).",
    ),
    "desktop_gpu_workstation": DeviceProfile(
        key="desktop_gpu_workstation",
        name="Desktop GPU Workstation",
        tdp_watts=250.0,
        ram_gb=64.0,
        cpu_cores=16,
        has_gpu=True,
        gpu_vram_gb=24.0,
        description=(
            "High-end GPU workstation (e.g. RTX 3090). "
            "Upper-bound reference for accuracy-first benchmarking."
        ),
    ),
}


def get_profile(key: str) -> DeviceProfile:
    """Retrieve a pre-defined :class:`DeviceProfile` by key.

    Args:
        key: Profile identifier string (see module docstring for full list).

    Returns:
        The matching :class:`DeviceProfile`.

    Raises:
        ValueError: If *key* is not in the registry.

    Example::

        profile = get_profile("jetson_nano")
        engine = BenchmarkEngine(tdp_watts=profile.tdp_watts)
    """
    if key not in DEVICE_PROFILES:
        available = sorted(DEVICE_PROFILES.keys())
        raise ValueError(
            f"Unknown device profile {key!r}. "
            f"Available profiles: {available}"
        )
    return DEVICE_PROFILES[key]


def list_profiles() -> Dict[str, DeviceProfile]:
    """Return a copy of the full device profile registry.

    Returns:
        Dict mapping profile key → :class:`DeviceProfile`.
    """
    return dict(DEVICE_PROFILES)
