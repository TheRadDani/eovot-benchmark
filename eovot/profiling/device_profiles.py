"""Edge device hardware profile registry for EOVOT.

Provides a catalogue of common edge deployment targets with their key
hardware constraints (TDP, memory limit, target throughput).  These profiles
enable hardware-aware benchmarking without physical access to the target
device: benchmark locally, then project cost and suitability onto each profile.

Usage::

    from eovot.profiling.device_profiles import DEVICE_PROFILES, get_device

    rpi4 = get_device("raspberry_pi_4")
    print(rpi4.tdp_watts)       # 5.1
    print(rpi4.target_fps)      # 15.0
    print(rpi4.fits_memory(200.0))  # True

    for key in list_devices():
        print(key)

Data sources:
    - Raspberry Pi 4 Model B:
        https://www.raspberrypi.com/products/raspberry-pi-4-model-b/
    - NVIDIA Jetson Nano / Orin Nano:
        https://developer.nvidia.com/embedded/jetson-nano
    - Google Coral USB Accelerator:
        https://coral.ai/products/accelerator
    - Intel NUC (i5-8265U) — TDP from Intel ARK database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DeviceProfile:
    """Hardware specification for an edge deployment target.

    Attributes:
        key:              Short machine-readable identifier
                          (e.g. ``"raspberry_pi_4"``).
        display_name:     Human-readable device name.
        tdp_watts:        CPU (+ integrated GPU) Thermal Design Power (W).
        memory_limit_mb:  Available RAM for the inference process (MB).
        target_fps:       Minimum acceptable frame-rate for real-time use.
        cpu_cores:        Physical CPU core count.
        has_gpu:          True if a GPU accelerator is present.
        has_npu:          True if a dedicated neural/tensor processing unit
                          is present.
        notes:            Free-form description / data-source note.
    """

    key: str
    display_name: str
    tdp_watts: float
    memory_limit_mb: int
    target_fps: float
    cpu_cores: int
    has_gpu: bool = False
    has_npu: bool = False
    notes: str = ""

    def fits_memory(self, peak_memory_mb: float) -> bool:
        """Return ``True`` if *peak_memory_mb* is within the device's limit."""
        return peak_memory_mb <= self.memory_limit_mb

    def meets_fps(self, fps: float) -> bool:
        """Return ``True`` if *fps* is at or above the device's target."""
        return fps >= self.target_fps

    def to_dict(self) -> Dict:
        """Serialize this profile to a plain dict."""
        return {
            "key": self.key,
            "display_name": self.display_name,
            "tdp_watts": self.tdp_watts,
            "memory_limit_mb": self.memory_limit_mb,
            "target_fps": self.target_fps,
            "cpu_cores": self.cpu_cores,
            "has_gpu": self.has_gpu,
            "has_npu": self.has_npu,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Built-in device catalogue
# ---------------------------------------------------------------------------

DEVICE_PROFILES: Dict[str, DeviceProfile] = {
    "raspberry_pi_4": DeviceProfile(
        key="raspberry_pi_4",
        display_name="Raspberry Pi 4 (4 GB)",
        tdp_watts=5.1,
        memory_limit_mb=4096,
        target_fps=15.0,
        cpu_cores=4,
        has_gpu=False,
        has_npu=False,
        notes=(
            "ARM Cortex-A72 @ 1.8 GHz, 4 GB LPDDR4-3200. "
            "Standard CPU-only tracking target for robotics."
        ),
    ),
    "jetson_nano": DeviceProfile(
        key="jetson_nano",
        display_name="NVIDIA Jetson Nano (4 GB)",
        tdp_watts=10.0,
        memory_limit_mb=4096,
        target_fps=30.0,
        cpu_cores=4,
        has_gpu=True,
        has_npu=False,
        notes=(
            "ARM Cortex-A57 + 128-core Maxwell GPU, 4 GB LPDDR4. "
            "10 W MAXN mode; 5 W low-power mode also available."
        ),
    ),
    "jetson_orin_nano": DeviceProfile(
        key="jetson_orin_nano",
        display_name="NVIDIA Jetson Orin Nano (8 GB)",
        tdp_watts=15.0,
        memory_limit_mb=8192,
        target_fps=60.0,
        cpu_cores=6,
        has_gpu=True,
        has_npu=True,
        notes=(
            "6× Cortex-A78AE + 1024-core Ampere GPU + NVDLA, 8 GB LPDDR5. "
            "40 TOPS INT8; successor to Jetson Nano."
        ),
    ),
    "coral_usb": DeviceProfile(
        key="coral_usb",
        display_name="Google Coral USB Accelerator",
        tdp_watts=2.5,
        memory_limit_mb=256,
        target_fps=30.0,
        cpu_cores=1,
        has_gpu=False,
        has_npu=True,
        notes=(
            "Edge TPU (4 TOPS INT8), USB 3.0. "
            "Only TFLite models compiled for Edge TPU are supported."
        ),
    ),
    "intel_nuc_i5": DeviceProfile(
        key="intel_nuc_i5",
        display_name="Intel NUC (Core i5-8265U)",
        tdp_watts=15.0,
        memory_limit_mb=16384,
        target_fps=60.0,
        cpu_cores=4,
        has_gpu=True,
        has_npu=False,
        notes=(
            "15 W TDP; Intel UHD 620 iGPU. "
            "Compact desktop / robotics compute platform."
        ),
    ),
    "desktop_cpu": DeviceProfile(
        key="desktop_cpu",
        display_name="Desktop CPU (baseline)",
        tdp_watts=65.0,
        memory_limit_mb=32768,
        target_fps=120.0,
        cpu_cores=8,
        has_gpu=False,
        has_npu=False,
        notes=(
            "Representative mid-range desktop (e.g. AMD Ryzen 5 5600X). "
            "Used as an unconstrained performance reference."
        ),
    ),
}


def list_devices() -> List[str]:
    """Return a sorted list of all registered device keys."""
    return sorted(DEVICE_PROFILES.keys())


def get_device(key: str) -> DeviceProfile:
    """Retrieve a :class:`DeviceProfile` by its key.

    Args:
        key: A key from :data:`DEVICE_PROFILES`
             (e.g. ``"jetson_nano"``).

    Raises:
        KeyError: If *key* is not in the registry.
    """
    if key not in DEVICE_PROFILES:
        available = ", ".join(sorted(DEVICE_PROFILES))
        raise KeyError(
            f"Unknown device '{key}'. Available: {available}"
        )
    return DEVICE_PROFILES[key]
