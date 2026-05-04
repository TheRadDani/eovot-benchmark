"""Hardware device profiles for edge deployment evaluation.

Defines :class:`HardwareProfile` — a dataclass encoding a target device's
deployment constraints (TDP, RAM, acceptable FPS/latency/memory) — along
with five built-in profiles covering the most common edge deployment targets.

Usage::

    from eovot.profiling.hardware_profiles import get_profile

    pi = get_profile("raspberry_pi4")
    ok = pi.is_tracker_suitable(fps=18.3, latency_ms=54.6, memory_mb=210.0)
    score = pi.deployment_score(fps=18.3, latency_ms=54.6, memory_mb=210.0)

Custom profiles can be loaded from YAML::

    profile = load_profile_from_yaml("configs/hardware/my_device.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class HardwareProfile:
    """Deployment constraints for a target edge device.

    Attributes:
        name:              Human-readable device name.
        cpu_tdp_watts:     CPU Thermal Design Power (W) from the datasheet.
        ram_mb:            Total device RAM in megabytes.
        cpu_cores:         Number of physical CPU cores.
        cpu_freq_ghz:      Base CPU clock frequency (GHz).
        has_gpu:           Whether the device has a usable GPU.
        gpu_tdp_watts:     GPU TDP (W); 0 when ``has_gpu`` is False.
        target_fps:        Minimum acceptable tracker FPS on this device.
        max_latency_ms:    Maximum acceptable per-frame latency (ms).
        max_memory_mb:     Maximum acceptable tracker memory footprint (MB).
        description:       Free-text device description.
    """

    name: str
    cpu_tdp_watts: float
    ram_mb: int
    cpu_cores: int
    cpu_freq_ghz: float
    has_gpu: bool = False
    gpu_tdp_watts: float = 0.0
    target_fps: float = 30.0
    max_latency_ms: float = 33.3
    max_memory_mb: float = 256.0
    description: str = ""

    def is_tracker_suitable(self, fps: float, latency_ms: float, memory_mb: float) -> bool:
        """Return True iff the tracker meets all three deployment constraints.

        Args:
            fps:        Measured tracker FPS.
            latency_ms: Measured mean per-frame latency (ms).
            memory_mb:  Measured peak memory usage (MB).
        """
        return (
            fps >= self.target_fps
            and latency_ms <= self.max_latency_ms
            and memory_mb <= self.max_memory_mb
        )

    def deployment_score(self, fps: float, latency_ms: float, memory_mb: float) -> float:
        """Compute a [0, 1] deployment feasibility score for this hardware.

        Each dimension is scored independently and clipped to [0, 1]:
        - FPS score:     min(fps / target_fps, 1.0)
        - Latency score: min(max_latency_ms / latency_ms, 1.0)
        - Memory score:  min(max_memory_mb / memory_mb, 1.0)

        Returns the unweighted mean of the three sub-scores.

        Args:
            fps:        Measured tracker FPS.
            latency_ms: Measured mean per-frame latency (ms).
            memory_mb:  Measured peak memory usage (MB).

        Returns:
            Float in [0, 1]; 1.0 means the tracker fully satisfies all
            constraints, values below 1.0 indicate which dimension fails.
        """
        fps_score = min(fps / self.target_fps, 1.0) if self.target_fps > 0 else 1.0
        lat_score = min(self.max_latency_ms / latency_ms, 1.0) if latency_ms > 0 else 1.0
        mem_score = min(self.max_memory_mb / memory_mb, 1.0) if memory_mb > 0 else 1.0
        return round((fps_score + lat_score + mem_score) / 3.0, 6)


# ---------------------------------------------------------------------------
# Built-in hardware profiles
# ---------------------------------------------------------------------------

RASPBERRY_PI_4 = HardwareProfile(
    name="Raspberry Pi 4 (4 GB)",
    cpu_tdp_watts=6.0,
    ram_mb=4096,
    cpu_cores=4,
    cpu_freq_ghz=1.5,
    has_gpu=False,
    gpu_tdp_watts=0.0,
    target_fps=15.0,
    max_latency_ms=66.7,
    max_memory_mb=512.0,
    description="ARM Cortex-A72, quad-core 1.5 GHz. Most common IoT / edge device.",
)

JETSON_NANO = HardwareProfile(
    name="NVIDIA Jetson Nano",
    cpu_tdp_watts=10.0,
    ram_mb=4096,
    cpu_cores=4,
    cpu_freq_ghz=1.43,
    has_gpu=True,
    gpu_tdp_watts=5.0,
    target_fps=30.0,
    max_latency_ms=33.3,
    max_memory_mb=1024.0,
    description="ARM Cortex-A57 + 128 Maxwell CUDA cores. Entry-level embedded GPU.",
)

INTEL_NUC = HardwareProfile(
    name="Intel NUC (Core i5)",
    cpu_tdp_watts=28.0,
    ram_mb=16384,
    cpu_cores=4,
    cpu_freq_ghz=3.6,
    has_gpu=False,
    gpu_tdp_watts=0.0,
    target_fps=60.0,
    max_latency_ms=16.7,
    max_memory_mb=2048.0,
    description="Intel Core i5 NUC. Mid-range edge compute platform.",
)

DESKTOP_CPU = HardwareProfile(
    name="Desktop CPU (x86 baseline)",
    cpu_tdp_watts=65.0,
    ram_mb=32768,
    cpu_cores=8,
    cpu_freq_ghz=4.0,
    has_gpu=False,
    gpu_tdp_watts=0.0,
    target_fps=100.0,
    max_latency_ms=10.0,
    max_memory_mb=8192.0,
    description="Standard desktop workstation. Used as upper-bound benchmark baseline.",
)

SMARTPHONE = HardwareProfile(
    name="Smartphone ARM SoC",
    cpu_tdp_watts=4.0,
    ram_mb=4096,
    cpu_cores=8,
    cpu_freq_ghz=2.4,
    has_gpu=False,
    gpu_tdp_watts=0.0,
    target_fps=24.0,
    max_latency_ms=41.7,
    max_memory_mb=256.0,
    description="Modern smartphone SoC (e.g. Snapdragon 8 Gen 2). Mobile deployment target.",
)


BUILTIN_PROFILES: Dict[str, HardwareProfile] = {
    "raspberry_pi4": RASPBERRY_PI_4,
    "jetson_nano": JETSON_NANO,
    "intel_nuc": INTEL_NUC,
    "desktop_cpu": DESKTOP_CPU,
    "smartphone": SMARTPHONE,
}


def get_profile(name: str) -> HardwareProfile:
    """Retrieve a built-in hardware profile by key.

    Args:
        name: One of ``'raspberry_pi4'``, ``'jetson_nano'``, ``'intel_nuc'``,
              ``'desktop_cpu'``, ``'smartphone'``.

    Returns:
        The corresponding :class:`HardwareProfile`.

    Raises:
        KeyError: If *name* is not found in the built-in registry.
    """
    if name not in BUILTIN_PROFILES:
        raise KeyError(
            f"Unknown hardware profile '{name}'. "
            f"Available profiles: {sorted(BUILTIN_PROFILES)}"
        )
    return BUILTIN_PROFILES[name]


def load_profile_from_yaml(path: str) -> HardwareProfile:
    """Load a :class:`HardwareProfile` from a YAML file.

    The YAML must contain keys matching the :class:`HardwareProfile` field
    names.  Unknown keys are silently ignored so that future fields can be
    added without breaking existing YAML files.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        A :class:`HardwareProfile` populated from the YAML data.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If a required field is missing from the YAML.
    """
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"Hardware profile YAML not found: {fpath}")
    with open(fpath, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    valid_fields = HardwareProfile.__dataclass_fields__
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return HardwareProfile(**filtered)
