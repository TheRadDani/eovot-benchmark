"""Predefined edge device profiles for constraint-aware tracker evaluation.

Each :class:`EdgeProfile` captures the deployment constraints of a target
hardware platform.  The :class:`~eovot.constraints.evaluator.ConstraintEvaluator`
compares benchmark measurements against these thresholds to produce a
deployability verdict for every tracker/device pair.

Predefined profiles cover the most common edge deployment targets in
robotics and embedded computer vision:

* :data:`RASPBERRY_PI_4`   — ARM Cortex-A72, ~6 W TDP
* :data:`JETSON_NANO`      — ARM Cortex-A57 + 128-core Maxwell GPU, ~10 W TDP
* :data:`MOBILE_CLASS`     — Smartphone-class ARM big.LITTLE, ~3 W CPU draw
* :data:`EMBEDDED_MICRO`   — Ultra-low-power MCU class, tight memory budget
* :data:`LAPTOP_CPU`       — Modern x86 laptop, unconstrained baseline reference

Custom profiles can be created with :class:`EdgeProfile` directly::

    from eovot.constraints.profiles import EdgeProfile

    my_profile = EdgeProfile(
        name="Custom FPGA Board",
        min_fps=20.0,
        max_memory_mb=128.0,
        max_latency_ms=50.0,
        max_energy_mj_per_frame=20.0,
        description="Xilinx ZCU104, 10 W power budget",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EdgeProfile:
    """Hardware deployment constraints for a target edge device.

    All numeric thresholds define the boundary a tracker must stay within
    to be considered deployable on the device.  FPS is a *lower* bound
    (the tracker must be fast enough); all others are *upper* bounds.

    Attributes:
        name: Human-readable device identifier used in reports.
        min_fps: Minimum acceptable throughput (frames per second).
            A tracker running slower than this cannot keep up with
            the camera stream in real-time.
        max_memory_mb: Maximum allowable peak RAM usage in megabytes.
            Reflects the usable RAM budget on the target device after
            the OS and other processes have claimed their share.
        max_latency_ms: Maximum allowable per-frame processing latency
            in milliseconds.  Derived from the required response time
            for the application (e.g. 33 ms ≈ 30 FPS real-time).
        max_energy_mj_per_frame: Maximum energy budget per frame in
            milli-Joules.  Set to ``None`` when energy is unconstrained
            (e.g. desktop / server deployments).
        description: Free-text notes about the device or profile origin.
    """

    name: str
    min_fps: float
    max_memory_mb: float
    max_latency_ms: float
    max_energy_mj_per_frame: Optional[float] = None
    description: str = field(default="")

    def __post_init__(self) -> None:
        if self.min_fps <= 0:
            raise ValueError(f"min_fps must be positive, got {self.min_fps}")
        if self.max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be positive, got {self.max_memory_mb}")
        if self.max_latency_ms <= 0:
            raise ValueError(f"max_latency_ms must be positive, got {self.max_latency_ms}")
        if self.max_energy_mj_per_frame is not None and self.max_energy_mj_per_frame <= 0:
            raise ValueError(
                f"max_energy_mj_per_frame must be positive or None, "
                f"got {self.max_energy_mj_per_frame}"
            )


# ---------------------------------------------------------------------------
# Predefined device profiles
# ---------------------------------------------------------------------------

RASPBERRY_PI_4 = EdgeProfile(
    name="Raspberry Pi 4",
    min_fps=10.0,
    max_memory_mb=512.0,
    max_latency_ms=100.0,
    max_energy_mj_per_frame=60.0,
    description="ARM Cortex-A72 @ 1.8 GHz, 4 GB LPDDR4, ~6 W TDP",
)
"""Raspberry Pi 4 Model B — the most common low-cost edge platform."""

JETSON_NANO = EdgeProfile(
    name="NVIDIA Jetson Nano",
    min_fps=15.0,
    max_memory_mb=1024.0,
    max_latency_ms=67.0,
    max_energy_mj_per_frame=100.0,
    description="ARM Cortex-A57 @ 1.43 GHz + 128-core Maxwell GPU, 4 GB RAM, ~10 W TDP",
)
"""NVIDIA Jetson Nano — entry-level GPU-accelerated edge platform."""

MOBILE_CLASS = EdgeProfile(
    name="Mobile Device",
    min_fps=25.0,
    max_memory_mb=256.0,
    max_latency_ms=40.0,
    max_energy_mj_per_frame=30.0,
    description="Smartphone ARM big.LITTLE ~2.5 GHz, 256 MB tracking budget, ~3 W CPU draw",
)
"""Smartphone-class CPU — represents augmented-reality / mobile CV applications."""

EMBEDDED_MICRO = EdgeProfile(
    name="Embedded Microcontroller",
    min_fps=5.0,
    max_memory_mb=64.0,
    max_latency_ms=200.0,
    max_energy_mj_per_frame=10.0,
    description="Ultra-low-power MCU class, 64 MB RAM hard limit, tight energy budget",
)
"""Ultra-constrained microcontroller class — e.g. STM32 H7 or similar."""

LAPTOP_CPU = EdgeProfile(
    name="Laptop CPU",
    min_fps=30.0,
    max_memory_mb=4096.0,
    max_latency_ms=33.0,
    max_energy_mj_per_frame=None,
    description="Modern x86 laptop CPU, unconstrained energy — baseline reference profile",
)
"""Laptop CPU baseline — unconstrained energy, used as a reference tier."""

PREDEFINED_PROFILES: Dict[str, EdgeProfile] = {
    "raspberry_pi_4": RASPBERRY_PI_4,
    "jetson_nano": JETSON_NANO,
    "mobile": MOBILE_CLASS,
    "embedded_micro": EMBEDDED_MICRO,
    "laptop_cpu": LAPTOP_CPU,
}
"""Registry mapping short keys to predefined :class:`EdgeProfile` instances.

Use this to look up a profile by name in scripts and config files::

    from eovot.constraints.profiles import PREDEFINED_PROFILES

    profile = PREDEFINED_PROFILES["raspberry_pi_4"]
"""
