"""Hardware platform detection for EOVOT energy-aware benchmarking.

Automatically identifies the host platform (Raspberry Pi, NVIDIA Jetson,
laptop, desktop, Apple Silicon) and returns the recommended CPU TDP value
for :class:`~eovot.profiling.energy.EnergyProfiler`.

Motivation
----------
Meaningful energy estimates require a device-specific TDP.  Manually
setting ``--tdp-watts`` on every run is error-prone and impedes
reproducibility.  This module introspects the host at runtime so the
benchmark can self-configure without user intervention.

Detection heuristics (applied in priority order)
-------------------------------------------------
1. ``/proc/device-tree/model`` — unambiguous on Raspberry Pi
2. ``/proc/device-tree/compatible`` — detects NVIDIA Jetson SoCs
3. Battery presence (``/sys/class/power_supply/BAT*/``) — laptop vs desktop
4. DMI chassis type — portable chassis codes (8–10, 14) → laptop
5. CPU architecture — ARM/AArch64 without device-tree → conservative edge TDP
6. ``platform.machine()`` fallback → laptop TDP (safe default)

All heuristics use read-only filesystem probes; no shell commands are
executed.

Example
-------
::

    from eovot.hardware import detect_platform, get_recommended_tdp

    platform = detect_platform()
    print(platform.name)          # e.g. "Raspberry Pi 4"
    print(platform.tdp_watts)     # e.g. 6.0

    tdp = get_recommended_tdp()   # convenience one-liner
"""

from __future__ import annotations

import glob
import os
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwarePlatform:
    """Descriptor for a detected or assumed hardware platform.

    Attributes
    ----------
    name:
        Human-readable platform name (e.g. ``"Raspberry Pi 4"``).
    arch:
        CPU architecture string (e.g. ``"aarch64"``, ``"x86_64"``).
    tdp_watts:
        Recommended TDP in Watts for CPU energy estimation.
    description:
        Short description of the primary CPU/SoC.
    detected:
        ``True`` when the platform was positively identified via filesystem
        probes; ``False`` when a fallback/default was used.
    """

    name: str
    arch: str
    tdp_watts: float
    description: str
    detected: bool = False

    def __str__(self) -> str:
        tag = "detected" if self.detected else "assumed"
        return (
            f"HardwarePlatform({self.name!r}, arch={self.arch!r}, "
            f"TDP={self.tdp_watts} W, {tag})"
        )


# ---------------------------------------------------------------------------
# Known platform presets
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict] = {
    "raspberry_pi": dict(
        name="Raspberry Pi 4",
        arch="aarch64",
        tdp_watts=6.0,
        description="ARM Cortex-A72 @ 1.8 GHz (Broadcom BCM2711)",
    ),
    "jetson_nano": dict(
        name="NVIDIA Jetson Nano",
        arch="aarch64",
        tdp_watts=10.0,
        description="ARM Cortex-A57 + 128-core Maxwell GPU (NVIDIA T210)",
    ),
    "jetson_orin": dict(
        name="NVIDIA Jetson Orin",
        arch="aarch64",
        tdp_watts=15.0,
        description="ARM Cortex-A78AE + 1792-core Ampere GPU (NVIDIA Orin)",
    ),
    "apple_silicon": dict(
        name="Apple Silicon (M-series)",
        arch="arm64",
        tdp_watts=20.0,
        description="Apple M-series unified memory architecture",
    ),
    "laptop": dict(
        name="Laptop CPU",
        arch="x86_64",
        tdp_watts=15.0,
        description="Typical mobile CPU (e.g. Intel Core i5/i7 U/H series)",
    ),
    "desktop": dict(
        name="Desktop CPU",
        arch="x86_64",
        tdp_watts=65.0,
        description="Typical desktop CPU (e.g. Intel Core i7/i9 or AMD Ryzen)",
    ),
    "edge_arm": dict(
        name="Generic ARM Edge Device",
        arch="aarch64",
        tdp_watts=8.0,
        description="Unidentified ARM SoC — conservative edge TDP assumed",
    ),
}


def _make(key: str, detected: bool = True) -> HardwarePlatform:
    return HardwarePlatform(**_PRESETS[key], detected=detected)


# ---------------------------------------------------------------------------
# Internal probe helpers (read-only filesystem, no subprocesses)
# ---------------------------------------------------------------------------

def _read_file(path: str) -> str:
    """Return stripped file contents, or empty string on any error."""
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read().strip().lower()
    except OSError:
        return ""


def _probe_device_tree_model() -> str:
    """Read /proc/device-tree/model (Linux ARM boards)."""
    return _read_file("/proc/device-tree/model")


def _probe_device_tree_compatible() -> str:
    """Read /proc/device-tree/compatible (Linux ARM SoC identifiers)."""
    return _read_file("/proc/device-tree/compatible")


def _has_battery() -> bool:
    """Return True if any battery power supply is visible in sysfs."""
    return bool(glob.glob("/sys/class/power_supply/BAT*"))


def _dmi_chassis_type() -> str:
    """Return the DMI chassis type string, or empty string."""
    return _read_file("/sys/class/dmi/id/chassis_type")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_platform() -> HardwarePlatform:
    """Detect the current hardware platform using read-only filesystem probes.

    Returns
    -------
    HardwarePlatform
        A fully-populated descriptor.  The ``detected`` attribute is
        ``True`` when the platform was positively identified, ``False``
        when a default was assumed.

    Notes
    -----
    The function never raises; any probe that fails is silently skipped and
    the next heuristic is tried.
    """
    arch = platform.machine().lower()

    # ------------------------------------------------------------------
    # 1. Raspberry Pi — /proc/device-tree/model is authoritative
    # ------------------------------------------------------------------
    model = _probe_device_tree_model()
    if model:
        if "raspberry pi" in model:
            return _make("raspberry_pi")

        # ------------------------------------------------------------------
        # 2. NVIDIA Jetson — device-tree model string
        # ------------------------------------------------------------------
        if "jetson" in model or "nvidia" in model:
            if "orin" in model:
                return _make("jetson_orin")
            return _make("jetson_nano")

    # ------------------------------------------------------------------
    # 3. Compatible string (catches Jetson when model is absent)
    # ------------------------------------------------------------------
    compat = _probe_device_tree_compatible()
    if compat:
        if "nvidia" in compat or "jetson" in compat:
            return _make("jetson_nano")

    # ------------------------------------------------------------------
    # 4. Apple Silicon (macOS / Asahi Linux)
    # ------------------------------------------------------------------
    if "arm64" in arch or (platform.system() == "Darwin" and "arm" in arch):
        return _make("apple_silicon")

    # ------------------------------------------------------------------
    # 5. Generic ARM without device-tree → conservative edge estimate
    # ------------------------------------------------------------------
    if "arm" in arch or ("aarch64" in arch and not model):
        return _make("edge_arm")

    # ------------------------------------------------------------------
    # 6. x86/x64: distinguish laptop from desktop via battery or DMI
    # ------------------------------------------------------------------
    if _has_battery():
        return _make("laptop")

    chassis = _dmi_chassis_type()
    # SMBIOS chassis types: 8=Portable, 9=Laptop, 10=Notebook, 14=Sub-Notebook
    if chassis in {"8", "9", "10", "14"}:
        return _make("laptop")

    # DMI chassis types that indicate a desktop/server
    if chassis in {"3", "4", "5", "6", "7", "11", "17", "23", "24", "25"}:
        return _make("desktop", detected=True)

    # ------------------------------------------------------------------
    # 7. Fallback — laptop TDP is a conservative safe default
    # ------------------------------------------------------------------
    return _make("laptop", detected=False)


def get_recommended_tdp(platform_override: str | None = None) -> float:
    """Return the recommended TDP (Watts) for the current or named platform.

    Args:
        platform_override: If provided, must be one of the preset keys:
            ``"raspberry_pi"``, ``"jetson_nano"``, ``"jetson_orin"``,
            ``"apple_silicon"``, ``"laptop"``, ``"desktop"``,
            ``"edge_arm"``.
            If ``None`` (default), the platform is auto-detected.

    Returns:
        TDP in Watts as a ``float``.

    Raises:
        ValueError: If *platform_override* is not a recognised preset key.

    Example::

        tdp = get_recommended_tdp()                    # auto-detect
        tdp = get_recommended_tdp("raspberry_pi")      # explicit override
    """
    if platform_override is not None:
        if platform_override not in _PRESETS:
            valid = ", ".join(sorted(_PRESETS))
            raise ValueError(
                f"Unknown platform preset {platform_override!r}. "
                f"Valid keys: {valid}"
            )
        return _PRESETS[platform_override]["tdp_watts"]

    return detect_platform().tdp_watts


def list_known_platforms() -> list[HardwarePlatform]:
    """Return all built-in platform presets as a list."""
    return [HardwarePlatform(**v, detected=True) for v in _PRESETS.values()]
