"""Profiling sub-package — hardware-aware latency, memory, energy, and deployment evaluation."""

from .profiler import Profiler, ProfilingResult
from .energy import EnergyProfiler, EnergyResult
from .hardware_profiles import (
    HardwareProfile,
    BUILTIN_PROFILES,
    RASPBERRY_PI_4,
    JETSON_NANO,
    INTEL_NUC,
    DESKTOP_CPU,
    SMARTPHONE,
    get_profile,
    load_profile_from_yaml,
)
from .deployment_report import evaluate_deployment, compare_trackers_on_hardware

__all__ = [
    "Profiler",
    "ProfilingResult",
    "EnergyProfiler",
    "EnergyResult",
    "HardwareProfile",
    "BUILTIN_PROFILES",
    "RASPBERRY_PI_4",
    "JETSON_NANO",
    "INTEL_NUC",
    "DESKTOP_CPU",
    "SMARTPHONE",
    "get_profile",
    "load_profile_from_yaml",
    "evaluate_deployment",
    "compare_trackers_on_hardware",
]
