"""Profiling sub-package — hardware-aware latency, memory, and energy measurement."""

from .profiler import Profiler, ProfilingResult
from .energy import EnergyProfiler, EnergyResult
from .device_profiles import (
    DeviceProfile,
    EdgeComplianceReport,
    Criterion,
    ALL_PROFILES,
    PROFILE_REGISTRY,
    RASPBERRY_PI_4B,
    JETSON_NANO,
    JETSON_ORIN_NANO,
    INTEL_NUC,
    LAPTOP_MID,
    DESKTOP_SERVER,
    get_profile,
    assess_edge_compliance,
    compliance_matrix,
)

__all__ = [
    "Profiler",
    "ProfilingResult",
    "EnergyProfiler",
    "EnergyResult",
    "DeviceProfile",
    "EdgeComplianceReport",
    "Criterion",
    "ALL_PROFILES",
    "PROFILE_REGISTRY",
    "RASPBERRY_PI_4B",
    "JETSON_NANO",
    "JETSON_ORIN_NANO",
    "INTEL_NUC",
    "LAPTOP_MID",
    "DESKTOP_SERVER",
    "get_profile",
    "assess_edge_compliance",
    "compliance_matrix",
]
