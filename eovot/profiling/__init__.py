"""Profiling sub-package — hardware-aware latency, memory, energy, and device simulation."""

from .profiler import Profiler, ProfilingResult
from .energy import EnergyProfiler, EnergyResult
from .device_sim import DeviceProfile, DeviceSimResult, DeviceSimulator, KNOWN_DEVICES
from .memory_profile import (
    FrameMemorySnapshot,
    MemoryProfileResult,
    MemoryProfiler,
    MemoryBudgetChecker,
    BudgetCheckReport,
)

__all__ = [
    "Profiler",
    "ProfilingResult",
    "EnergyProfiler",
    "EnergyResult",
    "DeviceProfile",
    "DeviceSimResult",
    "DeviceSimulator",
    "KNOWN_DEVICES",
    "FrameMemorySnapshot",
    "MemoryProfileResult",
    "MemoryProfiler",
    "MemoryBudgetChecker",
    "BudgetCheckReport",
]
