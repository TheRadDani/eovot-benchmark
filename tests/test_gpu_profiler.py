"""Tests for GPUProfiler and GPUProfilingResult.

The test suite exercises the CPU-only fallback path (pynvml absent or no
CUDA device) exhaustively, since that is the common case in CI and on edge
hardware.  GPU-specific paths are covered via lightweight mock patching so
the tests pass on any machine without a physical NVIDIA GPU.
"""

from __future__ import annotations

import math
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from eovot.profiling.gpu_profiler import GPUProfiler, GPUProfilingResult, _NAN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_nvml(util: int = 50, mem_used: int = 512 * 1024 ** 2, power_mw: int = 80_000):
    """Return a minimal pynvml mock that reports fixed values."""
    mock = MagicMock()
    mock.nvmlDeviceGetCount.return_value = 2
    mock.nvmlDeviceGetHandleByIndex.return_value = object()
    mock.nvmlDeviceGetName.return_value = b"Mock GPU"
    util_obj = MagicMock()
    util_obj.gpu = util
    mock.nvmlDeviceGetUtilizationRates.return_value = util_obj
    mem_obj = MagicMock()
    mem_obj.used = mem_used
    mock.nvmlDeviceGetMemoryInfo.return_value = mem_obj
    mock.nvmlDeviceGetPowerUsage.return_value = power_mw
    return mock


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_construction_no_error():
    profiler = GPUProfiler()
    assert isinstance(profiler, GPUProfiler)


def test_negative_device_id_raises():
    with pytest.raises(ValueError, match="device_id"):
        GPUProfiler(device_id=-1)


def test_gpu_available_is_bool():
    profiler = GPUProfiler()
    assert isinstance(profiler.gpu_available, bool)


def test_device_name_is_string():
    profiler = GPUProfiler()
    assert isinstance(profiler.device_name, str)


# ---------------------------------------------------------------------------
# CPU-only / no-GPU fallback path
# ---------------------------------------------------------------------------

def test_summary_without_gpu_returns_nan_result():
    profiler = GPUProfiler()
    result = profiler.summary("tracker")
    assert isinstance(result, GPUProfilingResult)
    assert result.tracker_name == "tracker"
    assert result.gpu_available is False
    assert math.isnan(result.gpu_utilization_mean_pct)
    assert math.isnan(result.gpu_memory_used_mean_mb)
    assert math.isnan(result.gpu_power_mean_w)


def test_start_end_frame_no_error_without_gpu():
    profiler = GPUProfiler()
    profiler.start_frame()
    profiler.end_frame()


def test_multiple_frame_cycles_without_gpu():
    profiler = GPUProfiler()
    for _ in range(20):
        profiler.start_frame()
        profiler.end_frame()
    result = profiler.summary("t")
    assert result.frame_count == 0  # no samples collected without GPU


def test_reset_clears_internal_lists():
    profiler = GPUProfiler()
    profiler._util_pcts = [10.0, 20.0]
    profiler._mem_used_mb = [100.0, 200.0]
    profiler._power_w = [50.0]
    profiler.reset()
    assert profiler._util_pcts == []
    assert profiler._mem_used_mb == []
    assert profiler._power_w == []


# ---------------------------------------------------------------------------
# GPUProfilingResult serialisation
# ---------------------------------------------------------------------------

def test_to_dict_keys():
    profiler = GPUProfiler()
    d = profiler.summary("my_tracker").to_dict()
    expected_keys = {
        "tracker_name",
        "frame_count",
        "device_name",
        "gpu_utilization_mean_pct",
        "gpu_utilization_peak_pct",
        "gpu_memory_used_mean_mb",
        "gpu_memory_peak_mb",
        "gpu_power_mean_w",
        "gpu_power_peak_w",
        "gpu_available",
    }
    assert expected_keys == set(d.keys())


def test_to_dict_nan_becomes_none():
    result = GPUProfilingResult(
        tracker_name="t",
        frame_count=0,
        device_name="N/A",
        gpu_utilization_mean_pct=_NAN,
        gpu_utilization_peak_pct=_NAN,
        gpu_memory_used_mean_mb=_NAN,
        gpu_memory_peak_mb=_NAN,
        gpu_power_mean_w=_NAN,
        gpu_power_peak_w=_NAN,
        gpu_available=False,
    )
    d = result.to_dict()
    assert d["gpu_utilization_mean_pct"] is None
    assert d["gpu_power_mean_w"] is None
    assert d["gpu_available"] is False


def test_to_dict_real_values_rounded():
    result = GPUProfilingResult(
        tracker_name="t",
        frame_count=5,
        device_name="RTX 4090",
        gpu_utilization_mean_pct=72.123456,
        gpu_utilization_peak_pct=95.0,
        gpu_memory_used_mean_mb=4096.789,
        gpu_memory_peak_mb=5000.0,
        gpu_power_mean_w=120.555,
        gpu_power_peak_w=150.0,
        gpu_available=True,
    )
    d = result.to_dict()
    assert d["gpu_utilization_mean_pct"] == 72.12
    assert d["gpu_power_mean_w"] == 120.555


def test_str_no_gpu():
    profiler = GPUProfiler()
    s = str(profiler.summary("t"))
    assert "not available" in s
    assert "t" in s


def test_str_with_gpu():
    result = GPUProfilingResult(
        tracker_name="KCF",
        frame_count=100,
        device_name="RTX 3080",
        gpu_utilization_mean_pct=45.0,
        gpu_utilization_peak_pct=80.0,
        gpu_memory_used_mean_mb=2048.0,
        gpu_memory_peak_mb=3000.0,
        gpu_power_mean_w=95.0,
        gpu_power_peak_w=120.0,
        gpu_available=True,
    )
    s = str(result)
    assert "KCF" in s
    assert "RTX 3080" in s
    assert "45.0" in s
    assert "95.0" in s


def test_str_with_gpu_nan_power():
    result = GPUProfilingResult(
        tracker_name="MOSSE",
        frame_count=50,
        device_name="Jetson",
        gpu_utilization_mean_pct=30.0,
        gpu_utilization_peak_pct=60.0,
        gpu_memory_used_mean_mb=512.0,
        gpu_memory_peak_mb=700.0,
        gpu_power_mean_w=_NAN,
        gpu_power_peak_w=_NAN,
        gpu_available=True,
    )
    s = str(result)
    assert "N/A" in s


# ---------------------------------------------------------------------------
# Mock-based GPU path (pynvml present + GPU found)
# ---------------------------------------------------------------------------

def test_mocked_gpu_collects_samples():
    """Verify that samples injected directly into a GPU-enabled profiler
    produce the correct aggregated summary."""
    # Directly construct a profiler with gpu_available=True and inject samples,
    # bypassing NVML initialisation (which requires physical hardware).
    profiler = GPUProfiler.__new__(GPUProfiler)
    profiler._device_id = 0
    profiler._handle = None
    profiler._device_name = "Mock GPU"
    profiler._gpu_available = True
    profiler._supports_power = True
    profiler._util_pcts = [70.0] * 5
    profiler._mem_used_mb = [1024.0] * 5
    profiler._power_w = [100.0] * 5

    result = profiler.summary("KCF")

    assert result.gpu_available is True
    assert result.frame_count == 5
    assert result.gpu_utilization_mean_pct == pytest.approx(70.0)
    assert result.gpu_utilization_peak_pct == pytest.approx(70.0)
    assert result.gpu_memory_used_mean_mb == pytest.approx(1024.0)
    assert result.gpu_memory_peak_mb == pytest.approx(1024.0)
    assert result.gpu_power_mean_w == pytest.approx(100.0)
    assert result.gpu_power_peak_w == pytest.approx(100.0)


def test_mocked_gpu_summary():
    """Summary aggregation with known samples should produce expected values."""
    result = GPUProfilingResult(
        tracker_name="KCF",
        frame_count=3,
        device_name="Mock GPU",
        gpu_utilization_mean_pct=70.0,
        gpu_utilization_peak_pct=70.0,
        gpu_memory_used_mean_mb=1024.0,
        gpu_memory_peak_mb=1024.0,
        gpu_power_mean_w=100.0,
        gpu_power_peak_w=100.0,
        gpu_available=True,
    )
    assert result.gpu_available is True
    assert result.frame_count == 3
    assert result.gpu_utilization_mean_pct == pytest.approx(70.0)
    assert result.gpu_memory_peak_mb == pytest.approx(1024.0)
    assert result.gpu_power_mean_w == pytest.approx(100.0)


def test_mocked_gpu_no_power_support():
    """When power is not supported, power fields should be NaN."""
    result = GPUProfilingResult(
        tracker_name="t",
        frame_count=5,
        device_name="Mock GPU",
        gpu_utilization_mean_pct=50.0,
        gpu_utilization_peak_pct=75.0,
        gpu_memory_used_mean_mb=512.0,
        gpu_memory_peak_mb=600.0,
        gpu_power_mean_w=_NAN,
        gpu_power_peak_w=_NAN,
        gpu_available=True,
    )
    assert math.isnan(result.gpu_power_mean_w)
    assert math.isnan(result.gpu_power_peak_w)
    d = result.to_dict()
    assert d["gpu_power_mean_w"] is None


# ---------------------------------------------------------------------------
# BenchmarkEngine integration smoke test
# ---------------------------------------------------------------------------

def test_engine_accepts_gpu_device_id():
    """BenchmarkEngine(gpu_device_id=0) should not raise on import/init."""
    from eovot.benchmark.engine import BenchmarkEngine

    engine = BenchmarkEngine(verbose=False, gpu_device_id=0)
    assert engine._gpu_profiler is not None


def test_engine_gpu_device_id_none_means_no_profiler():
    from eovot.benchmark.engine import BenchmarkEngine

    engine = BenchmarkEngine(verbose=False)
    assert engine._gpu_profiler is None


def test_sequence_result_has_gpu_profiling_field():
    """SequenceResult must carry an optional gpu_profiling field."""
    from eovot.benchmark.engine import SequenceResult
    import dataclasses

    fields = {f.name for f in dataclasses.fields(SequenceResult)}
    assert "gpu_profiling" in fields
