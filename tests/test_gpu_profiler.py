"""Tests for eovot.profiling.gpu_profiler.

All tests run on CPU-only environments — GPU-specific assertions are skipped
when CUDA is unavailable.  This ensures CI passes on machines without a GPU
while still exercising the full code path on GPU-equipped machines.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from eovot.profiling.gpu_profiler import (
    GpuProfiler,
    GpuProfilingResult,
    cuda_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_n_frames(profiler: GpuProfiler, n: int, sleep_ms: float = 1.0) -> None:
    """Simulate n tracker update calls."""
    for _ in range(n):
        profiler.start_frame()
        time.sleep(sleep_ms / 1_000.0)
        profiler.end_frame()


# ---------------------------------------------------------------------------
# Basic API contract
# ---------------------------------------------------------------------------

class TestGpuProfilerApi:
    def test_returns_result_after_frames(self):
        p = GpuProfiler()
        _run_n_frames(p, n=5)
        result = p.summary("test_tracker")
        assert isinstance(result, GpuProfilingResult)

    def test_frame_count_matches(self):
        p = GpuProfiler()
        _run_n_frames(p, n=10)
        result = p.summary()
        assert result.frame_count == 10

    def test_fps_positive(self):
        p = GpuProfiler()
        _run_n_frames(p, n=3)
        result = p.summary()
        assert result.fps > 0.0

    def test_latency_mean_positive(self):
        p = GpuProfiler()
        _run_n_frames(p, n=5, sleep_ms=2.0)
        result = p.summary()
        assert result.latency_mean_ms > 0.0

    def test_p95_gte_mean(self):
        p = GpuProfiler()
        _run_n_frames(p, n=20, sleep_ms=1.0)
        result = p.summary()
        assert result.latency_p95_ms >= result.latency_mean_ms - 0.1  # allow tiny float diff

    def test_cpu_memory_positive(self):
        p = GpuProfiler()
        _run_n_frames(p, n=5)
        result = p.summary()
        assert result.peak_cpu_memory_mb > 0.0

    def test_gpu_available_flag_matches_cuda(self):
        p = GpuProfiler()
        _run_n_frames(p, n=2)
        result = p.summary()
        assert result.gpu_available == cuda_available()

    def test_tracker_name_stored(self):
        p = GpuProfiler()
        _run_n_frames(p, n=2)
        result = p.summary("MOSSE")
        assert result.tracker_name == "MOSSE"

    def test_default_tracker_name(self):
        p = GpuProfiler()
        _run_n_frames(p, n=2)
        result = p.summary()
        assert result.tracker_name == "unknown"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestGpuProfilerErrors:
    def test_end_frame_without_start_raises(self):
        p = GpuProfiler()
        with pytest.raises(RuntimeError, match="end_frame"):
            p.end_frame()

    def test_summary_without_frames_raises(self):
        p = GpuProfiler()
        with pytest.raises(ValueError, match="No frames profiled"):
            p.summary()


# ---------------------------------------------------------------------------
# Reset behaviour
# ---------------------------------------------------------------------------

class TestGpuProfilerReset:
    def test_reset_clears_frame_count(self):
        p = GpuProfiler()
        _run_n_frames(p, n=5)
        p.reset()
        with pytest.raises(ValueError):
            p.summary()

    def test_reset_allows_reuse(self):
        p = GpuProfiler()
        _run_n_frames(p, n=3)
        p.reset()
        _run_n_frames(p, n=7)
        result = p.summary()
        assert result.frame_count == 7


# ---------------------------------------------------------------------------
# CPU-only fields (always present)
# ---------------------------------------------------------------------------

class TestCpuFields:
    def test_fps_consistent_with_latency(self):
        p = GpuProfiler()
        _run_n_frames(p, n=10, sleep_ms=5.0)
        result = p.summary()
        expected_fps = 1_000.0 / result.latency_mean_ms
        assert abs(result.fps - expected_fps) < 1.0  # within 1 FPS

    def test_std_is_nonnegative(self):
        p = GpuProfiler()
        _run_n_frames(p, n=10)
        result = p.summary()
        assert result.latency_std_ms >= 0.0


# ---------------------------------------------------------------------------
# GPU-specific fields (skipped on CPU-only machines)
# ---------------------------------------------------------------------------

class TestGpuFields:
    @pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
    def test_cuda_latency_positive(self):
        p = GpuProfiler()
        _run_n_frames(p, n=5)
        result = p.summary()
        assert result.cuda_latency_mean_ms >= 0.0

    @pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
    def test_device_name_present(self):
        p = GpuProfiler()
        _run_n_frames(p, n=2)
        result = p.summary()
        assert result.gpu_device_name is not None
        assert len(result.gpu_device_name) > 0

    @pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
    def test_gpu_memory_nonnegative(self):
        p = GpuProfiler()
        _run_n_frames(p, n=5)
        result = p.summary()
        assert result.peak_gpu_memory_allocated_mb >= 0.0
        assert result.peak_gpu_memory_reserved_mb >= 0.0

    @pytest.mark.skipif(cuda_available(), reason="Test CPU fallback when CUDA absent")
    def test_gpu_fields_zero_when_no_cuda(self):
        p = GpuProfiler()
        _run_n_frames(p, n=5)
        result = p.summary()
        assert result.cuda_latency_mean_ms == 0.0
        assert result.peak_gpu_memory_allocated_mb == 0.0
        assert result.gpu_device_name is None


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestGpuProfilingResultDict:
    def test_to_dict_has_required_keys(self):
        p = GpuProfiler()
        _run_n_frames(p, n=5)
        result = p.summary("KCF")
        d = result.to_dict()
        required = {
            "tracker_name", "frame_count", "fps",
            "latency_mean_ms", "latency_std_ms", "latency_p95_ms",
            "peak_cpu_memory_mb", "gpu_available",
        }
        assert required.issubset(set(d.keys()))

    def test_to_dict_gpu_keys_present_when_cuda(self):
        p = GpuProfiler()
        _run_n_frames(p, n=3)
        result = p.summary()
        d = result.to_dict()
        if result.gpu_available:
            gpu_keys = {
                "cuda_latency_mean_ms",
                "cuda_latency_std_ms",
                "cuda_latency_p95_ms",
                "peak_gpu_memory_allocated_mb",
                "peak_gpu_memory_reserved_mb",
            }
            assert gpu_keys.issubset(set(d.keys()))

    def test_to_dict_values_are_rounded(self):
        p = GpuProfiler()
        _run_n_frames(p, n=5)
        result = p.summary()
        d = result.to_dict()
        # fps should be a float with at most 2 decimal places
        fps_str = str(d["fps"])
        if "." in fps_str:
            decimals = len(fps_str.split(".")[1])
            assert decimals <= 2

    def test_str_repr_includes_tracker_name(self):
        p = GpuProfiler()
        _run_n_frames(p, n=3)
        result = p.summary("CSRT")
        assert "CSRT" in str(result)
