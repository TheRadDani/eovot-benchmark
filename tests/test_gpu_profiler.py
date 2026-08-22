"""Tests for eovot.profiling.gpu.GPUProfiler.

The tests do not require an NVIDIA GPU.  When nvidia-smi is absent the
profiler runs in fallback mode (gpu_available=False), which is the state
exercised here.  Tests verify:

- Profiler initialises without errors on any machine.
- start_frame / end_frame accumulate the expected number of frames.
- summary() returns a GPUProfilingResult with correct scalar types.
- GPU-unavailable path produces zero-valued GPU fields.
- reset() clears accumulated state cleanly.
- context-manager protocol calls close() on exit.
- BenchmarkEngine integrates GPUProfiler when gpu_profiling=True.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from eovot.profiling.gpu import GPUProfiler, GPUProfilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_frames(profiler: GPUProfiler, n: int = 5, sleep_s: float = 0.001) -> None:
    """Simulate n tracker-update frames through the profiler."""
    for _ in range(n):
        profiler.start_frame()
        time.sleep(sleep_s)
        profiler.end_frame()


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


def test_init_no_error():
    """GPUProfiler constructs without raising even without a GPU."""
    p = GPUProfiler(device_index=0, poll_interval_ms=5.0)
    p.close()


def test_context_manager():
    """Context-manager closes the profiler on exit."""
    with GPUProfiler() as p:
        _run_frames(p, n=3)
    # After __exit__ the poller should be stopped; close() is idempotent.
    p.close()  # must not raise


def test_frame_count():
    with GPUProfiler() as p:
        _run_frames(p, n=7)
        result = p.summary("test_tracker")
    assert result.frame_count == 7


def test_summary_returns_correct_type():
    with GPUProfiler() as p:
        _run_frames(p, n=4)
        result = p.summary("tracker_x")
    assert isinstance(result, GPUProfilingResult)
    assert result.tracker_name == "tracker_x"


def test_summary_scalars_non_negative():
    with GPUProfiler() as p:
        _run_frames(p, n=5)
        result = p.summary("t")
    assert result.peak_gpu_memory_mb >= 0.0
    assert result.mean_gpu_utilization_pct >= 0.0
    assert result.peak_gpu_utilization_pct >= 0.0
    assert result.mean_gpu_power_w >= 0.0
    assert result.peak_gpu_power_w >= 0.0
    assert result.total_gpu_energy_j >= 0.0
    assert result.mean_gpu_energy_per_frame_mj >= 0.0


def test_unavailable_gpu_fields_are_zero():
    """When no GPU is detected all power/energy fields must be zero."""
    p = GPUProfiler()
    if p._available:
        pytest.skip("NVIDIA GPU present — skipping unavailable-fallback test.")
    _run_frames(p, n=4)
    result = p.summary("t")
    assert result.gpu_available is False
    assert result.peak_gpu_memory_mb == 0.0
    assert result.mean_gpu_power_w == 0.0
    assert result.total_gpu_energy_j == 0.0
    p.close()


def test_to_dict_keys():
    with GPUProfiler() as p:
        _run_frames(p, n=3)
        d = p.summary("t").to_dict()
    required_keys = {
        "tracker_name", "frame_count", "device_index", "gpu_available",
        "gpu_name", "peak_gpu_memory_mb", "mean_gpu_utilization_pct",
        "peak_gpu_utilization_pct", "mean_gpu_power_w", "peak_gpu_power_w",
        "total_gpu_energy_j", "mean_gpu_energy_per_frame_mj",
    }
    assert required_keys.issubset(d.keys())


def test_str_representation():
    with GPUProfiler() as p:
        _run_frames(p, n=2)
        s = str(p.summary("t"))
    assert "GPUProfilingResult" in s


def test_reset_clears_frames():
    with GPUProfiler() as p:
        _run_frames(p, n=5)
        p.reset()
        _run_frames(p, n=3)
        result = p.summary("t")
    assert result.frame_count == 3


def test_summary_before_any_frame_raises():
    with GPUProfiler() as p:
        with pytest.raises(ValueError, match="No frames profiled"):
            p.summary("t")


def test_end_frame_without_start_raises():
    with GPUProfiler() as p:
        with pytest.raises(RuntimeError, match="end_frame"):
            p.end_frame()


# ---------------------------------------------------------------------------
# BenchmarkEngine integration
# ---------------------------------------------------------------------------


def test_benchmark_engine_gpu_profiling_flag():
    """BenchmarkEngine accepts gpu_profiling=True and populates gpu_profiling field."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker

    dataset = SyntheticDataset(num_sequences=1, num_frames=10, frame_size=(320, 240))
    engine = BenchmarkEngine(verbose=False, gpu_profiling=True)
    tracker = MOSSETracker()
    result = engine.run(tracker, dataset, dataset_name="synthetic")

    assert len(result.sequence_results) == 1
    seq_res = result.sequence_results[0]
    # gpu_profiling field is populated (GPUProfilingResult), not None.
    assert seq_res.gpu_profiling is not None
    assert isinstance(seq_res.gpu_profiling, GPUProfilingResult)
    # Aggregate properties exist on BenchmarkResult.
    assert result.peak_gpu_memory_mb is not None
    assert result.mean_gpu_power_w is not None
    assert result.total_gpu_energy_j is not None

    # Cleanup GPU profiler thread.
    if engine._gpu_profiler is not None:
        engine._gpu_profiler.close()
