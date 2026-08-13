"""Tests for eovot.profiling.gpu — GpuProfiler and GpuProfilingResult.

The test suite covers both the no-GPU path (the common CI/CD scenario) and,
when pynvml is available, a mocked-NVML path that validates sampling logic
without requiring real GPU hardware.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eovot.profiling.gpu import GpuProfiler, GpuProfilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_pynvml(
    device_count: int = 1,
    device_name: str = "Test GPU",
    used_memory_mb: float = 256.0,
    power_mw: float = 25_000,
    gpu_util_pct: int = 45,
):
    """Build a minimal pynvml mock with configurable return values."""
    mock = MagicMock()
    mock.nvmlInit = MagicMock()
    mock.nvmlDeviceGetCount = MagicMock(return_value=device_count)

    handle = MagicMock()
    mock.nvmlDeviceGetHandleByIndex = MagicMock(return_value=handle)
    mock.nvmlDeviceGetName = MagicMock(return_value=device_name)

    mem_info = MagicMock()
    mem_info.used = int(used_memory_mb * (1024 ** 2))
    mock.nvmlDeviceGetMemoryInfo = MagicMock(return_value=mem_info)
    mock.nvmlDeviceGetPowerUsage = MagicMock(return_value=int(power_mw))

    util_rates = MagicMock()
    util_rates.gpu = gpu_util_pct
    mock.nvmlDeviceGetUtilizationRates = MagicMock(return_value=util_rates)
    return mock


# ---------------------------------------------------------------------------
# No-GPU / no-pynvml path
# ---------------------------------------------------------------------------

class TestGpuProfilerNoGpu:
    """GpuProfiler behaviour when pynvml is absent or no GPU is present."""

    def test_no_pynvml_available_marks_unavailable(self):
        """ImportError from pynvml must produce available=False."""
        with patch.dict(sys.modules, {"pynvml": None}):
            profiler = GpuProfiler(device_index=0)
        assert not profiler.available

    def test_summary_returns_valid_dataclass_when_unavailable(self):
        with patch.dict(sys.modules, {"pynvml": None}):
            profiler = GpuProfiler(device_index=0)
        result = profiler.summary("TestTracker")
        assert isinstance(result, GpuProfilingResult)
        assert result.available is False
        assert result.tracker_name == "TestTracker"
        assert result.peak_gpu_memory_mb == 0.0
        assert result.mean_gpu_memory_mb == 0.0
        assert result.mean_gpu_power_w == 0.0
        assert result.peak_gpu_power_w == 0.0
        assert result.energy_per_frame_mj == 0.0

    def test_start_end_frame_noop_when_unavailable(self):
        """start_frame / end_frame must not raise when unavailable."""
        with patch.dict(sys.modules, {"pynvml": None}):
            profiler = GpuProfiler(device_index=0)
        profiler.start_frame()
        profiler.end_frame()  # should not raise

    def test_reset_noop_when_unavailable(self):
        with patch.dict(sys.modules, {"pynvml": None}):
            profiler = GpuProfiler(device_index=0)
        profiler.reset()  # should not raise

    def test_device_name_na_when_unavailable(self):
        with patch.dict(sys.modules, {"pynvml": None}):
            profiler = GpuProfiler(device_index=0)
        assert profiler.device_name == "N/A"

    def test_str_repr_when_unavailable(self):
        with patch.dict(sys.modules, {"pynvml": None}):
            profiler = GpuProfiler(device_index=0)
        result = profiler.summary("Tracker")
        text = str(result)
        assert "unavailable" in text.lower()

    def test_zero_devices_unavailable(self):
        """nvmlDeviceGetCount() == 0 must produce available=False."""
        mock_nvml = _make_mock_pynvml(device_count=0)
        with patch.dict(sys.modules, {"pynvml": mock_nvml}):
            profiler = GpuProfiler(device_index=0)
        assert not profiler.available

    def test_out_of_range_device_index_unavailable(self):
        """device_index >= device_count must produce available=False."""
        mock_nvml = _make_mock_pynvml(device_count=1)
        with patch.dict(sys.modules, {"pynvml": mock_nvml}):
            profiler = GpuProfiler(device_index=5)
        assert not profiler.available


# ---------------------------------------------------------------------------
# Mocked-NVML path
# ---------------------------------------------------------------------------

class TestGpuProfilerMocked:
    """GpuProfiler with a mocked pynvml — validates sampling and aggregation."""

    def _make_profiler(self, **kwargs) -> GpuProfiler:
        mock_nvml = _make_mock_pynvml(**kwargs)
        with patch.dict(sys.modules, {"pynvml": mock_nvml}):
            profiler = GpuProfiler(device_index=0)
        # Keep the mock accessible for later sample() calls
        profiler._pynvml_mock = mock_nvml
        return profiler

    def test_available_true_with_valid_mock(self):
        mock_nvml = _make_mock_pynvml(device_count=1)
        with patch.dict(sys.modules, {"pynvml": mock_nvml}):
            profiler = GpuProfiler(device_index=0)
        assert profiler.available

    def test_device_name_captured(self):
        mock_nvml = _make_mock_pynvml(device_name="NVIDIA RTX Test")
        with patch.dict(sys.modules, {"pynvml": mock_nvml}):
            profiler = GpuProfiler(device_index=0)
        assert "RTX Test" in profiler.device_name

    def test_memory_samples_aggregated_correctly(self):
        """Inject samples directly to validate aggregation math."""
        mock_nvml = _make_mock_pynvml(device_count=1)
        with patch.dict(sys.modules, {"pynvml": mock_nvml}):
            profiler = GpuProfiler(device_index=0)

        # Bypass _sample and inject directly for deterministic testing
        profiler._memory_samples = [100.0, 120.0, 110.0, 130.0]
        profiler._power_samples = [20.0, 22.0, 21.0, 23.0]
        profiler._util_samples = [40.0, 50.0, 45.0, 55.0]

        result = profiler.summary("TestTracker")
        assert result.available is True
        assert result.peak_gpu_memory_mb == pytest.approx(130.0)
        assert result.mean_gpu_memory_mb == pytest.approx(115.0)
        assert result.peak_gpu_power_w == pytest.approx(23.0)
        assert result.mean_gpu_power_w == pytest.approx(21.5)
        assert result.mean_gpu_util_pct == pytest.approx(47.5)

    def test_energy_per_frame_populated_with_latency(self):
        mock_nvml = _make_mock_pynvml(device_count=1)
        with patch.dict(sys.modules, {"pynvml": mock_nvml}):
            profiler = GpuProfiler(device_index=0)

        profiler._memory_samples = [256.0, 256.0]
        profiler._power_samples = [30.0, 30.0]   # 30 W

        result = profiler.summary("T", mean_latency_ms=10.0)  # 10 ms/frame
        # 30 W × 0.01 s × 1000 = 300 mJ
        assert result.energy_per_frame_mj == pytest.approx(300.0, rel=1e-3)

    def test_energy_zero_when_no_latency_given(self):
        mock_nvml = _make_mock_pynvml(device_count=1)
        with patch.dict(sys.modules, {"pynvml": mock_nvml}):
            profiler = GpuProfiler(device_index=0)

        profiler._memory_samples = [256.0]
        profiler._power_samples = [30.0]

        result = profiler.summary("T")  # no mean_latency_ms
        assert result.energy_per_frame_mj == 0.0

    def test_reset_clears_all_samples(self):
        mock_nvml = _make_mock_pynvml(device_count=1)
        with patch.dict(sys.modules, {"pynvml": mock_nvml}):
            profiler = GpuProfiler(device_index=0)

        profiler._memory_samples = [256.0, 300.0]
        profiler._power_samples = [20.0]
        profiler._util_samples = [50.0]
        profiler.reset()

        assert profiler._memory_samples == []
        assert profiler._power_samples == []
        assert profiler._util_samples == []

    def test_summary_without_power_samples(self):
        """Profilers that can't query power should still return valid result."""
        mock_nvml = _make_mock_pynvml(device_count=1)
        with patch.dict(sys.modules, {"pynvml": mock_nvml}):
            profiler = GpuProfiler(device_index=0)

        profiler._memory_samples = [100.0, 110.0]
        # _power_samples left empty
        result = profiler.summary("T")
        assert result.mean_gpu_power_w == 0.0
        assert result.peak_gpu_power_w == 0.0

    def test_frame_count_is_half_sample_count(self):
        """Each frame generates two samples (start + end), so frame_count = samples/2."""
        mock_nvml = _make_mock_pynvml(device_count=1)
        with patch.dict(sys.modules, {"pynvml": mock_nvml}):
            profiler = GpuProfiler(device_index=0)

        profiler._memory_samples = [100.0] * 10  # 10 samples → 5 frames
        result = profiler.summary("T")
        assert result.frame_count == 5


# ---------------------------------------------------------------------------
# GpuProfilingResult serialisation
# ---------------------------------------------------------------------------

class TestGpuProfilingResultToDict:
    def _make_result(self, available: bool = True) -> GpuProfilingResult:
        return GpuProfilingResult(
            tracker_name="MOSSE",
            device_name="Test GPU",
            device_index=0,
            frame_count=10,
            available=available,
            peak_gpu_memory_mb=512.5,
            mean_gpu_memory_mb=480.0,
            mean_gpu_power_w=25.3,
            peak_gpu_power_w=28.1,
            mean_gpu_util_pct=67.2,
            energy_per_frame_mj=0.25,
        )

    def test_to_dict_returns_all_fields(self):
        result = self._make_result()
        d = result.to_dict()
        expected_keys = {
            "tracker_name", "device_name", "device_index", "frame_count",
            "available", "peak_gpu_memory_mb", "mean_gpu_memory_mb",
            "mean_gpu_power_w", "peak_gpu_power_w", "mean_gpu_util_pct",
            "energy_per_frame_mj",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_are_rounded(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["peak_gpu_memory_mb"] == round(512.5, 2)
        assert d["mean_gpu_power_w"] == round(25.3, 3)
        assert d["energy_per_frame_mj"] == round(0.25, 4)

    def test_str_when_available(self):
        result = self._make_result(available=True)
        text = str(result)
        assert "MOSSE" in text
        assert "Test GPU" in text
        assert "MiB" in text

    def test_str_when_unavailable(self):
        result = self._make_result(available=False)
        text = str(result)
        assert "unavailable" in text.lower()


# ---------------------------------------------------------------------------
# BenchmarkEngine integration (CPU-only, no GPU)
# ---------------------------------------------------------------------------

class TestBenchmarkEngineGpuIntegration:
    """Ensure BenchmarkEngine properly handles gpu_device_index=None and with GPU."""

    def test_engine_without_gpu_param(self):
        from eovot.benchmark.engine import BenchmarkEngine
        engine = BenchmarkEngine(verbose=False)
        assert engine._gpu_profiler is None

    def test_engine_with_gpu_param_unavailable(self):
        """gpu_device_index=0 on a machine without GPU → profiler is created but unavailable."""
        with patch.dict(sys.modules, {"pynvml": None}):
            from eovot.benchmark.engine import BenchmarkEngine
            engine = BenchmarkEngine(verbose=False, gpu_device_index=0)
        # The profiler should exist but report unavailable
        assert engine._gpu_profiler is not None
        assert not engine._gpu_profiler.available

    def test_sequence_result_gpu_none_without_profiler(self):
        """SequenceResult.gpu_profiling must be None when no GPU profiler is set."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        engine = BenchmarkEngine(verbose=False)
        dataset = SyntheticDataset(num_sequences=1, num_frames=10)
        tracker = MOSSETracker()
        result = engine.run(tracker, dataset, dataset_name="Synth")

        for sr in result.sequence_results:
            assert sr.gpu_profiling is None

    def test_summary_no_gpu_keys_without_profiler(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        engine = BenchmarkEngine(verbose=False)
        dataset = SyntheticDataset(num_sequences=1, num_frames=10)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Synth")
        summary = result.summary()

        assert "mean_gpu_memory_mb" not in summary
        assert "mean_gpu_power_w" not in summary

    def test_sequence_result_gpu_unavailable_when_no_pynvml(self):
        """With gpu_device_index set but pynvml absent, result should have unavailable GPU."""
        with patch.dict(sys.modules, {"pynvml": None}):
            from eovot.benchmark.engine import BenchmarkEngine
            from eovot.datasets.synthetic import SyntheticDataset
            from eovot.trackers.mosse import MOSSETracker

            engine = BenchmarkEngine(verbose=False, gpu_device_index=0)
            dataset = SyntheticDataset(num_sequences=1, num_frames=10)
            result = engine.run(MOSSETracker(), dataset, dataset_name="Synth")

        for sr in result.sequence_results:
            assert sr.gpu_profiling is not None
            assert sr.gpu_profiling.available is False
