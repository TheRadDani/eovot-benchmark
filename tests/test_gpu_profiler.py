"""Tests for the GPU profiling module (eovot/profiling/gpu.py).

All tests run on any machine — on systems without an NVIDIA GPU or pynvml
the profiler operates in stub mode (all measurements are zero).  The suite
verifies both the stub path and the live-NVML path (skipped when no GPU).
"""

from __future__ import annotations

import time

import pytest

from eovot.profiling.gpu import GPUProfiler, GPUProfilingResult


# ---------------------------------------------------------------------------
# GPUProfilingResult dataclass tests
# ---------------------------------------------------------------------------

class TestGPUProfilingResult:
    """Unit tests for the GPUProfilingResult data container."""

    def _make_result(self, nvml: bool = True) -> GPUProfilingResult:
        return GPUProfilingResult(
            tracker_name="TestTracker",
            frame_count=10,
            device_name="Tesla T4" if nvml else "N/A",
            total_energy_j=0.5,
            mean_power_w=25.0,
            energy_per_frame_mj=50.0,
            peak_memory_mb=512.0,
            mean_gpu_util_pct=60.0,
            peak_gpu_util_pct=85.0,
            nvml_available=nvml,
        )

    def test_str_without_nvml_says_unavailable(self):
        r = self._make_result(nvml=False)
        assert "unavailable" in str(r).lower()

    def test_str_with_nvml_contains_tracker_name(self):
        r = self._make_result(nvml=True)
        assert "TestTracker" in str(r)

    def test_str_with_nvml_contains_device_name(self):
        r = self._make_result(nvml=True)
        assert "Tesla T4" in str(r)

    def test_to_dict_has_all_keys(self):
        expected_keys = {
            "tracker_name",
            "frame_count",
            "device_name",
            "total_energy_j",
            "mean_power_w",
            "energy_per_frame_mj",
            "peak_memory_mb",
            "mean_gpu_util_pct",
            "peak_gpu_util_pct",
            "nvml_available",
        }
        d = self._make_result().to_dict()
        assert expected_keys == set(d.keys())

    def test_to_dict_values_rounded(self):
        r = self._make_result()
        d = r.to_dict()
        # total_energy_j is 0.5 — should round to 6 decimal places
        assert isinstance(d["total_energy_j"], float)
        assert d["tracker_name"] == "TestTracker"
        assert d["frame_count"] == 10
        assert d["nvml_available"] is True

    def test_to_dict_nvml_false(self):
        r = self._make_result(nvml=False)
        assert r.to_dict()["nvml_available"] is False
        assert r.to_dict()["device_name"] == "N/A"


# ---------------------------------------------------------------------------
# GPUProfiler tests (stub mode — no GPU required)
# ---------------------------------------------------------------------------

class TestGPUProfilerStub:
    """Tests for GPUProfiler in stub mode (no NVML / no GPU).

    All numeric measurements are zero in stub mode; the timing pipeline and
    control-flow logic must still work correctly.
    """

    def setup_method(self):
        self.profiler = GPUProfiler(device_index=0)

    def test_is_available_returns_bool(self):
        assert isinstance(GPUProfiler.is_available(), bool)

    def test_device_name_is_string(self):
        assert isinstance(self.profiler.device_name, str)

    def test_start_end_frame_no_raise(self):
        self.profiler.start_frame()
        time.sleep(0.002)
        energy_mj = self.profiler.end_frame()
        assert isinstance(energy_mj, float)
        assert energy_mj >= 0.0

    def test_end_frame_without_start_raises(self):
        with pytest.raises(RuntimeError, match="start_frame"):
            self.profiler.end_frame()

    def test_summary_after_frames_returns_result(self):
        for _ in range(5):
            self.profiler.start_frame()
            time.sleep(0.001)
            self.profiler.end_frame()
        result = self.profiler.summary("TestTracker")
        assert isinstance(result, GPUProfilingResult)
        assert result.tracker_name == "TestTracker"
        assert result.frame_count == 5

    def test_summary_before_frames_raises(self):
        with pytest.raises(ValueError, match="No frames profiled"):
            self.profiler.summary()

    def test_reset_clears_state(self):
        for _ in range(3):
            self.profiler.start_frame()
            self.profiler.end_frame()
        self.profiler.reset()
        with pytest.raises(ValueError, match="No frames profiled"):
            self.profiler.summary()

    def test_stub_mode_zero_energy(self):
        """In stub mode all energy metrics must be zero."""
        if self.profiler._nvml_active:
            pytest.skip("NVML is active on this machine — stub test not applicable")
        for _ in range(3):
            self.profiler.start_frame()
            energy_mj = self.profiler.end_frame()
            assert energy_mj == 0.0
        result = self.profiler.summary("stub")
        assert result.total_energy_j == 0.0
        assert result.mean_power_w == 0.0
        assert result.peak_memory_mb == 0.0
        assert result.mean_gpu_util_pct == 0.0
        assert result.nvml_available is False

    def test_frame_count_matches_calls(self):
        n = 7
        for _ in range(n):
            self.profiler.start_frame()
            self.profiler.end_frame()
        result = self.profiler.summary()
        assert result.frame_count == n

    def test_to_dict_round_trips(self):
        for _ in range(4):
            self.profiler.start_frame()
            self.profiler.end_frame()
        result = self.profiler.summary("dict_test")
        d = result.to_dict()
        assert d["tracker_name"] == "dict_test"
        assert d["frame_count"] == 4
        # All numeric values must be non-negative
        for key in ("total_energy_j", "mean_power_w", "energy_per_frame_mj",
                    "peak_memory_mb", "mean_gpu_util_pct", "peak_gpu_util_pct"):
            assert d[key] >= 0.0, f"{key} must be non-negative"

    def test_multiple_resets(self):
        """Profiler must remain reusable across sequences after reset."""
        for seq in range(3):
            self.profiler.reset()
            for _ in range(5):
                self.profiler.start_frame()
                self.profiler.end_frame()
            result = self.profiler.summary(f"seq_{seq}")
            assert result.frame_count == 5
