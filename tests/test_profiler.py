"""Unit tests for eovot.profiling.profiler."""

import time

import pytest

from eovot.profiling.profiler import Profiler, ProfilingResult


class TestProfiler:
    """Tests for the Profiler timing and memory tracker."""

    def setup_method(self):
        self.profiler = Profiler()

    def test_single_frame(self):
        self.profiler.start_frame()
        time.sleep(0.005)  # 5 ms
        elapsed = self.profiler.end_frame()
        assert elapsed >= 4.0  # at least 4 ms recorded

    def test_summary_after_frames(self):
        for _ in range(5):
            self.profiler.start_frame()
            time.sleep(0.002)
            self.profiler.end_frame()

        result = self.profiler.summary("test_tracker")
        assert isinstance(result, ProfilingResult)
        assert result.tracker_name == "test_tracker"
        assert result.frame_count == 5
        assert result.fps > 0.0
        assert result.latency_mean_ms >= 1.0
        assert result.latency_std_ms >= 0.0
        assert result.latency_p95_ms >= result.latency_mean_ms - 1.0
        assert result.peak_memory_mb > 0.0

    def test_end_frame_before_start_raises(self):
        with pytest.raises(RuntimeError):
            self.profiler.end_frame()

    def test_summary_without_frames_raises(self):
        with pytest.raises(ValueError):
            self.profiler.summary()

    def test_reset_clears_state(self):
        self.profiler.start_frame()
        self.profiler.end_frame()
        self.profiler.reset()
        with pytest.raises(ValueError):
            self.profiler.summary()

    def test_fps_inversely_proportional_to_latency(self):
        self.profiler.start_frame()
        time.sleep(0.01)  # ~10 ms → ~100 FPS
        self.profiler.end_frame()

        result = self.profiler.summary()
        # Very loose bound: FPS should be roughly 100 ± big margin due to sleep imprecision
        assert result.fps > 10.0
        assert result.fps < 10_000.0

    def test_profiling_result_str(self):
        self.profiler.start_frame()
        self.profiler.end_frame()
        result = self.profiler.summary("my_tracker")
        text = str(result)
        assert "my_tracker" in text
        assert "FPS" in text
        assert "latency" in text
