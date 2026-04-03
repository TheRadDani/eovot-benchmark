"""Unit tests for eovot.profiling.profiler."""

import time
import pytest

from eovot.profiling.profiler import Profiler, ProfilingResult


class TestProfiler:
    def setup_method(self):
        self.profiler = Profiler()

    def _run_frames(self, n: int = 5, sleep_s: float = 0.001) -> ProfilingResult:
        for _ in range(n):
            self.profiler.start_frame()
            time.sleep(sleep_s)
            self.profiler.end_frame()
        return self.profiler.summary("test_tracker")

    def test_summary_after_frames(self):
        result = self._run_frames(5)
        assert isinstance(result, ProfilingResult)

    def test_frame_count(self):
        result = self._run_frames(7)
        assert result.frame_count == 7

    def test_fps_positive(self):
        result = self._run_frames(5)
        assert result.fps > 0.0

    def test_latency_mean_positive(self):
        result = self._run_frames(5)
        assert result.latency_mean_ms > 0.0

    def test_latency_p95_ge_mean(self):
        result = self._run_frames(10)
        assert result.latency_p95_ms >= result.latency_mean_ms - 1e-9

    def test_peak_memory_positive(self):
        result = self._run_frames(3)
        assert result.peak_memory_mb > 0.0

    def test_end_frame_before_start_raises(self):
        with pytest.raises(RuntimeError):
            self.profiler.end_frame()

    def test_summary_without_frames_raises(self):
        with pytest.raises(ValueError):
            self.profiler.summary()

    def test_reset_clears_state(self):
        self._run_frames(3)
        self.profiler.reset()
        with pytest.raises(ValueError):
            self.profiler.summary()

    def test_fps_consistent_with_latency(self):
        result = self._run_frames(5)
        # fps ≈ 1000 / mean_latency_ms  (within 10% due to measurement overhead)
        expected_fps = 1000.0 / result.latency_mean_ms
        assert abs(result.fps - expected_fps) / expected_fps < 0.1

    def test_tracker_name_preserved(self):
        result = self._run_frames(2)
        assert result.tracker_name == "test_tracker"
