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

    def test_latency_p50_positive(self):
        result = self._run_frames(10)
        assert result.latency_p50_ms > 0.0

    def test_latency_p99_positive(self):
        result = self._run_frames(10)
        assert result.latency_p99_ms > 0.0

    def test_latency_percentile_ordering(self):
        """p50 <= p95 <= p99 must hold for any sample."""
        result = self._run_frames(20)
        assert result.latency_p50_ms <= result.latency_p95_ms + 1e-9
        assert result.latency_p95_ms <= result.latency_p99_ms + 1e-9

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


class TestProfilerWarmup:
    """Tests for the warmup_frames feature."""

    def test_warmup_frames_excluded_from_count(self):
        """frame_count should reflect only post-warmup frames."""
        p = Profiler(warmup_frames=3)
        total = 10
        for _ in range(total):
            p.start_frame()
            time.sleep(0.001)
            p.end_frame()
        result = p.summary("tracker")
        assert result.frame_count == total - 3
        assert result.warmup_frames_excluded == 3

    def test_no_warmup_is_backward_compatible(self):
        """Default Profiler() behaves as before: all frames counted."""
        p = Profiler()
        for _ in range(5):
            p.start_frame()
            time.sleep(0.001)
            p.end_frame()
        result = p.summary("t")
        assert result.frame_count == 5
        assert result.warmup_frames_excluded == 0

    def test_warmup_all_frames_raises(self):
        """Consuming all frames as warm-up should raise ValueError."""
        p = Profiler(warmup_frames=5)
        for _ in range(5):
            p.start_frame()
            time.sleep(0.001)
            p.end_frame()
        with pytest.raises(ValueError, match="warm-up"):
            p.summary("t")

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError):
            Profiler(warmup_frames=-1)

    def test_warmup_zero_is_same_as_default(self):
        p1 = Profiler(warmup_frames=0)
        p2 = Profiler()
        for p in (p1, p2):
            for _ in range(5):
                p.start_frame()
                time.sleep(0.001)
                p.end_frame()
        r1 = p1.summary("t")
        r2 = p2.summary("t")
        assert r1.frame_count == r2.frame_count == 5
