"""Unit tests for eovot.profiling.profiler."""

import time

import numpy as np
import pytest

from eovot.profiling.profiler import Profiler, ProfilingResult


class TestProfiler:
    def setup_method(self):
        self.profiler = Profiler()

    def test_start_end_frame_returns_positive_ms(self):
        self.profiler.start_frame()
        time.sleep(0.005)  # 5 ms
        elapsed = self.profiler.end_frame()
        assert elapsed > 0.0

    def test_end_frame_without_start_raises(self):
        with pytest.raises(RuntimeError, match="end_frame"):
            self.profiler.end_frame()

    def test_summary_without_frames_raises(self):
        with pytest.raises(ValueError):
            self.profiler.summary()

    def test_summary_after_frames(self):
        for _ in range(5):
            self.profiler.start_frame()
            time.sleep(0.001)
            self.profiler.end_frame()

        result = self.profiler.summary(tracker_name="test_tracker")
        assert isinstance(result, ProfilingResult)
        assert result.tracker_name == "test_tracker"
        assert result.frame_count == 5
        assert result.fps > 0.0
        assert result.latency_mean_ms > 0.0
        assert result.latency_std_ms >= 0.0
        assert result.latency_p95_ms >= result.latency_mean_ms - 1e-9

    def test_fps_is_consistent_with_latency(self):
        for _ in range(10):
            self.profiler.start_frame()
            time.sleep(0.002)
            self.profiler.end_frame()

        result = self.profiler.summary()
        expected_fps = 1_000.0 / result.latency_mean_ms
        assert result.fps == pytest.approx(expected_fps, rel=1e-6)

    def test_reset_clears_state(self):
        for _ in range(3):
            self.profiler.start_frame()
            self.profiler.end_frame()

        self.profiler.reset()
        with pytest.raises(ValueError):
            self.profiler.summary()

    def test_p95_leq_max_latency(self):
        for _ in range(20):
            self.profiler.start_frame()
            self.profiler.end_frame()

        result = self.profiler.summary()
        assert result.latency_p95_ms <= result.latency_mean_ms + 5 * result.latency_std_ms

    def test_peak_memory_is_positive(self):
        for _ in range(3):
            self.profiler.start_frame()
            self.profiler.end_frame()

        result = self.profiler.summary()
        assert result.peak_memory_mb > 0.0

    def test_str_representation(self):
        self.profiler.start_frame()
        self.profiler.end_frame()
        result = self.profiler.summary(tracker_name="MOSSE")
        s = str(result)
        assert "MOSSE" in s
        assert "FPS" in s
        assert "latency" in s


class TestProfilingResult:
    def test_str_contains_key_fields(self):
        pr = ProfilingResult(
            tracker_name="KCF",
            frame_count=100,
            fps=250.0,
            latency_mean_ms=4.0,
            latency_std_ms=0.5,
            latency_p95_ms=5.0,
            peak_memory_mb=32.0,
        )
        s = str(pr)
        assert "KCF" in s
        assert "250.0" in s or "250" in s
