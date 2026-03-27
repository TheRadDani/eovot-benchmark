"""Unit tests for eovot.profiling.profiler."""

from __future__ import annotations

import time

import numpy as np
import pytest

from eovot.profiling.profiler import Profiler, ProfilingResult


class TestProfiler:
    def setup_method(self):
        self.profiler = Profiler()

    # ------------------------------------------------------------------
    # Basic timing
    # ------------------------------------------------------------------

    def test_single_frame_produces_result(self):
        self.profiler.start_frame()
        time.sleep(0.001)  # 1 ms sleep
        self.profiler.end_frame()
        result = self.profiler.summary("test")
        assert isinstance(result, ProfilingResult)
        assert result.frame_count == 1

    def test_latency_positive(self):
        self.profiler.start_frame()
        time.sleep(0.002)
        self.profiler.end_frame()
        result = self.profiler.summary("t")
        assert result.latency_mean_ms > 0.0

    def test_fps_approximately_matches_latency(self):
        self.profiler.start_frame()
        time.sleep(0.005)  # ~5 ms → ~200 FPS
        self.profiler.end_frame()
        result = self.profiler.summary("t")
        expected_fps = 1000.0 / result.latency_mean_ms
        assert result.fps == pytest.approx(expected_fps, rel=1e-5)

    def test_multiple_frames_count(self):
        n = 5
        for _ in range(n):
            self.profiler.start_frame()
            self.profiler.end_frame()
        result = self.profiler.summary("t")
        assert result.frame_count == n

    def test_p95_geq_mean(self):
        """p95 latency must be ≥ mean latency."""
        for _ in range(20):
            self.profiler.start_frame()
            self.profiler.end_frame()
        result = self.profiler.summary("t")
        assert result.latency_p95_ms >= result.latency_mean_ms

    def test_std_non_negative(self):
        for _ in range(10):
            self.profiler.start_frame()
            self.profiler.end_frame()
        result = self.profiler.summary("t")
        assert result.latency_std_ms >= 0.0

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def test_peak_memory_positive(self):
        self.profiler.start_frame()
        self.profiler.end_frame()
        result = self.profiler.summary("t")
        assert result.peak_memory_mb > 0.0

    # ------------------------------------------------------------------
    # Error conditions
    # ------------------------------------------------------------------

    def test_end_before_start_raises(self):
        with pytest.raises(RuntimeError, match="end_frame.*start_frame"):
            self.profiler.end_frame()

    def test_summary_without_frames_raises(self):
        with pytest.raises(ValueError):
            self.profiler.summary("t")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def test_reset_clears_frames(self):
        self.profiler.start_frame()
        self.profiler.end_frame()
        self.profiler.reset()
        with pytest.raises(ValueError):
            self.profiler.summary("t")

    def test_reset_allows_reuse(self):
        for _ in range(3):
            self.profiler.start_frame()
            self.profiler.end_frame()
        self.profiler.reset()
        self.profiler.start_frame()
        self.profiler.end_frame()
        result = self.profiler.summary("t")
        assert result.frame_count == 1

    # ------------------------------------------------------------------
    # ProfilingResult helpers
    # ------------------------------------------------------------------

    def test_profiling_result_str(self):
        self.profiler.start_frame()
        self.profiler.end_frame()
        result = self.profiler.summary("MyTracker")
        s = str(result)
        assert "MyTracker" in s
        assert "FPS" in s
        assert "latency" in s

    def test_end_frame_returns_elapsed_ms(self):
        self.profiler.start_frame()
        elapsed = self.profiler.end_frame()
        assert elapsed > 0.0
