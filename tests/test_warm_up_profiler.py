"""Tests for warm-up calibration and latency CI in the EOVOT Profiler."""

from __future__ import annotations

import time

import numpy as np
import pytest

from eovot.profiling.profiler import Profiler, ProfilingResult, _bootstrap_ci_mean


# ---------------------------------------------------------------------------
# _bootstrap_ci_mean
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_basic_coverage(self):
        """CI should contain the true mean for a simple distribution."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=10.0, scale=2.0, size=200)
        lo, hi = _bootstrap_ci_mean(data, n_resamples=2_000, seed=0)
        assert lo < data.mean() < hi

    def test_single_value(self):
        """Single-element array returns (value, value)."""
        data = np.array([5.0])
        lo, hi = _bootstrap_ci_mean(data, n_resamples=100)
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    def test_deterministic_with_seed(self):
        data = np.arange(50, dtype=float)
        r1 = _bootstrap_ci_mean(data, n_resamples=500, seed=7)
        r2 = _bootstrap_ci_mean(data, n_resamples=500, seed=7)
        assert r1 == r2

    def test_lower_less_than_upper(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = _bootstrap_ci_mean(data, n_resamples=1_000)
        assert lo < hi


# ---------------------------------------------------------------------------
# Profiler — warm_up_frames=0 (no warm-up, backward compatible)
# ---------------------------------------------------------------------------

class TestProfilerNoWarmUp:
    def _run(self, n_frames: int = 20, warm_up: int = 0) -> ProfilingResult:
        p = Profiler(warm_up_frames=warm_up, n_bootstrap=500)
        for _ in range(n_frames):
            p.start_frame()
            time.sleep(0.001)
            p.end_frame()
        return p.summary("tracker")

    def test_frame_count(self):
        result = self._run(n_frames=10, warm_up=0)
        assert result.frame_count == 10

    def test_fps_positive(self):
        result = self._run(n_frames=10, warm_up=0)
        assert result.fps > 0

    def test_latency_positive(self):
        result = self._run(n_frames=10, warm_up=0)
        assert result.latency_mean_ms > 0

    def test_ci_fields_present(self):
        result = self._run(n_frames=10, warm_up=0)
        lo, hi = result.latency_ci_95_ms
        assert lo > 0
        assert hi >= lo

    def test_warm_up_skipped_is_zero(self):
        result = self._run(n_frames=10, warm_up=0)
        assert result.warm_up_frames_skipped == 0


# ---------------------------------------------------------------------------
# Profiler — warm_up_frames > 0
# ---------------------------------------------------------------------------

class TestProfilerWithWarmUp:
    def _run(self, n_frames: int, warm_up: int) -> ProfilingResult:
        p = Profiler(warm_up_frames=warm_up, n_bootstrap=500)
        for _ in range(n_frames):
            p.start_frame()
            time.sleep(0.001)
            p.end_frame()
        return p.summary("tracker")

    def test_recorded_frames_reduced_by_warm_up(self):
        result = self._run(n_frames=15, warm_up=5)
        assert result.frame_count == 10

    def test_warm_up_skipped_reported(self):
        result = self._run(n_frames=15, warm_up=5)
        assert result.warm_up_frames_skipped == 5

    def test_fallback_when_all_frames_are_warm_up(self):
        """When seq < warm_up_frames, profiler uses all frames rather than raising."""
        p = Profiler(warm_up_frames=10, n_bootstrap=200)
        for _ in range(5):
            p.start_frame()
            time.sleep(0.0005)
            p.end_frame()
        result = p.summary("tracker")
        # Falls back to all 5 frames; warm_up_frames_skipped is 0 (none discarded).
        assert result.frame_count == 5
        assert result.warm_up_frames_skipped == 0
        assert result.fps > 0

    def test_large_warm_up_falls_back_to_all_frames(self):
        """warm_up_frames_skipped is 0 when sequence is shorter than warm-up window."""
        p = Profiler(warm_up_frames=100, n_bootstrap=200)
        for _ in range(20):
            p.start_frame()
            p.end_frame()
        result = p.summary("tracker")
        assert result.frame_count == 20
        assert result.warm_up_frames_skipped == 0

    def test_ci_contains_mean(self):
        result = self._run(n_frames=50, warm_up=5)
        lo, hi = result.latency_ci_95_ms
        assert lo <= result.latency_mean_ms <= hi

    def test_reset_clears_warm_up_counter(self):
        p = Profiler(warm_up_frames=3, n_bootstrap=200)
        for _ in range(5):
            p.start_frame()
            time.sleep(0.0005)
            p.end_frame()
        r1 = p.summary("t")
        assert r1.frame_count == 2

        p.reset()
        for _ in range(5):
            p.start_frame()
            time.sleep(0.0005)
            p.end_frame()
        r2 = p.summary("t")
        assert r2.frame_count == 2  # same behaviour after reset

    def test_repr_includes_ci(self):
        result = self._run(n_frames=10, warm_up=2)
        s = str(result)
        assert "95%CI" in s
        assert "warm_up" in s


# ---------------------------------------------------------------------------
# Profiler — invalid construction
# ---------------------------------------------------------------------------

class TestProfilerValidation:
    def test_negative_warm_up_raises(self):
        with pytest.raises(ValueError):
            Profiler(warm_up_frames=-1)

    def test_end_before_start_raises(self):
        p = Profiler(warm_up_frames=0)
        with pytest.raises(RuntimeError):
            p.end_frame()


# ---------------------------------------------------------------------------
# BenchmarkEngine integration
# ---------------------------------------------------------------------------

class TestBenchmarkEngineWarmUp:
    def test_engine_accepts_warm_up_frames(self):
        """BenchmarkEngine should forward warm_up_frames to the Profiler."""
        from eovot.benchmark.engine import BenchmarkEngine

        engine = BenchmarkEngine(verbose=False, warm_up_frames=3)
        # Access private profiler to verify configuration was passed through.
        assert engine._profiler._warm_up_frames == 3

    def test_default_warm_up_is_five_for_engine(self):
        """BenchmarkEngine defaults to 5 warm-up frames for reliable latency estimates."""
        from eovot.benchmark.engine import BenchmarkEngine

        engine = BenchmarkEngine(verbose=False)
        assert engine._profiler._warm_up_frames == 5
