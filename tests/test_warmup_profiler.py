"""Tests for warm-up-aware profiling and latency budget violation tracking."""

from __future__ import annotations

import time
import pytest

from eovot.profiling.profiler import Profiler, ProfilingResult


def _run_frames(profiler: Profiler, n: int, sleep_s: float = 0.001) -> ProfilingResult:
    for _ in range(n):
        profiler.start_frame()
        time.sleep(sleep_s)
        profiler.end_frame()
    return profiler.summary("test")


class TestWarmupFrames:
    def test_zero_warmup_is_backward_compatible(self):
        p = Profiler(warmup_frames=0)
        result = _run_frames(p, 5)
        assert result.warmup_frame_count == 0
        assert result.frame_count == 5
        # steady_state fields should mirror overall fields
        assert abs(result.steady_state_latency_mean_ms - result.latency_mean_ms) < 1e-9
        assert abs(result.steady_state_fps - result.fps) < 1e-3

    def test_warmup_frames_excluded_from_steady_state(self):
        # Run 3 warm-up frames (fast) then 7 steady-state frames (slower)
        warmup = 3
        p = Profiler(warmup_frames=warmup)
        # Warm-up: very fast
        for _ in range(warmup):
            p.start_frame()
            time.sleep(0.0005)
            p.end_frame()
        # Steady-state: noticeably slower
        for _ in range(7):
            p.start_frame()
            time.sleep(0.005)
            p.end_frame()
        result = p.summary("test")
        assert result.warmup_frame_count == warmup
        assert result.frame_count == 10
        # Steady-state mean should be higher than overall mean
        # (because warm-up was faster and pulled the overall mean down)
        assert result.steady_state_latency_mean_ms > result.latency_mean_ms - 1e-6

    def test_warmup_count_correct(self):
        p = Profiler(warmup_frames=4)
        result = _run_frames(p, 10)
        assert result.warmup_frame_count == 4
        assert result.frame_count == 10

    def test_warmup_larger_than_frame_count(self):
        # When we record fewer frames than warmup_frames, all are warm-up
        p = Profiler(warmup_frames=20)
        result = _run_frames(p, 5)
        assert result.warmup_frame_count == 5
        assert result.frame_count == 5
        # steady_state mirrors overall when no steady frames exist
        assert result.steady_state_latency_mean_ms == pytest.approx(
            result.latency_mean_ms, rel=1e-6
        )

    def test_invalid_warmup_frames_raises(self):
        with pytest.raises(ValueError):
            Profiler(warmup_frames=-1)

    def test_reset_clears_warmup_buckets(self):
        p = Profiler(warmup_frames=3)
        _run_frames(p, 5)
        p.reset()
        with pytest.raises(ValueError):
            p.summary()


class TestBudgetViolationRate:
    def test_no_budget_gives_zero_violation_rate(self):
        p = Profiler()
        result = _run_frames(p, 10)
        assert result.budget_ms is None
        assert result.budget_violation_rate == 0.0

    def test_tight_budget_causes_violations(self):
        # Budget = 0.1 ms — basically impossible to hit — all frames should violate
        p = Profiler(budget_ms=0.1)
        result = _run_frames(p, 10, sleep_s=0.002)
        assert result.budget_ms == pytest.approx(0.1)
        assert result.budget_violation_rate == pytest.approx(1.0)

    def test_loose_budget_no_violations(self):
        # Budget = 10 seconds — nothing should violate
        p = Profiler(budget_ms=10_000.0)
        result = _run_frames(p, 10, sleep_s=0.001)
        assert result.budget_violation_rate == pytest.approx(0.0)

    def test_partial_violation_rate(self):
        # Mix: first 5 frames fast, next 5 frames slow; budget between the two
        p = Profiler(budget_ms=2.0)
        for _ in range(5):
            p.start_frame()
            time.sleep(0.0005)   # ~0.5 ms — under budget
            p.end_frame()
        for _ in range(5):
            p.start_frame()
            time.sleep(0.005)    # ~5 ms — over budget
            p.end_frame()
        result = p.summary("test")
        # Roughly half should violate (timing imprecise, but clearly > 0 and < 1)
        assert 0.0 < result.budget_violation_rate < 1.0

    def test_budget_violation_respects_warmup(self):
        # Warm-up frames (slow) are excluded from violation count
        warmup = 3
        p = Profiler(warmup_frames=warmup, budget_ms=2.0)
        # Warm-up: slow (would violate budget if included)
        for _ in range(warmup):
            p.start_frame()
            time.sleep(0.01)
            p.end_frame()
        # Steady-state: fast (under budget)
        for _ in range(7):
            p.start_frame()
            time.sleep(0.0005)
            p.end_frame()
        result = p.summary("test")
        # Only steady-state frames count — all fast — so violation rate should be 0
        assert result.budget_violation_rate == pytest.approx(0.0)
        assert result.warmup_frame_count == warmup

    def test_invalid_budget_ms_raises(self):
        with pytest.raises(ValueError):
            Profiler(budget_ms=0.0)
        with pytest.raises(ValueError):
            Profiler(budget_ms=-5.0)

    def test_str_includes_budget_info(self):
        p = Profiler(budget_ms=33.3)
        result = _run_frames(p, 5)
        s = str(result)
        assert "budget=33.3ms" in s
        assert "violation=" in s

    def test_str_omits_budget_info_when_none(self):
        p = Profiler()
        result = _run_frames(p, 5)
        assert "budget=" not in str(result)


class TestBenchmarkEngineIntegration:
    """Integration smoke-test: BenchmarkEngine exposes warmup_frames and latency_budget_ms."""

    def test_engine_accepts_new_params(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        engine = BenchmarkEngine(
            verbose=False,
            warmup_frames=2,
            latency_budget_ms=50.0,
        )
        ds = SyntheticDataset(num_sequences=1, num_frames=10)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")
        sr = result.sequence_results[0]
        assert sr.profiling.warmup_frame_count == 2
        assert sr.profiling.budget_ms == pytest.approx(50.0)
        assert 0.0 <= sr.profiling.budget_violation_rate <= 1.0

    def test_engine_to_dict_round_trip(self):
        from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        engine = BenchmarkEngine(
            verbose=False,
            warmup_frames=1,
            latency_budget_ms=100.0,
        )
        ds = SyntheticDataset(num_sequences=1, num_frames=8)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")
        d = result.to_dict()

        seq = d["sequences"][0]
        assert "warmup_frame_count" in seq
        assert "steady_state_latency_mean_ms" in seq
        assert "steady_state_fps" in seq
        assert "latency_budget_ms" in seq
        assert "budget_violation_rate" in seq

        # Round-trip via from_dict
        restored = BenchmarkResult.from_dict(d)
        rp = restored.sequence_results[0].profiling
        assert rp.warmup_frame_count == seq["warmup_frame_count"]
        assert rp.budget_ms == pytest.approx(100.0)
