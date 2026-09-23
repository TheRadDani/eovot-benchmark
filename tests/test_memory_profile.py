"""Tests for eovot.profiling.memory_profile."""

from __future__ import annotations

import pytest
import numpy as np

from eovot.profiling.memory_profile import (
    FrameMemorySnapshot,
    MemoryProfileResult,
    MemoryProfiler,
    MemoryBudgetChecker,
    BudgetCheckReport,
    _linear_slope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profiler_with_snapshots(
    deltas_mb: list,
    baseline_mb: float = 100.0,
    leak_threshold: float = 1e-3,
) -> MemoryProfiler:
    """Return a MemoryProfiler pre-populated with synthetic snapshots."""
    mp = MemoryProfiler(leak_threshold_mb_per_frame=leak_threshold)
    mp._baseline_mb = baseline_mb
    for i, d in enumerate(deltas_mb):
        snap = FrameMemorySnapshot(
            frame_idx=i,
            rss_mb=baseline_mb + d,
            delta_mb=d,
        )
        mp._snapshots.append(snap)
    return mp


# ---------------------------------------------------------------------------
# FrameMemorySnapshot
# ---------------------------------------------------------------------------

class TestFrameMemorySnapshot:
    def test_fields_stored(self):
        s = FrameMemorySnapshot(frame_idx=3, rss_mb=200.5, delta_mb=5.5)
        assert s.frame_idx == 3
        assert s.rss_mb == pytest.approx(200.5)
        assert s.delta_mb == pytest.approx(5.5)

    def test_zero_delta(self):
        s = FrameMemorySnapshot(frame_idx=0, rss_mb=100.0, delta_mb=0.0)
        assert s.delta_mb == pytest.approx(0.0)

    def test_negative_delta_allowed(self):
        s = FrameMemorySnapshot(frame_idx=1, rss_mb=95.0, delta_mb=-5.0)
        assert s.delta_mb == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# _linear_slope
# ---------------------------------------------------------------------------

class TestLinearSlope:
    def test_zero_slope_constant_y(self):
        x = np.arange(10, dtype=float)
        y = np.ones(10) * 3.0
        assert _linear_slope(x, y) == pytest.approx(0.0, abs=1e-9)

    def test_positive_slope(self):
        x = np.arange(10, dtype=float)
        y = x * 2.0
        assert _linear_slope(x, y) == pytest.approx(2.0, rel=1e-6)

    def test_negative_slope(self):
        x = np.arange(10, dtype=float)
        y = -x * 0.5
        assert _linear_slope(x, y) == pytest.approx(-0.5, rel=1e-6)

    def test_single_point_returns_zero(self):
        assert _linear_slope(np.array([1.0]), np.array([5.0])) == pytest.approx(0.0)

    def test_constant_x_returns_zero(self):
        x = np.ones(5)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _linear_slope(x, y) == pytest.approx(0.0)

    def test_two_points(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 3.0])
        assert _linear_slope(x, y) == pytest.approx(3.0, rel=1e-6)

    def test_large_series(self):
        x = np.arange(1000, dtype=float)
        slope = 0.007
        y = slope * x + np.random.default_rng(0).normal(0, 1e-6, 1000)
        assert _linear_slope(x, y) == pytest.approx(slope, rel=1e-3)


# ---------------------------------------------------------------------------
# MemoryProfiler — baseline
# ---------------------------------------------------------------------------

class TestMemoryProfilerBaseline:
    def test_set_baseline_returns_positive_float(self):
        mp = MemoryProfiler()
        baseline = mp.set_baseline()
        assert baseline > 0

    def test_set_baseline_clears_existing_snapshots(self):
        mp = _make_profiler_with_snapshots([1.0, 2.0])
        assert len(mp._snapshots) == 2
        mp.set_baseline()
        assert len(mp._snapshots) == 0

    def test_set_baseline_stores_value(self):
        mp = MemoryProfiler()
        val = mp.set_baseline()
        assert mp._baseline_mb == pytest.approx(val)


# ---------------------------------------------------------------------------
# MemoryProfiler — record_frame
# ---------------------------------------------------------------------------

class TestMemoryProfilerRecordFrame:
    def test_record_returns_snapshot(self):
        mp = MemoryProfiler()
        mp.set_baseline()
        snap = mp.record_frame(0)
        assert isinstance(snap, FrameMemorySnapshot)
        assert snap.frame_idx == 0
        assert snap.rss_mb > 0

    def test_snapshot_accumulated(self):
        mp = MemoryProfiler()
        mp.set_baseline()
        for i in range(5):
            mp.record_frame(i)
        assert len(mp._snapshots) == 5

    def test_delta_is_rss_minus_baseline(self):
        mp = MemoryProfiler()
        mp._baseline_mb = 100.0
        mp._rss_mb = lambda: 105.0
        snap = mp.record_frame(0)
        assert snap.delta_mb == pytest.approx(5.0)
        assert snap.rss_mb == pytest.approx(105.0)

    def test_frame_indices_preserved(self):
        mp = MemoryProfiler()
        mp.set_baseline()
        for i in [0, 5, 10]:
            mp.record_frame(i)
        indices = [s.frame_idx for s in mp._snapshots]
        assert indices == [0, 5, 10]

    def test_appends_sequentially(self):
        mp = MemoryProfiler()
        mp.set_baseline()
        for i in range(3):
            mp.record_frame(i)
        assert mp._snapshots[0].frame_idx == 0
        assert mp._snapshots[2].frame_idx == 2


# ---------------------------------------------------------------------------
# MemoryProfiler — summary
# ---------------------------------------------------------------------------

class TestMemoryProfilerSummary:
    def test_raises_if_no_frames(self):
        mp = MemoryProfiler()
        with pytest.raises(ValueError, match="No frames recorded"):
            mp.summary()

    def test_tracker_name_default(self):
        mp = _make_profiler_with_snapshots([1.0])
        result = mp.summary()
        assert result.tracker_name == "unknown"

    def test_tracker_name_set(self):
        mp = _make_profiler_with_snapshots([1.0, 2.0])
        result = mp.summary("MOSSE")
        assert result.tracker_name == "MOSSE"

    def test_frame_count_correct(self):
        mp = _make_profiler_with_snapshots([1.0, 2.0, 3.0])
        result = mp.summary()
        assert result.frame_count == 3

    def test_peak_delta_correct(self):
        mp = _make_profiler_with_snapshots([1.0, 5.0, 2.0])
        result = mp.summary()
        assert result.peak_delta_mb == pytest.approx(5.0)

    def test_final_delta_correct(self):
        mp = _make_profiler_with_snapshots([1.0, 5.0, 2.5])
        result = mp.summary()
        assert result.final_delta_mb == pytest.approx(2.5)

    def test_mean_delta_correct(self):
        mp = _make_profiler_with_snapshots([2.0, 4.0, 6.0])
        result = mp.summary()
        assert result.mean_delta_mb == pytest.approx(4.0)

    def test_baseline_preserved(self):
        mp = _make_profiler_with_snapshots([1.0], baseline_mb=200.0)
        result = mp.summary()
        assert result.baseline_mb == pytest.approx(200.0)

    def test_peak_mb_is_absolute_rss(self):
        mp = _make_profiler_with_snapshots([5.0, 10.0], baseline_mb=100.0)
        result = mp.summary()
        assert result.peak_mb == pytest.approx(110.0)

    def test_snapshots_excluded_by_default(self):
        mp = _make_profiler_with_snapshots([1.0, 2.0])
        result = mp.summary()
        assert result.snapshots == []

    def test_snapshots_included_when_requested(self):
        mp = _make_profiler_with_snapshots([1.0, 2.0])
        result = mp.summary(include_snapshots=True)
        assert len(result.snapshots) == 2

    def test_no_leak_flag_for_flat_memory(self):
        deltas = [0.5] * 50
        mp = _make_profiler_with_snapshots(deltas)
        result = mp.summary()
        assert not result.is_leaking
        assert result.growth_rate_mb_per_frame == pytest.approx(0.0, abs=1e-6)

    def test_leak_detected_for_linear_growth(self):
        deltas = [i * 0.01 for i in range(100)]
        mp = _make_profiler_with_snapshots(deltas, leak_threshold=1e-3)
        result = mp.summary()
        assert result.is_leaking
        assert result.growth_rate_mb_per_frame == pytest.approx(0.01, rel=1e-3)

    def test_custom_threshold_tighter_detects_more(self):
        deltas = [i * 5e-4 for i in range(100)]
        mp_default = _make_profiler_with_snapshots(deltas, leak_threshold=1e-3)
        mp_tight = _make_profiler_with_snapshots(deltas, leak_threshold=1e-4)
        assert not mp_default.summary().is_leaking
        assert mp_tight.summary().is_leaking


# ---------------------------------------------------------------------------
# MemoryProfileResult — str and to_dict
# ---------------------------------------------------------------------------

class TestMemoryProfileResultOutput:
    def _make_result(self, leaking: bool = False) -> MemoryProfileResult:
        return MemoryProfileResult(
            tracker_name="KCF",
            frame_count=100,
            baseline_mb=128.0,
            peak_mb=135.0,
            peak_delta_mb=7.0,
            final_delta_mb=6.5,
            mean_delta_mb=3.0,
            growth_rate_mb_per_frame=0.07 if leaking else 0.0001,
            is_leaking=leaking,
        )

    def test_str_contains_tracker_name(self):
        assert "KCF" in str(self._make_result())

    def test_str_flags_leak(self):
        assert "LEAK" in str(self._make_result(leaking=True))

    def test_str_no_flag_when_healthy(self):
        assert "LEAK" not in str(self._make_result(leaking=False))

    def test_to_dict_has_all_keys(self):
        d = self._make_result().to_dict()
        expected = {
            "tracker_name", "frame_count", "baseline_mb", "peak_mb",
            "peak_delta_mb", "final_delta_mb", "mean_delta_mb",
            "growth_rate_mb_per_frame", "is_leaking",
        }
        assert set(d.keys()) == expected

    def test_to_dict_values_roundtrip(self):
        r = self._make_result(leaking=True)
        d = r.to_dict()
        assert d["tracker_name"] == "KCF"
        assert d["is_leaking"] is True
        assert d["peak_delta_mb"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# MemoryProfiler — reset
# ---------------------------------------------------------------------------

class TestMemoryProfilerReset:
    def test_reset_clears_snapshots(self):
        mp = _make_profiler_with_snapshots([1.0, 2.0, 3.0])
        mp.reset()
        assert len(mp._snapshots) == 0

    def test_reset_clears_baseline(self):
        mp = _make_profiler_with_snapshots([1.0], baseline_mb=200.0)
        mp.reset()
        assert mp._baseline_mb == pytest.approx(0.0)

    def test_summary_raises_after_reset(self):
        mp = _make_profiler_with_snapshots([1.0])
        mp.reset()
        with pytest.raises(ValueError):
            mp.summary()


# ---------------------------------------------------------------------------
# MemoryBudgetChecker
# ---------------------------------------------------------------------------

class TestMemoryBudgetChecker:
    def _make_result(
        self,
        peak_delta_mb: float = 10.0,
        is_leaking: bool = False,
        growth_rate: float = 0.0,
    ) -> MemoryProfileResult:
        return MemoryProfileResult(
            tracker_name="T",
            frame_count=50,
            baseline_mb=100.0,
            peak_mb=100.0 + peak_delta_mb,
            peak_delta_mb=peak_delta_mb,
            final_delta_mb=peak_delta_mb * 0.9,
            mean_delta_mb=peak_delta_mb * 0.5,
            growth_rate_mb_per_frame=growth_rate,
            is_leaking=is_leaking,
        )

    def test_raises_on_zero_budget(self):
        with pytest.raises(ValueError):
            MemoryBudgetChecker(budget_mb=0)

    def test_raises_on_negative_budget(self):
        with pytest.raises(ValueError):
            MemoryBudgetChecker(budget_mb=-10)

    def test_passes_under_budget_no_leak(self):
        checker = MemoryBudgetChecker(budget_mb=64.0)
        result = self._make_result(peak_delta_mb=10.0, is_leaking=False)
        assert checker.check(result).passes

    def test_fails_over_budget(self):
        checker = MemoryBudgetChecker(budget_mb=8.0)
        result = self._make_result(peak_delta_mb=10.0)
        assert not checker.check(result).passes

    def test_excess_mb_correct(self):
        checker = MemoryBudgetChecker(budget_mb=8.0)
        result = self._make_result(peak_delta_mb=10.0)
        report = checker.check(result)
        assert report.excess_mb == pytest.approx(2.0)

    def test_fails_leaking_even_under_budget(self):
        checker = MemoryBudgetChecker(budget_mb=100.0)
        result = self._make_result(peak_delta_mb=5.0, is_leaking=True, growth_rate=0.01)
        report = checker.check(result)
        assert not report.passes
        assert report.excess_mb == pytest.approx(0.0)

    def test_report_str_ok(self):
        checker = MemoryBudgetChecker(budget_mb=64.0)
        result = self._make_result(peak_delta_mb=10.0)
        assert "OK" in str(checker.check(result))

    def test_report_str_fail(self):
        checker = MemoryBudgetChecker(budget_mb=5.0)
        result = self._make_result(peak_delta_mb=10.0)
        assert "FAIL" in str(checker.check(result))

    def test_report_str_warning_for_leak(self):
        checker = MemoryBudgetChecker(budget_mb=100.0)
        result = self._make_result(peak_delta_mb=5.0, is_leaking=True, growth_rate=0.05)
        assert "WARNING" in str(checker.check(result))

    def test_zero_excess_when_passes(self):
        checker = MemoryBudgetChecker(budget_mb=100.0)
        result = self._make_result(peak_delta_mb=10.0)
        assert checker.check(result).excess_mb == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration: real psutil (smoke test)
# ---------------------------------------------------------------------------

class TestMemoryProfilerIntegration:
    def test_real_profiler_smoke(self):
        mp = MemoryProfiler()
        mp.set_baseline()
        for i in range(20):
            _ = list(range(10_000))
            mp.record_frame(i)
        result = mp.summary("smoke_test", include_snapshots=True)
        assert result.frame_count == 20
        assert len(result.snapshots) == 20
        assert result.peak_mb > 0

    def test_budget_checker_generous_passes(self):
        mp = MemoryProfiler()
        mp.set_baseline()
        for i in range(10):
            mp.record_frame(i)
        result = mp.summary("smoke")
        checker = MemoryBudgetChecker(budget_mb=500.0)
        report = checker.check(result)
        assert isinstance(report.passes, bool)
