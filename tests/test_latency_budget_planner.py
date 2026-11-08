"""Tests for the LatencyBudgetPlanner SLA compliance module."""

import math
import pytest

from eovot.metrics.latency_budget_planner import (
    LatencyBudgetPlanner,
    SLAComplianceResult,
    _lognormal_params,
    _lognormal_cdf,
    _lognormal_quantile,
    _erf_approx,
)
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_profiling(
    name: str = "TestTracker",
    mean_ms: float = 20.0,
    std_ms: float = 4.0,
    p95_ms: float = 27.0,
    p99_ms: float = 32.0,
    fps: float = 50.0,
    frames: int = 500,
    mem_mb: float = 128.0,
) -> ProfilingResult:
    cv = std_ms / mean_ms if mean_ms > 0 else 0.0
    return ProfilingResult(
        tracker_name=name,
        frame_count=frames,
        fps=fps,
        latency_mean_ms=mean_ms,
        latency_std_ms=std_ms,
        latency_p95_ms=p95_ms,
        latency_p99_ms=p99_ms,
        latency_cv=cv,
        peak_memory_mb=mem_mb,
    )


# ---------------------------------------------------------------------------
# Log-normal helper tests
# ---------------------------------------------------------------------------

class TestLognormalHelpers:
    def test_params_typical(self):
        mu, sigma = _lognormal_params(20.0, 4.0)
        assert mu is not None
        assert sigma > 0
        # Reconstruct mean from mu and sigma: E[X] = exp(mu + sigma²/2)
        reconstructed_mean = math.exp(mu + sigma ** 2 / 2)
        assert abs(reconstructed_mean - 20.0) < 0.5

    def test_params_zero_std(self):
        mu, sigma = _lognormal_params(15.0, 0.0)
        assert mu is not None
        assert sigma == 0.0
        assert abs(math.exp(mu) - 15.0) < 1e-9

    def test_params_invalid_mean(self):
        mu, sigma = _lognormal_params(0.0, 4.0)
        assert mu is None

    def test_params_negative_mean(self):
        mu, sigma = _lognormal_params(-5.0, 2.0)
        assert mu is None

    def test_cdf_monotone(self):
        mu, sigma = _lognormal_params(20.0, 4.0)
        vals = [_lognormal_cdf(x, mu, sigma) for x in [5.0, 10.0, 20.0, 30.0, 50.0]]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]

    def test_cdf_at_zero(self):
        mu, sigma = _lognormal_params(20.0, 4.0)
        assert _lognormal_cdf(0.0, mu, sigma) == 0.0

    def test_cdf_large_x(self):
        mu, sigma = _lognormal_params(20.0, 4.0)
        assert _lognormal_cdf(10_000.0, mu, sigma) > 0.9999

    def test_quantile_roundtrip(self):
        mu, sigma = _lognormal_params(20.0, 4.0)
        for p in [0.5, 0.9, 0.95, 0.99]:
            q = _lognormal_quantile(p, mu, sigma)
            assert abs(_lognormal_cdf(q, mu, sigma) - p) < 0.001

    def test_quantile_p0_p1(self):
        mu, sigma = _lognormal_params(20.0, 4.0)
        assert _lognormal_quantile(0.0, mu, sigma) == 0.0
        assert _lognormal_quantile(1.0, mu, sigma) == float("inf")

    def test_erf_approx_known_values(self):
        assert abs(_erf_approx(0.0)) < 1e-6
        assert abs(_erf_approx(1.0) - 0.842701) < 1e-5
        assert abs(_erf_approx(-1.0) + 0.842701) < 1e-5
        assert abs(_erf_approx(3.0) - 1.0) < 5e-5


# ---------------------------------------------------------------------------
# LatencyBudgetPlanner.compute_compliance tests
# ---------------------------------------------------------------------------

class TestComputeCompliance:
    def setup_method(self):
        self.planner = LatencyBudgetPlanner()

    def test_budget_well_above_mean_high_compliance(self):
        p = make_profiling(mean_ms=20.0, std_ms=3.0, p99_ms=28.0)
        result = self.planner.compute_compliance(p, budget_ms=50.0)
        assert result.compliance_rate > 0.99
        assert result.violation_rate < 0.01

    def test_budget_at_mean_roughly_half_compliance(self):
        p = make_profiling(mean_ms=20.0, std_ms=4.0)
        result = self.planner.compute_compliance(p, budget_ms=20.0)
        # Log-normal median < mean, so compliance at mean should be > 50 %.
        assert result.compliance_rate > 0.5

    def test_budget_below_mean_low_compliance(self):
        p = make_profiling(mean_ms=20.0, std_ms=3.0)
        result = self.planner.compute_compliance(p, budget_ms=5.0)
        assert result.compliance_rate < 0.1

    def test_tracker_name_preserved(self):
        p = make_profiling(name="MOSSE")
        result = self.planner.compute_compliance(p, budget_ms=33.0)
        assert result.tracker_name == "MOSSE"

    def test_budget_ms_preserved(self):
        p = make_profiling(mean_ms=10.0, std_ms=2.0)
        result = self.planner.compute_compliance(p, budget_ms=25.0)
        assert result.budget_ms == 25.0

    def test_fps_equivalent_correct(self):
        p = make_profiling()
        result = self.planner.compute_compliance(p, budget_ms=33.333)
        assert abs(result.fps_equivalent - 30.0) < 0.5

    def test_compliance_plus_violation_equals_one(self):
        p = make_profiling(mean_ms=20.0, std_ms=4.0)
        result = self.planner.compute_compliance(p, budget_ms=33.0)
        assert abs(result.compliance_rate + result.violation_rate - 1.0) < 1e-9

    def test_budget_ms_invalid_raises(self):
        p = make_profiling()
        with pytest.raises(ValueError):
            self.planner.compute_compliance(p, budget_ms=0.0)
        with pytest.raises(ValueError):
            self.planner.compute_compliance(p, budget_ms=-5.0)

    def test_zero_std_deterministic_distribution(self):
        p = make_profiling(mean_ms=20.0, std_ms=0.0, p99_ms=20.0)
        result = self.planner.compute_compliance(p, budget_ms=25.0)
        assert result.compliance_rate == pytest.approx(1.0, abs=1e-6)
        assert result.distribution in ("lognormal", "deterministic")

    def test_zero_std_budget_below_mean(self):
        p = make_profiling(mean_ms=20.0, std_ms=0.0, p99_ms=20.0)
        result = self.planner.compute_compliance(p, budget_ms=15.0)
        assert result.compliance_rate == pytest.approx(0.0, abs=1e-6)

    def test_min_budget_99pct_conservative_vs_p99(self):
        # min_budget_99pct_ms must be >= empirical p99.
        p = make_profiling(mean_ms=20.0, std_ms=4.0, p99_ms=35.0)
        result = self.planner.compute_compliance(p, budget_ms=33.0)
        assert result.min_budget_99pct_ms >= 35.0

    def test_str_representation(self):
        p = make_profiling(name="KCF", mean_ms=15.0, std_ms=3.0, p99_ms=25.0)
        result = self.planner.compute_compliance(p, budget_ms=33.0)
        s = str(result)
        assert "KCF" in s
        assert "33" in s

    def test_mean_latency_in_result(self):
        p = make_profiling(mean_ms=18.0, std_ms=3.0)
        result = self.planner.compute_compliance(p, budget_ms=25.0)
        assert result.mean_latency_ms == pytest.approx(18.0, abs=1e-6)

    def test_p99_latency_in_result(self):
        p = make_profiling(mean_ms=20.0, std_ms=4.0, p99_ms=33.0)
        result = self.planner.compute_compliance(p, budget_ms=33.0)
        assert result.p99_latency_ms == pytest.approx(33.0, abs=1e-6)


# ---------------------------------------------------------------------------
# LatencyBudgetPlanner.find_min_budget tests
# ---------------------------------------------------------------------------

class TestFindMinBudget:
    def setup_method(self):
        self.planner = LatencyBudgetPlanner()

    def test_budget_exceeds_mean(self):
        p = make_profiling(mean_ms=20.0, std_ms=4.0)
        budget = self.planner.find_min_budget(p, target_rate=0.99)
        assert budget > 20.0

    def test_p99_target_honors_empirical_p99(self):
        p = make_profiling(mean_ms=20.0, std_ms=4.0, p99_ms=40.0)
        budget = self.planner.find_min_budget(p, target_rate=0.99)
        assert budget >= 40.0

    def test_p95_target_honors_empirical_p95(self):
        p = make_profiling(mean_ms=20.0, std_ms=4.0, p95_ms=35.0, p99_ms=40.0)
        budget = self.planner.find_min_budget(p, target_rate=0.95)
        assert budget >= 35.0

    def test_higher_target_needs_larger_budget(self):
        p = make_profiling(mean_ms=20.0, std_ms=4.0)
        b99 = self.planner.find_min_budget(p, target_rate=0.99)
        b95 = self.planner.find_min_budget(p, target_rate=0.95)
        b90 = self.planner.find_min_budget(p, target_rate=0.90)
        assert b99 >= b95 >= b90

    def test_invalid_target_rate_raises(self):
        p = make_profiling()
        with pytest.raises(ValueError):
            self.planner.find_min_budget(p, target_rate=0.0)
        with pytest.raises(ValueError):
            self.planner.find_min_budget(p, target_rate=1.0)
        with pytest.raises(ValueError):
            self.planner.find_min_budget(p, target_rate=1.5)

    def test_deterministic_tracker_returns_at_least_mean(self):
        # When std=0 the degenerate log-normal collapses to the mean;
        # find_min_budget must still respect the empirical p95/p99 guard.
        p = make_profiling(mean_ms=20.0, std_ms=0.0, p95_ms=20.0, p99_ms=20.0)
        budget = self.planner.find_min_budget(p, target_rate=0.99)
        assert budget >= 20.0


# ---------------------------------------------------------------------------
# LatencyBudgetPlanner.compare_trackers tests
# ---------------------------------------------------------------------------

class TestCompareTrackers:
    def setup_method(self):
        self.planner = LatencyBudgetPlanner()

    def _make_result(self, tracker_name, mean_ms, std_ms):
        """Create a minimal BenchmarkResult-like object for compare_trackers."""
        from types import SimpleNamespace
        p = make_profiling(name=tracker_name, mean_ms=mean_ms, std_ms=std_ms)
        seq = SimpleNamespace(profiling=p)
        br = SimpleNamespace(
            tracker_name=tracker_name,
            sequence_results=[seq],
            peak_memory_mb=128.0,
        )
        return br

    def test_returns_sorted_by_compliance(self):
        fast = self._make_result("Fast", mean_ms=10.0, std_ms=1.0)
        slow = self._make_result("Slow", mean_ms=40.0, std_ms=8.0)
        results = self.planner.compare_trackers([fast, slow], budget_ms=33.0)
        assert results[0].tracker_name == "Fast"
        assert results[1].tracker_name == "Slow"

    def test_names_preserved(self):
        a = self._make_result("TrackerA", mean_ms=15.0, std_ms=3.0)
        b = self._make_result("TrackerB", mean_ms=25.0, std_ms=5.0)
        results = self.planner.compare_trackers([a, b], budget_ms=33.0)
        names = {r.tracker_name for r in results}
        assert names == {"TrackerA", "TrackerB"}

    def test_empty_sequence_results_skipped(self):
        from types import SimpleNamespace
        empty = SimpleNamespace(tracker_name="Empty", sequence_results=[], peak_memory_mb=0.0)
        a = self._make_result("TrackerA", mean_ms=15.0, std_ms=3.0)
        results = self.planner.compare_trackers([a, empty], budget_ms=33.0)
        assert len(results) == 1
        assert results[0].tracker_name == "TrackerA"


# ---------------------------------------------------------------------------
# LatencyBudgetPlanner.to_markdown_table tests
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def setup_method(self):
        self.planner = LatencyBudgetPlanner()

    def test_table_has_header(self):
        p = make_profiling(name="MOSSE")
        result = self.planner.compute_compliance(p, budget_ms=33.0)
        table = self.planner.to_markdown_table([result])
        assert "Tracker" in table
        assert "Compliance" in table

    def test_table_contains_tracker_name(self):
        p = make_profiling(name="MOSSE")
        result = self.planner.compute_compliance(p, budget_ms=33.0)
        table = self.planner.to_markdown_table([result])
        assert "MOSSE" in table

    def test_empty_list_returns_header_only(self):
        table = self.planner.to_markdown_table([])
        assert "Tracker" in table
        lines = table.strip().splitlines()
        assert len(lines) == 2  # header + separator

    def test_multiple_trackers(self):
        results = [
            self.planner.compute_compliance(make_profiling(name="A", mean_ms=10.0, std_ms=2.0), 33.0),
            self.planner.compute_compliance(make_profiling(name="B", mean_ms=20.0, std_ms=4.0), 33.0),
            self.planner.compute_compliance(make_profiling(name="C", mean_ms=35.0, std_ms=7.0), 33.0),
        ]
        table = self.planner.to_markdown_table(results)
        assert "A" in table
        assert "B" in table
        assert "C" in table


# ---------------------------------------------------------------------------
# LatencyBudgetPlanner.deployment_report tests
# ---------------------------------------------------------------------------

class TestDeploymentReport:
    def setup_method(self):
        self.planner = LatencyBudgetPlanner()

    def test_report_contains_fps_targets(self):
        result = self.planner.compute_compliance(make_profiling(name="KCF"), 33.0)
        report = self.planner.deployment_report([result], fps_targets=[30.0, 60.0])
        assert "30" in report
        assert "60" in report

    def test_fast_tracker_marked_compliant(self):
        p = make_profiling(name="MOSSE", mean_ms=5.0, std_ms=1.0, p99_ms=8.0)
        result = self.planner.compute_compliance(p, budget_ms=33.0)
        report = self.planner.deployment_report([result], fps_targets=[30.0])
        assert "MOSSE" in report

    def test_default_fps_targets(self):
        result = self.planner.compute_compliance(make_profiling(name="T"), 33.0)
        report = self.planner.deployment_report([result])
        # Default targets are 15, 24, 30, 60.
        assert "15" in report
        assert "30" in report
        assert "60" in report


# ---------------------------------------------------------------------------
# Integration: imports from package-level __init__
# ---------------------------------------------------------------------------

def test_package_level_import():
    from eovot.metrics import LatencyBudgetPlanner, SLAComplianceResult
    assert callable(LatencyBudgetPlanner)
    assert SLAComplianceResult.__dataclass_fields__  # is a dataclass


def test_end_to_end_30fps_scenario():
    """Fast tracker should comfortably pass 30 FPS SLA; slow tracker should fail."""
    planner = LatencyBudgetPlanner()
    fast = make_profiling(name="MOSSE", mean_ms=8.0, std_ms=1.5, p99_ms=12.0)
    slow = make_profiling(name="CSRT", mean_ms=45.0, std_ms=10.0, p99_ms=70.0)

    fast_result = planner.compute_compliance(fast, budget_ms=33.3)
    slow_result = planner.compute_compliance(slow, budget_ms=33.3)

    assert fast_result.compliance_rate > slow_result.compliance_rate
    assert fast_result.compliance_rate > 0.95

    table = planner.to_markdown_table([fast_result, slow_result])
    assert "MOSSE" in table
    assert "CSRT" in table
