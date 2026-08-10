"""Tests for eovot.analysis.latency_budget."""

import pytest
import numpy as np

from eovot.analysis.latency_budget import (
    LatencyBudgetAnalyzer,
    LatencyBudgetEntry,
    LatencyBudgetReport,
)
from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=30, motion="linear", seed=0)


@pytest.fixture(scope="module")
def mosse_result(synthetic_dataset):
    engine = BenchmarkEngine(verbose=False)
    return engine.run(MOSSETracker(), synthetic_dataset, dataset_name="Synthetic")


@pytest.fixture(scope="module")
def kcf_result(synthetic_dataset):
    engine = BenchmarkEngine(verbose=False)
    return engine.run(KCFTracker(), synthetic_dataset, dataset_name="Synthetic")


# ---------------------------------------------------------------------------
# analyze_single
# ---------------------------------------------------------------------------

class TestAnalyzeSingle:
    def test_returns_entry(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        entry = analyzer.analyze_single(mosse_result, target_fps=30.0)
        assert isinstance(entry, LatencyBudgetEntry)

    def test_budget_ms(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        entry = analyzer.analyze_single(mosse_result, target_fps=30.0)
        assert abs(entry.budget_ms - 1000.0 / 30.0) < 1e-6

    def test_tracker_name_preserved(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        entry = analyzer.analyze_single(mosse_result, target_fps=30.0)
        assert entry.tracker_name == mosse_result.tracker_name

    def test_budget_utilization_positive(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        entry = analyzer.analyze_single(mosse_result, target_fps=30.0)
        assert entry.budget_utilization > 0.0

    def test_p99_gte_mean(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        entry = analyzer.analyze_single(mosse_result, target_fps=30.0)
        assert entry.p99_latency_ms >= entry.mean_latency_ms

    def test_compliance_at_very_low_fps(self, mosse_result):
        """A tracker should be compliant at 1 FPS (budget = 1000 ms)."""
        analyzer = LatencyBudgetAnalyzer()
        entry = analyzer.analyze_single(mosse_result, target_fps=1.0)
        assert entry.real_time_compliant is True
        assert entry.safety_margin_ms > 0

    def test_noncompliance_at_very_high_fps(self, mosse_result):
        """At 10000 FPS budget = 0.1 ms, even the fastest tracker should fail."""
        analyzer = LatencyBudgetAnalyzer()
        entry = analyzer.analyze_single(mosse_result, target_fps=10_000.0)
        assert entry.real_time_compliant is False
        assert entry.safety_margin_ms < 0

    def test_invalid_target_fps(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        with pytest.raises(ValueError, match="target_fps must be > 0"):
            analyzer.analyze_single(mosse_result, target_fps=0.0)

    def test_safety_margin_consistent(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        entry = analyzer.analyze_single(mosse_result, target_fps=30.0)
        expected_margin = entry.budget_ms - entry.p99_latency_ms
        assert abs(entry.safety_margin_ms - expected_margin) < 1e-2

    def test_to_dict_keys(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        entry = analyzer.analyze_single(mosse_result, target_fps=30.0)
        d = entry.to_dict()
        required = {
            "tracker", "dataset", "target_fps", "budget_ms",
            "mean_latency_ms", "p95_latency_ms", "p99_latency_ms",
            "budget_utilization", "safety_margin_ms", "real_time_compliant",
            "observed_fps",
        }
        assert required.issubset(d.keys())

    def test_str_contains_pass_or_fail(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        entry = analyzer.analyze_single(mosse_result, target_fps=30.0)
        s = str(entry)
        assert "PASS" in s or "FAIL" in s

    def test_p95_compliance_percentile(self, mosse_result):
        """With compliance_percentile=95, p95 drives the compliance decision."""
        analyzer = LatencyBudgetAnalyzer(compliance_percentile=95)
        entry = analyzer.analyze_single(mosse_result, target_fps=1.0)
        assert entry.real_time_compliant is True

    def test_invalid_compliance_percentile(self):
        with pytest.raises(ValueError):
            LatencyBudgetAnalyzer(compliance_percentile=50)


# ---------------------------------------------------------------------------
# analyze (single tracker, multiple FPS targets)
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_report(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.analyze(mosse_result, [15.0, 30.0, 60.0])
        assert isinstance(report, LatencyBudgetReport)

    def test_entry_count(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.analyze(mosse_result, [10.0, 20.0, 30.0])
        assert len(report.entries) == 3

    def test_fps_targets_in_entries(self, mosse_result):
        targets = [15.0, 30.0, 60.0]
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.analyze(mosse_result, targets)
        found = {e.target_fps for e in report.entries}
        assert found == set(targets)

    def test_compliant_at_low_fps(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.analyze(mosse_result, [1.0])
        assert report.entries[0].real_time_compliant is True


# ---------------------------------------------------------------------------
# compare_trackers
# ---------------------------------------------------------------------------

class TestCompareTrackers:
    def test_entry_count(self, mosse_result, kcf_result):
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.compare_trackers([mosse_result, kcf_result], [30.0, 60.0])
        # 2 trackers × 2 FPS targets = 4 entries
        assert len(report.entries) == 4

    def test_trackers_list(self, mosse_result, kcf_result):
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.compare_trackers([mosse_result, kcf_result], [30.0])
        assert set(report.trackers) == {"MOSSE", "KCF"}

    def test_compliant_trackers_at_1_fps(self, mosse_result, kcf_result):
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.compare_trackers([mosse_result, kcf_result], [1.0])
        compliant = report.compliant_trackers(1.0)
        assert "MOSSE" in compliant
        assert "KCF" in compliant

    def test_fastest_compliant_returns_entry(self, mosse_result, kcf_result):
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.compare_trackers([mosse_result, kcf_result], [1.0])
        entry = report.fastest_compliant(1.0)
        assert entry is not None
        assert isinstance(entry, LatencyBudgetEntry)

    def test_fastest_compliant_none_at_high_fps(self, mosse_result, kcf_result):
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.compare_trackers([mosse_result, kcf_result], [10_000.0])
        entry = report.fastest_compliant(10_000.0)
        assert entry is None


# ---------------------------------------------------------------------------
# report formatting
# ---------------------------------------------------------------------------

class TestReportFormatting:
    def test_to_markdown_nonempty(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.analyze(mosse_result, [30.0])
        md = report.to_markdown()
        assert "Tracker" in md
        assert "MOSSE" in md

    def test_to_markdown_empty(self):
        report = LatencyBudgetReport()
        assert "No" in report.to_markdown()

    def test_compliance_matrix_contains_check(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.analyze(mosse_result, [1.0, 10_000.0])
        matrix = report.to_compliance_matrix()
        # At 1 FPS tracker passes, at 10000 FPS it fails
        assert "✓" in matrix
        assert "✗" in matrix

    def test_compliance_matrix_empty(self):
        report = LatencyBudgetReport()
        assert "No entries" in report.to_compliance_matrix()


# ---------------------------------------------------------------------------
# recommend_fps_budget
# ---------------------------------------------------------------------------

class TestRecommendFpsBudget:
    def test_returns_dict(self, mosse_result, kcf_result):
        analyzer = LatencyBudgetAnalyzer()
        rec = analyzer.recommend_fps_budget([mosse_result, kcf_result])
        assert "MOSSE" in rec
        assert "KCF" in rec

    def test_compliant_at_low_fps(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        rec = analyzer.recommend_fps_budget([mosse_result], candidate_fps=[1.0, 2.0])
        # Should be compliant at both 1 FPS and 2 FPS
        assert rec["MOSSE"] is not None
        assert rec["MOSSE"] >= 1.0

    def test_none_for_impossible_target(self, mosse_result):
        """When the only candidate is 10 000 FPS, should return None."""
        analyzer = LatencyBudgetAnalyzer()
        rec = analyzer.recommend_fps_budget([mosse_result], candidate_fps=[10_000.0])
        assert rec["MOSSE"] is None

    def test_min_feasible_fps_method(self, mosse_result):
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.compare_trackers([mosse_result], [1.0, 5.0, 10_000.0])
        fps = report.min_feasible_fps("MOSSE")
        # min_feasible returns the HIGHEST compliant fps
        assert fps is not None
        assert fps >= 1.0
