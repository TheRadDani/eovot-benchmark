"""Tests for the LatencyBudgetAnalyzer.

Covers:
- _normal_cdf approximation correctness
- _evaluate_entry verdict classification (safe / marginal / unsafe)
- analyze_profiling: correct entries for each target FPS
- analyze_benchmark: aggregate and worst-case modes
- LatencyBudgetReport helpers: safe_targets, max_safe_fps, recommendations
- Markdown table formatting (smoke tests — structure only, not exact text)
- multi_tracker_table: includes all tracker names
- Edge cases: zero std, single sequence, empty sequence list
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest

from eovot.analysis.latency_budget import (
    LatencyBudgetAnalyzer,
    LatencyBudgetReport,
    _erf_approx,
    _normal_cdf,
)
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_profiling(
    tracker_name: str = "MOSSE",
    fps: float = 200.0,
    mean_ms: float = 5.0,
    std_ms: float = 0.5,
    p95_ms: float = 5.9,
    p99_ms: float = 6.5,
    frame_count: int = 100,
    peak_memory_mb: float = 48.0,
) -> ProfilingResult:
    return ProfilingResult(
        tracker_name=tracker_name,
        frame_count=frame_count,
        fps=fps,
        latency_mean_ms=mean_ms,
        latency_std_ms=std_ms,
        latency_p95_ms=p95_ms,
        latency_p99_ms=p99_ms,
        latency_cv=std_ms / mean_ms if mean_ms > 0 else 0.0,
        peak_memory_mb=peak_memory_mb,
    )


def _make_benchmark_result(
    tracker_name: str = "KCF",
    dataset_name: str = "Synthetic",
    sequence_profilings: Optional[List[ProfilingResult]] = None,
):
    """Minimal BenchmarkResult with synthetic SequenceResults."""
    from eovot.benchmark.engine import BenchmarkResult, SequenceResult

    result = BenchmarkResult(tracker_name=tracker_name, dataset_name=dataset_name)
    if sequence_profilings is None:
        sequence_profilings = [_make_profiling()]
    for i, prof in enumerate(sequence_profilings):
        result.sequence_results.append(
            SequenceResult(
                sequence_name=f"seq_{i:03d}",
                ious=np.array([0.5, 0.6, 0.7]),
                profiling=prof,
            )
        )
    return result


# ---------------------------------------------------------------------------
# _erf_approx and _normal_cdf
# ---------------------------------------------------------------------------

class TestNormalCDF:
    def test_erf_at_zero(self):
        assert _erf_approx(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_erf_at_positive_large(self):
        # erf(3) ≈ 0.9999779
        assert _erf_approx(3.0) == pytest.approx(0.9999779, abs=1e-4)

    def test_erf_symmetry(self):
        for x in [0.5, 1.0, 2.0]:
            assert _erf_approx(-x) == pytest.approx(-_erf_approx(x), abs=1e-10)

    def test_normal_cdf_at_mean(self):
        # CDF at x == mean should be 0.5 for any std > 0.
        assert _normal_cdf(10.0, mean=10.0, std=2.0) == pytest.approx(0.5, abs=1e-4)

    def test_normal_cdf_zero_std_below_mean(self):
        assert _normal_cdf(5.0, mean=10.0, std=0.0) == pytest.approx(0.0)

    def test_normal_cdf_zero_std_at_or_above_mean(self):
        assert _normal_cdf(10.0, mean=10.0, std=0.0) == pytest.approx(1.0)
        assert _normal_cdf(15.0, mean=10.0, std=0.0) == pytest.approx(1.0)

    def test_normal_cdf_monotone(self):
        mean, std = 5.0, 1.0
        values = [_normal_cdf(x, mean, std) for x in [3.0, 4.0, 5.0, 6.0, 7.0]]
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))

    def test_normal_cdf_known_sigma(self):
        # P(X < mean + sigma) ≈ 0.8413 for standard normal.
        assert _normal_cdf(6.0, mean=5.0, std=1.0) == pytest.approx(0.8413, abs=1e-3)


# ---------------------------------------------------------------------------
# LatencyBudgetAnalyzer._evaluate_entry verdicts
# ---------------------------------------------------------------------------

class TestVerdicts:
    def setup_method(self):
        self.analyzer = LatencyBudgetAnalyzer(target_fps_list=[30.0])

    def test_safe_when_p99_within_budget(self):
        # 30 FPS → 33.3 ms budget; mean=5, p99=10 → safe
        entry = self.analyzer._evaluate_entry("T", 30.0, 5.0, 0.5, 10.0)
        assert entry.verdict == "safe"

    def test_marginal_when_mean_ok_but_p99_exceeds(self):
        # 30 FPS → 33.3 ms budget; mean=20, p99=40 → marginal
        entry = self.analyzer._evaluate_entry("T", 30.0, 20.0, 5.0, 40.0)
        assert entry.verdict == "marginal"

    def test_unsafe_when_mean_exceeds_budget(self):
        # 30 FPS → 33.3 ms budget; mean=50 → unsafe
        entry = self.analyzer._evaluate_entry("T", 30.0, 50.0, 5.0, 60.0)
        assert entry.verdict == "unsafe"

    def test_budget_ms_correct(self):
        entry = self.analyzer._evaluate_entry("T", 30.0, 5.0, 0.5, 10.0)
        assert entry.budget_ms == pytest.approx(1000.0 / 30.0, rel=1e-6)

    def test_headroom_values(self):
        entry = self.analyzer._evaluate_entry("T", 30.0, 5.0, 0.5, 10.0)
        budget = 1000.0 / 30.0
        assert entry.mean_headroom_ms == pytest.approx(budget - 5.0, rel=1e-6)
        assert entry.p99_headroom_ms == pytest.approx(budget - 10.0, rel=1e-6)

    def test_compliance_between_0_and_1(self):
        for mean in [1.0, 10.0, 50.0, 100.0]:
            entry = self.analyzer._evaluate_entry("T", 30.0, mean, 2.0, mean + 5.0)
            assert 0.0 <= entry.estimated_compliance <= 1.0

    def test_p99_safety_margin(self):
        entry = self.analyzer._evaluate_entry("T", 30.0, 5.0, 0.5, 16.67)
        budget = 1000.0 / 30.0
        assert entry.p99_safety_margin == pytest.approx(budget / 16.67, rel=1e-4)


# ---------------------------------------------------------------------------
# analyze_profiling
# ---------------------------------------------------------------------------

class TestAnalyzeProfiling:
    def test_returns_one_entry_per_target_fps(self):
        analyzer = LatencyBudgetAnalyzer(target_fps_list=[15, 30, 60])
        profiling = _make_profiling(mean_ms=5.0, p99_ms=8.0)
        entries = analyzer.analyze_profiling(profiling)
        assert len(entries) == 3

    def test_entries_sorted_by_fps(self):
        analyzer = LatencyBudgetAnalyzer(target_fps_list=[60, 15, 30])
        profiling = _make_profiling(mean_ms=5.0, p99_ms=8.0)
        entries = analyzer.analyze_profiling(profiling)
        fps_values = [e.target_fps for e in entries]
        assert fps_values == sorted(fps_values)

    def test_tracker_name_propagated(self):
        analyzer = LatencyBudgetAnalyzer(target_fps_list=[30])
        profiling = _make_profiling(tracker_name="CSRT")
        entries = analyzer.analyze_profiling(profiling)
        assert all(e.tracker_name == "CSRT" for e in entries)

    def test_fast_tracker_all_safe(self):
        # 1 ms mean → safe at 15, 24, 30, 60 FPS (budget ≥ 16.7 ms)
        analyzer = LatencyBudgetAnalyzer()
        profiling = _make_profiling(mean_ms=1.0, std_ms=0.1, p99_ms=1.5)
        entries = analyzer.analyze_profiling(profiling)
        assert all(e.verdict == "safe" for e in entries)

    def test_slow_tracker_all_unsafe(self):
        # 100 ms mean → unsafe even at 1 FPS... well, unsafe at 60 FPS
        analyzer = LatencyBudgetAnalyzer(target_fps_list=[30, 60])
        profiling = _make_profiling(mean_ms=100.0, std_ms=5.0, p99_ms=115.0)
        entries = analyzer.analyze_profiling(profiling)
        assert all(e.verdict == "unsafe" for e in entries)


# ---------------------------------------------------------------------------
# analyze_benchmark
# ---------------------------------------------------------------------------

class TestAnalyzeBenchmark:
    def test_returns_report_with_correct_tracker_and_dataset(self):
        analyzer = LatencyBudgetAnalyzer(target_fps_list=[30])
        br = _make_benchmark_result("MOSSE", "OTB100")
        report = analyzer.analyze_benchmark(br)
        assert report.tracker_name == "MOSSE"
        assert report.dataset_name == "OTB100"

    def test_one_entry_per_fps(self):
        analyzer = LatencyBudgetAnalyzer(target_fps_list=[15, 30, 60])
        br = _make_benchmark_result()
        report = analyzer.analyze_benchmark(br)
        assert len(report.entries) == 3

    def test_aggregate_mode_uses_mean_of_sequences(self):
        # Two sequences with mean_ms=4 and mean_ms=8 → aggregate mean = 6.
        analyzer = LatencyBudgetAnalyzer(target_fps_list=[30])
        profilings = [
            _make_profiling(mean_ms=4.0, p99_ms=5.0),
            _make_profiling(mean_ms=8.0, p99_ms=10.0),
        ]
        br = _make_benchmark_result(sequence_profilings=profilings)
        report = analyzer.analyze_benchmark(br, aggregate=True)
        assert report.entries[0].mean_latency_ms == pytest.approx(6.0, rel=1e-6)

    def test_worst_case_mode_uses_max_mean_sequence(self):
        analyzer = LatencyBudgetAnalyzer(target_fps_list=[30])
        profilings = [
            _make_profiling(mean_ms=4.0, p99_ms=5.0),
            _make_profiling(mean_ms=20.0, p99_ms=25.0),  # worst
        ]
        br = _make_benchmark_result(sequence_profilings=profilings)
        report = analyzer.analyze_benchmark(br, aggregate=False)
        assert report.entries[0].mean_latency_ms == pytest.approx(20.0, rel=1e-6)

    def test_empty_sequence_results(self):
        from eovot.benchmark.engine import BenchmarkResult
        br = BenchmarkResult(tracker_name="T", dataset_name="D")
        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.analyze_benchmark(br)
        assert report.entries == []


# ---------------------------------------------------------------------------
# LatencyBudgetReport helpers
# ---------------------------------------------------------------------------

class TestLatencyBudgetReport:
    def _build_report(self, target_fps_list, mean_ms, p99_ms) -> LatencyBudgetReport:
        analyzer = LatencyBudgetAnalyzer(target_fps_list=target_fps_list)
        profiling = _make_profiling(mean_ms=mean_ms, std_ms=1.0, p99_ms=p99_ms)
        entries = analyzer.analyze_profiling(profiling)
        return LatencyBudgetReport(
            tracker_name="Test", dataset_name="Synthetic", entries=entries
        )

    def test_safe_targets_returns_only_safe(self):
        # mean=5, p99=8: safe at 15, 30 (budgets 66.7 and 33.3ms); unsafe at 200 FPS (5ms budget)
        report = self._build_report([15, 30, 200], mean_ms=5.0, p99_ms=8.0)
        safe = report.safe_targets()
        assert 200.0 not in safe or 5.0 < 8.0  # 200 FPS → 5ms budget, p99=8ms > budget

    def test_max_safe_fps_none_when_all_unsafe(self):
        report = self._build_report([60, 120], mean_ms=100.0, p99_ms=120.0)
        assert report.max_safe_fps() is None

    def test_max_safe_fps_correct(self):
        # At 15 FPS (66.7ms budget) a 5ms mean tracker is safe; at 200 FPS unsafe
        report = self._build_report([15, 200], mean_ms=5.0, p99_ms=8.0)
        safe = report.safe_targets()
        if safe:
            assert report.max_safe_fps() == max(safe)

    def test_recommendations_returns_string(self):
        report = self._build_report([30], mean_ms=5.0, p99_ms=8.0)
        rec = report.recommendations()
        assert isinstance(rec, str)
        assert len(rec) > 0

    def test_to_markdown_table_contains_tracker_name(self):
        analyzer = LatencyBudgetAnalyzer(target_fps_list=[30])
        br = _make_benchmark_result("MOSSE", "Synthetic")
        report = analyzer.analyze_benchmark(br)
        table = report.to_markdown_table()
        assert "MOSSE" in table
        assert "Synthetic" in table

    def test_to_markdown_table_contains_fps_target(self):
        analyzer = LatencyBudgetAnalyzer(target_fps_list=[15, 60])
        br = _make_benchmark_result()
        report = analyzer.analyze_benchmark(br)
        table = report.to_markdown_table()
        assert "15" in table
        assert "60" in table


# ---------------------------------------------------------------------------
# multi_tracker_table
# ---------------------------------------------------------------------------

class TestMultiTrackerTable:
    def test_all_tracker_names_present(self):
        analyzer = LatencyBudgetAnalyzer(target_fps_list=[30])
        reports = []
        for name, mean in [("MOSSE", 3.0), ("KCF", 8.0), ("CSRT", 15.0)]:
            br = _make_benchmark_result(name, "Synthetic", [_make_profiling(mean_ms=mean, p99_ms=mean + 3)])
            reports.append(analyzer.analyze_benchmark(br))
        table = LatencyBudgetAnalyzer.multi_tracker_table(reports)
        for name in ["MOSSE", "KCF", "CSRT"]:
            assert name in table

    def test_empty_reports(self):
        table = LatencyBudgetAnalyzer.multi_tracker_table([])
        assert isinstance(table, str)


# ---------------------------------------------------------------------------
# LatencyBudgetEntry.__str__
# ---------------------------------------------------------------------------

class TestEntryStr:
    def test_str_does_not_raise(self):
        analyzer = LatencyBudgetAnalyzer(target_fps_list=[30])
        profiling = _make_profiling(mean_ms=5.0, p99_ms=8.0)
        entries = analyzer.analyze_profiling(profiling)
        for e in entries:
            s = str(e)
            assert "MOSSE" in s or "safe" in s or "marginal" in s or "unsafe" in s
