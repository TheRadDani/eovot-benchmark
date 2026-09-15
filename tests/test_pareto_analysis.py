"""Unit tests for eovot.analysis.pareto.

All tests are self-contained and require only numpy + the eovot package;
no dataset downloads or GPU are needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pytest

from eovot.analysis.pareto import ParetoFrontier, TrackerPoint, _dominates_2d, _dominates_3d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_point(
    name: str,
    accuracy: float,
    latency_ms: float,
    memory_mb: float = 100.0,
    fps: float = 0.0,
    dataset: str = "test",
) -> TrackerPoint:
    if fps == 0.0 and latency_ms > 0:
        fps = 1000.0 / latency_ms
    return TrackerPoint(
        tracker_name=name,
        accuracy=accuracy,
        latency_ms=latency_ms,
        memory_mb=memory_mb,
        fps=fps,
        dataset=dataset,
    )


@dataclass
class _FakeBenchmarkResult:
    """Minimal stub that mimics BenchmarkResult.summary()."""
    tracker: str
    dataset: str
    mean_iou: float
    mean_fps: float
    peak_memory_mb: float
    success_auc: Optional[float] = None

    def summary(self) -> Dict:
        d: Dict = {
            "tracker": self.tracker,
            "dataset": self.dataset,
            "mean_iou": self.mean_iou,
            "mean_fps": self.mean_fps,
            "peak_memory_mb": self.peak_memory_mb,
        }
        if self.success_auc is not None:
            d["success_auc"] = self.success_auc
        return d


# ---------------------------------------------------------------------------
# TrackerPoint
# ---------------------------------------------------------------------------

class TestTrackerPoint:
    def test_efficiency_ratio_normal(self):
        p = make_point("A", accuracy=0.6, latency_ms=10.0)
        assert p.efficiency_ratio == pytest.approx(0.06)

    def test_efficiency_ratio_zero_latency(self):
        p = make_point("A", accuracy=0.6, latency_ms=0.0)
        assert p.efficiency_ratio == 0.0

    def test_to_dict_keys(self):
        p = make_point("KCF", accuracy=0.55, latency_ms=8.0, memory_mb=120.0)
        d = p.to_dict()
        assert set(d.keys()) == {"tracker", "dataset", "accuracy", "latency_ms", "fps", "memory_mb", "efficiency_ratio"}
        assert d["tracker"] == "KCF"
        assert d["accuracy"] == pytest.approx(0.55, abs=1e-4)


# ---------------------------------------------------------------------------
# Dominance predicates
# ---------------------------------------------------------------------------

class TestDominance:
    def test_2d_a_dominates_b_strict_both(self):
        a = make_point("A", accuracy=0.7, latency_ms=5.0)
        b = make_point("B", accuracy=0.5, latency_ms=10.0)
        assert _dominates_2d(a, b) is True
        assert _dominates_2d(b, a) is False

    def test_2d_a_dominates_b_equal_accuracy(self):
        a = make_point("A", accuracy=0.5, latency_ms=5.0)
        b = make_point("B", accuracy=0.5, latency_ms=10.0)
        assert _dominates_2d(a, b) is True  # same acc, lower latency
        assert _dominates_2d(b, a) is False

    def test_2d_equal_points_no_dominance(self):
        a = make_point("A", accuracy=0.5, latency_ms=10.0)
        b = make_point("B", accuracy=0.5, latency_ms=10.0)
        assert _dominates_2d(a, b) is False
        assert _dominates_2d(b, a) is False

    def test_2d_trade_off_no_dominance(self):
        a = make_point("A", accuracy=0.7, latency_ms=20.0)  # more accurate but slower
        b = make_point("B", accuracy=0.5, latency_ms=5.0)   # less accurate but faster
        assert _dominates_2d(a, b) is False
        assert _dominates_2d(b, a) is False

    def test_3d_dominance_all_objectives(self):
        a = make_point("A", accuracy=0.7, latency_ms=5.0, memory_mb=50.0)
        b = make_point("B", accuracy=0.5, latency_ms=10.0, memory_mb=100.0)
        assert _dominates_3d(a, b) is True
        assert _dominates_3d(b, a) is False

    def test_3d_no_dominance_different_memory(self):
        # A: better accuracy+latency but worse memory
        a = make_point("A", accuracy=0.7, latency_ms=5.0, memory_mb=200.0)
        b = make_point("B", accuracy=0.5, latency_ms=10.0, memory_mb=50.0)
        assert _dominates_3d(a, b) is False
        assert _dominates_3d(b, a) is False


# ---------------------------------------------------------------------------
# frontier_2d
# ---------------------------------------------------------------------------

class TestFrontier2D:
    def test_all_on_front_classic_tradeoff(self):
        """MOSSE, KCF, CSRT all on the 2D front."""
        mosse = make_point("MOSSE", accuracy=0.30, latency_ms=2.0)
        kcf = make_point("KCF", accuracy=0.50, latency_ms=10.0)
        csrt = make_point("CSRT", accuracy=0.70, latency_ms=30.0)
        frontier = ParetoFrontier([mosse, kcf, csrt])
        f2 = frontier.frontier_2d
        names = {p.tracker_name for p in f2}
        assert names == {"MOSSE", "KCF", "CSRT"}

    def test_dominated_tracker_excluded(self):
        """A tracker that is strictly dominated must not appear on the front."""
        a = make_point("A", accuracy=0.7, latency_ms=5.0)
        b = make_point("B", accuracy=0.5, latency_ms=8.0)  # dominated by A
        c = make_point("C", accuracy=0.4, latency_ms=2.0)  # not dominated
        frontier = ParetoFrontier([a, b, c])
        names = {p.tracker_name for p in frontier.frontier_2d}
        assert "B" not in names
        assert "A" in names
        assert "C" in names

    def test_sorted_by_accuracy_descending(self):
        points = [
            make_point("A", accuracy=0.4, latency_ms=2.0),
            make_point("B", accuracy=0.6, latency_ms=8.0),
            make_point("C", accuracy=0.8, latency_ms=20.0),
        ]
        f2 = ParetoFrontier(points).frontier_2d
        accs = [p.accuracy for p in f2]
        assert accs == sorted(accs, reverse=True)

    def test_single_point_is_on_front(self):
        p = make_point("solo", accuracy=0.5, latency_ms=15.0)
        f2 = ParetoFrontier([p]).frontier_2d
        assert len(f2) == 1 and f2[0].tracker_name == "solo"

    def test_empty_frontier(self):
        frontier = ParetoFrontier([])
        assert frontier.frontier_2d == []

    def test_identical_accuracy_keeps_faster(self):
        """Two trackers with the same accuracy: only the faster should survive."""
        a = make_point("A", accuracy=0.5, latency_ms=5.0)   # faster
        b = make_point("B", accuracy=0.5, latency_ms=15.0)  # dominated by A
        f2 = ParetoFrontier([a, b]).frontier_2d
        names = {p.tracker_name for p in f2}
        assert "B" not in names
        assert "A" in names


# ---------------------------------------------------------------------------
# frontier_3d
# ---------------------------------------------------------------------------

class TestFrontier3D:
    def test_memory_constraint_removes_dominated(self):
        a = make_point("A", accuracy=0.7, latency_ms=5.0, memory_mb=50.0)
        b = make_point("B", accuracy=0.6, latency_ms=6.0, memory_mb=80.0)  # 3D dominated by A
        c = make_point("C", accuracy=0.5, latency_ms=2.0, memory_mb=30.0)  # not dominated
        frontier = ParetoFrontier([a, b, c])
        names = {p.tracker_name for p in frontier.frontier_3d}
        assert "B" not in names
        assert "A" in names
        assert "C" in names

    def test_memory_tradeoff_both_survive(self):
        """One tracker has better accuracy+latency but more memory — neither dominates."""
        a = make_point("A", accuracy=0.7, latency_ms=5.0, memory_mb=200.0)
        b = make_point("B", accuracy=0.5, latency_ms=10.0, memory_mb=50.0)
        names = {p.tracker_name for p in ParetoFrontier([a, b]).frontier_3d}
        assert names == {"A", "B"}

    def test_frontier_3d_subset_or_equal_2d(self):
        """Every 3D-front tracker is also on the 2D front (memory only adds constraints)."""
        points = [
            make_point("A", accuracy=0.7, latency_ms=5.0, memory_mb=50.0),
            make_point("B", accuracy=0.5, latency_ms=3.0, memory_mb=30.0),
            make_point("C", accuracy=0.4, latency_ms=2.0, memory_mb=200.0),
        ]
        frontier = ParetoFrontier(points)
        f2_names = {p.tracker_name for p in frontier.frontier_2d}
        for p in frontier.frontier_3d:
            assert p.tracker_name in f2_names


# ---------------------------------------------------------------------------
# Constraint queries
# ---------------------------------------------------------------------------

class TestConstraintQueries:
    def _make_frontier(self):
        return ParetoFrontier([
            make_point("MOSSE", accuracy=0.30, latency_ms=2.0, memory_mb=30.0),
            make_point("KCF",   accuracy=0.55, latency_ms=10.0, memory_mb=80.0),
            make_point("CSRT",  accuracy=0.72, latency_ms=35.0, memory_mb=150.0),
        ])

    def test_best_under_latency_finds_correct(self):
        frontier = self._make_frontier()
        best = frontier.best_under_latency(15.0)
        assert best is not None
        assert best.tracker_name == "KCF"  # CSRT (35ms) excluded; KCF beats MOSSE in acc

    def test_best_under_latency_very_tight(self):
        frontier = self._make_frontier()
        best = frontier.best_under_latency(3.0)
        assert best is not None
        assert best.tracker_name == "MOSSE"

    def test_best_under_latency_none_qualifies(self):
        frontier = self._make_frontier()
        assert frontier.best_under_latency(0.5) is None

    def test_best_under_memory_finds_correct(self):
        frontier = self._make_frontier()
        best = frontier.best_under_memory(100.0)
        assert best is not None
        assert best.tracker_name == "KCF"  # CSRT (150MB) excluded; KCF beats MOSSE in acc

    def test_best_under_memory_none_qualifies(self):
        frontier = self._make_frontier()
        assert frontier.best_under_memory(5.0) is None

    def test_best_under_constraints_combined(self):
        frontier = self._make_frontier()
        best = frontier.best_under_constraints(max_latency_ms=20.0, max_memory_mb=100.0)
        assert best is not None
        assert best.tracker_name == "KCF"

    def test_best_under_constraints_no_latency(self):
        """Only memory constraint."""
        frontier = self._make_frontier()
        best = frontier.best_under_constraints(max_memory_mb=100.0)
        assert best.tracker_name == "KCF"

    def test_best_under_constraints_none(self):
        frontier = self._make_frontier()
        assert frontier.best_under_constraints(max_latency_ms=1.0, max_memory_mb=10.0) is None


# ---------------------------------------------------------------------------
# Hypervolume indicator
# ---------------------------------------------------------------------------

class TestHypervolume2D:
    def test_empty_front_returns_zero(self):
        assert ParetoFrontier([]).hypervolume_2d() == 0.0

    def test_single_point(self):
        p = make_point("A", accuracy=0.5, latency_ms=10.0)
        hv = ParetoFrontier([p]).hypervolume_2d(ref_accuracy=0.0, ref_latency=200.0)
        # contribution = (0.5 - 0) * (1/10 - 1/200) = 0.5 * (0.1 - 0.005) = 0.5 * 0.095
        expected = 0.5 * (1.0 / 10.0 - 1.0 / 200.0)
        assert hv == pytest.approx(expected, rel=1e-6)

    def test_multi_point_matches_manual(self):
        """Three-tracker example with manually computed expected HV."""
        mosse = make_point("MOSSE", accuracy=0.30, latency_ms=2.0)
        kcf = make_point("KCF", accuracy=0.50, latency_ms=10.0)
        csrt = make_point("CSRT", accuracy=0.70, latency_ms=30.0)
        frontier = ParetoFrontier([mosse, kcf, csrt])

        ref_acc = 0.0
        ref_lat = 200.0
        ref_speed = 1.0 / ref_lat

        # Manual calculation (sorted by accuracy ascending):
        # MOSSE: (0.30 - 0) * (0.5 - ref_speed)
        # KCF:   (0.50 - 0.30) * (0.1 - ref_speed)
        # CSRT:  (0.70 - 0.50) * (1/30 - ref_speed)
        expected = (
            (0.30 - ref_acc) * (1.0/2.0 - ref_speed)
            + (0.50 - 0.30) * (1.0/10.0 - ref_speed)
            + (0.70 - 0.50) * (1.0/30.0 - ref_speed)
        )
        assert frontier.hypervolume_2d(ref_acc, ref_lat) == pytest.approx(expected, rel=1e-6)

    def test_hv_larger_better_front_scores_higher(self):
        """Replacing a weak tracker with a stronger one must increase HV."""
        weak = [
            make_point("A", accuracy=0.3, latency_ms=5.0),
            make_point("B", accuracy=0.5, latency_ms=20.0),
        ]
        strong = [
            make_point("A", accuracy=0.3, latency_ms=5.0),
            make_point("B+", accuracy=0.6, latency_ms=20.0),  # B improved
        ]
        hv_weak = ParetoFrontier(weak).hypervolume_2d()
        hv_strong = ParetoFrontier(strong).hypervolume_2d()
        assert hv_strong > hv_weak

    def test_point_outside_ref_excluded(self):
        """A front point worse than the reference must not contribute."""
        # Point with latency_ms > ref_latency
        slow = make_point("A", accuracy=0.5, latency_ms=500.0)  # > ref 200ms
        assert ParetoFrontier([slow]).hypervolume_2d(ref_accuracy=0.0, ref_latency=200.0) == 0.0

    def test_hv_nonnegative(self):
        points = [make_point(f"T{i}", accuracy=0.1*i, latency_ms=5.0*i+1) for i in range(1, 6)]
        hv = ParetoFrontier(points).hypervolume_2d()
        assert hv >= 0.0


# ---------------------------------------------------------------------------
# from_benchmark_results
# ---------------------------------------------------------------------------

class TestFromBenchmarkResults:
    def test_basic_construction(self):
        results = [
            _FakeBenchmarkResult("KCF", "GOT10k", mean_iou=0.50, mean_fps=100.0, peak_memory_mb=80.0),
            _FakeBenchmarkResult("MOSSE", "GOT10k", mean_iou=0.35, mean_fps=500.0, peak_memory_mb=30.0),
        ]
        frontier = ParetoFrontier.from_benchmark_results(results, accuracy_key="mean_iou")
        assert len(frontier) == 2

    def test_uses_success_auc_when_present(self):
        results = [
            _FakeBenchmarkResult(
                "CSRT", "GOT10k", mean_iou=0.60, mean_fps=30.0,
                peak_memory_mb=150.0, success_auc=0.72,
            ),
        ]
        frontier = ParetoFrontier.from_benchmark_results(results, accuracy_key="success_auc")
        p = frontier.all_points[0]
        assert p.accuracy == pytest.approx(0.72, abs=1e-4)

    def test_falls_back_to_mean_iou_when_key_missing(self):
        results = [
            _FakeBenchmarkResult("KCF", "GOT10k", mean_iou=0.50, mean_fps=100.0, peak_memory_mb=80.0),
        ]
        frontier = ParetoFrontier.from_benchmark_results(results, accuracy_key="success_auc")
        p = frontier.all_points[0]
        assert p.accuracy == pytest.approx(0.50, abs=1e-4)

    def test_fps_converts_to_latency(self):
        results = [
            _FakeBenchmarkResult("MOSSE", "GOT10k", mean_iou=0.35, mean_fps=200.0, peak_memory_mb=30.0),
        ]
        frontier = ParetoFrontier.from_benchmark_results(results)
        p = frontier.all_points[0]
        assert p.latency_ms == pytest.approx(5.0, rel=1e-4)  # 1000/200


# ---------------------------------------------------------------------------
# summary_table and to_markdown
# ---------------------------------------------------------------------------

class TestReporting:
    def _make_three_tracker_frontier(self):
        return ParetoFrontier([
            make_point("MOSSE", accuracy=0.30, latency_ms=2.0, memory_mb=30.0),
            make_point("KCF",   accuracy=0.55, latency_ms=10.0, memory_mb=80.0),
            make_point("CSRT",  accuracy=0.72, latency_ms=35.0, memory_mb=150.0),
        ])

    def test_summary_table_length(self):
        rows = self._make_three_tracker_frontier().summary_table()
        assert len(rows) == 3

    def test_summary_table_has_required_keys(self):
        for row in self._make_three_tracker_frontier().summary_table():
            assert "on_frontier_2d" in row
            assert "on_frontier_3d" in row

    def test_markdown_contains_header(self):
        md = self._make_three_tracker_frontier().to_markdown()
        assert "Tracker" in md
        assert "Latency" in md
        assert "Accuracy" in md

    def test_markdown_contains_all_tracker_names(self):
        md = self._make_three_tracker_frontier().to_markdown()
        for name in ["MOSSE", "KCF", "CSRT"]:
            assert name in md

    def test_repr(self):
        f = self._make_three_tracker_frontier()
        r = repr(f)
        assert "ParetoFrontier" in r
        assert "n=3" in r
