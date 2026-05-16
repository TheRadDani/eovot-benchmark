"""Tests for eovot.metrics.efficiency — EdgeEfficiencyAnalyzer and Pareto analysis."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.metrics.efficiency import EdgeEfficiencyAnalyzer, EdgeEfficiencyScore


# ---------------------------------------------------------------------------
# Helpers — minimal BenchmarkResult mock
# ---------------------------------------------------------------------------

def _make_result(
    tracker_name: str,
    mean_iou: float,
    mean_fps: float,
    peak_memory_mb: float,
    mean_energy_per_frame_mj: Optional[float] = None,
):
    r = MagicMock()
    r.tracker_name = tracker_name
    r.mean_iou = mean_iou
    r.mean_fps = mean_fps
    r.peak_memory_mb = peak_memory_mb
    r.mean_energy_per_frame_mj = mean_energy_per_frame_mj
    return r


# ---------------------------------------------------------------------------
# EdgeEfficiencyAnalyzer construction
# ---------------------------------------------------------------------------

class TestAnalyzerConstruction:
    def test_defaults(self):
        a = EdgeEfficiencyAnalyzer()
        assert a.target_fps == 30.0
        assert a.max_memory_mb == 512.0
        assert a.max_energy_mj is None

    def test_custom_params(self):
        a = EdgeEfficiencyAnalyzer(target_fps=15.0, max_memory_mb=256.0, max_energy_mj=5.0)
        assert a.target_fps == 15.0
        assert a.max_memory_mb == 256.0
        assert a.max_energy_mj == 5.0

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError):
            EdgeEfficiencyAnalyzer(target_fps=0.0)

    def test_invalid_memory_raises(self):
        with pytest.raises(ValueError):
            EdgeEfficiencyAnalyzer(max_memory_mb=-1.0)

    def test_invalid_energy_raises(self):
        with pytest.raises(ValueError):
            EdgeEfficiencyAnalyzer(max_energy_mj=-0.1)

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError):
            EdgeEfficiencyAnalyzer(fps_weight=0.0, memory_weight=0.0)


# ---------------------------------------------------------------------------
# EdgeEfficiencyAnalyzer.score
# ---------------------------------------------------------------------------

class TestScoreMethod:
    def setup_method(self):
        self.analyzer = EdgeEfficiencyAnalyzer(
            target_fps=30.0,
            max_memory_mb=512.0,
        )

    def test_returns_score_object(self):
        r = _make_result("T", 0.5, 60.0, 100.0)
        s = self.analyzer.score(r)
        assert isinstance(s, EdgeEfficiencyScore)

    def test_tracker_name_preserved(self):
        r = _make_result("MOSSE", 0.5, 60.0, 100.0)
        s = self.analyzer.score(r)
        assert s.tracker_name == "MOSSE"

    def test_fps_score_capped_at_one(self):
        r = _make_result("T", 0.5, 1000.0, 100.0)  # way above target
        s = self.analyzer.score(r)
        assert s.fps_score == pytest.approx(1.0)

    def test_fps_score_below_target(self):
        r = _make_result("T", 0.5, 15.0, 100.0)  # half of target=30
        s = self.analyzer.score(r)
        assert s.fps_score == pytest.approx(0.5)

    def test_fps_score_zero_when_zero_fps(self):
        r = _make_result("T", 0.5, 0.0, 100.0)
        s = self.analyzer.score(r)
        assert s.fps_score == pytest.approx(0.0)

    def test_memory_score_full_when_no_memory_used(self):
        r = _make_result("T", 0.5, 30.0, 0.0)
        s = self.analyzer.score(r)
        assert s.memory_score == pytest.approx(1.0)

    def test_memory_score_zero_when_budget_exceeded(self):
        r = _make_result("T", 0.5, 30.0, 600.0)  # > max_memory_mb=512
        s = self.analyzer.score(r)
        assert s.memory_score == pytest.approx(0.0)

    def test_memory_score_half(self):
        r = _make_result("T", 0.5, 30.0, 256.0)  # half of 512
        s = self.analyzer.score(r)
        assert s.memory_score == pytest.approx(0.5)

    def test_composite_in_zero_one(self):
        r = _make_result("T", 0.5, 30.0, 200.0)
        s = self.analyzer.score(r)
        assert 0.0 <= s.composite_score <= 1.0

    def test_perfect_tracker_high_composite(self):
        r = _make_result("T", 1.0, 300.0, 1.0)  # fast and light
        s = self.analyzer.score(r)
        assert s.composite_score > 0.9

    def test_slow_heavy_tracker_low_composite(self):
        r = _make_result("T", 0.7, 1.0, 510.0)  # slow and heavy
        s = self.analyzer.score(r)
        assert s.composite_score < 0.3

    def test_no_energy_has_energy_false(self):
        r = _make_result("T", 0.5, 30.0, 100.0, mean_energy_per_frame_mj=None)
        s = self.analyzer.score(r)
        assert not s.has_energy
        assert s.energy_score == pytest.approx(0.0)

    def test_with_energy_has_energy_true(self):
        analyzer = EdgeEfficiencyAnalyzer(max_energy_mj=10.0)
        r = _make_result("T", 0.5, 30.0, 100.0, mean_energy_per_frame_mj=5.0)
        s = analyzer.score(r)
        assert s.has_energy
        assert s.energy_score == pytest.approx(0.5)

    def test_energy_score_zero_when_budget_exceeded(self):
        analyzer = EdgeEfficiencyAnalyzer(max_energy_mj=4.0)
        r = _make_result("T", 0.5, 30.0, 100.0, mean_energy_per_frame_mj=8.0)
        s = analyzer.score(r)
        assert s.energy_score == pytest.approx(0.0)

    def test_mean_iou_preserved(self):
        r = _make_result("T", 0.63, 30.0, 100.0)
        s = self.analyzer.score(r)
        assert s.mean_iou == pytest.approx(0.63)

    def test_raw_fps_preserved(self):
        r = _make_result("T", 0.5, 47.3, 100.0)
        s = self.analyzer.score(r)
        assert s.mean_fps == pytest.approx(47.3)

    def test_raw_memory_preserved(self):
        r = _make_result("T", 0.5, 30.0, 213.7)
        s = self.analyzer.score(r)
        assert s.peak_memory_mb == pytest.approx(213.7)


# ---------------------------------------------------------------------------
# EdgeEfficiencyAnalyzer.analyze (batch)
# ---------------------------------------------------------------------------

class TestAnalyze:
    def setup_method(self):
        self.analyzer = EdgeEfficiencyAnalyzer()

    def test_returns_list_same_length(self):
        results = [_make_result(f"T{i}", 0.5, 30.0, 100.0) for i in range(4)]
        scores = self.analyzer.analyze(results)
        assert len(scores) == 4

    def test_order_preserved(self):
        names = ["MOSSE", "KCF", "CSRT"]
        results = [_make_result(n, 0.5, 30.0, 100.0) for n in names]
        scores = self.analyzer.analyze(results)
        assert [s.tracker_name for s in scores] == names

    def test_empty_input(self):
        scores = self.analyzer.analyze([])
        assert scores == []


# ---------------------------------------------------------------------------
# EdgeEfficiencyAnalyzer.pareto_frontier
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    def setup_method(self):
        self.analyzer = EdgeEfficiencyAnalyzer()

    def _score(self, name, iou, eff):
        s = MagicMock(spec=EdgeEfficiencyScore)
        s.tracker_name = name
        s.mean_iou = iou
        s.composite_score = eff
        return s

    def test_single_tracker_is_pareto(self):
        scores = [self._score("T", 0.5, 0.6)]
        frontier = self.analyzer.pareto_frontier(scores)
        assert len(frontier) == 1

    def test_dominated_tracker_excluded(self):
        # A dominates B in both dimensions
        a = self._score("A", 0.6, 0.7)
        b = self._score("B", 0.5, 0.6)
        frontier = self.analyzer.pareto_frontier([a, b])
        assert len(frontier) == 1
        assert frontier[0].tracker_name == "A"

    def test_trade_off_both_optimal(self):
        # High accuracy, low efficiency vs low accuracy, high efficiency
        hi_acc = self._score("HiAccuracy", 0.7, 0.3)
        hi_eff = self._score("HiEfficiency", 0.3, 0.9)
        frontier = self.analyzer.pareto_frontier([hi_acc, hi_eff])
        names = {s.tracker_name for s in frontier}
        assert "HiAccuracy" in names
        assert "HiEfficiency" in names

    def test_equal_trackers_both_in_frontier(self):
        a = self._score("A", 0.5, 0.5)
        b = self._score("B", 0.5, 0.5)
        frontier = self.analyzer.pareto_frontier([a, b])
        # Neither dominates the other (equal) → both on frontier
        assert len(frontier) == 2

    def test_frontier_sorted_by_iou_desc(self):
        scores = [
            self._score("Low",  0.3, 0.9),
            self._score("Mid",  0.6, 0.5),
            self._score("High", 0.8, 0.2),
        ]
        frontier = self.analyzer.pareto_frontier(scores)
        ious = [s.mean_iou for s in frontier]
        assert ious == sorted(ious, reverse=True)

    def test_empty_input(self):
        frontier = self.analyzer.pareto_frontier([])
        assert frontier == []

    def test_three_clear_dominated_one_on_frontier(self):
        # A dominates B, C, and D in both dimensions
        a = self._score("A", 0.8, 0.9)
        b = self._score("B", 0.6, 0.7)
        c = self._score("C", 0.4, 0.5)
        d = self._score("D", 0.2, 0.3)
        frontier = self.analyzer.pareto_frontier([a, b, c, d])
        assert len(frontier) == 1
        assert frontier[0].tracker_name == "A"


# ---------------------------------------------------------------------------
# EdgeEfficiencyAnalyzer.ranking_table
# ---------------------------------------------------------------------------

class TestRankingTable:
    def setup_method(self):
        self.analyzer = EdgeEfficiencyAnalyzer()

    def test_contains_tracker_names(self):
        results = [
            _make_result("MOSSE", 0.5, 500.0, 80.0),
            _make_result("KCF",   0.55, 200.0, 100.0),
        ]
        scores = self.analyzer.analyze(results)
        table = self.analyzer.ranking_table(scores)
        assert "MOSSE" in table
        assert "KCF" in table

    def test_sorted_by_efficiency(self):
        # MOSSE is faster and lighter → higher efficiency rank
        results = [
            _make_result("Slow",   0.65, 5.0,  480.0),
            _make_result("Fast",   0.50, 500.0, 80.0),
        ]
        scores = self.analyzer.analyze(results)
        table = self.analyzer.ranking_table(scores)
        # Fast should appear before Slow (higher efficiency)
        assert table.index("Fast") < table.index("Slow")

    def test_has_header_row(self):
        results = [_make_result("T", 0.5, 30.0, 100.0)]
        scores = self.analyzer.analyze(results)
        table = self.analyzer.ranking_table(scores)
        assert "Rank" in table
        assert "Tracker" in table
        assert "mIoU" in table

    def test_energy_columns_when_present(self):
        analyzer = EdgeEfficiencyAnalyzer(max_energy_mj=5.0)
        results = [_make_result("T", 0.5, 30.0, 100.0, mean_energy_per_frame_mj=2.0)]
        scores = analyzer.analyze(results)
        table = analyzer.ranking_table(scores)
        assert "Energy" in table


# ---------------------------------------------------------------------------
# EdgeEfficiencyScore helpers
# ---------------------------------------------------------------------------

class TestEdgeEfficiencyScore:
    def _make_score(self, name="T", iou=0.5, eff=0.7, fps=30.0, mem=100.0, energy=None):
        return EdgeEfficiencyScore(
            tracker_name=name,
            fps_score=0.8,
            memory_score=0.9,
            energy_score=0.5 if energy else 0.0,
            composite_score=eff,
            mean_iou=iou,
            mean_fps=fps,
            peak_memory_mb=mem,
            energy_per_frame_mj=energy,
            has_energy=energy is not None,
        )

    def test_str_contains_tracker_name(self):
        s = self._make_score(name="MOSSE")
        assert "MOSSE" in str(s)

    def test_to_dict_keys_present(self):
        s = self._make_score()
        d = s.to_dict()
        for key in ("tracker", "mean_iou", "composite_score", "fps_score", "memory_score",
                    "mean_fps", "peak_memory_mb"):
            assert key in d

    def test_to_dict_energy_included_when_present(self):
        s = self._make_score(energy=3.0)
        d = s.to_dict()
        assert "energy_per_frame_mj" in d
        assert "energy_score" in d

    def test_to_dict_energy_absent_when_none(self):
        s = self._make_score(energy=None)
        d = s.to_dict()
        assert "energy_per_frame_mj" not in d
