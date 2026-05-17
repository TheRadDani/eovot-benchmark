"""Unit tests for eovot.metrics.efficiency."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from eovot.metrics.efficiency import EfficiencyEntry, EfficiencyMetricsEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    tracker_name: str,
    mean_iou: float,
    fps: float,
    peak_memory_mb: float,
    ees: float = 0.0,
    on_pareto_front: bool = False,
) -> EfficiencyEntry:
    return EfficiencyEntry(
        tracker_name=tracker_name,
        dataset_name="TestDataset",
        mean_iou=mean_iou,
        fps=fps,
        peak_memory_mb=peak_memory_mb,
        ees=ees,
        on_pareto_front=on_pareto_front,
    )


def _make_mock_result(tracker_name: str, mean_iou: float, fps: float, memory_mb: float):
    r = MagicMock()
    r.tracker_name = tracker_name
    r.dataset_name = "TestDataset"
    r.mean_iou = mean_iou
    r.mean_fps = fps
    r.peak_memory_mb = memory_mb
    return r


# ---------------------------------------------------------------------------
# EfficiencyMetricsEngine
# ---------------------------------------------------------------------------

class TestEfficiencyMetricsEngine:

    def setup_method(self):
        self.engine = EfficiencyMetricsEngine(memory_budget_mb=512.0)

    def test_invalid_memory_budget_raises(self):
        with pytest.raises(ValueError):
            EfficiencyMetricsEngine(memory_budget_mb=0.0)
        with pytest.raises(ValueError):
            EfficiencyMetricsEngine(memory_budget_mb=-100.0)

    def test_ees_formula(self):
        ees = self.engine.edge_efficiency_score(mean_iou=1.0, fps=100.0, peak_memory_mb=100.0)
        expected = 1.0 * math.log1p(100.0) / (1.0 + 100.0 / 512.0)
        assert ees == pytest.approx(expected, rel=1e-6)

    def test_ees_zero_iou_returns_zero(self):
        assert self.engine.edge_efficiency_score(0.0, 500.0, 50.0) == pytest.approx(0.0)

    def test_ees_zero_fps_returns_zero(self):
        assert self.engine.edge_efficiency_score(0.8, 0.0, 100.0) == pytest.approx(0.0)

    def test_ees_negative_fps_returns_zero(self):
        assert self.engine.edge_efficiency_score(0.8, -1.0, 100.0) == pytest.approx(0.0)

    def test_ees_faster_tracker_higher_score(self):
        fast = self.engine.edge_efficiency_score(1.0, 200.0, 100.0)
        slow = self.engine.edge_efficiency_score(1.0, 10.0, 100.0)
        assert fast > slow

    def test_ees_leaner_memory_higher_score(self):
        lean = self.engine.edge_efficiency_score(1.0, 100.0, 50.0)
        heavy = self.engine.edge_efficiency_score(1.0, 100.0, 1000.0)
        assert lean > heavy

    def test_ees_higher_iou_higher_score(self):
        accurate = self.engine.edge_efficiency_score(0.9, 100.0, 100.0)
        inaccurate = self.engine.edge_efficiency_score(0.3, 100.0, 100.0)
        assert accurate > inaccurate

    def test_ees_is_non_negative(self):
        for iou in [0.0, 0.5, 1.0]:
            for fps in [1.0, 100.0, 500.0]:
                assert self.engine.edge_efficiency_score(iou, fps, 200.0) >= 0.0

    # ------------------------------------------------------------------
    # Pareto front
    # ------------------------------------------------------------------

    def test_pareto_single_entry_is_on_front(self):
        entries = [_make_entry("A", 0.6, 100.0, 100.0, ees=1.0)]
        result = self.engine.compute_pareto_front(entries)
        assert result[0].on_pareto_front is True

    def test_pareto_clear_dominance(self):
        """A dominates B in both objectives → B is not on the front."""
        entries = [
            _make_entry("A", mean_iou=0.8, fps=200.0, peak_memory_mb=100.0, ees=2.0),
            _make_entry("B", mean_iou=0.5, fps=100.0, peak_memory_mb=200.0, ees=1.0),
        ]
        self.engine.compute_pareto_front(entries)
        a = next(e for e in entries if e.tracker_name == "A")
        b = next(e for e in entries if e.tracker_name == "B")
        assert a.on_pareto_front is True
        assert b.on_pareto_front is False

    def test_pareto_trade_off_both_on_front(self):
        """A has higher IoU, B has higher EES → both Pareto-optimal."""
        entries = [
            _make_entry("A", mean_iou=0.9, fps=10.0, peak_memory_mb=500.0, ees=0.5),
            _make_entry("B", mean_iou=0.4, fps=500.0, peak_memory_mb=50.0, ees=2.0),
        ]
        self.engine.compute_pareto_front(entries)
        a = next(e for e in entries if e.tracker_name == "A")
        b = next(e for e in entries if e.tracker_name == "B")
        assert a.on_pareto_front is True
        assert b.on_pareto_front is True

    def test_pareto_modifies_in_place(self):
        entries = [_make_entry("A", 0.5, 100.0, 100.0)]
        returned = self.engine.compute_pareto_front(entries)
        assert returned is entries

    # ------------------------------------------------------------------
    # rank_trackers
    # ------------------------------------------------------------------

    def test_rank_trackers_sorted_by_ees_descending(self):
        results = [
            _make_mock_result("Slow",   mean_iou=0.7, fps=5.0,   memory_mb=100.0),
            _make_mock_result("Fast",   mean_iou=0.7, fps=500.0, memory_mb=100.0),
            _make_mock_result("Medium", mean_iou=0.7, fps=50.0,  memory_mb=100.0),
        ]
        ranking = self.engine.rank_trackers(results)
        ees_values = [e.ees for e in ranking]
        assert ees_values == sorted(ees_values, reverse=True)

    def test_rank_trackers_pareto_flags_set(self):
        results = [
            _make_mock_result("A", mean_iou=0.9, fps=10.0,  memory_mb=500.0),
            _make_mock_result("B", mean_iou=0.4, fps=500.0, memory_mb=50.0),
        ]
        ranking = self.engine.rank_trackers(results)
        assert any(e.on_pareto_front for e in ranking)

    def test_rank_trackers_empty_input(self):
        assert self.engine.rank_trackers([]) == []

    def test_rank_trackers_single_result(self):
        results = [_make_mock_result("MOSSE", 0.5, 200.0, 50.0)]
        ranking = self.engine.rank_trackers(results)
        assert len(ranking) == 1
        assert ranking[0].on_pareto_front is True

    def test_rank_trackers_ees_values_match_formula(self):
        result = _make_mock_result("T", mean_iou=0.6, fps=120.0, memory_mb=200.0)
        ranking = self.engine.rank_trackers([result])
        expected_ees = self.engine.edge_efficiency_score(0.6, 120.0, 200.0)
        assert ranking[0].ees == pytest.approx(expected_ees)

    # ------------------------------------------------------------------
    # to_markdown_table
    # ------------------------------------------------------------------

    def test_markdown_table_has_header(self):
        ranking = self.engine.rank_trackers([_make_mock_result("MOSSE", 0.5, 200.0, 50.0)])
        table = self.engine.to_markdown_table(ranking)
        assert "Tracker" in table
        assert "EES" in table
        assert "Pareto" in table

    def test_markdown_table_contains_tracker_names(self):
        results = [
            _make_mock_result("MOSSE", 0.5, 200.0, 50.0),
            _make_mock_result("KCF",   0.6, 150.0, 60.0),
        ]
        ranking = self.engine.rank_trackers(results)
        table = self.engine.to_markdown_table(ranking)
        assert "MOSSE" in table
        assert "KCF" in table

    def test_markdown_table_empty_input(self):
        table = self.engine.to_markdown_table([])
        assert "Rank" in table  # header rows still present


# ---------------------------------------------------------------------------
# EfficiencyEntry
# ---------------------------------------------------------------------------

class TestEfficiencyEntry:

    def test_str_contains_key_fields(self):
        e = _make_entry("MOSSE", 0.5, 200.0, 50.0, ees=1.5, on_pareto_front=True)
        s = str(e)
        assert "MOSSE" in s
        assert "EES" in s
        assert "pareto=yes" in s

    def test_str_non_pareto(self):
        e = _make_entry("KCF", 0.6, 100.0, 80.0, ees=1.2, on_pareto_front=False)
        assert "pareto=no" in str(e)
