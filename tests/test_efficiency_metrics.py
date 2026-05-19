"""Unit tests for eovot.metrics.efficiency (Edge Efficiency Score & Pareto front)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.metrics.efficiency import EfficiencyEntry, EfficiencyMetricsEngine


# ---------------------------------------------------------------------------
# Helpers — minimal BenchmarkResult stub
# ---------------------------------------------------------------------------

def _make_result(tracker: str, dataset: str, miou: float, fps: float, mem: float):
    """Build a lightweight mock that looks like a BenchmarkResult."""
    r = MagicMock()
    r.tracker_name = tracker
    r.dataset_name = dataset
    r.mean_iou = miou
    r.mean_fps = fps
    r.peak_memory_mb = mem
    return r


# ---------------------------------------------------------------------------
# EfficiencyMetricsEngine — construction
# ---------------------------------------------------------------------------

class TestEfficiencyMetricsEngineInit:
    def test_default_budget(self):
        engine = EfficiencyMetricsEngine()
        assert engine.memory_budget_mb == 512.0

    def test_custom_budget(self):
        engine = EfficiencyMetricsEngine(memory_budget_mb=256.0)
        assert engine.memory_budget_mb == 256.0

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError):
            EfficiencyMetricsEngine(memory_budget_mb=0.0)

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError):
            EfficiencyMetricsEngine(memory_budget_mb=-100.0)


# ---------------------------------------------------------------------------
# edge_efficiency_score
# ---------------------------------------------------------------------------

class TestEES:
    def setup_method(self):
        self.engine = EfficiencyMetricsEngine(memory_budget_mb=512.0)

    def test_zero_fps_returns_zero(self):
        assert self.engine.edge_efficiency_score(0.5, 0.0, 100.0) == pytest.approx(0.0)

    def test_negative_fps_returns_zero(self):
        assert self.engine.edge_efficiency_score(0.5, -10.0, 100.0) == pytest.approx(0.0)

    def test_negative_iou_returns_zero(self):
        assert self.engine.edge_efficiency_score(-0.1, 100.0, 100.0) == pytest.approx(0.0)

    def test_zero_iou_returns_zero(self):
        assert self.engine.edge_efficiency_score(0.0, 100.0, 100.0) == pytest.approx(0.0)

    def test_within_budget_no_memory_penalty(self):
        # If peak_memory_mb << memory_budget_mb, denominator ≈ 1
        ees = self.engine.edge_efficiency_score(1.0, math.e - 1, 0.0)
        # log1p(e-1) = 1.0; denominator = 1 + 0/512 = 1.0
        assert ees == pytest.approx(1.0, rel=1e-6)

    def test_memory_penalty_reduces_score(self):
        ees_small_mem = self.engine.edge_efficiency_score(0.8, 100.0, 0.0)
        ees_large_mem = self.engine.edge_efficiency_score(0.8, 100.0, 512.0)
        assert ees_small_mem > ees_large_mem

    def test_higher_fps_higher_ees(self):
        ees_low = self.engine.edge_efficiency_score(0.7, 10.0, 100.0)
        ees_high = self.engine.edge_efficiency_score(0.7, 500.0, 100.0)
        assert ees_high > ees_low

    def test_higher_iou_higher_ees(self):
        ees_low = self.engine.edge_efficiency_score(0.3, 100.0, 100.0)
        ees_high = self.engine.edge_efficiency_score(0.9, 100.0, 100.0)
        assert ees_high > ees_low

    def test_formula_correctness(self):
        miou, fps, mem, budget = 0.6, 100.0, 256.0, 512.0
        engine = EfficiencyMetricsEngine(memory_budget_mb=budget)
        expected = (miou * math.log1p(fps)) / (1.0 + mem / budget)
        assert engine.edge_efficiency_score(miou, fps, mem) == pytest.approx(expected)

    def test_non_negative(self):
        for _ in range(20):
            rng = np.random.default_rng(0)
            miou = float(rng.uniform(0, 1))
            fps = float(rng.uniform(0.1, 1000))
            mem = float(rng.uniform(0, 2000))
            assert self.engine.edge_efficiency_score(miou, fps, mem) >= 0.0


# ---------------------------------------------------------------------------
# rank_trackers
# ---------------------------------------------------------------------------

class TestRankTrackers:
    def setup_method(self):
        self.engine = EfficiencyMetricsEngine(memory_budget_mb=512.0)

    def test_returns_list_of_efficiency_entries(self):
        results = [
            _make_result("MOSSE", "OTB", 0.4, 500, 50),
            _make_result("KCF", "OTB", 0.55, 150, 80),
        ]
        ranking = self.engine.rank_trackers(results)
        assert all(isinstance(e, EfficiencyEntry) for e in ranking)

    def test_length_matches_input(self):
        results = [_make_result(f"T{i}", "OTB", 0.5, 100, 50) for i in range(5)]
        ranking = self.engine.rank_trackers(results)
        assert len(ranking) == 5

    def test_sorted_by_ees_descending(self):
        results = [
            _make_result("slow_accurate", "OTB", 0.9, 5, 50),
            _make_result("fast_inaccurate", "OTB", 0.3, 600, 50),
            _make_result("balanced", "OTB", 0.6, 100, 50),
        ]
        ranking = self.engine.rank_trackers(results)
        ees_values = [e.ees for e in ranking]
        assert ees_values == sorted(ees_values, reverse=True)

    def test_empty_input(self):
        ranking = self.engine.rank_trackers([])
        assert ranking == []

    def test_ees_computed_correctly(self):
        r = _make_result("T", "D", 0.7, 200.0, 100.0)
        ranking = self.engine.rank_trackers([r])
        expected_ees = self.engine.edge_efficiency_score(0.7, 200.0, 100.0)
        assert ranking[0].ees == pytest.approx(expected_ees)


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

class TestParetoFront:
    def setup_method(self):
        self.engine = EfficiencyMetricsEngine()

    def _make_entry(self, name, miou, ees):
        return EfficiencyEntry(
            tracker_name=name,
            dataset_name="D",
            mean_iou=miou,
            fps=100.0,
            peak_memory_mb=50.0,
            ees=ees,
        )

    def test_single_entry_is_pareto_optimal(self):
        entries = [self._make_entry("A", 0.7, 2.0)]
        self.engine.compute_pareto_front(entries)
        assert entries[0].on_pareto_front is True

    def test_dominated_entry_not_on_front(self):
        # B dominates A in both objectives
        entries = [
            self._make_entry("A", 0.5, 1.0),
            self._make_entry("B", 0.8, 2.0),
        ]
        self.engine.compute_pareto_front(entries)
        assert entries[0].on_pareto_front is False
        assert entries[1].on_pareto_front is True

    def test_trade_off_both_pareto_optimal(self):
        # A is more accurate; B is more efficient — neither dominates
        entries = [
            self._make_entry("A", 0.9, 1.0),
            self._make_entry("B", 0.5, 3.0),
        ]
        self.engine.compute_pareto_front(entries)
        assert entries[0].on_pareto_front is True
        assert entries[1].on_pareto_front is True

    def test_returns_same_list(self):
        entries = [self._make_entry("A", 0.7, 2.0)]
        result = self.engine.compute_pareto_front(entries)
        assert result is entries

    def test_all_identical_all_pareto(self):
        entries = [self._make_entry(f"T{i}", 0.6, 2.0) for i in range(3)]
        self.engine.compute_pareto_front(entries)
        # Strict dominance — ties don't dominate each other
        assert all(e.on_pareto_front for e in entries)


# ---------------------------------------------------------------------------
# to_markdown_table
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def setup_method(self):
        self.engine = EfficiencyMetricsEngine()

    def test_contains_header(self):
        entries = [
            EfficiencyEntry("MOSSE", "OTB", 0.5, 300.0, 60.0, 2.5, True)
        ]
        table = self.engine.to_markdown_table(entries)
        assert "Tracker" in table
        assert "EES" in table
        assert "Pareto" in table

    def test_contains_tracker_name(self):
        entries = [EfficiencyEntry("MyTracker", "OTB", 0.5, 100.0, 50.0, 1.5, False)]
        table = self.engine.to_markdown_table(entries)
        assert "MyTracker" in table

    def test_pareto_checkmark_present(self):
        entries = [EfficiencyEntry("A", "D", 0.7, 100.0, 50.0, 2.0, True)]
        table = self.engine.to_markdown_table(entries)
        assert "✓" in table

    def test_non_pareto_no_checkmark_on_row(self):
        entries = [EfficiencyEntry("A", "D", 0.7, 100.0, 50.0, 2.0, False)]
        table = self.engine.to_markdown_table(entries)
        assert "✓" not in table

    def test_empty_entries(self):
        table = self.engine.to_markdown_table([])
        assert "Rank" in table  # header should still be present
