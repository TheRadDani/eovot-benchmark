"""Tests for eovot.analysis.efficiency_budget."""

import pytest
from eovot.analysis.efficiency_budget import (
    BudgetAnalyzer,
    BudgetEntry,
    BudgetReport,
    HardwareBudget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _R:
    """Minimal BenchmarkResult stand-in."""

    def __init__(
        self,
        tracker_name="KCF",
        dataset_name="test",
        mean_iou=0.50,
        success_auc=0.48,
        fps=30.0,
        memory_mb=64.0,
        energy_mj_per_frame=0.5,
        latency_ms=33.3,
        latency_p95_ms=40.0,
    ):
        self.tracker_name = tracker_name
        self.dataset_name = dataset_name
        self.mean_iou = mean_iou
        self.success_auc = success_auc
        self.fps = fps
        self.memory_mb = memory_mb
        self.energy_mj_per_frame = energy_mj_per_frame
        self.latency_ms = latency_ms
        self.latency_p95_ms = latency_p95_ms


def _three_trackers():
    return [
        _R("fast_tracker",   "ds", 0.40, 0.38, fps=60.0, memory_mb=32.0,  energy_mj_per_frame=0.2),
        _R("medium_tracker", "ds", 0.55, 0.53, fps=25.0, memory_mb=96.0,  energy_mj_per_frame=0.6),
        _R("best_tracker",   "ds", 0.70, 0.68, fps=10.0, memory_mb=256.0, energy_mj_per_frame=2.0),
    ]


# ---------------------------------------------------------------------------
# HardwareBudget tests
# ---------------------------------------------------------------------------

class TestHardwareBudget:
    def test_empty_budget_always_satisfied(self):
        ok, v = HardwareBudget().is_satisfied_by(_R())
        assert ok and v == {}

    def test_min_fps_satisfied(self):
        ok, v = HardwareBudget(min_fps=20.0).is_satisfied_by(_R(fps=30.0))
        assert ok and "fps" not in v

    def test_min_fps_violated(self):
        ok, v = HardwareBudget(min_fps=50.0).is_satisfied_by(_R(fps=30.0))
        assert not ok and "fps" in v

    def test_max_latency_satisfied(self):
        ok, _ = HardwareBudget(max_latency_ms=50.0).is_satisfied_by(_R(latency_ms=33.3))
        assert ok

    def test_max_latency_violated(self):
        ok, v = HardwareBudget(max_latency_ms=20.0).is_satisfied_by(_R(latency_ms=33.3))
        assert not ok and "latency_ms" in v

    def test_max_latency_p95_satisfied(self):
        ok, _ = HardwareBudget(max_latency_p95_ms=50.0).is_satisfied_by(
            _R(latency_p95_ms=40.0)
        )
        assert ok

    def test_max_latency_p95_violated(self):
        ok, v = HardwareBudget(max_latency_p95_ms=30.0).is_satisfied_by(
            _R(latency_p95_ms=40.0)
        )
        assert not ok and "latency_p95_ms" in v

    def test_max_memory_satisfied(self):
        ok, _ = HardwareBudget(max_memory_mb=128.0).is_satisfied_by(_R(memory_mb=64.0))
        assert ok

    def test_max_memory_violated(self):
        ok, v = HardwareBudget(max_memory_mb=32.0).is_satisfied_by(_R(memory_mb=64.0))
        assert not ok and "memory_mb" in v

    def test_max_energy_satisfied(self):
        ok, _ = HardwareBudget(max_energy_mj_per_frame=1.0).is_satisfied_by(
            _R(energy_mj_per_frame=0.5)
        )
        assert ok

    def test_max_energy_violated(self):
        ok, v = HardwareBudget(max_energy_mj_per_frame=0.3).is_satisfied_by(
            _R(energy_mj_per_frame=0.5)
        )
        assert not ok and "energy_mj_per_frame" in v

    def test_multiple_violations(self):
        ok, v = HardwareBudget(min_fps=60.0, max_memory_mb=16.0).is_satisfied_by(
            _R(fps=30.0, memory_mb=64.0)
        )
        assert not ok and len(v) == 2

    def test_none_attribute_skipped(self):
        r = _R()
        r.memory_mb = None
        ok, _ = HardwareBudget(max_memory_mb=100.0).is_satisfied_by(r)
        assert ok

    def test_label_default(self):
        assert HardwareBudget().label == "target device"

    def test_label_custom(self):
        assert HardwareBudget(label="RPi4").label == "RPi4"


# ---------------------------------------------------------------------------
# BudgetAnalyzer.analyze tests
# ---------------------------------------------------------------------------

class TestBudgetAnalyzerAnalyze:
    def test_returns_budget_report(self):
        report = BudgetAnalyzer().analyze(_three_trackers(), HardwareBudget())
        assert isinstance(report, BudgetReport)

    def test_all_feasible_with_empty_budget(self):
        report = BudgetAnalyzer().analyze(_three_trackers(), HardwareBudget())
        assert len(report.feasible) == 3 and len(report.infeasible) == 0

    def test_none_feasible_with_impossible_budget(self):
        report = BudgetAnalyzer().analyze(
            _three_trackers(), HardwareBudget(min_fps=1000.0)
        )
        assert len(report.feasible) == 0 and len(report.infeasible) == 3

    def test_partial_feasibility(self):
        report = BudgetAnalyzer().analyze(
            _three_trackers(), HardwareBudget(min_fps=30.0)
        )
        assert len(report.feasible) == 1
        assert report.feasible[0].tracker_name == "fast_tracker"

    def test_accuracy_rank_order(self):
        report = BudgetAnalyzer().analyze(_three_trackers(), HardwareBudget())
        ranks = {e.tracker_name: e.accuracy_rank for e in report.entries}
        assert ranks["best_tracker"] == 1
        assert ranks["medium_tracker"] == 2
        assert ranks["fast_tracker"] == 3

    def test_budget_rank_among_feasible(self):
        report = BudgetAnalyzer().analyze(
            _three_trackers(), HardwareBudget(min_fps=20.0)
        )
        feasible_ranks = {e.tracker_name: e.budget_rank for e in report.feasible}
        assert feasible_ranks["medium_tracker"] == 1
        assert feasible_ranks["fast_tracker"] == 2

    def test_empty_results(self):
        report = BudgetAnalyzer().analyze([], HardwareBudget())
        assert report.entries == [] and report.feasible == []


# ---------------------------------------------------------------------------
# BudgetReport property tests
# ---------------------------------------------------------------------------

class TestBudgetReport:
    def _report(self, min_fps=None):
        return BudgetAnalyzer().analyze(
            _three_trackers(), HardwareBudget(min_fps=min_fps)
        )

    def test_best_tracker_no_constraint(self):
        assert self._report().best_tracker == "best_tracker"

    def test_best_tracker_with_fps_constraint(self):
        assert self._report(min_fps=30.0).best_tracker == "fast_tracker"

    def test_best_tracker_none_when_all_infeasible(self):
        assert self._report(min_fps=9999.0).best_tracker is None

    def test_accuracy_loss_zero_when_no_constraint(self):
        assert self._report().accuracy_loss_at_budget == 0.0

    def test_accuracy_loss_positive_when_constrained(self):
        assert self._report(min_fps=30.0).accuracy_loss_at_budget > 0

    def test_accuracy_loss_none_when_empty(self):
        report = BudgetAnalyzer().analyze([], HardwareBudget())
        assert report.accuracy_loss_at_budget is None

    def test_markdown_table_contains_headers(self):
        md = self._report().to_markdown_table()
        assert "| Tracker |" in md
        assert "| IoU |" in md
        assert "| Feasible |" in md

    def test_markdown_table_lists_all_trackers(self):
        md = self._report().to_markdown_table()
        for name in ["best_tracker", "medium_tracker", "fast_tracker"]:
            assert name in md


# ---------------------------------------------------------------------------
# BudgetAnalyzer.sweep_budgets tests
# ---------------------------------------------------------------------------

class TestSweepBudgets:
    def test_returns_dict_keyed_by_name(self):
        budgets = {
            "relaxed": HardwareBudget(min_fps=10.0),
            "strict":  HardwareBudget(min_fps=60.0),
        }
        reports = BudgetAnalyzer().sweep_budgets(_three_trackers(), budgets)
        assert set(reports.keys()) == {"relaxed", "strict"}
        assert isinstance(reports["relaxed"], BudgetReport)

    def test_sweep_feasibility_varies(self):
        budgets = {
            "all_pass":  HardwareBudget(),
            "none_pass": HardwareBudget(min_fps=9999.0),
        }
        reports = BudgetAnalyzer().sweep_budgets(_three_trackers(), budgets)
        assert len(reports["all_pass"].feasible) == 3
        assert len(reports["none_pass"].feasible) == 0


# ---------------------------------------------------------------------------
# BudgetAnalyzer.accuracy_cost_curve tests
# ---------------------------------------------------------------------------

class TestAccuracyCostCurve:
    def test_returns_list_of_tuples(self):
        curve = BudgetAnalyzer().accuracy_cost_curve(
            _three_trackers(), [10.0, 30.0, 60.0]
        )
        assert len(curve) == 3
        fps_vals = [c[0] for c in curve]
        assert fps_vals == sorted(fps_vals)

    def test_loss_increases_with_fps_requirement(self):
        curve = BudgetAnalyzer().accuracy_cost_curve(
            _three_trackers(), [10.0, 30.0, 100.0]
        )
        losses = [c[1] for c in curve]
        assert losses[-1] >= losses[0]

    def test_curve_sorted_by_fps(self):
        curve = BudgetAnalyzer().accuracy_cost_curve(
            _three_trackers(), [60.0, 10.0, 30.0]
        )
        fps_vals = [c[0] for c in curve]
        assert fps_vals == sorted(fps_vals)
