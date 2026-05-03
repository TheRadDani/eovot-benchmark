"""Unit tests for eovot.metrics.efficiency.EfficiencyAnalyzer."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from eovot.metrics.efficiency import EdgeFitnessScore, EfficiencyAnalyzer, _minmax_norm


# ---------------------------------------------------------------------------
# Helpers — synthetic result dicts
# ---------------------------------------------------------------------------

def _make_result(
    tracker: str,
    mean_iou: float,
    mean_fps: float,
    peak_memory_mb: float = 100.0,
    mean_latency_ms: float = 5.0,
    mean_energy_mj: float | None = None,
) -> Dict[str, Any]:
    s: Dict[str, Any] = {
        "tracker": tracker,
        "mean_iou": mean_iou,
        "mean_fps": mean_fps,
        "peak_memory_mb": peak_memory_mb,
        "mean_latency_ms": mean_latency_ms,
    }
    if mean_energy_mj is not None:
        s["mean_energy_per_frame_mj"] = mean_energy_mj
    return {"summary": s, "sequences": []}


# Representative 4-tracker comparison used across several tests.
RESULTS_4 = [
    _make_result("MOSSE",      mean_iou=0.40, mean_fps=500.0, peak_memory_mb=50.0),
    _make_result("KCF",        mean_iou=0.52, mean_fps=250.0, peak_memory_mb=80.0),
    _make_result("CSRT",       mean_iou=0.65, mean_fps=60.0,  peak_memory_mb=150.0),
    _make_result("MedianFlow", mean_iou=0.35, mean_fps=120.0, peak_memory_mb=60.0),
]


# ---------------------------------------------------------------------------
# _minmax_norm helper
# ---------------------------------------------------------------------------

class TestMinmaxNorm:
    def test_min_value_maps_to_zero(self):
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0])
        assert _minmax_norm(1.0, arr) == pytest.approx(0.0)

    def test_max_value_maps_to_one(self):
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0])
        assert _minmax_norm(3.0, arr) == pytest.approx(1.0)

    def test_midpoint(self):
        import numpy as np
        arr = np.array([0.0, 1.0])
        assert _minmax_norm(0.5, arr) == pytest.approx(0.5)

    def test_constant_array_returns_half(self):
        import numpy as np
        arr = np.array([5.0, 5.0, 5.0])
        assert _minmax_norm(5.0, arr) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# EfficiencyAnalyzer initialisation
# ---------------------------------------------------------------------------

class TestEfficiencyAnalyzerInit:
    def test_weights_normalised_to_one(self):
        analyzer = EfficiencyAnalyzer(
            accuracy_weight=2.0, fps_weight=1.0, memory_weight=1.0, energy_weight=0.0
        )
        total = (
            analyzer.accuracy_weight
            + analyzer.fps_weight
            + analyzer.memory_weight
            + analyzer.energy_weight
        )
        assert total == pytest.approx(1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            EfficiencyAnalyzer(accuracy_weight=-1.0)

    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError, match="positive"):
            EfficiencyAnalyzer(
                accuracy_weight=0.0, fps_weight=0.0,
                memory_weight=0.0, energy_weight=0.0
            )


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    def setup_method(self):
        self.analyzer = EfficiencyAnalyzer()

    def test_clear_domination(self):
        # A strictly dominates B on both axes.
        results = [
            _make_result("A", mean_iou=0.8, mean_fps=200.0),
            _make_result("B", mean_iou=0.5, mean_fps=100.0),
        ]
        frontier = self.analyzer.pareto_frontier(results)
        assert "A" in frontier
        assert "B" not in frontier

    def test_trade_off_both_on_frontier(self):
        # Fast-but-inaccurate and slow-but-accurate trackers are both Pareto-optimal.
        results = [
            _make_result("Fast",     mean_iou=0.40, mean_fps=500.0),
            _make_result("Accurate", mean_iou=0.70, mean_fps=50.0),
        ]
        frontier = self.analyzer.pareto_frontier(results)
        assert "Fast" in frontier
        assert "Accurate" in frontier

    def test_equal_trackers_both_on_frontier(self):
        results = [
            _make_result("T1", mean_iou=0.5, mean_fps=100.0),
            _make_result("T2", mean_iou=0.5, mean_fps=100.0),
        ]
        frontier = self.analyzer.pareto_frontier(results)
        assert "T1" in frontier
        assert "T2" in frontier

    def test_single_tracker_is_on_frontier(self):
        results = [_make_result("Solo", mean_iou=0.6, mean_fps=200.0)]
        frontier = self.analyzer.pareto_frontier(results)
        assert frontier == ["Solo"]

    def test_frontier_sorted_by_fps(self):
        results = [
            _make_result("Fast",     mean_iou=0.40, mean_fps=500.0),
            _make_result("Accurate", mean_iou=0.70, mean_fps=50.0),
        ]
        frontier = self.analyzer.pareto_frontier(results)
        fps_vals = [
            results[["Fast", "Accurate"].index(n)].get("summary", {}).get("mean_fps", 0)
            for n in frontier
        ]
        assert fps_vals == sorted(fps_vals)

    def test_dominated_tracker_excluded(self):
        results = [
            _make_result("Best",  mean_iou=0.80, mean_fps=300.0),
            _make_result("Weak",  mean_iou=0.50, mean_fps=100.0),  # dominated by Best
            _make_result("Niche", mean_iou=0.30, mean_fps=800.0),  # fast, low IoU
        ]
        frontier = self.analyzer.pareto_frontier(results)
        assert "Weak" not in frontier
        assert "Best" in frontier
        assert "Niche" in frontier


# ---------------------------------------------------------------------------
# Fitness score computation
# ---------------------------------------------------------------------------

class TestComputeFitness:
    def setup_method(self):
        self.analyzer = EfficiencyAnalyzer()

    def test_returns_edge_fitness_score(self):
        score = self.analyzer.compute_fitness(RESULTS_4[0], RESULTS_4)
        assert isinstance(score, EdgeFitnessScore)

    def test_fitness_in_unit_interval(self):
        for r in RESULTS_4:
            score = self.analyzer.compute_fitness(r, RESULTS_4)
            assert 0.0 <= score.fitness <= 1.0

    def test_tracker_name_propagated(self):
        score = self.analyzer.compute_fitness(RESULTS_4[0], RESULTS_4)
        assert score.tracker_name == "MOSSE"

    def test_pareto_flag_set_correctly(self):
        # MOSSE is fast and CSRT is accurate — both are Pareto-optimal in RESULTS_4.
        mosse_score = self.analyzer.compute_fitness(RESULTS_4[0], RESULTS_4)  # MOSSE
        csrt_score = self.analyzer.compute_fitness(RESULTS_4[2], RESULTS_4)   # CSRT
        assert mosse_score.is_pareto_optimal
        assert csrt_score.is_pareto_optimal

    def test_component_scores_present(self):
        score = self.analyzer.compute_fitness(RESULTS_4[0], RESULTS_4)
        assert "accuracy" in score.component_scores
        assert "throughput" in score.component_scores
        assert "memory_efficiency" in score.component_scores

    def test_best_on_all_axes_has_highest_fitness(self):
        # Construct a tracker that wins on all axes.
        best = _make_result("BestEver", mean_iou=1.0, mean_fps=1000.0, peak_memory_mb=1.0)
        results = RESULTS_4 + [best]
        best_score = self.analyzer.compute_fitness(best, results)
        for r in RESULTS_4:
            other_score = self.analyzer.compute_fitness(r, results)
            assert best_score.fitness >= other_score.fitness

    def test_energy_component_when_present(self):
        results = [
            _make_result("A", mean_iou=0.5, mean_fps=100.0, mean_energy_mj=1.0),
            _make_result("B", mean_iou=0.6, mean_fps=200.0, mean_energy_mj=2.0),
        ]
        score = self.analyzer.compute_fitness(results[0], results)
        assert "energy_efficiency" in score.component_scores

    def test_no_energy_component_when_absent(self):
        score = self.analyzer.compute_fitness(RESULTS_4[0], RESULTS_4)
        assert "energy_efficiency" not in score.component_scores


# ---------------------------------------------------------------------------
# rank_trackers
# ---------------------------------------------------------------------------

class TestRankTrackers:
    def setup_method(self):
        self.analyzer = EfficiencyAnalyzer()

    def test_returns_list_of_scores(self):
        ranked = self.analyzer.rank_trackers(RESULTS_4)
        assert len(ranked) == len(RESULTS_4)
        assert all(isinstance(s, EdgeFitnessScore) for s in ranked)

    def test_sorted_descending_by_fitness(self):
        ranked = self.analyzer.rank_trackers(RESULTS_4)
        fitnesses = [s.fitness for s in ranked]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_single_tracker(self):
        ranked = self.analyzer.rank_trackers([RESULTS_4[0]])
        assert len(ranked) == 1
        assert ranked[0].fitness == pytest.approx(0.5)  # all single-element ranges → 0.5


# ---------------------------------------------------------------------------
# Hardware constraint filtering
# ---------------------------------------------------------------------------

class TestFilterByConstraints:
    def setup_method(self):
        self.analyzer = EfficiencyAnalyzer()

    def test_no_constraints_returns_all(self):
        feasible = self.analyzer.filter_by_constraints(RESULTS_4)
        assert len(feasible) == len(RESULTS_4)

    def test_min_fps_filters_slow_trackers(self):
        # Only MOSSE (500 FPS) and KCF (250 FPS) exceed 200 FPS.
        feasible = self.analyzer.filter_by_constraints(RESULTS_4, min_fps=200.0)
        names = [r["summary"]["tracker"] for r in feasible]
        assert "MOSSE" in names
        assert "KCF" in names
        assert "CSRT" not in names
        assert "MedianFlow" not in names

    def test_max_memory_filters_heavy_trackers(self):
        # CSRT uses 150 MB → excluded at 100 MB limit.
        feasible = self.analyzer.filter_by_constraints(RESULTS_4, max_memory_mb=100.0)
        names = [r["summary"]["tracker"] for r in feasible]
        assert "CSRT" not in names

    def test_max_latency_constraint(self):
        results = [
            _make_result("Fast",  mean_iou=0.5, mean_fps=200.0, mean_latency_ms=5.0),
            _make_result("Slow",  mean_iou=0.7, mean_fps=20.0,  mean_latency_ms=50.0),
        ]
        feasible = self.analyzer.filter_by_constraints(results, max_latency_ms=10.0)
        names = [r["summary"]["tracker"] for r in feasible]
        assert "Fast" in names
        assert "Slow" not in names

    def test_energy_constraint(self):
        results = [
            _make_result("Green",  mean_iou=0.5, mean_fps=100.0, mean_energy_mj=0.5),
            _make_result("Hungry", mean_iou=0.6, mean_fps=100.0, mean_energy_mj=5.0),
        ]
        feasible = self.analyzer.filter_by_constraints(results, max_energy_mj=1.0)
        names = [r["summary"]["tracker"] for r in feasible]
        assert "Green" in names
        assert "Hungry" not in names

    def test_combined_constraints(self):
        feasible = self.analyzer.filter_by_constraints(
            RESULTS_4, min_fps=100.0, max_memory_mb=100.0
        )
        names = [r["summary"]["tracker"] for r in feasible]
        # MOSSE: 500 FPS, 50 MB → pass
        # KCF:   250 FPS, 80 MB → pass
        # CSRT:  60 FPS (fails min_fps) → excluded
        # MedianFlow: 120 FPS, 60 MB → pass
        assert "MOSSE" in names
        assert "KCF" in names
        assert "CSRT" not in names
        assert "MedianFlow" in names

    def test_impossible_constraints_return_empty(self):
        feasible = self.analyzer.filter_by_constraints(RESULTS_4, min_fps=10_000.0)
        assert feasible == []


# ---------------------------------------------------------------------------
# score_with_constraints
# ---------------------------------------------------------------------------

class TestScoreWithConstraints:
    def setup_method(self):
        self.analyzer = EfficiencyAnalyzer()

    def test_violates_flag_set_for_slow_trackers(self):
        scored = self.analyzer.score_with_constraints(RESULTS_4, min_fps=200.0)
        violations = {s.tracker_name: s.violates_constraints for s in scored}
        assert not violations["MOSSE"]
        assert not violations["KCF"]
        assert violations["CSRT"]

    def test_all_pass_no_constraints(self):
        scored = self.analyzer.score_with_constraints(RESULTS_4)
        assert all(not s.violates_constraints for s in scored)
