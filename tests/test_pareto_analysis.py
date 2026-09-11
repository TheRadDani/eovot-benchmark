"""Tests for the accuracy-efficiency Pareto front analysis module."""

import pytest

from eovot.analysis.pareto import (
    ParetoAnalyzer,
    ParetoEntry,
    TrackerPoint,
    compute_pareto_front,
)


def _pt(name: str, acc: float, fps: float) -> TrackerPoint:
    return TrackerPoint(name=name, accuracy=acc, fps=fps)


# ---------------------------------------------------------------------------
# TrackerPoint
# ---------------------------------------------------------------------------

def test_latency_derived_from_fps():
    p = _pt("A", 0.5, 100.0)
    assert abs(p.latency_mean_ms - 10.0) < 1e-9


def test_zero_fps_gives_infinite_latency():
    p = _pt("A", 0.5, 0.0)
    assert p.latency_mean_ms == float("inf")


def test_from_benchmark_result_fallback_to_iou():
    class FakeResult:
        tracker_name = "mock"
        dataset_name = "test"
        mean_fps = 50.0
        mean_iou = 0.4
    pt = TrackerPoint.from_benchmark_result(FakeResult())
    assert pt.name == "mock"
    assert pt.accuracy == pytest.approx(0.4)
    assert pt.fps == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# compute_pareto_front
# ---------------------------------------------------------------------------

def test_single_point_is_pareto_optimal():
    entries = compute_pareto_front([_pt("A", 0.6, 50.0)])
    assert len(entries) == 1
    assert entries[0].is_pareto_optimal
    assert entries[0].pareto_rank == 1


def test_two_incomparable_points_are_both_optimal():
    # A: higher accuracy, B: higher FPS — neither dominates the other
    entries = {e.name: e for e in compute_pareto_front([
        _pt("A", 0.8, 30.0),
        _pt("B", 0.5, 120.0),
    ])}
    assert entries["A"].is_pareto_optimal
    assert entries["B"].is_pareto_optimal


def test_dominated_point_is_not_optimal():
    # C dominates D on both accuracy and FPS
    entries = {e.name: e for e in compute_pareto_front([
        _pt("C", 0.7, 100.0),
        _pt("D", 0.5, 80.0),
    ])}
    assert entries["C"].is_pareto_optimal
    assert not entries["D"].is_pareto_optimal


def test_pareto_ranks_layered_correctly():
    # Layer 1: A and B non-dominated; Layer 2: C
    entries = {e.name: e for e in compute_pareto_front([
        _pt("A", 0.9, 20.0),
        _pt("B", 0.5, 90.0),
        _pt("C", 0.4, 15.0),   # dominated by A (both objectives)
    ])}
    assert entries["A"].pareto_rank == 1
    assert entries["B"].pareto_rank == 1
    assert entries["C"].pareto_rank > 1


def test_identical_points_are_both_optimal():
    # Equal on all objectives: neither strictly dominates the other
    entries = {e.name: e for e in compute_pareto_front([
        _pt("X", 0.6, 60.0),
        _pt("Y", 0.6, 60.0),
    ])}
    assert entries["X"].is_pareto_optimal
    assert entries["Y"].is_pareto_optimal


# ---------------------------------------------------------------------------
# ParetoAnalyzer
# ---------------------------------------------------------------------------

def test_empty_points_raises():
    with pytest.raises(ValueError):
        ParetoAnalyzer([])


def test_all_points_represented_in_entries():
    points = [_pt(f"T{i}", 0.1 * i, float(i * 10)) for i in range(1, 6)]
    analyzer = ParetoAnalyzer(points)
    assert len(analyzer.pareto_entries) == len(points)


def test_pareto_optimal_is_subset_of_all_entries():
    points = [
        _pt("Fast",     0.4, 200.0),
        _pt("Accurate", 0.9, 20.0),
        _pt("Bad",      0.3, 10.0),   # dominated by Fast
    ]
    analyzer = ParetoAnalyzer(points)
    optimal_names = {e.name for e in analyzer.pareto_optimal}
    assert "Fast" in optimal_names
    assert "Accurate" in optimal_names
    assert "Bad" not in optimal_names


def test_rank_for_device_invalid_fps_raises():
    analyzer = ParetoAnalyzer([_pt("A", 0.5, 30.0)])
    with pytest.raises(ValueError):
        analyzer.rank_for_device(target_fps=0.0)
    with pytest.raises(ValueError):
        analyzer.rank_for_device(target_fps=-1.0)


def test_rank_for_device_ees_in_unit_interval():
    points = [_pt("A", 0.8, 60.0), _pt("B", 0.6, 15.0)]
    analyzer = ParetoAnalyzer(points)
    for e in analyzer.rank_for_device(target_fps=30.0):
        assert e.ees is not None
        assert 0.0 <= e.ees <= 1.0


def test_rank_for_device_higher_fps_ranks_better_at_equal_accuracy():
    # Same accuracy, Fast meets target — Slow does not
    points = [_pt("Fast", 0.6, 100.0), _pt("Slow", 0.6, 10.0)]
    analyzer = ParetoAnalyzer(points)
    ranked = analyzer.rank_for_device(target_fps=30.0)
    assert ranked[0].name == "Fast"


def test_rank_for_device_no_extra_reward_beyond_target_fps():
    # Both exceed target FPS; higher accuracy should win
    points = [_pt("HighFPS", 0.5, 1000.0), _pt("MedFPS", 0.7, 60.0)]
    analyzer = ParetoAnalyzer(points)
    ranked = analyzer.rank_for_device(target_fps=30.0)
    assert ranked[0].name == "MedFPS"


def test_summary_table_is_string_containing_tracker_names():
    points = [_pt("MyTracker", 0.55, 45.0)]
    analyzer = ParetoAnalyzer(points)
    table = analyzer.summary_table()
    assert isinstance(table, str)
    assert "MyTracker" in table
