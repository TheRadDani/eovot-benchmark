"""Tests for eovot.analysis.pareto."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from eovot.analysis.pareto import (
    ParetoFrontierAnalyzer,
    ParetoReport,
    TrackerPoint,
    _pareto_frontier_mask,
    _normalize,
    _harmonic_mean_pair,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summary(name: str, fps: float, iou: float, sauc: float = None) -> dict:
    s: dict = {"tracker": name, "mean_fps": fps, "mean_iou": iou}
    if sauc is not None:
        s["success_auc"] = sauc
    return s


def _make_result(name: str, fps: float, iou: float, sauc: float = None) -> dict:
    return {"summary": _make_summary(name, fps, iou, sauc)}


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

class TestParetoMask:
    def test_single_point_is_pareto_optimal(self):
        fps = np.array([10.0])
        acc = np.array([0.5])
        mask = _pareto_frontier_mask(fps, acc)
        assert mask[0] is np.bool_(True)

    def test_dominated_point_excluded(self):
        # A dominates B: A has more FPS AND more accuracy.
        fps = np.array([100.0, 50.0])
        acc = np.array([0.8,   0.7])
        mask = _pareto_frontier_mask(fps, acc)
        assert mask[0] is np.bool_(True)
        assert mask[1] is np.bool_(False)

    def test_two_incomparable_points_both_pareto(self):
        # A: fast but less accurate.  B: slow but more accurate.  Neither dominates.
        fps = np.array([200.0, 50.0])
        acc = np.array([0.5,   0.9])
        mask = _pareto_frontier_mask(fps, acc)
        assert mask[0] is np.bool_(True)
        assert mask[1] is np.bool_(True)

    def test_three_point_frontier(self):
        # Three points: corner1 (fast, low acc), corner2 (balanced), corner3 (slow, high acc)
        fps = np.array([300.0, 150.0,  30.0])
        acc = np.array([0.3,   0.65,   0.95])
        mask = _pareto_frontier_mask(fps, acc)
        assert mask.all(), "All three should be Pareto-optimal."

    def test_strictly_dominated_interior_point(self):
        fps = np.array([300.0, 150.0, 100.0,  30.0])
        acc = np.array([0.3,   0.65,  0.50,   0.95])
        mask = _pareto_frontier_mask(fps, acc)
        assert mask[0] and mask[1] and mask[3], "Frontier points should be optimal."
        assert not mask[2], "Interior point (100 FPS, 0.50) dominated by (150, 0.65)."


class TestNormalize:
    def test_all_same_returns_zeros(self):
        arr = np.array([5.0, 5.0, 5.0])
        assert np.allclose(_normalize(arr), 0.0)

    def test_range_maps_to_zero_one(self):
        arr = np.array([0.0, 5.0, 10.0])
        n = _normalize(arr)
        assert n[0] == pytest.approx(0.0)
        assert n[2] == pytest.approx(1.0)
        assert n[1] == pytest.approx(0.5)


class TestHarmonicMeanPair:
    def test_equal_values(self):
        a = np.array([0.5])
        b = np.array([0.5])
        assert _harmonic_mean_pair(a, b)[0] == pytest.approx(0.5)

    def test_zero_element_gives_zero(self):
        a = np.array([0.0, 1.0])
        b = np.array([1.0, 0.0])
        h = _harmonic_mean_pair(a, b)
        assert h[0] == pytest.approx(0.0)
        assert h[1] == pytest.approx(0.0)

    def test_known_harmonic_mean(self):
        a = np.array([1.0])
        b = np.array([0.5])
        # HM = 2*1*0.5 / (1+0.5) = 1.0 / 1.5 ≈ 0.6667
        assert _harmonic_mean_pair(a, b)[0] == pytest.approx(2 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# ParetoFrontierAnalyzer tests
# ---------------------------------------------------------------------------

class TestParetoFrontierAnalyzer:
    def _three_trackers(self):
        return [
            _make_result("FAST",  fps=400, iou=0.50, sauc=0.55),
            _make_result("MED",   fps=150, iou=0.72, sauc=0.68),
            _make_result("SLOW",  fps=30,  iou=0.90, sauc=0.87),
        ]

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            ParetoFrontierAnalyzer().analyze([])

    def test_single_tracker_is_pareto_optimal(self):
        result = [_make_result("A", fps=100, iou=0.7, sauc=0.65)]
        report = ParetoFrontierAnalyzer().analyze(result)
        assert len(report.pareto_points) == 1
        assert report.pareto_points[0].tracker_name == "A"

    def test_all_incomparable_are_pareto_optimal(self):
        report = ParetoFrontierAnalyzer(accuracy_metric="success_auc").analyze(
            self._three_trackers()
        )
        assert len(report.pareto_points) == 3

    def test_dominated_tracker_excluded_from_frontier(self):
        results = self._three_trackers() + [
            _make_result("WORST", fps=100, iou=0.30, sauc=0.25),
        ]
        report = ParetoFrontierAnalyzer(accuracy_metric="success_auc").analyze(results)
        names = {p.tracker_name for p in report.pareto_points}
        assert "WORST" not in names

    def test_best_by_eds_returns_tracker(self):
        report = ParetoFrontierAnalyzer(accuracy_metric="success_auc").analyze(
            self._three_trackers()
        )
        best = report.best_by_eds()
        assert best is not None
        assert best.tracker_name in {"FAST", "MED", "SLOW"}

    def test_optimal_for_fps_budget(self):
        report = ParetoFrontierAnalyzer(accuracy_metric="success_auc").analyze(
            self._three_trackers()
        )
        # Budget 200 FPS: only FAST (400) qualifies.
        rec = report.optimal_for_fps_budget(min_fps=200)
        assert rec is not None
        assert rec.tracker_name == "FAST"

    def test_optimal_for_fps_budget_no_match(self):
        report = ParetoFrontierAnalyzer(accuracy_metric="success_auc").analyze(
            self._three_trackers()
        )
        assert report.optimal_for_fps_budget(min_fps=1000) is None

    def test_eds_in_unit_interval(self):
        report = ParetoFrontierAnalyzer(accuracy_metric="success_auc").analyze(
            self._three_trackers()
        )
        for pt in report.all_points:
            assert 0.0 <= pt.eds <= 1.0

    def test_accepts_benchmark_result_objects(self):
        """Analyzer should work with BenchmarkResult objects directly."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker
        from eovot.trackers.kcf import KCFTracker

        ds = SyntheticDataset(num_sequences=2, num_frames=20, seed=1)
        engine = BenchmarkEngine(verbose=False)
        results = [
            engine.run(MOSSETracker(), ds, dataset_name="Synthetic"),
            engine.run(KCFTracker(),   ds, dataset_name="Synthetic"),
        ]
        report = ParetoFrontierAnalyzer(accuracy_metric="mean_iou").analyze(results)
        assert len(report.all_points) == 2
        assert len(report.pareto_points) >= 1

    def test_accuracy_metric_mean_iou(self):
        results = [
            _make_result("A", fps=200, iou=0.80, sauc=0.75),
            _make_result("B", fps=100, iou=0.70, sauc=0.85),
        ]
        # Using mean_iou: A has higher iou AND more fps → A dominates B.
        report = ParetoFrontierAnalyzer(accuracy_metric="mean_iou").analyze(results)
        pareto_names = {p.tracker_name for p in report.pareto_points}
        assert "A" in pareto_names

    def test_pareto_points_sorted_by_fps(self):
        report = ParetoFrontierAnalyzer(accuracy_metric="success_auc").analyze(
            self._three_trackers()
        )
        fpss = [p.fps for p in report.pareto_points]
        assert fpss == sorted(fpss)


# ---------------------------------------------------------------------------
# ParetoReport output format tests
# ---------------------------------------------------------------------------

class TestParetoReportOutput:
    def _report(self) -> ParetoReport:
        results = [
            _make_result("FAST", fps=400, iou=0.50, sauc=0.55),
            _make_result("MED",  fps=150, iou=0.72, sauc=0.68),
            _make_result("SLOW", fps=30,  iou=0.90, sauc=0.87),
        ]
        return ParetoFrontierAnalyzer(accuracy_metric="success_auc").analyze(results)

    def test_markdown_contains_tracker_names(self):
        md = self._report().to_markdown()
        assert "FAST" in md
        assert "MED" in md
        assert "SLOW" in md

    def test_markdown_contains_pareto_marker(self):
        md = self._report().to_markdown()
        assert "✓" in md

    def test_csv_written_correctly(self, tmp_path):
        path = self._report().to_csv(tmp_path / "test")
        assert path.exists()
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3
        names = {r["tracker"] for r in rows}
        assert names == {"FAST", "MED", "SLOW"}

    def test_csv_adds_extension(self, tmp_path):
        path = self._report().to_csv(tmp_path / "out")
        assert path.suffix == ".csv"

    def test_dominated_points_list(self):
        results = [
            _make_result("BEST",  fps=400, iou=0.90, sauc=0.90),
            _make_result("WORST", fps=10,  iou=0.10, sauc=0.10),
        ]
        report = ParetoFrontierAnalyzer(accuracy_metric="success_auc").analyze(results)
        dominated = report.dominated_points
        assert any(p.tracker_name == "WORST" for p in dominated)
