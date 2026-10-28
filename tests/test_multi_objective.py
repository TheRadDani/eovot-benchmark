"""Tests for eovot.metrics.multi_objective."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pytest

from eovot.metrics.multi_objective import (
    MultiObjectiveRanker,
    MultiObjectiveScore,
    ObjectiveWeights,
    _hypervolume_2d_sweep,
    _minmax_normalize,
)


# ---------------------------------------------------------------------------
# Minimal BenchmarkResult stub (avoids importing the full engine in tests)
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    tracker_name: str
    dataset_name: str = "TestDataset"
    mean_iou: float = 0.5
    mean_fps: float = 30.0
    peak_memory_mb: float = 200.0
    mean_energy_per_frame_mj: Optional[float] = None


# ---------------------------------------------------------------------------
# ObjectiveWeights
# ---------------------------------------------------------------------------


class TestObjectiveWeights:
    def test_normalises_to_sum_one(self):
        w = ObjectiveWeights(iou=2.0, fps=1.0, memory=1.0)
        assert abs(w.iou + w.fps + w.memory + w.energy - 1.0) < 1e-9

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ObjectiveWeights(iou=-0.1, fps=1.0, memory=0.0)

    def test_all_zero_raises(self):
        with pytest.raises(ValueError):
            ObjectiveWeights(iou=0.0, fps=0.0, memory=0.0, energy=0.0)

    def test_single_nonzero(self):
        w = ObjectiveWeights(iou=1.0, fps=0.0, memory=0.0)
        assert w.iou == pytest.approx(1.0)
        assert w.fps == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _minmax_normalize
# ---------------------------------------------------------------------------


class TestMinmaxNormalize:
    def test_known_range(self):
        arr = np.array([0.0, 5.0, 10.0])
        out = _minmax_normalize(arr)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])

    def test_constant_returns_half(self):
        arr = np.array([3.0, 3.0, 3.0])
        out = _minmax_normalize(arr)
        np.testing.assert_allclose(out, [0.5, 0.5, 0.5])

    def test_single_element(self):
        arr = np.array([7.0])
        out = _minmax_normalize(arr)
        assert out[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _hypervolume_2d_sweep
# ---------------------------------------------------------------------------


class TestHypervolume2d:
    def test_single_point(self):
        pts = np.array([[0.8, 0.6]])
        hv = _hypervolume_2d_sweep(pts, ref=(0.0, 0.0))
        assert hv == pytest.approx(0.8 * 0.6)

    def test_two_non_dominated_points(self):
        # (0.8, 0.3) and (0.5, 0.7) — both on front vs (0,0)
        pts = np.array([[0.8, 0.3], [0.5, 0.7]])
        hv = _hypervolume_2d_sweep(pts, ref=(0.0, 0.0))
        # Rectangle from x=0.5..0.8, y=0..0.3  PLUS rectangle 0..0.5, 0..0.7
        expected = 0.3 * 0.3 + 0.5 * 0.7
        assert hv == pytest.approx(expected, abs=1e-9)

    def test_dominated_point_excluded(self):
        # (0.9, 0.9) dominates (0.5, 0.5)
        pts = np.array([[0.9, 0.9], [0.5, 0.5]])
        hv = _hypervolume_2d_sweep(pts, ref=(0.0, 0.0))
        assert hv == pytest.approx(0.9 * 0.9)

    def test_ref_filters_all(self):
        pts = np.array([[0.5, 0.5]])
        hv = _hypervolume_2d_sweep(pts, ref=(0.6, 0.6))
        assert hv == pytest.approx(0.0)

    def test_empty_input(self):
        pts = np.empty((0, 2), dtype=np.float64)
        hv = _hypervolume_2d_sweep(pts, ref=(0.0, 0.0))
        assert hv == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MultiObjectiveRanker.score_trackers
# ---------------------------------------------------------------------------


def _make_results(specs):
    """specs: list of (name, iou, fps, memory_mb, energy_mj_or_None)."""
    return [
        _FakeResult(
            tracker_name=name,
            mean_iou=iou,
            mean_fps=fps,
            peak_memory_mb=mem,
            mean_energy_per_frame_mj=eng,
        )
        for name, iou, fps, mem, eng in specs
    ]


class TestMultiObjectiveRanker:
    def test_three_trackers_sorted_by_wcs(self):
        results = _make_results([
            ("Best",  0.9, 60.0, 100.0, None),
            ("Mid",   0.6, 30.0, 200.0, None),
            ("Worst", 0.3, 10.0, 400.0, None),
        ])
        ranker = MultiObjectiveRanker(ObjectiveWeights(iou=0.5, fps=0.3, memory=0.2))
        scores = ranker.score_trackers(results)

        assert scores[0].tracker_name == "Best"
        assert scores[-1].tracker_name == "Worst"
        # WCS is sorted descending
        assert scores[0].wcs >= scores[1].wcs >= scores[2].wcs

    def test_all_norm_scores_in_unit_interval(self):
        results = _make_results([
            ("A", 0.7, 50.0, 150.0, None),
            ("B", 0.5, 25.0, 300.0, None),
            ("C", 0.9, 75.0,  80.0, None),
        ])
        ranker = MultiObjectiveRanker()
        scores = ranker.score_trackers(results)
        for s in scores:
            assert 0.0 <= s.norm_iou <= 1.0
            assert 0.0 <= s.norm_fps <= 1.0
            assert 0.0 <= s.norm_memory <= 1.0
            assert 0.0 <= s.wcs <= 1.0

    def test_wcs_with_energy(self):
        results = _make_results([
            ("LowEnergy",  0.7, 40.0, 200.0, 1.0),
            ("HighEnergy", 0.7, 40.0, 200.0, 5.0),
        ])
        ranker = MultiObjectiveRanker(ObjectiveWeights(iou=0.3, fps=0.3, memory=0.2, energy=0.2))
        scores = ranker.score_trackers(results)
        low_e = next(s for s in scores if s.tracker_name == "LowEnergy")
        high_e = next(s for s in scores if s.tracker_name == "HighEnergy")
        assert low_e.wcs > high_e.wcs

    def test_pareto_front_single_tracker(self):
        results = _make_results([("Solo", 0.6, 30.0, 200.0, None)])
        scores = MultiObjectiveRanker().score_trackers(results)
        assert scores[0].on_pareto_front

    def test_pareto_front_two_non_dominated(self):
        results = _make_results([
            ("HighIoU",  0.9, 10.0, 500.0, None),
            ("HighSpeed", 0.3, 80.0,  50.0, None),
        ])
        scores = MultiObjectiveRanker().score_trackers(results)
        # Neither strictly dominates the other in (IoU, WCS) space
        pareto_names = {s.tracker_name for s in scores if s.on_pareto_front}
        assert len(pareto_names) >= 1

    def test_empty_results_returns_empty(self):
        assert MultiObjectiveRanker().score_trackers([]) == []

    def test_identical_trackers_constant_normalisation(self):
        results = _make_results([
            ("T1", 0.5, 30.0, 200.0, None),
            ("T2", 0.5, 30.0, 200.0, None),
        ])
        scores = MultiObjectiveRanker().score_trackers(results)
        # All norms should be 0.5 (constant-array fallback)
        for s in scores:
            assert s.norm_iou == pytest.approx(0.5)
            assert s.norm_fps == pytest.approx(0.5)

    def test_hypervolume_2d_method(self):
        results = _make_results([
            ("Fast",    0.6, 60.0, 100.0, None),
            ("Precise", 0.9, 20.0, 200.0, None),
            ("Balanced",0.75, 40.0, 150.0, None),
        ])
        ranker = MultiObjectiveRanker()
        scores = ranker.score_trackers(results)
        hv = ranker.hypervolume_2d(scores)
        assert hv >= 0.0

    def test_markdown_table_contains_tracker_names(self):
        results = _make_results([
            ("MOSSE", 0.4, 200.0, 80.0,  None),
            ("KCF",   0.6,  50.0, 120.0, None),
        ])
        scores = MultiObjectiveRanker().score_trackers(results)
        table = MultiObjectiveRanker().to_markdown_table(scores)
        assert "MOSSE" in table
        assert "KCF" in table
        assert "WCS" in table

    def test_sensitivity_analysis_returns_all_trackers(self):
        results = _make_results([
            ("A", 0.8, 40.0, 150.0, None),
            ("B", 0.5, 70.0, 100.0, None),
        ])
        ranker = MultiObjectiveRanker()
        sa = ranker.sensitivity_analysis(results, weight_grid_steps=3)
        assert "A" in sa and "B" in sa
        assert len(sa["A"]) == 3  # one entry per weight step
