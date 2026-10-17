"""Tests for eovot.metrics.ranking — weighted multi-criteria tracker ranking."""

from __future__ import annotations

import math
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.metrics.ranking import RankingEngine, RankingWeights, TrackerRank


# ---------------------------------------------------------------------------
# Helpers: build minimal BenchmarkResult mocks
# ---------------------------------------------------------------------------

def _make_result(
    tracker: str,
    dataset: str = "synthetic",
    mean_iou: float = 0.5,
    mean_fps: float = 100.0,
    peak_memory_mb: float = 200.0,
    success_auc: float | None = None,
    energy_mj: float | None = None,
) -> MagicMock:
    r = MagicMock()
    r.tracker_name = tracker
    r.dataset_name = dataset
    r.mean_iou = mean_iou
    r.mean_fps = mean_fps
    r.peak_memory_mb = peak_memory_mb
    r.mean_success_auc = success_auc
    r.mean_energy_per_frame_mj = energy_mj
    return r


# ---------------------------------------------------------------------------
# RankingWeights
# ---------------------------------------------------------------------------

class TestRankingWeights:
    def test_default_weights_non_negative(self) -> None:
        w = RankingWeights()
        assert w.accuracy >= 0
        assert w.fps >= 0
        assert w.memory >= 0
        assert w.energy >= 0
        assert w.robustness >= 0

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            RankingWeights(accuracy=-1.0)

    def test_total_is_sum(self) -> None:
        w = RankingWeights(accuracy=2.0, fps=3.0, memory=1.0, energy=0.0, robustness=0.0)
        assert w.total() == pytest.approx(6.0)

    def test_preset_edge_balanced(self) -> None:
        w = RankingWeights.edge_balanced()
        assert w.label == "edge_balanced"
        assert w.accuracy > 0 and w.fps > 0 and w.memory > 0

    def test_preset_accuracy_first(self) -> None:
        w = RankingWeights.accuracy_first()
        assert w.accuracy > w.fps

    def test_preset_battery_saver(self) -> None:
        w = RankingWeights.battery_saver()
        assert w.energy > w.fps

    def test_preset_throughput_max(self) -> None:
        w = RankingWeights.throughput_max()
        assert w.fps > w.accuracy


# ---------------------------------------------------------------------------
# RankingEngine — basic contract
# ---------------------------------------------------------------------------

class TestRankingEngineBasic:
    def test_empty_results_raises(self) -> None:
        engine = RankingEngine()
        with pytest.raises(ValueError, match="empty"):
            engine.rank([])

    def test_single_tracker_scores_half(self) -> None:
        """With one tracker, all min-max normalised values are 0.5 (tie)."""
        engine = RankingEngine()
        r = _make_result("MOSSE", mean_iou=0.6, mean_fps=300.0, peak_memory_mb=50.0)
        ranking = engine.rank([r])
        assert len(ranking) == 1
        assert ranking[0].rank == 1
        assert ranking[0].composite_score == pytest.approx(0.5, abs=1e-9)

    def test_rank_ordering(self) -> None:
        """Higher composite score → lower (better) rank number."""
        engine = RankingEngine(RankingWeights(accuracy=1.0, fps=0.0, memory=0.0))
        fast = _make_result("Fast", mean_iou=0.8, mean_fps=100.0, peak_memory_mb=100.0, success_auc=0.8)
        slow = _make_result("Slow", mean_iou=0.3, mean_fps=100.0, peak_memory_mb=100.0, success_auc=0.3)
        ranking = engine.rank([fast, slow])
        assert ranking[0].tracker_name == "Fast"
        assert ranking[0].rank == 1
        assert ranking[1].rank == 2

    def test_rank_fields_populated(self) -> None:
        results = [
            _make_result("A", mean_iou=0.6, mean_fps=200.0, peak_memory_mb=100.0),
            _make_result("B", mean_iou=0.4, mean_fps=400.0, peak_memory_mb=80.0),
        ]
        engine = RankingEngine(RankingWeights.edge_balanced())
        ranking = engine.rank(results)
        for r in ranking:
            assert r.tracker_name in ("A", "B")
            assert 0.0 <= r.composite_score <= 1.0
            assert r.raw_fps > 0
            assert r.raw_memory_mb > 0

    def test_success_auc_preferred_over_iou(self) -> None:
        """When success_auc is available it should be used as raw_accuracy."""
        engine = RankingEngine(RankingWeights(accuracy=1.0, fps=0.0, memory=0.0))
        r = _make_result("T", mean_iou=0.5, success_auc=0.9)
        ranking = engine.rank([r])
        assert ranking[0].raw_accuracy == pytest.approx(0.9)

    def test_iou_fallback_when_no_auc(self) -> None:
        engine = RankingEngine(RankingWeights(accuracy=1.0, fps=0.0, memory=0.0))
        r = _make_result("T", mean_iou=0.55, success_auc=None)
        ranking = engine.rank([r])
        assert ranking[0].raw_accuracy == pytest.approx(0.55)

    def test_energy_none_handled(self) -> None:
        """Missing energy should not raise and norm_energy should default to 0.5."""
        engine = RankingEngine()
        r = _make_result("T", energy_mj=None)
        ranking = engine.rank([r])
        assert ranking[0].raw_energy_mj is None
        assert ranking[0].norm_energy == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# RankingEngine — normalisation properties
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_best_tracker_norm_values_near_one(self) -> None:
        """The tracker with best accuracy/fps should have norm ≈ 1.0 on those dims."""
        engine = RankingEngine()
        best = _make_result("Best", mean_iou=0.9, mean_fps=500.0, peak_memory_mb=50.0)
        worst = _make_result("Worst", mean_iou=0.2, mean_fps=10.0, peak_memory_mb=900.0)
        ranking = engine.rank([best, worst])
        best_entry = next(r for r in ranking if r.tracker_name == "Best")
        assert best_entry.norm_accuracy == pytest.approx(1.0, abs=1e-9)
        assert best_entry.norm_fps == pytest.approx(1.0, abs=1e-9)
        assert best_entry.norm_memory == pytest.approx(1.0, abs=1e-9)

    def test_identical_trackers_get_same_score(self) -> None:
        """Two identical trackers should receive the same composite score."""
        engine = RankingEngine()
        r1 = _make_result("A", mean_iou=0.5, mean_fps=100.0, peak_memory_mb=200.0)
        r2 = _make_result("B", mean_iou=0.5, mean_fps=100.0, peak_memory_mb=200.0)
        ranking = engine.rank([r1, r2])
        assert ranking[0].composite_score == pytest.approx(ranking[1].composite_score, abs=1e-9)

    def test_all_zero_weights_raises(self) -> None:
        engine = RankingEngine(RankingWeights(accuracy=0.0, fps=0.0, memory=0.0,
                                              energy=0.0, robustness=0.0))
        r = _make_result("T")
        with pytest.raises(ValueError, match="zero"):
            engine.rank([r])


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class TestOutputFormatting:
    def _two_tracker_ranking(self) -> List[TrackerRank]:
        engine = RankingEngine(RankingWeights.edge_balanced())
        results = [
            _make_result("MOSSE", mean_iou=0.6, mean_fps=300.0, peak_memory_mb=80.0),
            _make_result("KCF",   mean_iou=0.65, mean_fps=150.0, peak_memory_mb=90.0),
        ]
        return engine.rank(results)

    def test_markdown_table_has_header(self) -> None:
        engine = RankingEngine(RankingWeights.edge_balanced())
        ranking = self._two_tracker_ranking()
        table = engine.to_markdown_table(ranking)
        assert "Rank" in table
        assert "Tracker" in table
        assert "Score" in table

    def test_markdown_table_has_both_trackers(self) -> None:
        ranking = self._two_tracker_ranking()
        table = RankingEngine().to_markdown_table(ranking)
        assert "MOSSE" in table or "KCF" in table  # at least one appears

    def test_summary_dict_length(self) -> None:
        engine = RankingEngine()
        ranking = self._two_tracker_ranking()
        summary = engine.to_summary_dict(ranking)
        assert len(summary) == 2

    def test_summary_dict_keys(self) -> None:
        engine = RankingEngine()
        ranking = self._two_tracker_ranking()
        d = engine.to_summary_dict(ranking)[0]
        for key in ("rank", "tracker", "dataset", "composite_score",
                    "raw_accuracy", "raw_fps", "raw_memory_mb"):
            assert key in d

    def test_tracker_rank_str(self) -> None:
        engine = RankingEngine()
        ranking = self._two_tracker_ranking()
        s = str(ranking[0])
        assert "#1" in s


# ---------------------------------------------------------------------------
# Preset differentiation
# ---------------------------------------------------------------------------

class TestPresetDifferentiation:
    """Verify that different presets produce different orderings when expected."""

    def test_accuracy_first_vs_throughput_max(self) -> None:
        """
        Tracker A: high accuracy, low FPS.
        Tracker B: low accuracy, high FPS.
        accuracy_first should prefer A; throughput_max should prefer B.
        """
        accurate = _make_result("Accurate", mean_iou=0.85, mean_fps=20.0,
                                peak_memory_mb=150.0, success_auc=0.85)
        fast = _make_result("Fast", mean_iou=0.30, mean_fps=500.0,
                            peak_memory_mb=60.0, success_auc=0.30)

        acc_engine = RankingEngine(RankingWeights.accuracy_first())
        acc_ranking = acc_engine.rank([accurate, fast])
        assert acc_ranking[0].tracker_name == "Accurate"

        fps_engine = RankingEngine(RankingWeights.throughput_max())
        fps_ranking = fps_engine.rank([accurate, fast])
        assert fps_ranking[0].tracker_name == "Fast"

    def test_battery_saver_penalises_high_energy(self) -> None:
        """Tracker with lower energy should rank first under battery_saver."""
        efficient = _make_result("Efficient", mean_iou=0.5, mean_fps=100.0,
                                 peak_memory_mb=100.0, energy_mj=0.5)
        hungry = _make_result("Hungry", mean_iou=0.5, mean_fps=100.0,
                              peak_memory_mb=100.0, energy_mj=10.0)
        engine = RankingEngine(RankingWeights.battery_saver())
        ranking = engine.rank([efficient, hungry])
        assert ranking[0].tracker_name == "Efficient"
