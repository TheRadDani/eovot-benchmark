"""Tests for the Edge Efficiency Score module."""

from __future__ import annotations

import pytest

from eovot.metrics.efficiency import (
    EdgeEfficiencyScorer,
    EfficiencyResult,
    score_from_summary,
)


# ---------------------------------------------------------------------------
# EdgeEfficiencyScorer construction
# ---------------------------------------------------------------------------


class TestEdgeEfficiencyScorerInit:
    def test_default_construction(self):
        scorer = EdgeEfficiencyScorer()
        assert scorer.fps_ref == 30.0
        assert scorer.memory_ref_mb == 512.0

    def test_custom_refs(self):
        scorer = EdgeEfficiencyScorer(fps_ref=60.0, memory_ref_mb=256.0)
        assert scorer.fps_ref == 60.0
        assert scorer.memory_ref_mb == 256.0

    def test_invalid_fps_ref_raises(self):
        with pytest.raises(ValueError):
            EdgeEfficiencyScorer(fps_ref=0.0)

    def test_invalid_memory_ref_raises(self):
        with pytest.raises(ValueError):
            EdgeEfficiencyScorer(memory_ref_mb=-10.0)

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            EdgeEfficiencyScorer(weight_accuracy=0.0)

    def test_weights_normalised(self):
        scorer = EdgeEfficiencyScorer(
            weight_accuracy=2.0,
            weight_speed=2.0,
            weight_memory=2.0,
        )
        # After normalisation all three should be equal (1/3 each)
        assert abs(scorer._w_acc - scorer._w_spd) < 1e-9
        assert abs(scorer._w_acc + scorer._w_spd + scorer._w_mem - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# score() method
# ---------------------------------------------------------------------------


class TestEdgeEfficiencyScorerScore:
    def setup_method(self):
        self.scorer = EdgeEfficiencyScorer(fps_ref=30.0, memory_ref_mb=512.0)

    def test_returns_efficiency_result(self):
        r = self.scorer.score("MOSSE", mean_iou=0.6, fps=300.0, memory_mb=50.0)
        assert isinstance(r, EfficiencyResult)

    def test_perfect_tracker_near_one(self):
        # IoU=1, FPS >> ref, memory=0
        r = self.scorer.score("perfect", mean_iou=1.0, fps=1000.0, memory_mb=0.0)
        assert abs(r.edge_score - 1.0) < 1e-6

    def test_zero_tracker_near_zero(self):
        # IoU=0, FPS=0, memory=ref (worst case)
        r = self.scorer.score("zero", mean_iou=0.0, fps=0.0, memory_mb=512.0)
        assert r.edge_score == 0.0

    def test_score_in_valid_range(self):
        r = self.scorer.score("tracker", mean_iou=0.55, fps=45.0, memory_mb=120.0)
        assert 0.0 <= r.edge_score <= 1.0

    def test_speed_capped_at_one(self):
        r = self.scorer.score("fast", mean_iou=0.5, fps=10000.0, memory_mb=100.0)
        assert r.speed_score == 1.0

    def test_memory_score_zero_at_ref(self):
        r = self.scorer.score("heavy", mean_iou=0.5, fps=30.0, memory_mb=512.0)
        assert r.memory_score == 0.0

    def test_memory_score_above_ref_clamped_to_zero(self):
        r = self.scorer.score("heavy", mean_iou=0.5, fps=30.0, memory_mb=1000.0)
        assert r.memory_score == 0.0

    def test_invalid_iou_raises(self):
        with pytest.raises(ValueError):
            self.scorer.score("x", mean_iou=1.5, fps=30.0, memory_mb=50.0)

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError):
            self.scorer.score("x", mean_iou=0.5, fps=-1.0, memory_mb=50.0)

    def test_invalid_memory_raises(self):
        with pytest.raises(ValueError):
            self.scorer.score("x", mean_iou=0.5, fps=30.0, memory_mb=-50.0)

    def test_tracker_name_preserved(self):
        r = self.scorer.score("KCF", mean_iou=0.6, fps=200.0, memory_mb=80.0)
        assert r.tracker_name == "KCF"

    def test_to_dict_has_required_keys(self):
        r = self.scorer.score("T", mean_iou=0.6, fps=60.0, memory_mb=100.0)
        d = r.to_dict()
        assert "tracker" in d
        assert "edge_score" in d
        assert "mean_iou" in d
        assert "fps" in d
        assert "memory_mb" in d

    def test_str_representation(self):
        r = self.scorer.score("MOSSE", mean_iou=0.55, fps=300.0, memory_mb=45.0)
        text = str(r)
        assert "MOSSE" in text
        assert "score=" in text

    def test_weight_effect_accuracy_dominant(self):
        """Higher accuracy weight should make IoU differences more impactful."""
        scorer_acc = EdgeEfficiencyScorer(
            fps_ref=30.0, memory_ref_mb=512.0,
            weight_accuracy=0.9, weight_speed=0.05, weight_memory=0.05,
        )
        scorer_spd = EdgeEfficiencyScorer(
            fps_ref=30.0, memory_ref_mb=512.0,
            weight_accuracy=0.05, weight_speed=0.9, weight_memory=0.05,
        )
        # High IoU, low FPS — accuracy-biased scorer should rank it higher
        high_iou = dict(mean_iou=0.9, fps=5.0, memory_mb=100.0)
        low_iou = dict(mean_iou=0.3, fps=500.0, memory_mb=100.0)

        r_acc_hi = scorer_acc.score("hi", **high_iou)
        r_acc_lo = scorer_acc.score("lo", **low_iou)
        r_spd_hi = scorer_spd.score("hi", **high_iou)
        r_spd_lo = scorer_spd.score("lo", **low_iou)

        assert r_acc_hi.edge_score > r_acc_lo.edge_score
        assert r_spd_lo.edge_score > r_spd_hi.edge_score


# ---------------------------------------------------------------------------
# rank() method
# ---------------------------------------------------------------------------


class TestEdgeEfficiencyScorerRank:
    def setup_method(self):
        self.scorer = EdgeEfficiencyScorer(fps_ref=30.0, memory_ref_mb=512.0)
        self.trackers = {
            "MOSSE": {"mean_iou": 0.45, "fps": 350.0, "memory_mb": 30.0},
            "KCF":   {"mean_iou": 0.55, "fps": 180.0, "memory_mb": 60.0},
            "CSRT":  {"mean_iou": 0.70, "fps": 40.0,  "memory_mb": 90.0},
        }

    def test_returns_sorted_list(self):
        ranked = self.scorer.rank(self.trackers)
        scores = [r.edge_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_all_trackers_present(self):
        ranked = self.scorer.rank(self.trackers)
        names = {r.tracker_name for r in ranked}
        assert names == set(self.trackers.keys())

    def test_empty_input_returns_empty(self):
        assert self.scorer.rank({}) == []

    def test_missing_key_raises(self):
        with pytest.raises(ValueError):
            self.scorer.rank({"T": {"mean_iou": 0.5, "fps": 30.0}})  # no memory_mb

    def test_single_tracker(self):
        ranked = self.scorer.rank({"solo": {"mean_iou": 0.6, "fps": 50.0, "memory_mb": 100.0}})
        assert len(ranked) == 1
        assert ranked[0].tracker_name == "solo"


# ---------------------------------------------------------------------------
# summary_table() method
# ---------------------------------------------------------------------------


class TestSummaryTable:
    def test_table_contains_tracker_names(self):
        scorer = EdgeEfficiencyScorer()
        trackers = {
            "MOSSE": {"mean_iou": 0.45, "fps": 350.0, "memory_mb": 30.0},
            "CSRT":  {"mean_iou": 0.70, "fps": 40.0,  "memory_mb": 90.0},
        }
        table = scorer.summary_table(trackers, title="Test Ranking")
        assert "MOSSE" in table
        assert "CSRT" in table
        assert "Test Ranking" in table

    def test_table_is_string(self):
        scorer = EdgeEfficiencyScorer()
        table = scorer.summary_table({"T": {"mean_iou": 0.5, "fps": 30.0, "memory_mb": 50.0}})
        assert isinstance(table, str)


# ---------------------------------------------------------------------------
# score_from_summary() convenience function
# ---------------------------------------------------------------------------


class TestScoreFromSummary:
    def test_basic_usage(self):
        summary = {
            "tracker": "MOSSE",
            "mean_iou": 0.55,
            "mean_fps": 300.0,
            "peak_memory_mb": 45.0,
        }
        result = score_from_summary(summary)
        assert isinstance(result, EfficiencyResult)
        assert result.tracker_name == "MOSSE"
        assert 0.0 <= result.edge_score <= 1.0

    def test_custom_scorer(self):
        scorer = EdgeEfficiencyScorer(fps_ref=60.0, memory_ref_mb=256.0)
        summary = {"tracker": "T", "mean_iou": 0.6, "mean_fps": 60.0, "peak_memory_mb": 0.0}
        result = score_from_summary(summary, scorer=scorer)
        assert result.speed_score == 1.0

    def test_missing_keys_use_defaults(self):
        """Missing keys should not raise — defaults of 0.0 are used for numeric fields."""
        result = score_from_summary({})
        assert result.tracker_name == "unknown"
        # memory_mb defaults to 0 → memory_score = 1.0; accuracy and speed are 0
        # edge_score = w_mem * 1.0 (non-zero due to perfect memory efficiency)
        assert result.edge_score > 0.0
        assert result.accuracy_score == 0.0
        assert result.speed_score == 0.0
        assert result.memory_score == 1.0
