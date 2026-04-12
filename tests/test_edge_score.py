"""Unit tests for the EdgeScore composite metric.

Covers :func:`~eovot.metrics.edge_score.compute_edge_score` and
:func:`~eovot.metrics.edge_score.rank_by_edge_score`.
"""

from __future__ import annotations

import pytest

from eovot.metrics.edge_score import (
    EdgeScoreWeights,
    compute_edge_score,
    rank_by_edge_score,
)


class TestEdgeScoreWeights:
    def test_default_weights_are_positive(self):
        w = EdgeScoreWeights()
        assert w.iou > 0
        assert w.fps > 0
        assert w.memory > 0
        assert w.energy > 0

    def test_zero_weight_allowed(self):
        # Energy can be zeroed out (disabled dimension)
        w = EdgeScoreWeights(energy=0.0)
        assert w.energy == 0.0

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            EdgeScoreWeights(iou=-1.0)

    def test_negative_fps_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            EdgeScoreWeights(fps=-0.1)


class TestComputeEdgeScore:
    # ------------------------------------------------------------------
    # Basic contract
    # ------------------------------------------------------------------

    def test_returns_float(self):
        s = compute_edge_score(mean_iou=0.5, fps=30.0, memory_mb=100.0)
        assert isinstance(s, float)

    def test_score_in_unit_interval(self):
        for iou in (0.0, 0.3, 0.6, 1.0):
            s = compute_edge_score(mean_iou=iou, fps=30.0, memory_mb=128.0)
            assert 0.0 <= s <= 1.0, f"score={s} out of [0,1] for iou={iou}"

    def test_perfect_tracker_high_score(self):
        """IoU=1, FPS >> ref, tiny memory should yield a high score."""
        s = compute_edge_score(
            mean_iou=1.0,
            fps=300.0,
            memory_mb=10.0,
            energy_mj_per_frame=0.1,
            fps_ref=30.0,
            mem_ref_mb=256.0,
            energy_ref_mj=5.0,
        )
        assert s > 0.8, f"Expected high score, got {s}"

    def test_zero_iou_collapses_score(self):
        """Geometric mean with iou_norm=0 must give 0."""
        s = compute_edge_score(mean_iou=0.0, fps=30.0, memory_mb=128.0)
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_fps_at_reference_contributes_one(self):
        """fps == fps_ref should normalise fps_norm to 1.0."""
        # With iou=1, fps=ref, mem -> 0 MB (mem_norm ~1), score should be high.
        s = compute_edge_score(
            mean_iou=1.0,
            fps=30.0,
            memory_mb=1.0,
            fps_ref=30.0,
            mem_ref_mb=256.0,
        )
        assert s > 0.85

    def test_fps_above_reference_capped_at_one(self):
        """fps > fps_ref should not exceed the reference score."""
        s_fast = compute_edge_score(mean_iou=0.7, fps=1000.0, memory_mb=50.0, fps_ref=30.0)
        s_ref = compute_edge_score(mean_iou=0.7, fps=30.0, memory_mb=50.0, fps_ref=30.0)
        assert s_fast == pytest.approx(s_ref, abs=1e-9)

    def test_high_memory_lowers_score(self):
        s_low = compute_edge_score(mean_iou=0.6, fps=60.0, memory_mb=20.0)
        s_high = compute_edge_score(mean_iou=0.6, fps=60.0, memory_mb=200.0)
        assert s_low > s_high

    def test_memory_exceeding_ref_gives_zero_mem_norm(self):
        """memory_mb >= mem_ref_mb should clamp mem_norm to 0 -> score collapses."""
        s = compute_edge_score(
            mean_iou=0.8, fps=60.0, memory_mb=256.0, mem_ref_mb=256.0
        )
        assert s == pytest.approx(0.0, abs=1e-6)

    # ------------------------------------------------------------------
    # Energy dimension
    # ------------------------------------------------------------------

    def test_energy_none_omits_dimension(self):
        """Score without energy should differ from score with high energy."""
        s_no_e = compute_edge_score(mean_iou=0.6, fps=60.0, memory_mb=80.0, energy_mj_per_frame=None)
        s_bad_e = compute_edge_score(mean_iou=0.6, fps=60.0, memory_mb=80.0, energy_mj_per_frame=4.9)
        assert s_no_e > s_bad_e

    def test_low_energy_improves_score(self):
        s_low = compute_edge_score(mean_iou=0.6, fps=60.0, memory_mb=80.0, energy_mj_per_frame=0.5)
        s_high = compute_edge_score(mean_iou=0.6, fps=60.0, memory_mb=80.0, energy_mj_per_frame=4.5)
        assert s_low > s_high

    def test_energy_zero_norm_collapses_score(self):
        s = compute_edge_score(
            mean_iou=0.9, fps=60.0, memory_mb=40.0,
            energy_mj_per_frame=5.0, energy_ref_mj=5.0,
        )
        assert s == pytest.approx(0.0, abs=1e-6)

    # ------------------------------------------------------------------
    # Custom weights
    # ------------------------------------------------------------------

    def test_custom_weights_used(self):
        """Accuracy-only weights should rank a slower tracker with higher IoU above a faster one."""
        acc_weights = EdgeScoreWeights(iou=10.0, fps=0.0, memory=0.0, energy=0.0)
        s_accurate = compute_edge_score(mean_iou=0.9, fps=5.0, memory_mb=200.0, weights=acc_weights)
        s_fast = compute_edge_score(mean_iou=0.3, fps=500.0, memory_mb=10.0, weights=acc_weights)
        assert s_accurate > s_fast

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError, match="fps"):
            compute_edge_score(mean_iou=0.5, fps=0.0, memory_mb=100.0)

    def test_negative_fps_raises(self):
        with pytest.raises(ValueError, match="fps"):
            compute_edge_score(mean_iou=0.5, fps=-1.0, memory_mb=100.0)

    def test_invalid_memory_raises(self):
        with pytest.raises(ValueError, match="memory_mb"):
            compute_edge_score(mean_iou=0.5, fps=30.0, memory_mb=0.0)


class TestRankByEdgeScore:
    TRACKERS = {
        "MOSSE": {"mean_iou": 0.52, "mean_fps": 520.0, "peak_memory_mb": 35.0},
        "KCF":   {"mean_iou": 0.58, "mean_fps": 220.0, "peak_memory_mb": 42.0},
        "CSRT":  {"mean_iou": 0.68, "mean_fps": 45.0,  "peak_memory_mb": 65.0},
    }

    def test_returns_list_of_tuples(self):
        ranking = rank_by_edge_score(self.TRACKERS)
        assert isinstance(ranking, list)
        for item in ranking:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_all_trackers_present(self):
        ranking = rank_by_edge_score(self.TRACKERS)
        names = [name for name, _ in ranking]
        assert set(names) == set(self.TRACKERS)

    def test_sorted_descending(self):
        ranking = rank_by_edge_score(self.TRACKERS)
        scores = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_unit_interval(self):
        ranking = rank_by_edge_score(self.TRACKERS)
        for _, score in ranking:
            assert 0.0 <= score <= 1.0

    def test_empty_results(self):
        assert rank_by_edge_score({}) == []

    def test_with_energy_data(self):
        trackers_with_energy = {
            "A": {"mean_iou": 0.6, "mean_fps": 100.0, "peak_memory_mb": 50.0,
                  "mean_energy_per_frame_mj": 1.0},
            "B": {"mean_iou": 0.6, "mean_fps": 100.0, "peak_memory_mb": 50.0,
                  "mean_energy_per_frame_mj": 4.0},
        }
        ranking = rank_by_edge_score(trackers_with_energy)
        top_name = ranking[0][0]
        assert top_name == "A"  # lower energy should rank higher
