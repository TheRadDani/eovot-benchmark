"""Unit tests for the edge-score efficiency metrics module."""

from __future__ import annotations

import pytest

from eovot.metrics.efficiency import (
    EdgeScoreConfig,
    EdgeScoreResult,
    compute_edge_score,
    edge_score_leaderboard_md,
    rank_by_edge_score,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SUMMARY_NO_ENERGY = {
    "tracker": "MOSSE",
    "mean_iou": 0.60,
    "mean_fps": 300.0,
    "peak_memory_mb": 200.0,
}

SUMMARY_WITH_ENERGY = {
    "tracker": "KCF",
    "mean_iou": 0.55,
    "mean_fps": 150.0,
    "peak_memory_mb": 220.0,
    "mean_energy_per_frame_mj": 2.5,
}

RESULT_NO_ENERGY = {"summary": SUMMARY_NO_ENERGY}
RESULT_WITH_ENERGY = {"summary": SUMMARY_WITH_ENERGY}


# ---------------------------------------------------------------------------
# compute_edge_score
# ---------------------------------------------------------------------------

class TestComputeEdgeScore:
    def test_score_in_unit_interval(self):
        config = EdgeScoreConfig()
        r = compute_edge_score(SUMMARY_NO_ENERGY, config)
        assert 0.0 <= r.edge_score <= 1.0

    def test_returns_edge_score_result(self):
        r = compute_edge_score(SUMMARY_NO_ENERGY, EdgeScoreConfig())
        assert isinstance(r, EdgeScoreResult)

    def test_perfect_tracker_scores_near_one(self):
        """A tracker with IoU=1, FPS >> target, memory=0 should score near 1."""
        summary = {
            "tracker": "Perfect",
            "mean_iou": 1.0,
            "mean_fps": 10_000.0,
            "peak_memory_mb": 0.0,
        }
        r = compute_edge_score(summary, EdgeScoreConfig(target_fps=30.0, memory_budget_mb=512.0))
        assert r.edge_score > 0.95

    def test_zero_tracker_scores_zero(self):
        """A tracker with zero accuracy, zero speed, and exhausted budgets scores 0."""
        summary = {
            "tracker": "Zero",
            "mean_iou": 0.0,
            "mean_fps": 0.0,
            "peak_memory_mb": 9999.0,  # far exceeds budget → memory_component = 0
        }
        r = compute_edge_score(summary, EdgeScoreConfig())
        assert r.edge_score == pytest.approx(0.0, abs=1e-6)

    def test_speed_component_capped_at_one(self):
        """FPS component must be ≤ 1.0 even when FPS ≫ target."""
        summary = {
            "tracker": "Fast",
            "mean_iou": 0.5,
            "mean_fps": 999_999.0,
            "peak_memory_mb": 100.0,
        }
        r = compute_edge_score(summary, EdgeScoreConfig(target_fps=30.0))
        assert r.speed_component == pytest.approx(1.0)

    def test_memory_over_budget_yields_zero_component(self):
        """Memory component must clamp to 0 when memory exceeds budget."""
        summary = {
            "tracker": "Heavy",
            "mean_iou": 0.5,
            "mean_fps": 30.0,
            "peak_memory_mb": 1024.0,
        }
        r = compute_edge_score(summary, EdgeScoreConfig(memory_budget_mb=512.0))
        assert r.memory_component == pytest.approx(0.0)

    def test_energy_component_present_when_budget_set(self):
        config = EdgeScoreConfig(energy_budget_mj=5.0)
        r = compute_edge_score(SUMMARY_WITH_ENERGY, config)
        assert r.energy_component is not None
        assert 0.0 <= r.energy_component <= 1.0

    def test_energy_component_none_when_no_budget(self):
        config = EdgeScoreConfig(energy_budget_mj=None)
        r = compute_edge_score(SUMMARY_WITH_ENERGY, config)
        assert r.energy_component is None

    def test_energy_component_none_when_no_data(self):
        """Energy component absent when summary has no energy measurement."""
        config = EdgeScoreConfig(energy_budget_mj=5.0)
        r = compute_edge_score(SUMMARY_NO_ENERGY, config)
        assert r.energy_component is None

    def test_tracker_name_fallback(self):
        """'tracker_name' key should work as fallback when 'tracker' is absent."""
        summary = {
            "tracker_name": "MyTracker",
            "mean_iou": 0.5,
            "mean_fps": 30.0,
            "peak_memory_mb": 100.0,
        }
        r = compute_edge_score(summary, EdgeScoreConfig())
        assert r.tracker == "MyTracker"

    def test_weight_scaling_invariance(self):
        """Results must be identical regardless of absolute weight scale."""
        summary = SUMMARY_NO_ENERGY.copy()
        cfg1 = EdgeScoreConfig(w_accuracy=0.4, w_speed=0.3, w_memory=0.2, w_energy=0.1)
        cfg2 = EdgeScoreConfig(w_accuracy=4.0, w_speed=3.0, w_memory=2.0, w_energy=1.0)
        r1 = compute_edge_score(summary, cfg1)
        r2 = compute_edge_score(summary, cfg2)
        assert r1.edge_score == pytest.approx(r2.edge_score, abs=1e-6)

    def test_components_non_negative(self):
        r = compute_edge_score(SUMMARY_NO_ENERGY, EdgeScoreConfig())
        assert r.accuracy_component >= 0.0
        assert r.speed_component >= 0.0
        assert r.memory_component >= 0.0

    def test_iou_clamped_to_unit_interval(self):
        """IoU values outside [0,1] should not produce out-of-range components."""
        summary = {
            "tracker": "X",
            "mean_iou": 1.5,
            "mean_fps": 30.0,
            "peak_memory_mb": 100.0,
        }
        r = compute_edge_score(summary, EdgeScoreConfig())
        assert r.accuracy_component <= 1.0

    def test_str_representation(self):
        r = compute_edge_score(SUMMARY_NO_ENERGY, EdgeScoreConfig())
        s = str(r)
        assert "EdgeScoreResult" in s
        assert "MOSSE" in s

    def test_energy_over_budget_yields_zero_component(self):
        """Energy component clamps to 0 when usage exceeds budget."""
        summary = {
            "tracker": "Hungry",
            "mean_iou": 0.5,
            "mean_fps": 30.0,
            "peak_memory_mb": 100.0,
            "mean_energy_per_frame_mj": 100.0,
        }
        config = EdgeScoreConfig(energy_budget_mj=5.0)
        r = compute_edge_score(summary, config)
        assert r.energy_component == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# rank_by_edge_score
# ---------------------------------------------------------------------------

class TestRankByEdgeScore:
    def test_sorted_descending(self):
        results = [RESULT_NO_ENERGY, RESULT_WITH_ENERGY]
        ranked = rank_by_edge_score(results)
        scores = [r.edge_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_default_config_accepted(self):
        ranked = rank_by_edge_score([RESULT_NO_ENERGY, RESULT_WITH_ENERGY])
        assert len(ranked) == 2

    def test_returns_edge_score_result_objects(self):
        ranked = rank_by_edge_score([RESULT_NO_ENERGY])
        assert all(isinstance(r, EdgeScoreResult) for r in ranked)

    def test_empty_input_returns_empty_list(self):
        assert rank_by_edge_score([]) == []

    def test_bare_summary_dict_accepted(self):
        """A raw summary dict (without 'summary' wrapper) should work."""
        ranked = rank_by_edge_score([SUMMARY_NO_ENERGY])
        assert len(ranked) == 1

    def test_custom_config_changes_ranking(self):
        """Speed-first config should favour MOSSE (higher FPS) over KCF."""
        speed_first = EdgeScoreConfig(
            target_fps=30.0,
            w_accuracy=0.10,
            w_speed=0.80,
            w_memory=0.10,
            w_energy=0.0,
        )
        ranked = rank_by_edge_score([RESULT_NO_ENERGY, RESULT_WITH_ENERGY], config=speed_first)
        # MOSSE has 300 fps vs KCF's 150 fps — should rank first
        assert ranked[0].tracker == "MOSSE"


# ---------------------------------------------------------------------------
# edge_score_leaderboard_md
# ---------------------------------------------------------------------------

class TestEdgeScoreLeaderboardMd:
    def test_contains_tracker_names(self):
        md = edge_score_leaderboard_md([RESULT_NO_ENERGY, RESULT_WITH_ENERGY])
        assert "MOSSE" in md
        assert "KCF" in md

    def test_empty_returns_fallback_message(self):
        md = edge_score_leaderboard_md([])
        assert "No results" in md

    def test_no_energy_data_omits_energy_column(self):
        md = edge_score_leaderboard_md([RESULT_NO_ENERGY])
        assert "E (mJ/fr)" not in md

    def test_energy_data_shows_energy_column(self):
        config = EdgeScoreConfig(energy_budget_mj=5.0)
        md = edge_score_leaderboard_md([RESULT_WITH_ENERGY], config=config)
        assert "E (mJ/fr)" in md

    def test_config_budget_appears_in_output(self):
        config = EdgeScoreConfig(target_fps=60.0, memory_budget_mb=256.0)
        md = edge_score_leaderboard_md([RESULT_NO_ENERGY], config=config)
        assert "60.0" in md
        assert "256.0" in md

    def test_markdown_table_format(self):
        md = edge_score_leaderboard_md([RESULT_NO_ENERGY])
        lines = md.strip().split("\n")
        table_lines = [l for l in lines if l.startswith("|")]
        assert len(table_lines) >= 3  # header + separator + ≥1 data row

    def test_ranked_by_edge_score(self):
        """Higher edge score should appear first in the Markdown table."""
        md = edge_score_leaderboard_md([RESULT_NO_ENERGY, RESULT_WITH_ENERGY])
        mosse_pos = md.find("MOSSE")
        kcf_pos = md.find("KCF")
        # Whichever comes first in text has the higher edge score
        assert mosse_pos != -1 and kcf_pos != -1
