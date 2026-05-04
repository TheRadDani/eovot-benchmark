"""Unit tests for multi-objective tracker scoring and Pareto frontier analysis."""

import math

import numpy as np
import pytest

from eovot.metrics.scoring import (
    BALANCED_WEIGHTS,
    EDGE_WEIGHTS,
    ENERGY_WEIGHTS,
    PRESET_WEIGHTS,
    RESEARCH_WEIGHTS,
    ScoringWeights,
    compute_composite_scores,
    pareto_frontier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_metrics():
    """Four representative classical trackers for testing."""
    return {
        "MOSSE": {"auc": 0.48, "precision": 0.62, "fps": 520.0, "peak_memory_mb": 42.0, "mean_energy_j": 0.001},
        "KCF": {"auc": 0.55, "precision": 0.70, "fps": 280.0, "peak_memory_mb": 68.0, "mean_energy_j": 0.003},
        "CSRT": {"auc": 0.68, "precision": 0.82, "fps": 45.0, "peak_memory_mb": 130.0, "mean_energy_j": 0.018},
        "MedianFlow": {"auc": 0.51, "precision": 0.65, "fps": 150.0, "peak_memory_mb": 55.0, "mean_energy_j": 0.005},
    }


# ---------------------------------------------------------------------------
# ScoringWeights
# ---------------------------------------------------------------------------


class TestScoringWeights:
    def test_valid_weights_do_not_raise(self):
        w = ScoringWeights(accuracy=0.4, speed=0.3, memory=0.2, energy=0.1)
        assert w.accuracy == 0.4

    def test_invalid_weights_raise_value_error(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringWeights(accuracy=0.5, speed=0.5, memory=0.1, energy=0.1)

    def test_all_presets_sum_to_one(self):
        for name, preset in PRESET_WEIGHTS.items():
            total = preset.accuracy + preset.speed + preset.memory + preset.energy
            assert math.isclose(total, 1.0, abs_tol=1e-9), (
                f"Preset '{name}' sums to {total}, not 1.0"
            )

    def test_four_presets_exist(self):
        assert set(PRESET_WEIGHTS.keys()) == {"balanced", "edge", "research", "energy"}

    def test_research_weights_accuracy_highest(self):
        w = RESEARCH_WEIGHTS
        assert w.accuracy > w.speed
        assert w.accuracy > w.memory
        assert w.accuracy > w.energy

    def test_edge_weights_speed_highest(self):
        w = EDGE_WEIGHTS
        assert w.speed >= w.accuracy

    def test_energy_weights_energy_highest(self):
        w = ENERGY_WEIGHTS
        assert w.energy > w.accuracy
        assert w.energy > w.speed
        assert w.energy > w.memory


# ---------------------------------------------------------------------------
# compute_composite_scores
# ---------------------------------------------------------------------------


class TestCompositeScores:
    def test_returns_dataframe_with_expected_columns(self, sample_metrics):
        import pandas as pd
        df = compute_composite_scores(sample_metrics)
        assert isinstance(df, pd.DataFrame)
        for col in ("tracker", "composite_score", "accuracy_score", "speed_score",
                    "memory_score", "energy_score"):
            assert col in df.columns, f"Missing column: {col}"

    def test_sorted_by_composite_score_descending(self, sample_metrics):
        df = compute_composite_scores(sample_metrics)
        scores = df["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_all_scores_in_unit_interval(self, sample_metrics):
        df = compute_composite_scores(sample_metrics)
        for col in ("accuracy_score", "speed_score", "memory_score", "energy_score", "composite_score"):
            assert (df[col] >= 0.0).all(), f"{col} has negative values"
            assert (df[col] <= 1.0 + 1e-9).all(), f"{col} exceeds 1.0"

    def test_all_trackers_present(self, sample_metrics):
        df = compute_composite_scores(sample_metrics)
        assert set(df["tracker"]) == set(sample_metrics.keys())

    def test_research_weights_favour_accuracy(self, sample_metrics):
        """With RESEARCH_WEIGHTS, CSRT (highest AUC) should rank first."""
        df = compute_composite_scores(sample_metrics, weights=RESEARCH_WEIGHTS)
        assert df.iloc[0]["tracker"] == "CSRT"

    def test_edge_weights_favour_fast_trackers(self, sample_metrics):
        """With EDGE_WEIGHTS, speed/memory matter most, so MOSSE should rank first."""
        df = compute_composite_scores(sample_metrics, weights=EDGE_WEIGHTS)
        assert df.iloc[0]["tracker"] == "MOSSE"

    def test_empty_input_returns_empty_dataframe(self):
        import pandas as pd
        df = compute_composite_scores({})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_single_tracker_gets_score_one(self):
        """A single tracker is best on all objectives — should receive 1.0."""
        df = compute_composite_scores(
            {"only_tracker": {"auc": 0.5, "fps": 100.0, "peak_memory_mb": 80.0}}
        )
        assert math.isclose(df.iloc[0]["composite_score"], 1.0, abs_tol=1e-6)

    def test_missing_metrics_handled_gracefully(self):
        """Trackers with partial metric dicts should not raise."""
        metrics = {
            "TrackA": {"auc": 0.6},
            "TrackB": {"fps": 200.0},
        }
        df = compute_composite_scores(metrics)
        assert len(df) == 2

    def test_zero_energy_gives_full_energy_score(self):
        """When no tracker has energy data, all should get energy_score=1.0."""
        metrics = {
            "A": {"auc": 0.5, "fps": 100.0},
            "B": {"auc": 0.6, "fps": 80.0},
        }
        df = compute_composite_scores(metrics)
        assert (df["energy_score"] == 1.0).all()


# ---------------------------------------------------------------------------
# pareto_frontier
# ---------------------------------------------------------------------------


class TestParetoFrontier:
    def test_returns_two_element_tuple(self, sample_metrics):
        result = pareto_frontier(sample_metrics, "fps", "auc")
        assert isinstance(result, tuple) and len(result) == 2

    def test_pareto_names_are_subset_of_all_trackers(self, sample_metrics):
        pareto_names, _ = pareto_frontier(sample_metrics, "fps", "auc")
        assert set(pareto_names).issubset(set(sample_metrics.keys()))

    def test_pareto_dataframe_has_expected_columns(self, sample_metrics):
        _, df = pareto_frontier(sample_metrics, "fps", "auc")
        assert "tracker" in df.columns
        assert "on_pareto_frontier" in df.columns
        assert "fps" in df.columns
        assert "auc" in df.columns

    def test_mosse_on_pareto_frontier_fps_vs_auc(self, sample_metrics):
        """MOSSE has the highest FPS; no other tracker dominates it on FPS+AUC."""
        pareto_names, _ = pareto_frontier(sample_metrics, "fps", "auc")
        assert "MOSSE" in pareto_names

    def test_csrt_on_pareto_frontier_fps_vs_auc(self, sample_metrics):
        """CSRT has the highest AUC; no other tracker dominates it on FPS+AUC."""
        pareto_names, _ = pareto_frontier(sample_metrics, "fps", "auc")
        assert "CSRT" in pareto_names

    def test_dominated_tracker_not_on_frontier(self):
        """B is dominated by A (A >= B on both objectives with A > B on one)."""
        metrics = {
            "A": {"fps": 100.0, "auc": 0.7},
            "B": {"fps": 80.0, "auc": 0.6},   # dominated by A
            "C": {"fps": 200.0, "auc": 0.5},  # not dominated (high fps)
        }
        pareto_names, _ = pareto_frontier(metrics, "fps", "auc")
        assert "B" not in pareto_names
        assert "A" in pareto_names
        assert "C" in pareto_names

    def test_lower_is_better_direction(self):
        """Test x_higher_is_better=False (e.g. latency vs AUC)."""
        metrics = {
            "Fast": {"mean_latency_ms": 5.0, "auc": 0.6},
            "Slow": {"mean_latency_ms": 50.0, "auc": 0.5},  # dominated: worse on both
            "Accurate": {"mean_latency_ms": 20.0, "auc": 0.8},
        }
        pareto_names, _ = pareto_frontier(
            metrics, "mean_latency_ms", "auc",
            x_higher_is_better=False, y_higher_is_better=True,
        )
        assert "Slow" not in pareto_names
        assert "Fast" in pareto_names
        assert "Accurate" in pareto_names

    def test_empty_input_returns_empty(self):
        names, df = pareto_frontier({}, "fps", "auc")
        assert names == []
        assert len(df) == 0

    def test_single_tracker_is_on_frontier(self):
        metrics = {"only": {"fps": 100.0, "auc": 0.5}}
        names, _ = pareto_frontier(metrics, "fps", "auc")
        assert names == ["only"]

    def test_identical_trackers_both_on_frontier(self):
        """When two trackers are identical, neither dominates the other."""
        metrics = {
            "A": {"fps": 100.0, "auc": 0.5},
            "B": {"fps": 100.0, "auc": 0.5},
        }
        names, _ = pareto_frontier(metrics, "fps", "auc")
        assert set(names) == {"A", "B"}
