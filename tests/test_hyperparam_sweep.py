"""Tests for HyperparamSweeper."""

from __future__ import annotations

import pytest
from pathlib import Path

from eovot.analysis.hyperparam_sweep import HyperparamSweeper, SweepEntry
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=20)


@pytest.fixture
def sweeper(small_dataset):
    return HyperparamSweeper(
        dataset=small_dataset,
        dataset_name="SmallSynthetic",
        max_sequences=2,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# SweepEntry dataclass
# ---------------------------------------------------------------------------

class TestSweepEntry:
    def test_str_repr(self):
        e = SweepEntry(
            params={"lr": 0.1},
            mean_iou=0.5,
            success_auc=0.4,
            mean_fps=30.0,
            peak_memory_mb=100.0,
            ees=0.8,
            wall_time_s=1.0,
        )
        s = str(e)
        assert "SweepEntry" in s
        assert "mIoU=" in s
        assert "FPS=" in s


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

class TestGridSearch:
    def test_single_param_value(self, sweeper):
        results = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.10]})
        assert len(results) == 1
        assert isinstance(results[0], SweepEntry)

    def test_multiple_values(self, sweeper):
        results = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.05, 0.10, 0.20]})
        assert len(results) == 3

    def test_cross_product(self, sweeper):
        # 2 × 2 = 4 configurations
        results = sweeper.grid_search(MOSSETracker, {
            "learning_rate": [0.05, 0.10],
            "sigma": [0.5, 1.0],
        })
        assert len(results) == 4

    def test_sorted_by_ees_descending(self, sweeper):
        results = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.05, 0.10, 0.20]})
        for a, b in zip(results, results[1:]):
            assert a.ees >= b.ees

    def test_ees_non_negative(self, sweeper):
        results = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.10]})
        assert all(e.ees >= 0 for e in results)

    def test_mean_iou_in_range(self, sweeper):
        results = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.10]})
        assert 0.0 <= results[0].mean_iou <= 1.0

    def test_wall_time_positive(self, sweeper):
        results = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.10]})
        assert results[0].wall_time_s > 0.0

    def test_params_preserved(self, sweeper):
        results = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.05, 0.20]})
        param_vals = {e.params["learning_rate"] for e in results}
        assert param_vals == {0.05, 0.20}

    def test_empty_param_grid(self, sweeper):
        """Empty grid → one run with no params."""
        results = sweeper.grid_search(MOSSETracker, {})
        assert len(results) == 1

    def test_kcf_grid_search(self, sweeper):
        results = sweeper.grid_search(KCFTracker, {"learning_rate": [0.075, 0.125]})
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

class TestRandomSearch:
    def test_n_trials_count(self, sweeper):
        results = sweeper.random_search(
            MOSSETracker,
            {"learning_rate": (0.05, 0.30)},
            n_trials=5,
            seed=42,
        )
        assert len(results) == 5

    def test_list_distribution(self, sweeper):
        results = sweeper.random_search(
            MOSSETracker,
            {"learning_rate": [0.05, 0.10, 0.20]},
            n_trials=4,
            seed=7,
        )
        assert all(e.params["learning_rate"] in [0.05, 0.10, 0.20] for e in results)

    def test_tuple_distribution_bounds(self, sweeper):
        results = sweeper.random_search(
            MOSSETracker,
            {"learning_rate": (0.01, 0.50)},
            n_trials=10,
            seed=0,
        )
        for e in results:
            assert 0.01 <= e.params["learning_rate"] <= 0.50

    def test_sorted_by_ees(self, sweeper):
        results = sweeper.random_search(
            MOSSETracker,
            {"learning_rate": (0.05, 0.30)},
            n_trials=5,
            seed=99,
        )
        for a, b in zip(results, results[1:]):
            assert a.ees >= b.ees


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

class TestParetoFront:
    def test_pareto_subset_of_entries(self, sweeper):
        entries = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.05, 0.10, 0.20]})
        pareto = sweeper.pareto_front(entries)
        assert all(e in entries for e in pareto)

    def test_pareto_not_empty(self, sweeper):
        entries = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.05, 0.10, 0.20]})
        pareto = sweeper.pareto_front(entries)
        assert len(pareto) >= 1

    def test_pareto_flag_set(self, sweeper):
        entries = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.05, 0.10]})
        sweeper.pareto_front(entries)
        assert any(e.on_pareto_front for e in entries)

    def test_pareto_at_most_total(self, sweeper):
        entries = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.05, 0.10, 0.20]})
        pareto = sweeper.pareto_front(entries)
        assert len(pareto) <= len(entries)


# ---------------------------------------------------------------------------
# Sensitivity scores
# ---------------------------------------------------------------------------

class TestSensitivityScores:
    def test_returns_dict(self, sweeper):
        entries = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.05, 0.10, 0.20]})
        scores = sweeper.sensitivity_scores(entries)
        assert isinstance(scores, dict)

    def test_key_in_scores(self, sweeper):
        entries = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.05, 0.20]})
        scores = sweeper.sensitivity_scores(entries)
        assert "learning_rate" in scores

    def test_single_value_zero_sensitivity(self, sweeper):
        entries = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.10]})
        scores = sweeper.sensitivity_scores(entries)
        assert scores.get("learning_rate", 0.0) == pytest.approx(0.0, abs=1e-9)

    def test_empty_entries(self, sweeper):
        scores = sweeper.sensitivity_scores([])
        assert scores == {}


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

class TestExports:
    def test_to_markdown_non_empty(self, sweeper):
        entries = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.05, 0.20]})
        md = sweeper.to_markdown(entries)
        assert "|" in md
        assert "mIoU" in md

    def test_to_markdown_empty(self, sweeper):
        md = sweeper.to_markdown([])
        assert "No sweep results" in md

    def test_to_csv_creates_file(self, sweeper, tmp_path):
        entries = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.05, 0.20]})
        out = sweeper.to_csv(entries, str(tmp_path / "sweep.csv"))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_to_csv_has_headers(self, sweeper, tmp_path):
        entries = sweeper.grid_search(MOSSETracker, {"learning_rate": [0.10]})
        out = sweeper.to_csv(entries, str(tmp_path / "sweep.csv"))
        first_line = out.read_text().split("\n")[0]
        assert "mean_iou" in first_line
        assert "ees" in first_line
