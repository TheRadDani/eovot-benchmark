"""Tests for the hyperparameter ablation engine."""

from __future__ import annotations

import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.experiment.ablation import (
    AblationConfig,
    AblationResult,
    AblationStudy,
    SensitivityEntry,
)
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_dataset():
    """Minimal synthetic dataset: 3 sequences × 25 frames."""
    return SyntheticDataset(num_sequences=3, num_frames=25, seed=99)


# ---------------------------------------------------------------------------
# AblationStudy — single-parameter sweep
# ---------------------------------------------------------------------------


class TestAblationStudySingleParam:
    def test_correct_number_of_configs(self, tiny_dataset):
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125, "sigma": 2.0},
            param_grid={"learning_rate": [0.05, 0.125, 0.20]},
            dataset=tiny_dataset,
            dataset_name="Synthetic",
            verbose=False,
        )
        result = study.run()
        assert len(result.configs) == 3

    def test_all_configs_have_results(self, tiny_dataset):
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125},
            param_grid={"learning_rate": [0.05, 0.10, 0.15]},
            dataset=tiny_dataset,
            dataset_name="Synthetic",
            verbose=False,
        )
        result = study.run()
        for cfg in result.configs:
            assert cfg.result is not None
            assert cfg.success_auc >= 0.0
            assert cfg.mean_fps > 0.0
            assert cfg.peak_memory_mb > 0.0

    def test_configs_sorted_by_success_auc_descending(self, tiny_dataset):
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125},
            param_grid={"learning_rate": [0.05, 0.10, 0.15]},
            dataset=tiny_dataset,
            dataset_name="Synthetic",
            verbose=False,
        )
        result = study.run()
        aucs = [cfg.success_auc for cfg in result.configs]
        assert aucs == sorted(aucs, reverse=True)

    def test_best_config_is_first(self, tiny_dataset):
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125},
            param_grid={"learning_rate": [0.075, 0.125, 0.175]},
            dataset=tiny_dataset,
            dataset_name="Synthetic",
            verbose=False,
        )
        result = study.run()
        best = result.best_config()
        assert best.success_auc == result.configs[0].success_auc

    def test_wall_time_recorded(self, tiny_dataset):
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125},
            param_grid={"learning_rate": [0.05, 0.15]},
            dataset=tiny_dataset,
            dataset_name="Synthetic",
            verbose=False,
        )
        result = study.run()
        for cfg in result.configs:
            assert cfg.wall_time_s > 0.0


# ---------------------------------------------------------------------------
# AblationStudy — multi-parameter grid
# ---------------------------------------------------------------------------


class TestAblationStudyMultiParam:
    def test_cartesian_product_size(self, tiny_dataset):
        """3 × 2 grid → 6 configurations."""
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125, "sigma": 2.0},
            param_grid={
                "learning_rate": [0.05, 0.125, 0.20],
                "sigma": [1.0, 2.0],
            },
            dataset=tiny_dataset,
            dataset_name="Synthetic",
            verbose=False,
        )
        result = study.run()
        assert len(result.configs) == 6

    def test_param_combinations_are_unique(self, tiny_dataset):
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125, "sigma": 2.0},
            param_grid={
                "learning_rate": [0.05, 0.125],
                "sigma": [1.0, 2.0],
            },
            dataset=tiny_dataset,
            dataset_name="Synthetic",
            verbose=False,
        )
        result = study.run()
        param_tuples = [
            (cfg.params["learning_rate"], cfg.params["sigma"])
            for cfg in result.configs
        ]
        assert len(set(param_tuples)) == len(param_tuples)


# ---------------------------------------------------------------------------
# AblationStudy — max_sequences limiting
# ---------------------------------------------------------------------------


class TestAblationStudyMaxSequences:
    def test_max_sequences_limits_evaluation(self):
        dataset = SyntheticDataset(num_sequences=10, num_frames=20, seed=7)
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125},
            param_grid={"learning_rate": [0.10, 0.15]},
            dataset=dataset,
            dataset_name="Synthetic",
            verbose=False,
            max_sequences=3,
        )
        result = study.run()
        for cfg in result.configs:
            assert cfg.result is not None
            assert len(cfg.result.sequence_results) == 3


# ---------------------------------------------------------------------------
# AblationStudy — invalid inputs
# ---------------------------------------------------------------------------


class TestAblationStudyValidation:
    def test_empty_param_grid_raises(self, tiny_dataset):
        with pytest.raises(ValueError, match="param_grid"):
            AblationStudy(
                tracker_cls=MOSSETracker,
                base_params={"learning_rate": 0.125},
                param_grid={},
                dataset=tiny_dataset,
                dataset_name="Synthetic",
            )

    def test_invalid_tracker_params_raises(self, tiny_dataset):
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125},
            param_grid={"nonexistent_param": [1, 2]},
            dataset=tiny_dataset,
            dataset_name="Synthetic",
            verbose=False,
        )
        with pytest.raises((ValueError, TypeError)):
            study.run()


# ---------------------------------------------------------------------------
# AblationResult — sensitivity analysis
# ---------------------------------------------------------------------------


class TestSensitivityAnalysis:
    def _run_mosse_lr_study(self, dataset):
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125, "sigma": 2.0},
            param_grid={"learning_rate": [0.05, 0.125, 0.25]},
            dataset=dataset,
            dataset_name="Synthetic",
            verbose=False,
        )
        return study.run()

    def test_sensitivity_returns_one_entry_for_single_sweep(self, tiny_dataset):
        result = self._run_mosse_lr_study(tiny_dataset)
        entries = result.sensitivity_analysis()
        assert len(entries) == 1
        assert entries[0].param_name == "learning_rate"

    def test_sensitivity_entry_fields(self, tiny_dataset):
        result = self._run_mosse_lr_study(tiny_dataset)
        entry = result.sensitivity_analysis()[0]
        assert isinstance(entry, SensitivityEntry)
        assert len(entry.values_tested) == 3
        assert len(entry.success_aucs) == 3
        assert entry.impact >= 0.0
        assert entry.optimal_value in entry.values_tested

    def test_sensitivity_sorted_by_impact(self, tiny_dataset):
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125, "sigma": 2.0},
            param_grid={
                "learning_rate": [0.05, 0.125, 0.25],
                "sigma": [1.0, 2.0, 3.0],
            },
            dataset=tiny_dataset,
            dataset_name="Synthetic",
            verbose=False,
        )
        result = study.run()
        entries = result.sensitivity_analysis()
        impacts = [e.impact for e in entries]
        assert impacts == sorted(impacts, reverse=True)

    def test_sensitivity_to_markdown_contains_param_name(self, tiny_dataset):
        result = self._run_mosse_lr_study(tiny_dataset)
        md = result.sensitivity_to_markdown()
        assert "learning_rate" in md
        assert "Impact" in md


# ---------------------------------------------------------------------------
# AblationResult — reporting
# ---------------------------------------------------------------------------


class TestAblationResultReporting:
    def _make_result(self, dataset):
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125},
            param_grid={"learning_rate": [0.05, 0.125, 0.20]},
            dataset=dataset,
            dataset_name="Synthetic",
            verbose=False,
        )
        return study.run()

    def test_to_markdown_table_contains_headers(self, tiny_dataset):
        result = self._make_result(tiny_dataset)
        md = result.to_markdown_table()
        assert "Rank" in md
        assert "Success AUC" in md
        assert "learning_rate" in md

    def test_to_markdown_table_has_correct_row_count(self, tiny_dataset):
        result = self._make_result(tiny_dataset)
        md = result.to_markdown_table()
        # Header lines: title + blank + header + separator = 4 lines before data rows
        data_lines = [
            ln for ln in md.splitlines()
            if ln.startswith("| ") and "Rank" not in ln and "---" not in ln
            and "Ablation" not in ln
        ]
        assert len(data_lines) == 3

    def test_to_dict_structure(self, tiny_dataset):
        result = self._make_result(tiny_dataset)
        d = result.to_dict()
        assert "tracker" in d
        assert "best_config" in d
        assert "all_configs" in d
        assert "sensitivity" in d
        assert len(d["all_configs"]) == 3
        assert d["num_configs"] == 3

    def test_to_dict_best_config_has_highest_auc(self, tiny_dataset):
        result = self._make_result(tiny_dataset)
        d = result.to_dict()
        all_aucs = [c["success_auc"] for c in d["all_configs"]]
        assert d["best_config"]["success_auc"] == max(all_aucs)


# ---------------------------------------------------------------------------
# AblationConfig properties
# ---------------------------------------------------------------------------


class TestAblationConfigProperties:
    def test_defaults_when_result_is_none(self):
        cfg = AblationConfig(params={"learning_rate": 0.1})
        assert cfg.success_auc == 0.0
        assert cfg.mean_iou == 0.0
        assert cfg.mean_fps == 0.0
        assert cfg.peak_memory_mb == 0.0
