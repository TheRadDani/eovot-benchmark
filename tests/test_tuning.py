"""Tests for the hyperparameter tuning module (eovot/tuning/).

Covers GridSearchRunner, SensitivityAnalyzer, and their data-classes using
the built-in SyntheticDataset so no external data is required.
"""

from __future__ import annotations

import math
import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.tuning.grid_search import GridSearchRunner, TuningEntry, TuningResult
from eovot.tuning.sensitivity import (
    SensitivityAnalyzer,
    SensitivityReport,
    ParameterSensitivity,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=20, seed=0)


# ---------------------------------------------------------------------------
# GridSearchRunner
# ---------------------------------------------------------------------------

class TestGridSearchRunner:
    def test_basic_run_returns_tuning_result(self, small_dataset):
        runner = GridSearchRunner(
            tracker_class=MOSSETracker,
            param_grid={"learning_rate": [0.05, 0.125], "sigma": [1.5, 2.0]},
            metric="mean_iou",
            verbose=False,
        )
        result = runner.run(small_dataset, dataset_name="Synthetic")
        assert isinstance(result, TuningResult)

    def test_total_combinations_matches_grid(self, small_dataset):
        runner = GridSearchRunner(
            tracker_class=MOSSETracker,
            param_grid={"learning_rate": [0.05, 0.1, 0.125], "sigma": [1.0, 2.0]},
            metric="mean_iou",
            verbose=False,
        )
        result = runner.run(small_dataset)
        assert result.total_combinations == 6
        assert len(result.entries) == 6

    def test_entries_sorted_best_first(self, small_dataset):
        runner = GridSearchRunner(
            tracker_class=MOSSETracker,
            param_grid={"learning_rate": [0.05, 0.125]},
            metric="mean_iou",
            verbose=False,
        )
        result = runner.run(small_dataset)
        values = [e.metric_value for e in result.entries]
        assert values == sorted(values, reverse=True), "Entries should be sorted best-first"

    def test_lower_is_better_sorted_ascending(self, small_dataset):
        runner = GridSearchRunner(
            tracker_class=MOSSETracker,
            param_grid={"learning_rate": [0.05, 0.125]},
            metric="peak_memory_mb",
            verbose=False,
        )
        result = runner.run(small_dataset)
        values = [e.metric_value for e in result.entries]
        assert values == sorted(values), "peak_memory_mb should sort ascending (lower=better)"

    def test_best_params_keys_match_grid(self, small_dataset):
        param_grid = {"learning_rate": [0.05, 0.125], "sigma": [1.5, 2.0]}
        runner = GridSearchRunner(
            tracker_class=MOSSETracker,
            param_grid=param_grid,
            metric="mean_iou",
            verbose=False,
        )
        result = runner.run(small_dataset)
        assert set(result.best_params.keys()) == set(param_grid.keys())

    def test_metric_values_are_finite(self, small_dataset):
        runner = GridSearchRunner(
            tracker_class=MOSSETracker,
            param_grid={"sigma": [1.0, 2.0, 3.0]},
            metric="mean_iou",
            verbose=False,
        )
        result = runner.run(small_dataset)
        for e in result.entries:
            assert math.isfinite(e.metric_value)

    def test_empty_param_grid_raises(self):
        with pytest.raises(ValueError):
            GridSearchRunner(
                tracker_class=MOSSETracker,
                param_grid={},
                metric="mean_iou",
            )

    def test_invalid_metric_raises(self, small_dataset):
        runner = GridSearchRunner(
            tracker_class=MOSSETracker,
            param_grid={"sigma": [2.0]},
            metric="nonexistent_metric",
            verbose=False,
        )
        with pytest.raises(AttributeError):
            runner.run(small_dataset)

    def test_kcf_grid_search(self, small_dataset):
        runner = GridSearchRunner(
            tracker_class=KCFTracker,
            param_grid={"learning_rate": [0.05, 0.075], "kernel_sigma": [0.3, 0.5]},
            metric="mean_iou",
            verbose=False,
        )
        result = runner.run(small_dataset)
        assert len(result.entries) == 4

    def test_to_markdown_contains_metric_name(self, small_dataset):
        runner = GridSearchRunner(
            tracker_class=MOSSETracker,
            param_grid={"sigma": [1.0, 2.0]},
            metric="mean_iou",
            verbose=False,
        )
        result = runner.run(small_dataset)
        md = result.to_markdown()
        assert "mean_iou" in md
        assert "MOSSETracker" in md

    def test_save_csv(self, small_dataset, tmp_path):
        runner = GridSearchRunner(
            tracker_class=MOSSETracker,
            param_grid={"sigma": [1.0, 2.0]},
            metric="mean_iou",
            verbose=False,
        )
        result = runner.run(small_dataset)
        csv_path = str(tmp_path / "grid.csv")
        result.save_csv(csv_path)
        import csv
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert "sigma" in rows[0]
        assert "mean_iou" in rows[0]

    def test_max_sequences_respected(self, small_dataset):
        runner = GridSearchRunner(
            tracker_class=MOSSETracker,
            param_grid={"sigma": [1.5, 2.0]},
            metric="mean_iou",
            verbose=False,
            max_sequences=2,
        )
        result = runner.run(small_dataset)
        assert len(result.entries) == 2


# ---------------------------------------------------------------------------
# SensitivityAnalyzer
# ---------------------------------------------------------------------------

class TestSensitivityAnalyzer:
    def test_basic_run_returns_report(self, small_dataset):
        analyzer = SensitivityAnalyzer(
            tracker_class=MOSSETracker,
            base_params={"learning_rate": 0.125, "sigma": 2.0},
            param_ranges={
                "learning_rate": [0.05, 0.125, 0.2],
                "sigma": [1.0, 2.0, 3.0],
            },
            metric="mean_iou",
            verbose=False,
        )
        report = analyzer.run(small_dataset, dataset_name="Synthetic")
        assert isinstance(report, SensitivityReport)

    def test_report_has_correct_number_of_sensitivities(self, small_dataset):
        analyzer = SensitivityAnalyzer(
            tracker_class=MOSSETracker,
            base_params={"learning_rate": 0.125, "sigma": 2.0},
            param_ranges={
                "learning_rate": [0.05, 0.125, 0.2],
                "sigma": [1.0, 2.0, 3.0],
            },
            metric="mean_iou",
            verbose=False,
        )
        report = analyzer.run(small_dataset)
        assert len(report.sensitivities) == 2

    def test_sensitivities_sorted_by_absolute_score(self, small_dataset):
        analyzer = SensitivityAnalyzer(
            tracker_class=MOSSETracker,
            base_params={"learning_rate": 0.125, "sigma": 2.0},
            param_ranges={
                "learning_rate": [0.05, 0.10, 0.125, 0.175],
                "sigma": [1.0, 2.0, 3.0, 4.0],
            },
            metric="mean_iou",
            verbose=False,
        )
        report = analyzer.run(small_dataset)
        scores = [abs(s.sensitivity_score) for s in report.sensitivities]
        assert scores == sorted(scores, reverse=True)

    def test_metric_range_nonnegative(self, small_dataset):
        analyzer = SensitivityAnalyzer(
            tracker_class=MOSSETracker,
            base_params={"learning_rate": 0.125, "sigma": 2.0},
            param_ranges={"learning_rate": [0.05, 0.125, 0.2]},
            metric="mean_iou",
            verbose=False,
        )
        report = analyzer.run(small_dataset)
        for s in report.sensitivities:
            assert s.metric_range >= 0.0

    def test_empty_param_ranges_raises(self):
        with pytest.raises(ValueError):
            SensitivityAnalyzer(
                tracker_class=MOSSETracker,
                base_params={"learning_rate": 0.125},
                param_ranges={},
                metric="mean_iou",
            )

    def test_most_sensitive_param_is_first(self, small_dataset):
        analyzer = SensitivityAnalyzer(
            tracker_class=MOSSETracker,
            base_params={"learning_rate": 0.125, "sigma": 2.0},
            param_ranges={
                "learning_rate": [0.05, 0.125, 0.2],
                "sigma": [1.5, 2.0, 2.5],
            },
            metric="mean_iou",
            verbose=False,
        )
        report = analyzer.run(small_dataset)
        assert report.most_sensitive_param == report.sensitivities[0].param_name

    def test_to_markdown_contains_key_sections(self, small_dataset):
        analyzer = SensitivityAnalyzer(
            tracker_class=MOSSETracker,
            base_params={"learning_rate": 0.125, "sigma": 2.0},
            param_ranges={
                "learning_rate": [0.05, 0.125, 0.2],
                "sigma": [1.0, 2.0, 3.0],
            },
            metric="mean_iou",
            verbose=False,
        )
        report = analyzer.run(small_dataset)
        md = report.to_markdown()
        assert "Sensitivity Ranking" in md
        assert "learning_rate" in md
        assert "sigma" in md

    def test_to_dict_structure(self, small_dataset):
        analyzer = SensitivityAnalyzer(
            tracker_class=KCFTracker,
            base_params={"learning_rate": 0.075, "kernel_sigma": 0.5},
            param_ranges={"kernel_sigma": [0.3, 0.5, 0.7]},
            metric="mean_iou",
            verbose=False,
        )
        report = analyzer.run(small_dataset)
        d = report.to_dict()
        assert d["tracker"] == "KCFTracker"
        assert "baseline_metric" in d
        assert "sensitivities" in d
        assert isinstance(d["sensitivities"], list)

    def test_kcf_sensitivity(self, small_dataset):
        analyzer = SensitivityAnalyzer(
            tracker_class=KCFTracker,
            base_params={"learning_rate": 0.075, "kernel_sigma": 0.5},
            param_ranges={
                "learning_rate": [0.05, 0.075, 0.1],
                "kernel_sigma": [0.3, 0.5, 0.7],
            },
            metric="mean_fps",
            verbose=False,
        )
        report = analyzer.run(small_dataset)
        assert len(report.sensitivities) == 2
        for s in report.sensitivities:
            assert s.sensitivity_score >= 0.0


# ---------------------------------------------------------------------------
# ParameterSensitivity helpers
# ---------------------------------------------------------------------------

class TestParameterSensitivity:
    def _make(self, values, metrics):
        return ParameterSensitivity(
            param_name="lr",
            values_tested=values,
            metric_values=metrics,
            baseline_value=0.1,
            baseline_metric=0.5,
            sensitivity_score=(max(metrics) - min(metrics)) / 0.5,
            metric_name="mean_iou",
        )

    def test_best_value_is_argmax(self):
        s = self._make([0.05, 0.1, 0.2], [0.3, 0.6, 0.4])
        assert s.best_value == 0.1

    def test_worst_value_is_argmin(self):
        s = self._make([0.05, 0.1, 0.2], [0.3, 0.6, 0.4])
        assert s.worst_value == 0.05

    def test_metric_range(self):
        s = self._make([0.05, 0.1, 0.2], [0.3, 0.6, 0.4])
        assert abs(s.metric_range - 0.3) < 1e-9
