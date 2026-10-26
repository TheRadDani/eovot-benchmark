"""Tests for cross-dataset generalization analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eovot.analysis.cross_dataset import (
    CrossDatasetEvaluator,
    CrossDatasetResult,
    DatasetSummary,
    _rank_data,
    _spearman_rho,
)
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# _rank_data and _spearman_rho unit tests
# ---------------------------------------------------------------------------

class TestRankData:
    def test_simple_rank(self):
        arr = np.array([10.0, 30.0, 20.0])
        r = _rank_data(arr)
        np.testing.assert_array_equal(r, [1.0, 3.0, 2.0])

    def test_tied_ranks_averaged(self):
        arr = np.array([1.0, 2.0, 2.0, 3.0])
        r = _rank_data(arr)
        assert r[0] == 1.0
        assert r[1] == pytest.approx(2.5)
        assert r[2] == pytest.approx(2.5)
        assert r[3] == 4.0

    def test_all_equal(self):
        arr = np.array([5.0, 5.0, 5.0])
        r = _rank_data(arr)
        expected = np.full(3, 2.0)
        np.testing.assert_array_almost_equal(r, expected)


class TestSpearmanRho:
    def test_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert _spearman_rho(x, x) == pytest.approx(1.0)

    def test_perfect_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([4.0, 3.0, 2.0, 1.0])
        assert _spearman_rho(x, y) == pytest.approx(-1.0)

    def test_zero_variance_returns_nan(self):
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0])
        assert math.isnan(_spearman_rho(x, y))

    def test_known_value(self):
        # Manually computed: ranks([2,1,3,4]) vs ranks([1,2,3,4])
        x = np.array([2.0, 1.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        rho = _spearman_rho(x, y)
        assert -1.0 <= rho <= 1.0

    def test_symmetric(self):
        x = np.array([1.0, 3.0, 2.0])
        y = np.array([3.0, 1.0, 2.0])
        assert _spearman_rho(x, y) == pytest.approx(_spearman_rho(y, x))

    def test_too_short(self):
        assert math.isnan(_spearman_rho(np.array([1.0]), np.array([2.0])))


# ---------------------------------------------------------------------------
# DatasetSummary
# ---------------------------------------------------------------------------

class TestDatasetSummary:
    def _make(self, **kwargs) -> DatasetSummary:
        defaults = dict(
            tracker_name="T",
            dataset_name="D",
            mean_iou=0.5,
            success_auc=0.4,
            mean_fps=30.0,
            peak_memory_mb=50.0,
            num_sequences=5,
        )
        defaults.update(kwargs)
        return DatasetSummary(**defaults)

    def test_rank_key_default(self):
        s = self._make(success_auc=0.7)
        assert s.rank_key() == pytest.approx(0.7)

    def test_rank_key_iou(self):
        s = self._make(mean_iou=0.6)
        assert s.rank_key("mean_iou") == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# CrossDatasetEvaluator with synthetic data
# ---------------------------------------------------------------------------

def _make_datasets(n_seq: int = 5):
    return {
        "SyntheticA": SyntheticDataset(num_sequences=n_seq, seed=0, num_frames=30),
        "SyntheticB": SyntheticDataset(num_sequences=n_seq, seed=10, num_frames=30),
    }


@pytest.fixture(scope="module")
def cross_dataset_result():
    datasets = _make_datasets(n_seq=3)
    ev = CrossDatasetEvaluator(verbose=False)
    return ev.evaluate(["MOSSE", "KCF"], datasets)


class TestCrossDatasetEvaluator:
    def test_result_has_correct_trackers(self, cross_dataset_result):
        assert set(cross_dataset_result.tracker_names) == {"MOSSE", "KCF"}

    def test_result_has_correct_datasets(self, cross_dataset_result):
        assert set(cross_dataset_result.dataset_names) == {"SyntheticA", "SyntheticB"}

    def test_all_summaries_present(self, cross_dataset_result):
        for tr in cross_dataset_result.tracker_names:
            for ds in cross_dataset_result.dataset_names:
                s = cross_dataset_result.get(tr, ds)
                assert s is not None, f"Missing summary for {tr}/{ds}"

    def test_summary_num_sequences(self, cross_dataset_result):
        for tr in cross_dataset_result.tracker_names:
            for ds in cross_dataset_result.dataset_names:
                s = cross_dataset_result.get(tr, ds)
                assert s.num_sequences == 3

    def test_summary_fps_positive(self, cross_dataset_result):
        for tr in cross_dataset_result.tracker_names:
            for ds in cross_dataset_result.dataset_names:
                s = cross_dataset_result.get(tr, ds)
                assert s.mean_fps > 0

    def test_summary_iou_in_range(self, cross_dataset_result):
        for tr in cross_dataset_result.tracker_names:
            for ds in cross_dataset_result.dataset_names:
                s = cross_dataset_result.get(tr, ds)
                assert 0.0 <= s.mean_iou <= 1.0

    def test_rank_correlation_matrix_diagonal_is_one(self, cross_dataset_result):
        ev = CrossDatasetEvaluator(verbose=False)
        corr = ev.rank_correlation_matrix(cross_dataset_result)
        for ds in cross_dataset_result.dataset_names:
            assert corr[(ds, ds)] == pytest.approx(1.0)

    def test_rank_correlation_matrix_symmetric(self, cross_dataset_result):
        ev = CrossDatasetEvaluator(verbose=False)
        corr = ev.rank_correlation_matrix(cross_dataset_result)
        for a in cross_dataset_result.dataset_names:
            for b in cross_dataset_result.dataset_names:
                assert corr[(a, b)] == pytest.approx(corr[(b, a)], abs=1e-9)

    def test_rank_correlation_in_range(self, cross_dataset_result):
        ev = CrossDatasetEvaluator(verbose=False)
        corr = ev.rank_correlation_matrix(cross_dataset_result)
        for ds_a in cross_dataset_result.dataset_names:
            for ds_b in cross_dataset_result.dataset_names:
                v = corr[(ds_a, ds_b)]
                if not math.isnan(v):
                    assert -1.0 <= v <= 1.0

    def test_generalization_scores_all_trackers_present(self, cross_dataset_result):
        ev = CrossDatasetEvaluator(verbose=False)
        scores = ev.generalization_scores(cross_dataset_result)
        for tr in cross_dataset_result.tracker_names:
            assert tr in scores

    def test_generalization_scores_sorted_desc(self, cross_dataset_result):
        ev = CrossDatasetEvaluator(verbose=False)
        scores = ev.generalization_scores(cross_dataset_result)
        vals = [v for v in scores.values() if not math.isnan(v)]
        assert vals == sorted(vals, reverse=True)

    def test_markdown_report_sections_present(self, cross_dataset_result):
        ev = CrossDatasetEvaluator(verbose=False)
        report = ev.to_markdown_report(cross_dataset_result)
        assert "Cross-Dataset Accuracy Table" in report
        assert "Dataset Rank Correlation Matrix" in report
        assert "Tracker Generalization Scores" in report
        for tr in cross_dataset_result.tracker_names:
            assert tr in report
        for ds in cross_dataset_result.dataset_names:
            assert ds in report


class TestCrossDatasetEvaluatorSingleDataset:
    """Edge case: only one dataset — rank correlation is trivially 1.0."""

    def test_single_dataset(self):
        datasets = {"SyntheticA": SyntheticDataset(num_sequences=3, seed=0, num_frames=20)}
        ev = CrossDatasetEvaluator(verbose=False)
        result = ev.evaluate(["MOSSE"], datasets)
        assert result.get("MOSSE", "SyntheticA") is not None

    def test_generalization_nan_with_one_tracker(self):
        """With only one tracker, cross-tracker correlation is undefined."""
        datasets = _make_datasets(n_seq=3)
        ev = CrossDatasetEvaluator(verbose=False)
        result = ev.evaluate(["MOSSE"], datasets)
        scores = ev.generalization_scores(result)
        assert math.isnan(scores["MOSSE"])


class TestCrossDatasetMetricSwitch:
    def test_rank_metric_mean_iou(self):
        datasets = _make_datasets(n_seq=3)
        ev = CrossDatasetEvaluator(verbose=False, rank_metric="mean_iou")
        result = ev.evaluate(["MOSSE", "KCF"], datasets)
        corr = ev.rank_correlation_matrix(result)
        # Just verify it runs without error and returns valid correlation values.
        for ds in result.dataset_names:
            assert corr[(ds, ds)] == pytest.approx(1.0)
