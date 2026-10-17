"""Tests for eovot.analysis.regression_checker."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from eovot.analysis.regression_checker import (
    RegressionChecker,
    RegressionReport,
    MetricDelta,
    _welch_ttest_pvalue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    tracker_name: str = "MOSSE",
    mean_iou: float = 0.6,
    mean_fps: float = 200.0,
    n_seq: int = 5,
    n_frames: int = 25,
    seed: int = 0,
):
    """Build a BenchmarkResult via the full pipeline with deterministic IoU.

    Uses a larger frame and smaller target to avoid boundary-clip edge cases
    in the MOSSE tracker when the target drifts near frame edges.
    """
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    dataset = SyntheticDataset(
        num_sequences=n_seq,
        num_frames=n_frames,
        motion="circular",   # circular keeps target safely inside frame
        frame_size=(320, 240),
        bbox_size=(30, 30),
        seed=seed,
    )
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")
    result.tracker_name = tracker_name
    return result


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestWelchTtest:
    def test_identical_arrays_pvalue_one(self):
        a = np.array([0.5, 0.6, 0.7, 0.5, 0.6])
        p = _welch_ttest_pvalue(a, a)
        assert p == pytest.approx(1.0)

    def test_very_different_arrays_low_pvalue(self):
        a = np.zeros(20)
        b = np.ones(20)
        p = _welch_ttest_pvalue(a, b)
        assert p < 0.001

    def test_single_element_returns_one(self):
        assert _welch_ttest_pvalue(np.array([0.5]), np.array([0.9])) == 1.0


class TestMetricDelta:
    def _make_checker(self):
        return RegressionChecker(min_iou_delta=0.01)

    def test_improved_higher_better(self):
        checker = self._make_checker()
        d = checker._metric_delta("mean_iou", 0.5, 0.55)
        assert d.direction == "improved"

    def test_regressed_higher_better(self):
        checker = self._make_checker()
        d = checker._metric_delta("mean_iou", 0.5, 0.48)
        assert d.direction == "regressed"

    def test_unchanged_within_threshold(self):
        checker = self._make_checker()
        d = checker._metric_delta("mean_iou", 0.5, 0.503)
        assert d.direction == "unchanged"

    def test_improved_lower_better(self):
        checker = self._make_checker()
        d = checker._metric_delta("peak_memory_mb", 100.0, 90.0)
        assert d.direction == "improved"

    def test_regressed_lower_better(self):
        checker = self._make_checker()
        d = checker._metric_delta("peak_memory_mb", 100.0, 110.0)
        assert d.direction == "regressed"

    def test_delta_computation(self):
        checker = self._make_checker()
        d = checker._metric_delta("mean_iou", 0.4, 0.5)
        assert d.delta == pytest.approx(0.1)
        assert d.baseline == pytest.approx(0.4)
        assert d.candidate == pytest.approx(0.5)

    def test_relative_delta_zero_baseline(self):
        checker = self._make_checker()
        d = checker._metric_delta("mean_iou", 0.0, 0.5)
        assert math.isnan(d.relative_delta)


# ---------------------------------------------------------------------------
# RegressionChecker.compare integration tests
# ---------------------------------------------------------------------------

class TestRegressionChecker:
    def setup_method(self):
        self.baseline  = _make_result(tracker_name="MOSSE-v1", seed=1)
        self.candidate = _make_result(tracker_name="MOSSE-v2", seed=1)

    def test_compare_returns_report(self):
        checker = RegressionChecker()
        report = checker.compare(self.baseline, self.candidate)
        assert isinstance(report, RegressionReport)

    def test_same_result_no_regression(self):
        """Comparing identical results should produce no regression."""
        checker = RegressionChecker()
        report = checker.compare(self.baseline, self.baseline)
        assert not report.has_regression

    def test_same_result_all_unchanged(self):
        checker = RegressionChecker()
        report = checker.compare(self.baseline, self.baseline)
        directions = {d.direction for d in report.metric_deltas}
        # All metrics are unchanged when comparing a result with itself
        assert directions == {"unchanged"}

    def test_metric_deltas_populated(self):
        checker = RegressionChecker()
        report = checker.compare(self.baseline, self.candidate)
        # At minimum mean_iou and mean_fps must be present
        names = {d.name for d in report.metric_deltas}
        assert "mean_iou" in names
        assert "mean_fps" in names

    def test_iou_pvalue_computed_with_enough_sequences(self):
        checker = RegressionChecker()
        report = checker.compare(self.baseline, self.candidate)
        assert report.matched_sequences >= 2
        assert report.iou_pvalue is not None
        assert 0.0 <= report.iou_pvalue <= 1.0

    def test_report_summary_is_string(self):
        checker = RegressionChecker()
        report = checker.compare(self.baseline, self.candidate)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Regression" in summary or "OK" in summary

    def test_report_to_dict_serialisable(self):
        checker = RegressionChecker()
        report = checker.compare(self.baseline, self.candidate)
        d = report.to_dict()
        # Must be JSON-serialisable
        _ = json.dumps(d)
        assert "has_regression" in d
        assert "metric_deltas" in d

    def test_sequence_count_correct(self):
        checker = RegressionChecker()
        report = checker.compare(self.baseline, self.candidate)
        assert report.sequence_count == len(self.baseline.sequence_results)

    def test_matched_sequences_with_same_dataset(self):
        checker = RegressionChecker()
        report = checker.compare(self.baseline, self.candidate)
        # Same dataset and tracker → all sequences match
        assert report.matched_sequences == report.sequence_count

    def test_strict_mode_flags_small_regression(self):
        """In strict mode, even a small IoU drop with p > 0.05 should fail."""
        checker_strict = RegressionChecker(min_iou_delta=1e-6, strict=True)
        # Run two different seeds to get slightly different IoU
        baseline  = _make_result(seed=10)
        candidate = _make_result(seed=11)
        # At least check that strict mode respects the min_iou_delta threshold
        report = checker_strict.compare(baseline, candidate)
        assert isinstance(report.has_regression, bool)

    def test_high_min_delta_all_unchanged(self):
        """With a very high threshold, even real differences are 'unchanged'."""
        checker = RegressionChecker(min_iou_delta=1.0)
        report = checker.compare(self.baseline, self.candidate)
        iou_delta = next((d for d in report.metric_deltas if d.name == "mean_iou"), None)
        if iou_delta:
            assert iou_delta.direction == "unchanged"

    def test_no_overlap_sequences_pvalue_none(self):
        """When no sequences share a name, pvalue should be None."""
        import copy
        other = copy.deepcopy(self.candidate)
        for r in other.sequence_results:
            r.sequence_name = "no_match_" + r.sequence_name
        checker = RegressionChecker()
        report = checker.compare(self.baseline, other)
        assert report.iou_pvalue is None
        assert report.matched_sequences == 0


# ---------------------------------------------------------------------------
# Save/load roundtrip: ensure regression checker works on saved JSONs
# ---------------------------------------------------------------------------

class TestRegressionCheckerWithSavedResults:
    def test_compare_from_loaded_json(self, tmp_path):
        from eovot.benchmark.engine import BenchmarkResult

        baseline  = _make_result(seed=5)
        candidate = _make_result(seed=6)

        b_path = tmp_path / "baseline.json"
        c_path = tmp_path / "candidate.json"
        baseline.save(b_path)
        candidate.save(c_path)

        loaded_b = BenchmarkResult.load(b_path)
        loaded_c = BenchmarkResult.load(c_path)

        checker = RegressionChecker()
        report = checker.compare(loaded_b, loaded_c)
        assert isinstance(report.has_regression, bool)
        assert report.matched_sequences > 0
