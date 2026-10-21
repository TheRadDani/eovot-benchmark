"""Tests for TrackerReliabilityAnalyzer and ReliabilityResult."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.reliability import TrackerReliabilityAnalyzer, ReliabilityResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def perfect_scores(n: int = 100) -> list:
    """Scores and IoUs both = 1.0 — perfectly calibrated, perfectly confident."""
    return [1.0] * n, [1.0] * n


def anticorrelated_scores(n: int = 100) -> list:
    """Scores = 1 when IoU = 0, and vice versa — worst-case calibration."""
    scores = [1.0 if i % 2 == 0 else 0.0 for i in range(n)]
    ious = [0.0 if i % 2 == 0 else 1.0 for i in range(n)]
    return scores, ious


def linear_scores(n: int = 100) -> tuple:
    """Scores linearly correlated with IoU."""
    v = np.linspace(0.0, 1.0, n)
    return v.tolist(), v.tolist()


# ---------------------------------------------------------------------------
# spearman_r
# ---------------------------------------------------------------------------

class TestSpearmanR:
    def test_perfect_correlation(self):
        analyzer = TrackerReliabilityAnalyzer()
        s, v = linear_scores(50)
        r = analyzer.spearman_r(np.array(s), np.array(v))
        assert r == pytest.approx(1.0, abs=0.01)

    def test_anticorrelation(self):
        analyzer = TrackerReliabilityAnalyzer()
        n = 50
        s = np.linspace(1.0, 0.0, n)
        v = np.linspace(0.0, 1.0, n)
        r = analyzer.spearman_r(s, v)
        assert r < -0.9

    def test_no_correlation(self):
        analyzer = TrackerReliabilityAnalyzer()
        rng = np.random.default_rng(0)
        s = rng.random(200)
        v = rng.random(200)
        r = analyzer.spearman_r(s, v)
        assert abs(r) < 0.3  # statistical noise only

    def test_fewer_than_two_frames_returns_zero(self):
        analyzer = TrackerReliabilityAnalyzer()
        assert analyzer.spearman_r(np.array([0.5]), np.array([0.5])) == pytest.approx(0.0)
        assert analyzer.spearman_r(np.array([]), np.array([])) == pytest.approx(0.0)

    def test_random_uncorrelated_near_zero(self):
        """Random uncorrelated arrays should give |r| < 0.3 for large n."""
        analyzer = TrackerReliabilityAnalyzer()
        rng = np.random.default_rng(99)
        s = rng.random(500)
        v = rng.random(500)
        r = analyzer.spearman_r(s, v)
        assert abs(r) < 0.15


# ---------------------------------------------------------------------------
# calibration_curve
# ---------------------------------------------------------------------------

class TestCalibrationCurve:
    def test_output_shapes(self):
        analyzer = TrackerReliabilityAnalyzer(n_bins=10)
        s, v = linear_scores(100)
        mids, accs, wts = analyzer.calibration_curve(np.array(s), np.array(v))
        assert mids.shape == (10,)
        assert accs.shape == (10,)
        assert wts.shape == (10,)

    def test_weights_sum_to_one(self):
        analyzer = TrackerReliabilityAnalyzer(n_bins=5)
        s, v = linear_scores(50)
        _, _, wts = analyzer.calibration_curve(np.array(s), np.array(v))
        assert wts.sum() == pytest.approx(1.0, abs=0.01)

    def test_empty_bins_have_zero_accuracy_and_weight(self):
        analyzer = TrackerReliabilityAnalyzer(n_bins=10)
        # All scores in [0, 0.2], so bins 2-9 are empty
        s = np.linspace(0.0, 0.2, 50)
        v = np.linspace(0.0, 0.2, 50)
        _, accs, wts = analyzer.calibration_curve(s, v)
        assert accs[8] == pytest.approx(0.0)
        assert wts[8] == pytest.approx(0.0)

    def test_perfect_calibration_near_diagonal(self):
        analyzer = TrackerReliabilityAnalyzer(n_bins=5)
        s = np.linspace(0.0, 1.0, 500)
        v = np.linspace(0.0, 1.0, 500)
        mids, accs, wts = analyzer.calibration_curve(s, v)
        for m, a, w in zip(mids, accs, wts):
            if w > 0:
                assert abs(m - a) < 0.12  # close to diagonal


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------

class TestECE:
    def test_perfect_calibration_low_ece(self):
        analyzer = TrackerReliabilityAnalyzer(n_bins=10)
        s = np.linspace(0.0, 1.0, 1000)
        v = np.linspace(0.0, 1.0, 1000)
        ece = analyzer.expected_calibration_error(s, v)
        assert ece < 0.05

    def test_worst_case_calibration_high_ece(self):
        analyzer = TrackerReliabilityAnalyzer(n_bins=10)
        # Overconfident: always says 1.0 but IoU is 0.0
        s = np.ones(100)
        v = np.zeros(100)
        ece = analyzer.expected_calibration_error(s, v)
        assert ece > 0.5

    def test_ece_non_negative(self):
        analyzer = TrackerReliabilityAnalyzer()
        rng = np.random.default_rng(1)
        s = rng.random(100)
        v = rng.random(100)
        ece = analyzer.expected_calibration_error(s, v)
        assert ece >= 0.0


# ---------------------------------------------------------------------------
# optimal_threshold
# ---------------------------------------------------------------------------

class TestOptimalThreshold:
    def test_threshold_in_unit_interval(self):
        analyzer = TrackerReliabilityAnalyzer()
        s, v = linear_scores(100)
        thr, f1 = analyzer.optimal_threshold(np.array(s), np.array(v))
        assert 0.0 <= thr <= 1.0

    def test_f1_in_unit_interval(self):
        analyzer = TrackerReliabilityAnalyzer()
        s, v = linear_scores(100)
        _, f1 = analyzer.optimal_threshold(np.array(s), np.array(v))
        assert 0.0 <= f1 <= 1.0

    def test_perfect_classifier_high_f1(self):
        analyzer = TrackerReliabilityAnalyzer(iou_good_threshold=0.5)
        n = 100
        # Scores perfectly separate good/bad frames
        s = np.array([1.0] * 50 + [0.0] * 50)
        v = np.array([0.8] * 50 + [0.1] * 50)
        _, f1 = analyzer.optimal_threshold(s, v)
        assert f1 > 0.95

    def test_fewer_than_two_frames(self):
        analyzer = TrackerReliabilityAnalyzer()
        thr, f1 = analyzer.optimal_threshold(np.array([0.5]), np.array([0.5]))
        assert thr == pytest.approx(0.5)
        assert f1 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# analyze (end-to-end)
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_reliability_result(self):
        analyzer = TrackerReliabilityAnalyzer()
        s, v = linear_scores(50)
        result = analyzer.analyze(s, v, tracker_name="Test")
        assert isinstance(result, ReliabilityResult)

    def test_tracker_name_stored(self):
        analyzer = TrackerReliabilityAnalyzer()
        result = analyzer.analyze([0.5], [0.5], tracker_name="Foo")
        assert result.tracker_name == "Foo"

    def test_n_frames_correct(self):
        analyzer = TrackerReliabilityAnalyzer()
        s, v = linear_scores(80)
        result = analyzer.analyze(s, v)
        assert result.n_frames == 80

    def test_high_spearman_for_correlated_scores(self):
        analyzer = TrackerReliabilityAnalyzer()
        s, v = linear_scores(200)
        result = analyzer.analyze(s, v)
        assert result.spearman_r > 0.95

    def test_iou_good_threshold_stored(self):
        analyzer = TrackerReliabilityAnalyzer(iou_good_threshold=0.6)
        result = analyzer.analyze([0.5, 0.5], [0.7, 0.3])
        assert result.iou_good_threshold == pytest.approx(0.6)

    def test_bin_arrays_shape(self):
        analyzer = TrackerReliabilityAnalyzer(n_bins=8)
        s, v = linear_scores(100)
        result = analyzer.analyze(s, v)
        assert result.bin_confidences.shape == (8,)
        assert result.bin_accuracies.shape == (8,)
        assert result.bin_weights.shape == (8,)

    def test_str_representation(self):
        analyzer = TrackerReliabilityAnalyzer()
        s, v = linear_scores(50)
        result = analyzer.analyze(s, v, tracker_name="MOSSE")
        text = str(result)
        assert "MOSSE" in text
        assert "Spearman_r" in text
        assert "ECE" in text


# ---------------------------------------------------------------------------
# Integration with eovot.metrics public API
# ---------------------------------------------------------------------------

class TestPublicApiExport:
    def test_importable_from_metrics(self):
        from eovot.metrics import TrackerReliabilityAnalyzer, ReliabilityResult
        assert TrackerReliabilityAnalyzer is not None
        assert ReliabilityResult is not None
