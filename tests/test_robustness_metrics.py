"""Unit tests for eovot.metrics.robustness."""

import numpy as np
import pytest

from eovot.metrics.robustness import (
    RobustnessMetrics,
    compute_robustness,
    normalized_precision_curve,
    success_rate_at_threshold,
)


class TestComputeRobustness:
    """Tests for the core failure-detection logic."""

    def test_empty_ious_returns_zero_metrics(self):
        r = compute_robustness(np.array([]))
        assert r.failure_count == 0
        assert r.failure_rate == pytest.approx(0.0)
        assert r.tracking_success_rate == pytest.approx(0.0)
        assert r.failure_frames == []

    def test_perfect_tracking_no_failures(self):
        ious = np.ones(50)
        r = compute_robustness(ious)
        assert r.failure_count == 0
        assert r.failure_rate == pytest.approx(0.0)
        assert r.tracking_success_rate == pytest.approx(1.0)
        assert r.mean_iou_until_first_failure == pytest.approx(1.0)

    def test_one_failure_detected(self):
        # Frames 10–17 all have IoU=0 (8 consecutive → ≥ min_failure_length=5)
        ious = np.ones(30)
        ious[10:18] = 0.0
        r = compute_robustness(ious, failure_threshold=0.1, min_failure_length=5)
        assert r.failure_count == 1
        assert r.failure_frames == [10]

    def test_short_dip_below_threshold_not_counted(self):
        # Only 3 consecutive failure frames → below min_failure_length=5
        ious = np.ones(20)
        ious[5:8] = 0.0
        r = compute_robustness(ious, failure_threshold=0.1, min_failure_length=5)
        assert r.failure_count == 0

    def test_two_separate_failures(self):
        ious = np.ones(60)
        ious[5:12] = 0.0   # 7 frames — counts
        ious[40:47] = 0.0  # 7 frames — counts
        r = compute_robustness(ious, failure_threshold=0.1, min_failure_length=5)
        assert r.failure_count == 2
        assert r.failure_frames[0] == 5
        assert r.failure_frames[1] == 40

    def test_failure_rate_normalised_correctly(self):
        # 1 failure in 100 frames → 1.0 per 100 frames
        ious = np.ones(100)
        ious[10:20] = 0.0  # 10 consecutive failure frames
        r = compute_robustness(ious, failure_threshold=0.1, min_failure_length=5)
        assert r.failure_count == 1
        assert r.failure_rate == pytest.approx(1.0)

    def test_mean_iou_until_first_failure(self):
        ious = np.array([0.8, 0.75, 0.7, 0.72, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        r = compute_robustness(ious, failure_threshold=0.1, min_failure_length=4)
        assert r.failure_count == 1
        assert r.failure_frames == [4]
        expected_mean = float(ious[:4].mean())
        assert r.mean_iou_until_first_failure == pytest.approx(expected_mean)

    def test_failure_at_start_gives_none_mean(self):
        ious = np.zeros(10)
        r = compute_robustness(ious, failure_threshold=0.1, min_failure_length=5)
        # Failure starts at frame 0 → no "before failure" portion
        assert r.mean_iou_until_first_failure is None

    def test_failure_at_end_of_sequence(self):
        ious = np.concatenate([np.ones(15), np.zeros(8)])
        r = compute_robustness(ious, failure_threshold=0.1, min_failure_length=5)
        assert r.failure_count == 1
        assert r.failure_frames == [15]

    def test_tracking_success_rate(self):
        ious = np.array([0.8, 0.5, 0.05, 0.05, 0.9])  # 2/5 below threshold=0.1
        r = compute_robustness(ious, failure_threshold=0.1, min_failure_length=1)
        assert r.tracking_success_rate == pytest.approx(3.0 / 5.0)

    def test_to_dict_serialisable(self):
        ious = np.ones(20)
        r = compute_robustness(ious)
        d = r.to_dict()
        assert "failure_count" in d
        assert "failure_rate_per_100_frames" in d
        assert "tracking_success_rate" in d


class TestNormalizedPrecisionCurve:
    """Tests for the LaSOT normalised precision metric."""

    def test_perfect_predictions_max_rate(self):
        boxes = np.array([[10.0, 10.0, 20.0, 20.0]] * 10)
        thr, rates = normalized_precision_curve(boxes, boxes)
        # At any threshold > 0, distance = 0 < threshold → rate = 1.0
        assert rates[-1] == pytest.approx(1.0)

    def test_empty_arrays(self):
        thr, rates = normalized_precision_curve(np.empty((0, 4)), np.empty((0, 4)))
        assert np.all(rates == 0.0)

    def test_output_shape_matches_thresholds(self):
        preds = np.random.rand(30, 4) * 50 + 1
        gts = np.random.rand(30, 4) * 50 + 1
        thr, rates = normalized_precision_curve(preds, gts)
        assert len(thr) == len(rates)
        assert np.all(rates >= 0.0) and np.all(rates <= 1.0)

    def test_zero_gt_area_treated_as_inf(self):
        preds = np.array([[0.0, 0.0, 10.0, 10.0]] * 5)
        gts = np.array([[0.0, 0.0, 0.0, 0.0]] * 5)  # degenerate GT
        thr, rates = normalized_precision_curve(preds, gts)
        # All normalised distances are inf → no frame passes any threshold
        assert np.all(rates == 0.0)

    def test_rates_monotonically_non_decreasing(self):
        preds = np.random.rand(50, 4) * 100 + 1
        gts = np.random.rand(50, 4) * 100 + 1
        thr, rates = normalized_precision_curve(preds, gts)
        assert np.all(np.diff(rates) >= -1e-9)

    def test_scale_invariance(self):
        """Same relative displacement at 2× scale should give same normalised dist."""
        # 10×10 box, centre offset by (5, 0) → norm_dist = 5 / diag(10,10) ≈ 0.354
        preds_small = np.array([[15.0, 10.0, 10.0, 10.0]])
        gts_small = np.array([[10.0, 10.0, 10.0, 10.0]])
        # 20×20 box, centre offset by (10, 0) → norm_dist = 10 / diag(20,20) ≈ 0.354
        preds_large = np.array([[30.0, 20.0, 20.0, 20.0]])
        gts_large = np.array([[20.0, 20.0, 20.0, 20.0]])
        thr = np.array([0.0, 0.3, 0.4, 1.0])
        _, rates_small = normalized_precision_curve(preds_small, gts_small, thr)
        _, rates_large = normalized_precision_curve(preds_large, gts_large, thr)
        np.testing.assert_allclose(rates_small, rates_large, atol=1e-9)


class TestSuccessRateAtThreshold:
    """Tests for the SR@0.5 helper."""

    def test_all_above_threshold(self):
        ious = np.full(20, 0.9)
        assert success_rate_at_threshold(ious, 0.5) == pytest.approx(1.0)

    def test_all_below_threshold(self):
        ious = np.full(20, 0.3)
        assert success_rate_at_threshold(ious, 0.5) == pytest.approx(0.0)

    def test_half_above(self):
        ious = np.array([0.8] * 10 + [0.2] * 10)
        assert success_rate_at_threshold(ious, 0.5) == pytest.approx(0.5)

    def test_empty_array_returns_zero(self):
        assert success_rate_at_threshold(np.array([]), 0.5) == pytest.approx(0.0)

    def test_custom_threshold(self):
        # Use explicit values to avoid floating-point boundary ambiguity.
        ious = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # IoU > 0.5: values 0.6, 0.8, 1.0 → 3 out of 6
        sr = success_rate_at_threshold(ious, 0.5)
        assert sr == pytest.approx(0.5)


class TestAccuracyMetricsSuccessRate50:
    """Regression tests ensuring compute_all() now populates success_rate_50."""

    def test_success_rate_50_perfect(self):
        from eovot.metrics.accuracy import MetricsEngine
        engine = MetricsEngine()
        boxes = np.tile([0.0, 0.0, 10.0, 10.0], (20, 1))
        result = engine.compute_all(boxes, boxes)
        assert result.success_rate_50 == pytest.approx(1.0)

    def test_success_rate_50_zero(self):
        from eovot.metrics.accuracy import MetricsEngine
        engine = MetricsEngine()
        preds = np.tile([0.0, 0.0, 10.0, 10.0], (20, 1))
        gts = np.tile([100.0, 100.0, 10.0, 10.0], (20, 1))
        result = engine.compute_all(preds, gts)
        assert result.success_rate_50 == pytest.approx(0.0)

    def test_success_rate_50_in_range(self):
        from eovot.metrics.accuracy import MetricsEngine
        rng = np.random.default_rng(0)
        engine = MetricsEngine()
        preds = rng.uniform(0, 80, (40, 4))
        gts = rng.uniform(0, 80, (40, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 5
        gts[:, 2:] = np.abs(gts[:, 2:]) + 5
        result = engine.compute_all(preds, gts)
        assert 0.0 <= result.success_rate_50 <= 1.0
