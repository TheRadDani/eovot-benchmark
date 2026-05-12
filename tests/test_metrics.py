"""Unit tests for eovot.metrics.accuracy."""

import numpy as np
import pytest

from eovot.metrics.accuracy import MetricsEngine, iou, center_distance


class TestIoU:
    def test_perfect_overlap(self):
        box = (10.0, 10.0, 50.0, 50.0)
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (20.0, 20.0, 10.0, 10.0)
        assert iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # Two 10×10 boxes overlapping by a 5×10 region
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 0.0, 10.0, 10.0)
        # intersection = 5*10 = 50, union = 100+100-50 = 150
        assert iou(a, b) == pytest.approx(50.0 / 150.0)

    def test_one_inside_other(self):
        outer = (0.0, 0.0, 100.0, 100.0)
        inner = (25.0, 25.0, 50.0, 50.0)
        # intersection = 50*50 = 2500, union = 10000+2500-2500 = 10000
        assert iou(outer, inner) == pytest.approx(2500.0 / 10000.0)

    def test_zero_area_box(self):
        # A degenerate box with zero area should return 0.0
        a = (0.0, 0.0, 0.0, 10.0)
        b = (0.0, 0.0, 10.0, 10.0)
        assert iou(a, b) == pytest.approx(0.0)

    def test_symmetry(self):
        a = (5.0, 5.0, 30.0, 20.0)
        b = (15.0, 10.0, 30.0, 20.0)
        assert iou(a, b) == pytest.approx(iou(b, a))


class TestCenterDistance:
    def test_same_box(self):
        box = (10.0, 10.0, 20.0, 20.0)
        assert center_distance(box, box) == pytest.approx(0.0)

    def test_known_distance(self):
        # Centers: (20, 20) and (23, 24) → distance = 5
        a = (10.0, 10.0, 20.0, 20.0)
        b = (13.0, 14.0, 20.0, 20.0)
        assert center_distance(a, b) == pytest.approx(5.0)


class TestBatchIoU:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_identical_sequences(self):
        boxes = np.array([[0, 0, 10, 10], [5, 5, 10, 10]], dtype=float)
        result = self.engine.batch_iou(boxes, boxes)
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_shape(self):
        preds = np.random.rand(20, 4) * 100
        gts = np.random.rand(20, 4) * 100
        # Ensure w/h are positive
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1
        result = self.engine.batch_iou(preds, gts)
        assert result.shape == (20,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestMetricsEngine:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_batch_iou_delegates(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=float)
        result = self.engine.batch_iou(boxes, boxes)
        assert result[0] == pytest.approx(1.0)

    def test_success_curve_perfect(self):
        # All IoU = 1.0 → success rate = 1.0 at every threshold ≤ 1
        ious = np.ones(10)
        thresholds, rates = self.engine.success_curve(ious)
        # At threshold=0.0, all frames succeed; at threshold=1.0, strict > check fails
        assert rates[0] == pytest.approx(1.0)

    def test_success_curve_zero(self):
        # All IoU = 0.0 → success rate = 0 for all thresholds > 0
        ious = np.zeros(10)
        thresholds, rates = self.engine.success_curve(ious)
        # rates[0] at threshold=0.0: IoU > 0 is false for zeros
        assert rates[-1] == pytest.approx(0.0)

    def test_success_curve_shape(self):
        ious = np.linspace(0.0, 1.0, 50)
        thresholds, rates = self.engine.success_curve(ious)
        assert len(thresholds) == len(rates)
        assert np.all(rates >= 0.0) and np.all(rates <= 1.0)

    def test_precision_curve_perfect(self):
        # Identical boxes → center distance = 0 → precision = 1 everywhere except threshold=0
        preds = np.array([[5.0, 5.0, 10.0, 10.0]] * 10)
        gts = np.array([[5.0, 5.0, 10.0, 10.0]] * 10)
        thresholds, rates = self.engine.precision_curve(preds, gts)
        # At threshold > 0, all distances are < threshold → precision = 1.0
        assert rates[-1] == pytest.approx(1.0)

    def test_compute_all_returns_valid_metrics(self):
        preds = np.tile([0.0, 0.0, 10.0, 10.0], (30, 1))
        gts = np.tile([0.0, 0.0, 10.0, 10.0], (30, 1))
        result = self.engine.compute_all(preds, gts)
        assert result.mean_iou == pytest.approx(1.0)
        assert 0.0 <= result.success_auc <= 1.0
        assert 0.0 <= result.precision_auc <= 1.0

    def test_compute_all_auc_range(self):
        rng = np.random.default_rng(42)
        preds = rng.uniform(0, 100, (50, 4))
        gts = rng.uniform(0, 100, (50, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1
        result = self.engine.compute_all(preds, gts)
        assert 0.0 <= result.mean_iou <= 1.0
        assert 0.0 <= result.success_auc <= 1.0
        assert 0.0 <= result.precision_auc <= 1.0

    def test_compute_all_includes_norm_prec(self):
        preds = np.tile([0.0, 0.0, 50.0, 50.0], (20, 1))
        gts = np.tile([0.0, 0.0, 50.0, 50.0], (20, 1))
        result = self.engine.compute_all(preds, gts)
        # Perfect predictions → NP AUC should be near 1.0 (threshold=0 always
        # gives 0 precision, so the achievable max is slightly below 1.0).
        assert result.norm_prec_auc >= 0.98
        assert result.norm_prec_at_01 == pytest.approx(1.0)


class TestNormalizedPrecisionCurve:
    """Tests for MetricsEngine.normalized_precision_curve."""

    def setup_method(self):
        self.engine = MetricsEngine()

    def test_perfect_predictions_high_precision(self):
        # Identical boxes → centre distance = 0 → all normalized dists = 0.
        # Precision should be 1.0 at every threshold > 0.
        preds = np.tile([10.0, 10.0, 40.0, 30.0], (30, 1))
        gts = np.tile([10.0, 10.0, 40.0, 30.0], (30, 1))
        thresholds, rates = self.engine.normalized_precision_curve(preds, gts)
        # At threshold 0 nothing passes (strict <), above 0 everything should.
        assert rates[-1] == pytest.approx(1.0)

    def test_zero_precision_far_predictions(self):
        # Predictions 100 px away from a 10×10 target → normalized dist ≈ 10.0,
        # which exceeds even the maximum threshold of 0.5.
        preds = np.tile([110.0, 110.0, 10.0, 10.0], (20, 1))
        gts = np.tile([0.0, 0.0, 10.0, 10.0], (20, 1))
        thresholds, rates = self.engine.normalized_precision_curve(preds, gts)
        # All normalised distances >> 0.5 → precision = 0 everywhere.
        assert rates[-1] == pytest.approx(0.0)

    def test_threshold_range(self):
        preds = np.random.default_rng(7).uniform(0, 50, (40, 4))
        gts = np.random.default_rng(13).uniform(0, 50, (40, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 5
        gts[:, 2:] = np.abs(gts[:, 2:]) + 5
        thresholds, rates = self.engine.normalized_precision_curve(preds, gts)
        assert len(thresholds) == len(rates)
        assert thresholds[0] == pytest.approx(0.0)
        assert thresholds[-1] == pytest.approx(0.5)
        assert np.all(rates >= 0.0) and np.all(rates <= 1.0)

    def test_monotone_increasing(self):
        # Precision rate must be non-decreasing as threshold increases.
        preds = np.random.default_rng(99).uniform(0, 80, (60, 4))
        gts = np.random.default_rng(55).uniform(0, 80, (60, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 5
        gts[:, 2:] = np.abs(gts[:, 2:]) + 5
        _, rates = self.engine.normalized_precision_curve(preds, gts)
        assert np.all(np.diff(rates) >= -1e-9), "Precision curve must be non-decreasing"

    def test_scale_invariance(self):
        # A prediction that is 5 % of the target size away from centre should
        # score the same NP regardless of the absolute target size.
        def _build(target_size: float):
            half = target_size / 2.0
            # Centre of GT at (half, half); pred shifted by 5 % of size.
            shift = 0.05 * target_size
            preds = np.array([[shift, 0.0, target_size, target_size]] * 20)
            gts = np.array([[0.0, 0.0, target_size, target_size]] * 20)
            return preds, gts

        preds_small, gts_small = _build(20.0)
        preds_large, gts_large = _build(200.0)
        _, rates_small = self.engine.normalized_precision_curve(preds_small, gts_small)
        _, rates_large = self.engine.normalized_precision_curve(preds_large, gts_large)
        # Both should have the same precision curve because relative shift is identical.
        np.testing.assert_allclose(rates_small, rates_large, atol=1e-6)

    def test_custom_thresholds(self):
        preds = np.tile([0.0, 0.0, 40.0, 40.0], (10, 1))
        gts = np.tile([0.0, 0.0, 40.0, 40.0], (10, 1))
        custom_thr = np.array([0.0, 0.1, 0.2, 0.3])
        thresholds, rates = self.engine.normalized_precision_curve(
            preds, gts, thresholds=custom_thr
        )
        np.testing.assert_array_equal(thresholds, custom_thr)
        assert len(rates) == len(custom_thr)

    def test_np_at_01_via_compute_all(self):
        # Shift prediction by exactly 5 % of target width/height.
        # Expected norm_dist ≈ shift / sqrt(w*h); shift chosen so it's < 0.1.
        w, h = 50.0, 50.0
        shift = 4.0  # norm_dist = 4 / sqrt(50*50) = 4/50 = 0.08 < 0.1
        preds = np.array([[shift, 0.0, w, h]] * 30)
        gts = np.array([[0.0, 0.0, w, h]] * 30)
        result = self.engine.compute_all(preds, gts)
        # Every frame has norm_dist = 0.08 < 0.1 → NP@0.1 should be 1.0.
        assert result.norm_prec_at_01 == pytest.approx(1.0)
