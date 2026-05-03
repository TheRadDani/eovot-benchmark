"""Unit tests for eovot.metrics.accuracy."""

import numpy as np
import pytest

from eovot.metrics.accuracy import MetricsEngine, iou, center_distance, normalized_center_distance


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

    def test_compute_all_includes_normalized_precision(self):
        preds = np.tile([0.0, 0.0, 10.0, 10.0], (20, 1))
        gts = np.tile([0.0, 0.0, 10.0, 10.0], (20, 1))
        result = self.engine.compute_all(preds, gts)
        assert hasattr(result, "normalized_precision_auc")
        assert hasattr(result, "normalized_precision_at_20")
        assert 0.0 <= result.normalized_precision_auc <= 1.0
        assert 0.0 <= result.normalized_precision_at_20 <= 1.0

    def test_compute_all_perfect_tracker_np_at_20(self):
        """Perfect predictions → NP@0.20 = 1.0."""
        preds = np.tile([5.0, 5.0, 20.0, 20.0], (30, 1))
        gts = np.tile([5.0, 5.0, 20.0, 20.0], (30, 1))
        result = self.engine.compute_all(preds, gts)
        assert result.normalized_precision_at_20 == pytest.approx(1.0)


class TestNormalizedCenterDistance:
    def test_identical_boxes_zero_distance(self):
        box = (10.0, 10.0, 20.0, 20.0)
        assert normalized_center_distance(box, box) == pytest.approx(0.0)

    def test_scale_invariance(self):
        """Shifting by half the diagonal gives ~0.5 regardless of box size."""
        # Small 10×10 box: diagonal = 10*sqrt(2), half = 5*sqrt(2) ≈ 7.07 px
        # NCD = 7.07 / sqrt(10*10) = 7.07/10 ≈ 0.707
        pred_small = (17.07, 10.0, 10.0, 10.0)  # shifted x by ≈ 7.07 from (10,10,10,10)
        gt_small = (10.0, 10.0, 10.0, 10.0)
        d_small = normalized_center_distance(pred_small, gt_small)

        # Same shift fraction for a 100×100 box: shift x by 70.7
        pred_large = (170.7, 100.0, 100.0, 100.0)
        gt_large = (100.0, 100.0, 100.0, 100.0)
        d_large = normalized_center_distance(pred_large, gt_large)

        assert d_small == pytest.approx(d_large, abs=1e-3)

    def test_zero_area_gt_returns_inf(self):
        pred = (0.0, 0.0, 10.0, 10.0)
        gt_degenerate = (0.0, 0.0, 0.0, 0.0)
        assert normalized_center_distance(pred, gt_degenerate) == float("inf")

    def test_known_value(self):
        # GT box: (0,0,10,10) — centre (5,5), area=100, sqrt_area=10
        # Pred box: (8,8,10,10) — centre (13,13)
        # dist = sqrt((13-5)^2 + (13-5)^2) = sqrt(128) ≈ 11.31
        # NCD = 11.31 / 10 ≈ 1.131
        gt = (0.0, 0.0, 10.0, 10.0)
        pred = (8.0, 8.0, 10.0, 10.0)
        expected = (8.0**2 + 8.0**2) ** 0.5 / 10.0
        assert normalized_center_distance(pred, gt) == pytest.approx(expected, abs=1e-6)


class TestNormalizedPrecisionCurve:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_perfect_tracker_np_curve_is_one(self):
        preds = np.tile([0.0, 0.0, 50.0, 50.0], (20, 1))
        gts = np.tile([0.0, 0.0, 50.0, 50.0], (20, 1))
        thr, rates = self.engine.normalized_precision_curve(preds, gts)
        # All distances are 0, so every threshold > 0 gives rate 1.0
        assert rates[-1] == pytest.approx(1.0)

    def test_curve_shape(self):
        preds = np.random.rand(30, 4) * 100
        gts = np.random.rand(30, 4) * 100
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1
        thr, rates = self.engine.normalized_precision_curve(preds, gts)
        assert thr.shape == rates.shape
        assert np.all(rates >= 0.0) and np.all(rates <= 1.0)

    def test_thresholds_default_range(self):
        preds = np.tile([0.0, 0.0, 10.0, 10.0], (10, 1))
        gts = np.tile([0.0, 0.0, 10.0, 10.0], (10, 1))
        thr, _ = self.engine.normalized_precision_curve(preds, gts)
        assert thr[0] == pytest.approx(0.0)
        assert thr[-1] == pytest.approx(0.5)
