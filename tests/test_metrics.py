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


class TestNormalizedCenterDistance:
    def test_same_box_is_zero(self):
        box = (0.0, 0.0, 30.0, 40.0)
        assert normalized_center_distance(box, box) == pytest.approx(0.0)

    def test_zero_diagonal_returns_zero(self):
        pred = (5.0, 5.0, 10.0, 10.0)
        gt = (0.0, 0.0, 0.0, 0.0)  # degenerate box
        assert normalized_center_distance(pred, gt) == pytest.approx(0.0)

    def test_known_normalised_value(self):
        # GT box: w=30, h=40 → diagonal = 50
        # pred center shifted by (30, 40) → raw dist = 50, norm dist = 50/50 = 1.0
        gt = (0.0, 0.0, 30.0, 40.0)
        pred = (30.0, 40.0, 30.0, 40.0)
        assert normalized_center_distance(pred, gt) == pytest.approx(1.0)

    def test_result_is_non_negative(self):
        a = (5.0, 5.0, 20.0, 20.0)
        b = (100.0, 100.0, 20.0, 20.0)
        assert normalized_center_distance(a, b) >= 0.0


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

    def test_norm_precision_curve_perfect(self):
        # Identical boxes → normalised distance = 0 → precision = 1 at all thresholds > 0
        preds = np.array([[5.0, 5.0, 30.0, 40.0]] * 10)
        gts = np.array([[5.0, 5.0, 30.0, 40.0]] * 10)
        thresholds, rates = self.engine.norm_precision_curve(preds, gts)
        assert rates[-1] == pytest.approx(1.0)

    def test_norm_precision_curve_shape(self):
        preds = np.tile([0.0, 0.0, 20.0, 20.0], (20, 1))
        gts = np.tile([5.0, 5.0, 20.0, 20.0], (20, 1))
        thresholds, rates = self.engine.norm_precision_curve(preds, gts)
        assert len(thresholds) == len(rates)
        assert np.all(rates >= 0.0) and np.all(rates <= 1.0)

    def test_compute_all_includes_norm_precision(self):
        preds = np.tile([0.0, 0.0, 30.0, 40.0], (20, 1))
        gts = np.tile([0.0, 0.0, 30.0, 40.0], (20, 1))
        result = self.engine.compute_all(preds, gts)
        assert 0.0 <= result.norm_precision_auc <= 1.0

    def test_compute_all_perfect_tracker_norm_precision(self):
        # Perfect predictions → norm_precision_auc should be high (close to 1)
        preds = np.tile([0.0, 0.0, 30.0, 40.0], (30, 1))
        gts = np.tile([0.0, 0.0, 30.0, 40.0], (30, 1))
        result = self.engine.compute_all(preds, gts)
        assert result.norm_precision_auc > 0.9
