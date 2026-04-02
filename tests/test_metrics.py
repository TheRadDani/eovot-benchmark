"""Unit tests for eovot.metrics.accuracy."""

import numpy as np
import pytest

from eovot.metrics.accuracy import (
    AccuracyMetrics,
    MetricsEngine,
    center_distance,
    iou,
)


class TestIoU:
    """Tests for the scalar iou() function."""

    def test_perfect_overlap(self):
        box = (10.0, 10.0, 50.0, 50.0)
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = (0.0, 0.0, 10.0, 10.0)
        gt = (20.0, 20.0, 10.0, 10.0)
        assert iou(pred, gt) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # Pred covers [0,100] x [0,100]; GT covers [50,150] x [50,150]
        pred = (0.0, 0.0, 100.0, 100.0)
        gt = (50.0, 50.0, 100.0, 100.0)
        # Intersection: [50,100] x [50,100] = 50*50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        expected = 2500.0 / 17500.0
        assert iou(pred, gt) == pytest.approx(expected)

    def test_zero_area_pred(self):
        pred = (10.0, 10.0, 0.0, 50.0)
        gt = (10.0, 10.0, 50.0, 50.0)
        assert iou(pred, gt) == pytest.approx(0.0)

    def test_zero_area_gt(self):
        pred = (10.0, 10.0, 50.0, 50.0)
        gt = (10.0, 10.0, 50.0, 0.0)
        assert iou(pred, gt) == pytest.approx(0.0)

    def test_containment(self):
        outer = (0.0, 0.0, 100.0, 100.0)
        inner = (25.0, 25.0, 50.0, 50.0)
        # Intersection == inner area, union == outer area
        expected = (50.0 * 50.0) / (100.0 * 100.0)
        assert iou(inner, outer) == pytest.approx(expected)
        assert iou(outer, inner) == pytest.approx(expected)

    def test_symmetry(self):
        a = (5.0, 10.0, 30.0, 40.0)
        b = (20.0, 15.0, 30.0, 40.0)
        assert iou(a, b) == pytest.approx(iou(b, a))

    def test_result_in_unit_interval(self):
        for _ in range(20):
            pred = (np.random.uniform(0, 50), np.random.uniform(0, 50),
                    np.random.uniform(1, 60), np.random.uniform(1, 60))
            gt = (np.random.uniform(0, 50), np.random.uniform(0, 50),
                  np.random.uniform(1, 60), np.random.uniform(1, 60))
            val = iou(pred, gt)
            assert 0.0 <= val <= 1.0


class TestCenterDistance:
    """Tests for center_distance()."""

    def test_same_box(self):
        box = (10.0, 20.0, 30.0, 40.0)
        assert center_distance(box, box) == pytest.approx(0.0)

    def test_horizontal_shift(self):
        pred = (0.0, 0.0, 10.0, 10.0)   # centre (5, 5)
        gt = (10.0, 0.0, 10.0, 10.0)    # centre (15, 5)
        assert center_distance(pred, gt) == pytest.approx(10.0)

    def test_diagonal(self):
        pred = (0.0, 0.0, 0.0, 0.0)   # centre (0, 0)
        gt = (3.0, 4.0, 0.0, 0.0)     # centre (3, 4)
        assert center_distance(pred, gt) == pytest.approx(5.0)


class TestMetricsEngine:
    """Tests for MetricsEngine."""

    def setup_method(self):
        self.engine = MetricsEngine()

    def _make_arrays(self, preds, gts):
        return np.array(preds, dtype=np.float64), np.array(gts, dtype=np.float64)

    def test_batch_iou_perfect(self):
        boxes = [[0, 0, 10, 10], [5, 5, 20, 20]]
        preds, gts = self._make_arrays(boxes, boxes)
        ious = self.engine.batch_iou(preds, gts)
        np.testing.assert_allclose(ious, [1.0, 1.0])

    def test_batch_iou_shape(self):
        preds = np.zeros((7, 4))
        gts = np.ones((7, 4))
        ious = self.engine.batch_iou(preds, gts)
        assert ious.shape == (7,)

    def test_batch_iou_length_mismatch(self):
        preds = np.zeros((5, 4))
        gts = np.zeros((3, 4))
        ious = self.engine.batch_iou(preds, gts)
        assert len(ious) == 3

    def test_success_curve_all_ones(self):
        ious = np.ones(10)
        thresholds, rates = self.engine.success_curve(ious)
        # All frames have IoU=1, so success rate = 1 except at t=1
        assert rates[0] == pytest.approx(1.0)   # threshold 0: all pass
        assert rates[-1] == pytest.approx(0.0)   # threshold 1: IoU > 1 is false

    def test_success_curve_auc_perfect(self):
        ious = np.ones(100)
        thresholds, rates = self.engine.success_curve(ious)
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        auc = float(_trapz(rates, thresholds))
        # Except at the very last threshold, all pass → AUC ≈ 1.0
        assert auc > 0.95

    def test_success_curve_all_zeros(self):
        ious = np.zeros(10)
        thresholds, rates = self.engine.success_curve(ious)
        # IoU=0, threshold=0: IoU > 0 is False → rate=0
        np.testing.assert_allclose(rates, 0.0)

    def test_precision_curve_zero_distance(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]] * 5)
        thresholds, rates = self.engine.precision_curve(boxes, boxes)
        # All distances = 0, so for any threshold > 0 all frames pass.
        assert rates[-1] == pytest.approx(1.0)

    def test_compute_all_returns_accuracy_metrics(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]] * 10)
        result = self.engine.compute_all(boxes, boxes)
        assert isinstance(result, AccuracyMetrics)
        assert result.mean_iou == pytest.approx(1.0)
        assert result.success_auc > 0.0
        assert result.precision_auc > 0.0

    def test_compute_all_no_overlap(self):
        preds = np.array([[0.0, 0.0, 5.0, 5.0]] * 10)
        gts = np.array([[100.0, 100.0, 5.0, 5.0]] * 10)
        result = self.engine.compute_all(preds, gts)
        assert result.mean_iou == pytest.approx(0.0)
        assert result.success_auc == pytest.approx(0.0)
