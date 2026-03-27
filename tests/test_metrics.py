"""Unit tests for eovot.metrics.accuracy."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.accuracy import (
    AccuracyMetrics,
    MetricsEngine,
    center_distance,
    iou,
)


# ---------------------------------------------------------------------------
# iou()
# ---------------------------------------------------------------------------

class TestIoU:
    def test_perfect_overlap(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0.0, 0.0, 5.0, 5.0)
        b = (10.0, 10.0, 5.0, 5.0)
        assert iou(a, b) == pytest.approx(0.0)

    def test_half_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)   # area = 100
        b = (5.0, 0.0, 10.0, 10.0)   # area = 100, intersection = 50
        result = iou(a, b)
        assert result == pytest.approx(50.0 / 150.0, rel=1e-5)

    def test_touching_edge_no_overlap(self):
        a = (0.0, 0.0, 5.0, 5.0)
        b = (5.0, 0.0, 5.0, 5.0)
        assert iou(a, b) == pytest.approx(0.0)

    def test_zero_width_pred(self):
        a = (0.0, 0.0, 0.0, 10.0)
        b = (0.0, 0.0, 10.0, 10.0)
        assert iou(a, b) == pytest.approx(0.0)

    def test_zero_height_gt(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (0.0, 0.0, 10.0, 0.0)
        assert iou(a, b) == pytest.approx(0.0)

    def test_contained_box(self):
        outer = (0.0, 0.0, 20.0, 20.0)
        inner = (5.0, 5.0, 10.0, 10.0)
        # intersection = 100, union = 400 + 100 - 100 = 400
        assert iou(outer, inner) == pytest.approx(100.0 / 400.0, rel=1e-5)

    def test_return_in_unit_interval(self):
        import random
        rng = random.Random(0)
        for _ in range(50):
            box = lambda: (rng.uniform(0, 50), rng.uniform(0, 50),
                           rng.uniform(1, 30), rng.uniform(1, 30))
            result = iou(box(), box())
            assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# center_distance()
# ---------------------------------------------------------------------------

class TestCenterDistance:
    def test_same_box(self):
        box = (10.0, 20.0, 30.0, 40.0)
        assert center_distance(box, box) == pytest.approx(0.0)

    def test_horizontal_shift(self):
        a = (0.0, 0.0, 10.0, 10.0)   # center (5, 5)
        b = (10.0, 0.0, 10.0, 10.0)  # center (15, 5)
        assert center_distance(a, b) == pytest.approx(10.0)

    def test_diagonal(self):
        a = (0.0, 0.0, 0.0, 0.0)  # center (0, 0)
        b = (3.0, 4.0, 0.0, 0.0)  # center (3, 4)
        assert center_distance(a, b) == pytest.approx(5.0)

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            a = tuple(rng.uniform(0, 100, 4))
            b = tuple(rng.uniform(0, 100, 4))
            assert center_distance(a, b) >= 0.0


# ---------------------------------------------------------------------------
# MetricsEngine
# ---------------------------------------------------------------------------

class TestMetricsEngine:
    def setup_method(self):
        self.engine = MetricsEngine()

    # batch_iou
    def test_batch_iou_shape(self, pred_boxes, gt_boxes):
        result = self.engine.batch_iou(pred_boxes, gt_boxes)
        assert result.shape == (10,)

    def test_batch_iou_perfect(self, gt_boxes):
        result = self.engine.batch_iou(gt_boxes, gt_boxes)
        np.testing.assert_allclose(result, 1.0)

    def test_batch_iou_values_in_range(self, pred_boxes, gt_boxes):
        result = self.engine.batch_iou(pred_boxes, gt_boxes)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_batch_iou_shorter_array_truncated(self, gt_boxes):
        preds = gt_boxes[:7]
        result = self.engine.batch_iou(preds, gt_boxes)
        assert result.shape == (7,)

    # success_curve
    def test_success_curve_default_thresholds(self, gt_boxes):
        ious = np.ones(20)
        thrs, rates = self.engine.success_curve(ious)
        assert thrs.shape == rates.shape
        assert len(thrs) == 101  # default 0 … 1 in 101 steps
        # all IoU == 1 → all rates == 1 except the last (IoU > 1.0 is impossible)
        assert rates[0] == pytest.approx(1.0)

    def test_success_curve_zero_ious(self):
        ious = np.zeros(10)
        thrs, rates = self.engine.success_curve(ious)
        # IoU == 0, so IoU > 0 is False for every frame → rates[1:] == 0
        assert rates[-1] == pytest.approx(0.0)

    def test_success_curve_rates_in_unit_interval(self):
        ious = np.random.default_rng(3).uniform(0, 1, 50)
        _, rates = self.engine.success_curve(ious)
        assert np.all(rates >= 0.0) and np.all(rates <= 1.0)

    # precision_curve
    def test_precision_curve_default_thresholds(self, gt_boxes):
        thrs, rates = self.engine.precision_curve(gt_boxes, gt_boxes)
        assert thrs.shape == rates.shape
        assert len(thrs) == 51  # default 0 … 50 px

    def test_precision_curve_perfect_predictions(self, gt_boxes):
        _, rates = self.engine.precision_curve(gt_boxes, gt_boxes)
        # dist == 0 → dist < t for all t > 0 → rate == 1 for t > 0
        assert rates[-1] == pytest.approx(1.0)

    # compute_all
    def test_compute_all_type(self, pred_boxes, gt_boxes):
        result = self.engine.compute_all(pred_boxes, gt_boxes)
        assert isinstance(result, AccuracyMetrics)

    def test_compute_all_perfect(self, gt_boxes):
        result = self.engine.compute_all(gt_boxes, gt_boxes)
        assert result.mean_iou == pytest.approx(1.0)
        assert result.success_auc == pytest.approx(1.0, rel=1e-2)

    def test_compute_all_fields_non_negative(self, pred_boxes, gt_boxes):
        result = self.engine.compute_all(pred_boxes, gt_boxes)
        assert result.mean_iou >= 0.0
        assert result.success_auc >= 0.0
        assert result.precision_auc >= 0.0

    def test_accuracy_metrics_str(self, pred_boxes, gt_boxes):
        result = self.engine.compute_all(pred_boxes, gt_boxes)
        s = str(result)
        assert "mIoU" in s
        assert "success_AUC" in s
        assert "precision_AUC" in s
