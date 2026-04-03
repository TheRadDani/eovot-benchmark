"""Unit tests for eovot.metrics.accuracy."""

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
        box = (10.0, 10.0, 50.0, 50.0)
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = (0.0, 0.0, 10.0, 10.0)
        gt   = (20.0, 20.0, 10.0, 10.0)
        assert iou(pred, gt) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # Two 10×10 boxes offset by 5 in both axes → 5×5 intersection.
        pred = (0.0, 0.0, 10.0, 10.0)
        gt   = (5.0, 5.0, 10.0, 10.0)
        inter = 5.0 * 5.0          # 25
        union = 10*10 + 10*10 - 25  # 175
        assert iou(pred, gt) == pytest.approx(inter / union)

    def test_zero_area_pred(self):
        assert iou((0.0, 0.0, 0.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 0.0

    def test_zero_area_gt(self):
        assert iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 0.0)) == 0.0

    def test_contained_box(self):
        outer = (0.0, 0.0, 20.0, 20.0)
        inner = (5.0, 5.0, 10.0, 10.0)
        inter = 100.0
        union = 400.0 + 100.0 - 100.0
        assert iou(outer, inner) == pytest.approx(inter / union)

    def test_symmetry(self):
        a = (3.0, 7.0, 15.0, 20.0)
        b = (8.0, 12.0, 10.0, 10.0)
        assert iou(a, b) == pytest.approx(iou(b, a))


# ---------------------------------------------------------------------------
# center_distance()
# ---------------------------------------------------------------------------

class TestCenterDistance:
    def test_same_box(self):
        box = (10.0, 10.0, 20.0, 20.0)
        assert center_distance(box, box) == pytest.approx(0.0)

    def test_known_distance(self):
        # pred centre: (5, 5), gt centre: (8, 9) → dist = 5
        pred = (0.0, 0.0, 10.0, 10.0)
        gt   = (3.0, 4.0, 10.0, 10.0)
        assert center_distance(pred, gt) == pytest.approx(5.0)

    def test_horizontal_shift(self):
        pred = (0.0, 0.0, 10.0, 10.0)
        gt   = (10.0, 0.0, 10.0, 10.0)
        assert center_distance(pred, gt) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# MetricsEngine
# ---------------------------------------------------------------------------

class TestMetricsEngine:
    def setup_method(self):
        self.engine = MetricsEngine()

    # --- batch_iou ---

    def test_batch_iou_perfect(self):
        boxes = np.array([[0, 0, 10, 10], [5, 5, 10, 10]], dtype=float)
        result = self.engine.batch_iou(boxes, boxes)
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_batch_iou_no_overlap(self):
        preds = np.array([[0, 0, 10, 10]], dtype=float)
        gts   = np.array([[20, 20, 10, 10]], dtype=float)
        result = self.engine.batch_iou(preds, gts)
        assert result[0] == pytest.approx(0.0)

    def test_batch_iou_length_mismatch(self):
        preds = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
        gts   = np.array([[0, 0, 10, 10]], dtype=float)
        result = self.engine.batch_iou(preds, gts)
        assert len(result) == 1

    # --- success_curve ---

    def test_success_curve_all_ones(self):
        ious = np.ones(50)
        thr, rates = self.engine.success_curve(ious)
        # All frames have IoU=1.0, so for any threshold ≤ 1 the rate should be 1
        # except at threshold=1.0 where IoU > 1 is False (strict inequality).
        assert rates[0] == pytest.approx(1.0)   # threshold=0 → 100%
        assert rates[-1] == pytest.approx(0.0)  # threshold=1 → 0% (strict >)

    def test_success_curve_all_zeros(self):
        ious = np.zeros(50)
        thr, rates = self.engine.success_curve(ious)
        # IoU=0, threshold=0: 0 > 0 is False → 0%
        assert rates[0] == pytest.approx(0.0)

    def test_success_curve_shape(self):
        ious = np.linspace(0, 1, 100)
        thr, rates = self.engine.success_curve(ious)
        assert len(thr) == len(rates) == 101

    def test_success_curve_monotone(self):
        ious = np.random.default_rng(0).uniform(0, 1, 200)
        thr, rates = self.engine.success_curve(ious)
        assert np.all(np.diff(rates) <= 0)  # non-increasing with threshold

    # --- precision_curve ---

    def test_precision_curve_perfect(self):
        boxes = np.array([[0, 0, 10, 10]] * 20, dtype=float)
        thr, rates = self.engine.precision_curve(boxes, boxes)
        # All distances = 0, so for any threshold > 0 the rate is 1.0
        assert rates[-1] == pytest.approx(1.0)

    def test_precision_curve_shape(self):
        preds = np.random.default_rng(1).uniform(0, 100, (50, 4))
        gts   = np.random.default_rng(2).uniform(0, 100, (50, 4))
        thr, rates = self.engine.precision_curve(preds, gts)
        assert len(thr) == len(rates) == 51

    # --- compute_all ---

    def test_compute_all_perfect(self):
        boxes = np.array([[10, 10, 50, 50]] * 30, dtype=float)
        result = self.engine.compute_all(boxes, boxes)
        assert isinstance(result, AccuracyMetrics)
        assert result.mean_iou == pytest.approx(1.0)
        assert result.success_auc > 0.0
        assert result.precision_auc > 0.0

    def test_compute_all_no_overlap(self):
        preds = np.array([[0, 0, 5, 5]] * 10, dtype=float)
        gts   = np.array([[100, 100, 5, 5]] * 10, dtype=float)
        result = self.engine.compute_all(preds, gts)
        assert result.mean_iou == pytest.approx(0.0)

    def test_compute_all_returns_accuracy_metrics(self):
        boxes = np.array([[0, 0, 10, 10]] * 5, dtype=float)
        result = self.engine.compute_all(boxes, boxes)
        assert hasattr(result, "mean_iou")
        assert hasattr(result, "success_auc")
        assert hasattr(result, "precision_auc")
