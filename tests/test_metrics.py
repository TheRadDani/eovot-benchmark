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


class TestGOT10kMetrics:
    """Tests for GOT-10k and LaSOT protocol metrics: SR@0.5, SR@0.75, NP AUC."""

    def setup_method(self):
        self.engine = MetricsEngine()

    # ------------------------------------------------------------------ #
    # batch_normalized_center_distance                                     #
    # ------------------------------------------------------------------ #

    def test_norm_dist_identical_boxes_is_zero(self):
        boxes = np.tile([10.0, 10.0, 40.0, 40.0], (10, 1))
        result = self.engine.batch_normalized_center_distance(boxes, boxes)
        np.testing.assert_allclose(result, np.zeros(10))

    def test_norm_dist_known_value(self):
        # GT box: 40×40 → sqrt(area) = 40. Pred centre displaced by (30, 40) → dist=50.
        # Normalised distance = 50 / 40 = 1.25
        gt = np.array([[0.0, 0.0, 40.0, 40.0]])   # centre at (20, 20)
        pred = np.array([[30.0, 30.0, 40.0, 40.0]])  # centre at (50, 50), dx=30, dy=30
        result = self.engine.batch_normalized_center_distance(pred, gt)
        # pixel dist = sqrt(30²+30²) = 30√2 ≈ 42.43; / 40 ≈ 1.0607
        expected = float(np.sqrt(30**2 + 30**2)) / 40.0
        np.testing.assert_allclose(result, [expected], rtol=1e-5)

    def test_norm_dist_zero_gt_area_returns_pixel_dist(self):
        # Zero-area GT → sqrt_area defaults to 1.0, so norm_dist == pixel_dist
        gt = np.array([[0.0, 0.0, 0.0, 0.0]])   # degenerate
        pred = np.array([[3.0, 4.0, 0.0, 0.0]])
        result = self.engine.batch_normalized_center_distance(pred, gt)
        # centres: pred=(3,4), gt=(0,0) → pixel dist=5; gt_area=0 → divisor=1
        np.testing.assert_allclose(result, [5.0], rtol=1e-5)

    def test_norm_dist_shape(self):
        n = 25
        preds = np.random.default_rng(0).uniform(0, 50, (n, 4))
        gts = np.random.default_rng(1).uniform(0, 50, (n, 4))
        preds[:, 2:] += 1; gts[:, 2:] += 1
        result = self.engine.batch_normalized_center_distance(preds, gts)
        assert result.shape == (n,)
        assert np.all(result >= 0.0)

    def test_norm_dist_empty_input(self):
        result = self.engine.batch_normalized_center_distance(
            np.empty((0, 4)), np.empty((0, 4))
        )
        assert result.shape == (0,)

    # ------------------------------------------------------------------ #
    # normalized_precision_curve                                           #
    # ------------------------------------------------------------------ #

    def test_np_curve_perfect_tracking(self):
        # Identical boxes → norm_dist = 0 everywhere → precision = 1 for all t > 0
        boxes = np.tile([5.0, 5.0, 20.0, 20.0], (20, 1))
        thresholds, rates = self.engine.normalized_precision_curve(boxes, boxes)
        assert rates[-1] == pytest.approx(1.0)
        assert rates[0] == pytest.approx(0.0)  # t=0: strict < 0 is always false

    def test_np_curve_default_thresholds(self):
        boxes = np.tile([0.0, 0.0, 30.0, 30.0], (10, 1))
        thresholds, rates = self.engine.normalized_precision_curve(boxes, boxes)
        assert len(thresholds) == 51
        assert float(thresholds[0]) == pytest.approx(0.0)
        assert float(thresholds[-1]) == pytest.approx(0.5)

    def test_np_curve_monotone_non_decreasing(self):
        rng = np.random.default_rng(7)
        preds = rng.uniform(0, 80, (40, 4))
        gts = rng.uniform(0, 80, (40, 4))
        preds[:, 2:] += 5; gts[:, 2:] += 5
        _, rates = self.engine.normalized_precision_curve(preds, gts)
        assert np.all(np.diff(rates) >= -1e-10), "NP curve must be non-decreasing"

    def test_np_curve_empty_input(self):
        thresholds, rates = self.engine.normalized_precision_curve(
            np.empty((0, 4)), np.empty((0, 4))
        )
        np.testing.assert_allclose(rates, np.zeros_like(thresholds))

    # ------------------------------------------------------------------ #
    # compute_all — SR@0.5, SR@0.75, normalized_precision_auc             #
    # ------------------------------------------------------------------ #

    def test_compute_all_perfect_has_sr50_sr75_one(self):
        boxes = np.tile([0.0, 0.0, 20.0, 20.0], (50, 1))
        result = self.engine.compute_all(boxes, boxes)
        assert result.success_rate_50 == pytest.approx(1.0)
        assert result.success_rate_75 == pytest.approx(1.0)
        # NP AUC for perfect tracking is ~0.99 (not 1.0) because threshold=0
        # yields rate=0 in the trapezoidal integral (strict < comparison).
        assert result.normalized_precision_auc > 0.98

    def test_compute_all_no_overlap_has_sr50_sr75_zero(self):
        preds = np.tile([0.0, 0.0, 5.0, 5.0], (30, 1))
        gts = np.tile([100.0, 100.0, 5.0, 5.0], (30, 1))
        result = self.engine.compute_all(preds, gts)
        assert result.success_rate_50 == pytest.approx(0.0)
        assert result.success_rate_75 == pytest.approx(0.0)

    def test_compute_all_sr50_threshold(self):
        # Half the frames have IoU ≈ 0.8 (> 0.5 and > 0.75), half ≈ 0.0
        boxes = np.tile([0.0, 0.0, 10.0, 10.0], (20, 1))
        good = np.tile([0.0, 0.0, 10.0, 10.0], (10, 1))   # IoU = 1.0
        bad = np.tile([100.0, 100.0, 10.0, 10.0], (10, 1))  # IoU = 0.0
        preds = np.vstack([good, bad])
        gts = np.vstack([good, good])
        result = self.engine.compute_all(preds, gts)
        assert result.success_rate_50 == pytest.approx(0.5)
        assert result.success_rate_75 == pytest.approx(0.5)

    def test_compute_all_np_auc_range(self):
        rng = np.random.default_rng(99)
        preds = rng.uniform(0, 50, (40, 4))
        gts = rng.uniform(0, 50, (40, 4))
        preds[:, 2:] += 2; gts[:, 2:] += 2
        result = self.engine.compute_all(preds, gts)
        assert 0.0 <= result.normalized_precision_auc <= 1.0

    def test_compute_all_new_fields_present(self):
        boxes = np.tile([1.0, 1.0, 8.0, 8.0], (15, 1))
        result = self.engine.compute_all(boxes, boxes)
        assert hasattr(result, "success_rate_50")
        assert hasattr(result, "success_rate_75")
        assert hasattr(result, "normalized_precision_auc")

    def test_sr75_stricter_than_sr50(self):
        # SR@0.75 ≤ SR@0.5 always holds because 0.75 > 0.5
        rng = np.random.default_rng(13)
        preds = rng.uniform(0, 60, (60, 4))
        gts = rng.uniform(0, 60, (60, 4))
        preds[:, 2:] += 3; gts[:, 2:] += 3
        result = self.engine.compute_all(preds, gts)
        assert result.success_rate_75 <= result.success_rate_50 + 1e-9
