"""Unit tests for eovot.metrics.accuracy."""

import numpy as np
import pytest

from eovot.metrics.accuracy import AccuracyMetrics, MetricsEngine, iou, center_distance


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


class TestVectorizedBatchIoU:
    """Verify that vectorized batch_iou matches the scalar iou() on known inputs."""

    def setup_method(self):
        self.engine = MetricsEngine()

    def test_matches_scalar_iou_partial_overlap(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[5.0, 0.0, 10.0, 10.0]])
        expected = iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 10.0, 10.0))
        result = self.engine.batch_iou(a, b)
        assert result[0] == pytest.approx(expected)

    def test_matches_scalar_iou_contained(self):
        outer = np.array([[0.0, 0.0, 100.0, 100.0]])
        inner = np.array([[25.0, 25.0, 50.0, 50.0]])
        expected = iou((0.0, 0.0, 100.0, 100.0), (25.0, 25.0, 50.0, 50.0))
        result = self.engine.batch_iou(outer, inner)
        assert result[0] == pytest.approx(expected)

    def test_degenerate_boxes_yield_zero(self):
        preds = np.array([[0.0, 0.0, 0.0, 10.0], [0.0, 0.0, 10.0, 0.0]])
        gts = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
        result = self.engine.batch_iou(preds, gts)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_no_overlap_bulk(self):
        # 100 non-overlapping pairs
        rng = np.random.default_rng(0)
        preds = np.column_stack([
            rng.uniform(0, 50, 100),
            rng.uniform(0, 50, 100),
            np.full(100, 5.0),
            np.full(100, 5.0),
        ])
        # GT boxes far away
        gts = np.column_stack([
            rng.uniform(200, 300, 100),
            rng.uniform(200, 300, 100),
            np.full(100, 5.0),
            np.full(100, 5.0),
        ])
        result = self.engine.batch_iou(preds, gts)
        np.testing.assert_array_equal(result, np.zeros(100))

    def test_perfect_overlap_bulk(self):
        boxes = np.tile([10.0, 20.0, 30.0, 40.0], (50, 1))
        result = self.engine.batch_iou(boxes, boxes)
        np.testing.assert_allclose(result, np.ones(50))

    def test_symmetry(self):
        rng = np.random.default_rng(7)
        preds = rng.uniform(0, 80, (30, 4))
        gts = rng.uniform(0, 80, (30, 4))
        preds[:, 2:] += 5
        gts[:, 2:] += 5
        ab = self.engine.batch_iou(preds, gts)
        ba = self.engine.batch_iou(gts, preds)
        np.testing.assert_allclose(ab, ba, atol=1e-12)

    def test_output_range(self):
        rng = np.random.default_rng(99)
        preds = rng.uniform(0, 200, (200, 4))
        gts = rng.uniform(0, 200, (200, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1
        result = self.engine.batch_iou(preds, gts)
        assert result.shape == (200,)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_empty_input(self):
        preds = np.empty((0, 4))
        gts = np.empty((0, 4))
        result = self.engine.batch_iou(preds, gts)
        assert result.shape == (0,)


class TestNormalizedPrecision:
    """Tests for scale-invariant normalized precision curve."""

    def setup_method(self):
        self.engine = MetricsEngine()

    def test_identical_boxes_perfect_precision(self):
        boxes = np.tile([10.0, 10.0, 50.0, 50.0], (20, 1))
        thresholds, rates = self.engine.normalized_precision_curve(boxes, boxes)
        # Zero normalized distance → precision = 1.0 at all thresholds > 0
        assert rates[-1] == pytest.approx(1.0)
        # Threshold = 0 → distance is NOT < 0, so rate = 0
        assert rates[0] == pytest.approx(0.0)

    def test_output_shape_and_range(self):
        rng = np.random.default_rng(42)
        preds = rng.uniform(0, 100, (50, 4))
        gts = rng.uniform(0, 100, (50, 4))
        preds[:, 2:] += 10
        gts[:, 2:] += 10
        thresholds, rates = self.engine.normalized_precision_curve(preds, gts)
        assert thresholds.shape == rates.shape
        assert float(rates.min()) >= 0.0
        assert float(rates.max()) <= 1.0

    def test_scale_invariance(self):
        """Doubling the image scale should not change the normalized precision."""
        rng = np.random.default_rng(5)
        preds = rng.uniform(0, 50, (40, 4))
        gts = rng.uniform(0, 50, (40, 4))
        preds[:, 2:] += 5
        gts[:, 2:] += 5

        preds_2x = preds * 2.0
        gts_2x = gts * 2.0

        _, rates_1x = self.engine.normalized_precision_curve(preds, gts)
        _, rates_2x = self.engine.normalized_precision_curve(preds_2x, gts_2x)

        np.testing.assert_allclose(rates_1x, rates_2x, atol=1e-10)

    def test_auc_in_compute_all(self):
        boxes = np.tile([0.0, 0.0, 30.0, 30.0], (20, 1))
        result = self.engine.compute_all(boxes, boxes)
        assert 0.0 <= result.norm_precision_auc <= 1.0
        assert result.norm_precision_auc == pytest.approx(1.0, abs=0.02)

    def test_norm_precision_auc_range_random(self):
        rng = np.random.default_rng(13)
        preds = rng.uniform(0, 100, (60, 4))
        gts = rng.uniform(0, 100, (60, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 5
        gts[:, 2:] = np.abs(gts[:, 2:]) + 5
        result = self.engine.compute_all(preds, gts)
        assert 0.0 <= result.norm_precision_auc <= 1.0

    def test_accuracy_metrics_has_norm_field(self):
        m = AccuracyMetrics(mean_iou=0.5, success_auc=0.4, precision_auc=0.6)
        assert hasattr(m, "norm_precision_auc")
        assert m.norm_precision_auc == pytest.approx(0.0)

    def test_empty_input_returns_zero_rates(self):
        preds = np.empty((0, 4))
        gts = np.empty((0, 4))
        thresholds, rates = self.engine.normalized_precision_curve(preds, gts)
        np.testing.assert_array_equal(rates, np.zeros_like(thresholds))


class TestBatchCenterDistances:
    """Tests for vectorized centre-distance computation."""

    def setup_method(self):
        self.engine = MetricsEngine()

    def test_same_boxes_zero_distance(self):
        boxes = np.tile([5.0, 5.0, 20.0, 20.0], (10, 1))
        dists = self.engine.batch_center_distances(boxes, boxes)
        np.testing.assert_allclose(dists, np.zeros(10))

    def test_known_distance(self):
        # Centers: (20, 20) and (23, 24) → distance = 5
        preds = np.array([[10.0, 10.0, 20.0, 20.0]])
        gts = np.array([[13.0, 14.0, 20.0, 20.0]])
        dists = self.engine.batch_center_distances(preds, gts)
        assert dists[0] == pytest.approx(5.0)

    def test_output_shape(self):
        preds = np.random.rand(15, 4) * 50 + 5
        gts = np.random.rand(15, 4) * 50 + 5
        dists = self.engine.batch_center_distances(preds, gts)
        assert dists.shape == (15,)
        assert np.all(dists >= 0.0)
