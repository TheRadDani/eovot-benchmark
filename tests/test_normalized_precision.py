"""Tests for LaSOT normalized precision and GOT-10k SR metrics."""

import numpy as np
import pytest

from eovot.metrics.accuracy import AccuracyMetrics, MetricsEngine


class TestNormalizedPrecisionCurve:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_perfect_prediction_gives_high_nprec(self):
        """Identical pred and GT → norm distance = 0 → N-PREC = 1 at any threshold > 0."""
        boxes = np.array([[10.0, 10.0, 40.0, 40.0]] * 20)
        thresholds, rates = self.engine.normalized_precision_curve(boxes, boxes)
        # At threshold > 0, all norm distances (=0) are below threshold
        assert rates[-1] == pytest.approx(1.0)
        assert rates[1] == pytest.approx(1.0)

    def test_zero_overlap_gives_low_nprec(self):
        """Large displacement → large norm distance → precision near 0 at tight threshold."""
        # GT: 40×40 box at (0,0); pred center displaced by 200 px
        gt = np.array([[0.0, 0.0, 40.0, 40.0]] * 10)
        pred = np.array([[200.0, 200.0, 40.0, 40.0]] * 10)
        # norm_dist = center_dist / sqrt(gt_w*gt_h) = sqrt(200^2+200^2) / 40 ≈ 7.07
        # All thresholds [0, 0.5] are << 7.07 → precision should be 0
        thresholds, rates = self.engine.normalized_precision_curve(pred, gt)
        assert rates[-1] == pytest.approx(0.0)

    def test_scale_invariance(self):
        """N-PREC is scale-invariant: same relative error on two scales gives same rate."""
        # Small target (10×10), error of 2px → norm_dist = 2/10 = 0.2
        gt_small = np.array([[0.0, 0.0, 10.0, 10.0]] * 5)
        pred_small = np.array([[2.0, 0.0, 10.0, 10.0]] * 5)  # cx displaced 2px

        # Large target (100×100), error of 20px → norm_dist = 20/100 = 0.2
        gt_large = np.array([[0.0, 0.0, 100.0, 100.0]] * 5)
        pred_large = np.array([[20.0, 0.0, 100.0, 100.0]] * 5)  # cx displaced 20px

        thr = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
        _, rates_small = self.engine.normalized_precision_curve(pred_small, gt_small, thr)
        _, rates_large = self.engine.normalized_precision_curve(pred_large, gt_large, thr)

        # Both should yield rate=1 at threshold >= 0.2 (norm_dist=0.2 < threshold)
        np.testing.assert_allclose(rates_small, rates_large, atol=1e-10)

    def test_empty_input(self):
        """Empty arrays should return all-zero rates."""
        empty = np.empty((0, 4))
        thresholds, rates = self.engine.normalized_precision_curve(empty, empty)
        assert len(rates) == len(thresholds)
        assert np.all(rates == 0.0)

    def test_default_thresholds_range(self):
        """Default thresholds should span [0, 0.5]."""
        boxes = np.ones((5, 4))
        thresholds, _ = self.engine.normalized_precision_curve(boxes, boxes)
        assert thresholds[0] == pytest.approx(0.0)
        assert thresholds[-1] == pytest.approx(0.5)

    def test_different_target_sizes(self):
        """Metric handles mixed-size GT boxes within a sequence."""
        gt = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 50.0, 50.0],
            [0.0, 0.0, 100.0, 100.0],
        ])
        pred = gt.copy()  # perfect prediction
        thr = np.array([0.0, 0.1, 0.5])
        _, rates = self.engine.normalized_precision_curve(pred, gt, thr)
        assert rates[-1] == pytest.approx(1.0)


class TestSuccessRateAt:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_all_above_threshold(self):
        ious = np.array([0.6, 0.7, 0.8, 0.9])
        assert self.engine.success_rate_at(ious, 0.5) == pytest.approx(1.0)

    def test_all_below_threshold(self):
        ious = np.array([0.1, 0.2, 0.3])
        assert self.engine.success_rate_at(ious, 0.5) == pytest.approx(0.0)

    def test_half_above(self):
        ious = np.array([0.4, 0.6, 0.4, 0.6])
        assert self.engine.success_rate_at(ious, 0.5) == pytest.approx(0.5)

    def test_boundary_exact_threshold(self):
        """IoU exactly at threshold counts as a success (>=)."""
        ious = np.array([0.5, 0.5, 0.5])
        assert self.engine.success_rate_at(ious, 0.5) == pytest.approx(1.0)

    def test_empty_array(self):
        assert self.engine.success_rate_at(np.array([]), 0.5) == pytest.approx(0.0)

    def test_sr50_higher_than_sr75(self):
        """SR@0.50 should be >= SR@0.75 for any IoU sequence."""
        rng = np.random.default_rng(7)
        ious = rng.uniform(0, 1, 100)
        assert self.engine.success_rate_at(ious, 0.50) >= self.engine.success_rate_at(ious, 0.75)


class TestComputeAllExtended:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_compute_all_includes_new_fields(self):
        """compute_all should return an AccuracyMetrics with all 6 fields."""
        boxes = np.tile([0.0, 0.0, 30.0, 30.0], (20, 1))
        result = self.engine.compute_all(boxes, boxes)
        assert isinstance(result, AccuracyMetrics)
        # All fields should be populated
        assert hasattr(result, "norm_precision_auc")
        assert hasattr(result, "sr_50")
        assert hasattr(result, "sr_75")

    def test_perfect_tracker_gives_sr_100(self):
        """A tracker that always predicts IoU=1 should have SR@0.5=SR@0.75=1.0."""
        boxes = np.tile([5.0, 5.0, 40.0, 40.0], (30, 1))
        result = self.engine.compute_all(boxes, boxes)
        assert result.sr_50 == pytest.approx(1.0)
        assert result.sr_75 == pytest.approx(1.0)
        # At threshold=0 the strict-< condition excludes even zero-distance frames,
        # so the AUC is slightly below 1.0 (by one trapezoid step ≈ 0.01).
        assert result.norm_precision_auc == pytest.approx(1.0, abs=0.02)

    def test_zero_iou_tracker_gives_sr_zero(self):
        """Completely wrong predictions → SR@0.5 = SR@0.75 = 0."""
        gt = np.tile([0.0, 0.0, 10.0, 10.0], (10, 1))
        pred = np.tile([200.0, 200.0, 10.0, 10.0], (10, 1))
        result = self.engine.compute_all(pred, gt)
        assert result.sr_50 == pytest.approx(0.0)
        assert result.sr_75 == pytest.approx(0.0)

    def test_sr50_geq_sr75(self):
        """SR@0.50 >= SR@0.75 always holds."""
        rng = np.random.default_rng(0)
        preds = rng.uniform(0, 80, (40, 4))
        gts = rng.uniform(0, 80, (40, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 5
        gts[:, 2:] = np.abs(gts[:, 2:]) + 5
        result = self.engine.compute_all(preds, gts)
        assert result.sr_50 >= result.sr_75

    def test_norm_precision_in_range(self):
        """N-PREC AUC should always be in [0, 1]."""
        rng = np.random.default_rng(1)
        for _ in range(5):
            preds = rng.uniform(0, 100, (25, 4))
            gts = rng.uniform(0, 100, (25, 4))
            preds[:, 2:] = np.abs(preds[:, 2:]) + 1
            gts[:, 2:] = np.abs(gts[:, 2:]) + 1
            result = self.engine.compute_all(preds, gts)
            assert 0.0 <= result.norm_precision_auc <= 1.0

    def test_str_includes_new_metrics(self):
        """__str__ should display N-PREC AUC and SR values."""
        boxes = np.tile([0.0, 0.0, 20.0, 20.0], (5, 1))
        result = self.engine.compute_all(boxes, boxes)
        s = str(result)
        assert "N-PREC" in s
        assert "SR@0.5" in s
        assert "SR@0.75" in s
