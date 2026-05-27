"""Unit tests for normalized precision metric and extended profiling fields."""

from __future__ import annotations

import time

import numpy as np
import pytest

from eovot.metrics.accuracy import AccuracyMetrics, MetricsEngine
from eovot.profiling.profiler import Profiler, ProfilingResult


class TestNormalizedPrecisionCurve:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_perfect_prediction_max_rate(self):
        # Identical boxes → norm distance = 0 → rate = 1.0 at every threshold > 0
        boxes = np.tile([10.0, 10.0, 40.0, 40.0], (20, 1))
        thresholds, rates = self.engine.normalized_precision_curve(boxes, boxes)
        assert rates[-1] == pytest.approx(1.0)

    def test_far_prediction_zero_rate(self):
        # GT: small 4×4 box, pred: far away → norm dist >> 0.5 → rate at 0.5 ≈ 0
        gt = np.tile([0.0, 0.0, 4.0, 4.0], (10, 1))
        pred = np.tile([1000.0, 1000.0, 4.0, 4.0], (10, 1))
        thresholds, rates = self.engine.normalized_precision_curve(pred, gt)
        assert rates[-1] == pytest.approx(0.0)

    def test_output_shapes(self):
        boxes = np.random.default_rng(0).uniform(1, 100, (30, 4))
        boxes[:, 2:] = np.abs(boxes[:, 2:]) + 1.0
        thresholds, rates = self.engine.normalized_precision_curve(boxes, boxes)
        assert thresholds.shape == rates.shape
        assert len(thresholds) == 51  # default 51 points

    def test_rates_in_unit_interval(self):
        rng = np.random.default_rng(7)
        preds = rng.uniform(0, 200, (50, 4))
        gts = rng.uniform(0, 200, (50, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1.0
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1.0
        _, rates = self.engine.normalized_precision_curve(preds, gts)
        assert np.all(rates >= 0.0) and np.all(rates <= 1.0)

    def test_monotone_non_decreasing(self):
        # Larger threshold → at least as many frames qualify → rate non-decreasing
        rng = np.random.default_rng(13)
        preds = rng.uniform(0, 100, (40, 4))
        gts = rng.uniform(0, 100, (40, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1.0
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1.0
        _, rates = self.engine.normalized_precision_curve(preds, gts)
        assert np.all(np.diff(rates) >= -1e-9)

    def test_scale_invariance(self):
        # Scale gt box by 4× while keeping relative displacement constant →
        # normalized distance stays the same → same AUC.
        rng = np.random.default_rng(99)
        n = 30
        gt_small = rng.uniform(10, 50, (n, 4))
        gt_small[:, 2:] = 10.0  # 10×10 boxes
        gt_large = gt_small.copy()
        gt_large[:, 2:] = 40.0  # 40×40 boxes — 4× bigger

        # Displacement equal to 10% of target size for each
        disp_small = 1.0   # 10% of 10
        disp_large = 4.0   # 10% of 40

        pred_small = gt_small.copy()
        pred_small[:, 0] += disp_small
        pred_large = gt_large.copy()
        pred_large[:, 0] += disp_large

        _, rates_small = self.engine.normalized_precision_curve(pred_small, gt_small)
        _, rates_large = self.engine.normalized_precision_curve(pred_large, gt_large)
        np.testing.assert_allclose(rates_small, rates_large, atol=1e-9)

    def test_custom_thresholds(self):
        boxes = np.tile([0.0, 0.0, 20.0, 20.0], (5, 1))
        custom_thr = np.array([0.0, 0.1, 0.2, 0.3])
        thr_out, rates = self.engine.normalized_precision_curve(boxes, boxes,
                                                                thresholds=custom_thr)
        np.testing.assert_array_equal(thr_out, custom_thr)
        assert len(rates) == len(custom_thr)


class TestComputeAllIncludesNormPrecision:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_perfect_prediction_norm_prec_close_to_one(self):
        boxes = np.tile([5.0, 5.0, 30.0, 30.0], (20, 1))
        result = self.engine.compute_all(boxes, boxes)
        assert result.normalized_precision_auc == pytest.approx(1.0, abs=0.05)

    def test_returns_accuracy_metrics_instance(self):
        boxes = np.tile([0.0, 0.0, 10.0, 10.0], (10, 1))
        result = self.engine.compute_all(boxes, boxes)
        assert isinstance(result, AccuracyMetrics)
        assert hasattr(result, "normalized_precision_auc")

    def test_norm_prec_in_unit_interval(self):
        rng = np.random.default_rng(5)
        preds = rng.uniform(0, 100, (40, 4))
        gts = rng.uniform(0, 100, (40, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1.0
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1.0
        result = self.engine.compute_all(preds, gts)
        assert 0.0 <= result.normalized_precision_auc <= 1.0

    def test_str_includes_norm_precision(self):
        boxes = np.tile([0.0, 0.0, 10.0, 10.0], (5, 1))
        result = self.engine.compute_all(boxes, boxes)
        assert "norm_precision" in str(result)


class TestAccuracyMetricsBackwardCompat:
    """Ensure old code that omits normalized_precision_auc still works."""

    def test_default_zero(self):
        m = AccuracyMetrics(mean_iou=0.5, success_auc=0.4, precision_auc=0.3)
        assert m.normalized_precision_auc == 0.0

    def test_explicit_value(self):
        m = AccuracyMetrics(mean_iou=0.5, success_auc=0.4, precision_auc=0.3,
                            normalized_precision_auc=0.75)
        assert m.normalized_precision_auc == pytest.approx(0.75)


class TestExtendedProfilingFields:
    def setup_method(self):
        self.profiler = Profiler()

    def _run(self, n: int = 10) -> ProfilingResult:
        for _ in range(n):
            self.profiler.start_frame()
            time.sleep(0.002)
            self.profiler.end_frame()
        return self.profiler.summary("test")

    def test_latency_p99_present(self):
        result = self._run(10)
        assert hasattr(result, "latency_p99_ms")
        assert result.latency_p99_ms > 0.0

    def test_p99_ge_p95(self):
        result = self._run(20)
        assert result.latency_p99_ms >= result.latency_p95_ms - 1e-9

    def test_fps_std_present(self):
        result = self._run(10)
        assert hasattr(result, "fps_std")
        assert result.fps_std >= 0.0

    def test_str_includes_fps_std(self):
        result = self._run(5)
        assert "±" in str(result)

    def test_backward_compat_default_values(self):
        # ProfilingResult can still be built without new fields (they have defaults)
        pr = ProfilingResult(
            tracker_name="t",
            frame_count=5,
            fps=100.0,
            latency_mean_ms=10.0,
            latency_std_ms=1.0,
            latency_p95_ms=12.0,
            peak_memory_mb=50.0,
        )
        assert pr.latency_p99_ms == 0.0
        assert pr.fps_std == 0.0
