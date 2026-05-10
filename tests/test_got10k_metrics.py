"""Tests for GOT-10k protocol metrics: NormPrec, SR@0.5, SR@0.75, AO.

Covers:
- normalized_center_error()
- MetricsEngine.normalized_precision_curve()
- MetricsEngine.sr_at_threshold()
- AccuracyMetrics new fields (norm_precision_auc, sr_05, sr_075)
- BenchmarkResult.mean_sr_05 / mean_sr_075 / mean_norm_precision_auc
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.accuracy import (
    AccuracyMetrics,
    MetricsEngine,
    normalized_center_error,
)
from eovot.metrics import normalized_center_error as imported_nce


# ---------------------------------------------------------------------------
# normalized_center_error
# ---------------------------------------------------------------------------

class TestNormalizedCenterError:
    def test_same_box_returns_zero(self):
        box = (10.0, 10.0, 20.0, 20.0)
        assert normalized_center_error(box, box) == pytest.approx(0.0)

    def test_degenerate_gt_returns_inf(self):
        pred = (0.0, 0.0, 10.0, 10.0)
        gt_zero = (0.0, 0.0, 0.0, 0.0)
        assert normalized_center_error(pred, gt_zero) == float("inf")

    def test_known_value(self):
        # GT: 10×10 box centred at (15, 15) → scale = sqrt(10*10) = 10
        # pred centre at (18, 19), GT centre at (15, 15) → dist = sqrt(9+16) = 5
        # norm_error = 5 / 10 = 0.5
        gt = (10.0, 10.0, 10.0, 10.0)
        pred = (13.0, 14.0, 10.0, 10.0)
        assert normalized_center_error(pred, gt) == pytest.approx(0.5)

    def test_scale_invariance(self):
        # Doubling the GT box should halve the normalised error.
        gt_small = (0.0, 0.0, 10.0, 10.0)
        gt_large = (0.0, 0.0, 20.0, 20.0)
        pred = (5.0, 0.0, 10.0, 10.0)  # centre displaced 5px from GT centre

        err_small = normalized_center_error(pred, gt_small)
        err_large = normalized_center_error(pred, gt_large)
        assert err_small == pytest.approx(err_large * 2.0, rel=1e-6)

    def test_public_import(self):
        """normalized_center_error must be importable from the package root."""
        box = (0.0, 0.0, 10.0, 10.0)
        assert imported_nce(box, box) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MetricsEngine.normalized_precision_curve
# ---------------------------------------------------------------------------

class TestNormalizedPrecisionCurve:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_perfect_prediction_reaches_one(self):
        boxes = np.tile([0.0, 0.0, 20.0, 20.0], (10, 1))
        thresholds, rates = self.engine.normalized_precision_curve(boxes, boxes)
        # At any threshold > 0, all norm errors = 0 → rate = 1.0
        assert rates[-1] == pytest.approx(1.0)
        assert rates[1] == pytest.approx(1.0)

    def test_output_shapes(self):
        preds = np.tile([0.0, 0.0, 10.0, 10.0], (20, 1))
        gts = np.tile([5.0, 0.0, 10.0, 10.0], (20, 1))
        thresholds, rates = self.engine.normalized_precision_curve(preds, gts)
        assert len(thresholds) == len(rates)
        assert np.all(rates >= 0.0) and np.all(rates <= 1.0)

    def test_custom_thresholds(self):
        preds = np.tile([0.0, 0.0, 10.0, 10.0], (10, 1))
        gts = np.tile([0.0, 0.0, 10.0, 10.0], (10, 1))
        custom = np.array([0.0, 0.1, 0.2, 0.5])
        thresholds, rates = self.engine.normalized_precision_curve(preds, gts, custom)
        assert len(thresholds) == 4
        assert rates[-1] == pytest.approx(1.0)

    def test_degenerate_gt_does_not_crash(self):
        preds = np.tile([0.0, 0.0, 10.0, 10.0], (5, 1))
        gts = np.zeros((5, 4))  # all zero-area — all inf errors
        thresholds, rates = self.engine.normalized_precision_curve(preds, gts)
        # All frames have inf error → precision = 0 everywhere
        assert np.all(rates == pytest.approx(0.0))

    def test_monotonically_nondecreasing(self):
        """NormPrec rate must not decrease as threshold increases."""
        rng = np.random.default_rng(7)
        preds = rng.uniform(0, 50, (30, 4))
        gts = rng.uniform(0, 50, (30, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1
        thresholds, rates = self.engine.normalized_precision_curve(preds, gts)
        diffs = np.diff(rates)
        assert np.all(diffs >= -1e-9), "Normalized precision curve must be non-decreasing"


# ---------------------------------------------------------------------------
# MetricsEngine.sr_at_threshold
# ---------------------------------------------------------------------------

class TestSRAtThreshold:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_all_above_threshold(self):
        ious = np.array([0.6, 0.7, 0.8, 0.9])
        assert self.engine.sr_at_threshold(ious, 0.5) == pytest.approx(1.0)

    def test_none_above_threshold(self):
        ious = np.array([0.1, 0.2, 0.3])
        assert self.engine.sr_at_threshold(ious, 0.5) == pytest.approx(0.0)

    def test_half_above_threshold(self):
        ious = np.array([0.0, 0.0, 1.0, 1.0])
        assert self.engine.sr_at_threshold(ious, 0.5) == pytest.approx(0.5)

    def test_empty_array_returns_zero(self):
        assert self.engine.sr_at_threshold(np.array([]), 0.5) == pytest.approx(0.0)

    def test_boundary_value_is_counted(self):
        # IoU exactly == threshold should count (>= check).
        ious = np.array([0.5])
        assert self.engine.sr_at_threshold(ious, 0.5) == pytest.approx(1.0)

    def test_sr_05_vs_sr_075(self):
        ious = np.array([0.6, 0.6, 0.6, 0.8])
        sr05 = self.engine.sr_at_threshold(ious, 0.5)
        sr075 = self.engine.sr_at_threshold(ious, 0.75)
        assert sr05 == pytest.approx(1.0)
        assert sr075 == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# AccuracyMetrics (compute_all)
# ---------------------------------------------------------------------------

class TestAccuracyMetricsComputeAll:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_perfect_prediction_fields(self):
        boxes = np.tile([0.0, 0.0, 20.0, 20.0], (20, 1)).astype(float)
        result = self.engine.compute_all(boxes, boxes)
        assert isinstance(result, AccuracyMetrics)
        assert result.mean_iou == pytest.approx(1.0)
        assert result.sr_05 == pytest.approx(1.0)
        assert result.sr_075 == pytest.approx(1.0)
        assert result.norm_precision_auc == pytest.approx(1.0, abs=0.02)

    def test_all_fields_in_range(self):
        rng = np.random.default_rng(99)
        preds = rng.uniform(0, 100, (40, 4))
        gts = rng.uniform(0, 100, (40, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1
        result = self.engine.compute_all(preds, gts)
        for field_name in ("mean_iou", "success_auc", "precision_auc",
                           "norm_precision_auc", "sr_05", "sr_075"):
            val = getattr(result, field_name)
            assert 0.0 <= val <= 1.0, f"{field_name}={val} out of [0, 1]"

    def test_sr_ordering(self):
        """SR@0.75 must be <= SR@0.5 since it uses a stricter threshold."""
        rng = np.random.default_rng(3)
        preds = rng.uniform(0, 50, (50, 4))
        gts = rng.uniform(0, 50, (50, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1
        result = self.engine.compute_all(preds, gts)
        assert result.sr_075 <= result.sr_05 + 1e-9

    def test_str_representation(self):
        boxes = np.tile([5.0, 5.0, 10.0, 10.0], (5, 1)).astype(float)
        result = self.engine.compute_all(boxes, boxes)
        s = str(result)
        assert "SR@0.5" in s
        assert "SR@0.75" in s
        assert "norm_prec" in s

    def test_backward_compatibility_existing_fields(self):
        """mean_iou, success_auc, precision_auc must still be computed."""
        boxes = np.tile([0.0, 0.0, 10.0, 10.0], (10, 1)).astype(float)
        result = self.engine.compute_all(boxes, boxes)
        assert hasattr(result, "mean_iou")
        assert hasattr(result, "success_auc")
        assert hasattr(result, "precision_auc")
        assert result.mean_iou == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BenchmarkResult integration (mock-based, no real dataset needed)
# ---------------------------------------------------------------------------

class TestBenchmarkResultNewMetrics:
    """Verify BenchmarkResult exposes mean_sr_05, mean_sr_075, mean_norm_precision_auc."""

    def _make_result(self):
        from eovot.benchmark.engine import BenchmarkResult, SequenceResult
        from eovot.profiling.profiler import ProfilingResult

        prof = ProfilingResult(
            tracker_name="test",
            frame_count=10,
            fps=100.0,
            latency_mean_ms=10.0,
            latency_std_ms=0.5,
            latency_p95_ms=12.0,
            peak_memory_mb=50.0,
        )
        acc = AccuracyMetrics(
            mean_iou=0.6,
            success_auc=0.55,
            precision_auc=0.7,
            norm_precision_auc=0.65,
            sr_05=0.8,
            sr_075=0.4,
        )
        ious = np.full(10, 0.6)
        seq = SequenceResult(
            sequence_name="seq1",
            ious=ious,
            profiling=prof,
            accuracy=acc,
        )
        br = BenchmarkResult(tracker_name="test", dataset_name="mock")
        br.sequence_results.append(seq)
        return br

    def test_mean_sr_05(self):
        br = self._make_result()
        assert br.mean_sr_05 == pytest.approx(0.8)

    def test_mean_sr_075(self):
        br = self._make_result()
        assert br.mean_sr_075 == pytest.approx(0.4)

    def test_mean_norm_precision_auc(self):
        br = self._make_result()
        assert br.mean_norm_precision_auc == pytest.approx(0.65)

    def test_summary_contains_new_keys(self):
        br = self._make_result()
        summary = br.summary()
        assert "sr_05" in summary
        assert "sr_075" in summary
        assert "norm_precision_auc" in summary

    def test_no_accuracy_returns_none(self):
        from eovot.benchmark.engine import BenchmarkResult, SequenceResult
        from eovot.profiling.profiler import ProfilingResult

        prof = ProfilingResult(
            tracker_name="t", frame_count=5, fps=30.0,
            latency_mean_ms=33.0, latency_std_ms=1.0, latency_p95_ms=35.0,
            peak_memory_mb=20.0,
        )
        seq = SequenceResult(
            sequence_name="s", ious=np.zeros(5), profiling=prof
        )
        br = BenchmarkResult(tracker_name="t", dataset_name="d")
        br.sequence_results.append(seq)
        assert br.mean_sr_05 is None
        assert br.mean_sr_075 is None
        assert br.mean_norm_precision_auc is None
