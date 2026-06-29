"""Tests for extended latency distribution (P50/P99) and normalized precision metric."""

import time

import numpy as np
import pytest

from eovot.profiling.profiler import Profiler, ProfilingResult
from eovot.metrics.accuracy import MetricsEngine, AccuracyMetrics

# NumPy 2.0 renamed trapz → trapezoid; keep tests compatible with both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------------------------------------
# Latency distribution (profiler)
# ---------------------------------------------------------------------------

class TestLatencyDistribution:
    def setup_method(self):
        self.profiler = Profiler()

    def _run_frames(self, n: int = 20, sleep_s: float = 0.001) -> ProfilingResult:
        for _ in range(n):
            self.profiler.start_frame()
            time.sleep(sleep_s)
            self.profiler.end_frame()
        return self.profiler.summary("test")

    def test_result_has_p50(self):
        result = self._run_frames(10)
        assert hasattr(result, "latency_p50_ms")
        assert result.latency_p50_ms > 0.0

    def test_result_has_p99(self):
        result = self._run_frames(10)
        assert hasattr(result, "latency_p99_ms")
        assert result.latency_p99_ms > 0.0

    def test_percentile_ordering(self):
        """p50 ≤ p95 ≤ p99 must hold for any positive latency distribution."""
        result = self._run_frames(50)
        assert result.latency_p50_ms <= result.latency_p95_ms + 1e-9
        assert result.latency_p95_ms <= result.latency_p99_ms + 1e-9

    def test_p50_near_mean_for_uniform(self):
        """For a near-uniform distribution p50 should be close to the mean."""
        result = self._run_frames(100, sleep_s=0.002)
        ratio = result.latency_p50_ms / result.latency_mean_ms
        assert 0.5 < ratio < 2.0, f"p50/mean ratio {ratio:.3f} out of expected range"

    def test_p99_ge_p95(self):
        result = self._run_frames(20)
        assert result.latency_p99_ms >= result.latency_p95_ms - 1e-9

    def test_str_contains_p50_p99(self):
        result = self._run_frames(5)
        s = str(result)
        assert "p50=" in s
        assert "p99=" in s

    def test_skewed_distribution_tail(self):
        """P99 must exceed P50 in a right-skewed distribution.

        50 fast frames (~1 ms) + 10 slow frames (100 ms each) = 60 total.
        The slow frames are 16 % of all frames, so they appear at/above the 84th
        percentile.  P99 must therefore exceed P50 by a large margin.
        """
        for _ in range(50):
            self.profiler.start_frame()
            time.sleep(0.001)
            self.profiler.end_frame()
        for _ in range(10):
            self.profiler.start_frame()
            time.sleep(0.100)
            self.profiler.end_frame()
        result = self.profiler.summary("skewed")
        assert result.latency_p99_ms > result.latency_p50_ms * 5, (
            f"p99 should be >> p50 for a right-skewed distribution; "
            f"got p50={result.latency_p50_ms:.2f} ms, p99={result.latency_p99_ms:.2f} ms"
        )


# ---------------------------------------------------------------------------
# Normalized precision metric
# ---------------------------------------------------------------------------

class TestNormalizedPrecisionCurve:
    def setup_method(self):
        self.engine = MetricsEngine()

    def _make_boxes(self, n=50, offset=0.0, gt_size=40.0):
        """Generate (n, 4) arrays; predicted box is shifted by `offset` px."""
        gts = np.tile([100.0, 100.0, gt_size, gt_size], (n, 1))
        cx_gt = 100.0 + gt_size / 2
        cy_gt = 100.0 + gt_size / 2
        preds = np.tile(
            [cx_gt - gt_size / 2 + offset, cy_gt - gt_size / 2, gt_size, gt_size],
            (n, 1),
        )
        return preds, gts

    def test_perfect_tracking_np_auc_near_one(self):
        preds, gts = self._make_boxes(offset=0.0)
        thr, rates = self.engine.normalized_precision_curve(preds, gts)
        auc = float(_trapz(rates, thr) / thr[-1])
        assert auc > 0.95, f"Perfect tracking should give NP-AUC near 1, got {auc:.4f}"

    def test_large_offset_np_auc_low(self):
        """A large offset relative to sqrt(GT area) yields a low NP-AUC."""
        preds, gts = self._make_boxes(offset=200.0, gt_size=40.0)
        thr, rates = self.engine.normalized_precision_curve(preds, gts)
        auc = float(_trapz(rates, thr) / thr[-1])
        assert auc < 0.1, f"Large offset should give low NP-AUC, got {auc:.4f}"

    def test_scale_invariance(self):
        """Same relative offset should give the same NP-AUC regardless of target size."""
        small_preds, small_gts = self._make_boxes(offset=5.0, gt_size=20.0)
        large_preds, large_gts = self._make_boxes(offset=10.0, gt_size=40.0)

        _, small_rates = self.engine.normalized_precision_curve(small_preds, small_gts)
        _, large_rates = self.engine.normalized_precision_curve(large_preds, large_gts)

        # Both have offset / sqrt(area) = 5/sqrt(400)=0.25 and 10/sqrt(1600)=0.25
        np.testing.assert_allclose(small_rates, large_rates, atol=1e-9)

    def test_curve_returns_correct_shapes(self):
        preds, gts = self._make_boxes()
        thr, rates = self.engine.normalized_precision_curve(preds, gts)
        assert thr.shape == rates.shape
        assert len(thr) == 51  # default 51 points

    def test_default_threshold_range(self):
        preds, gts = self._make_boxes()
        thr, _ = self.engine.normalized_precision_curve(preds, gts)
        assert abs(thr[0]) < 1e-9
        assert abs(thr[-1] - 0.5) < 1e-9

    def test_rates_in_zero_one(self):
        preds, gts = self._make_boxes(offset=5.0)
        _, rates = self.engine.normalized_precision_curve(preds, gts)
        assert np.all(rates >= 0.0)
        assert np.all(rates <= 1.0 + 1e-9)

    def test_empty_input_returns_zeros(self):
        preds = np.empty((0, 4))
        gts = np.empty((0, 4))
        thr, rates = self.engine.normalized_precision_curve(preds, gts)
        assert np.all(rates == 0.0)

    def test_compute_all_includes_normalized_precision_auc(self):
        preds, gts = self._make_boxes(offset=2.0)
        result = self.engine.compute_all(preds, gts)
        assert isinstance(result, AccuracyMetrics)
        assert hasattr(result, "normalized_precision_auc")
        assert 0.0 <= result.normalized_precision_auc <= 1.0

    def test_np_auc_higher_for_better_tracker(self):
        """A tracker with smaller offset should have higher NP-AUC."""
        good_preds, gts = self._make_boxes(offset=1.0)
        poor_preds, _ = self._make_boxes(offset=15.0)
        good_result = self.engine.compute_all(good_preds, gts)
        poor_result = self.engine.compute_all(poor_preds, gts)
        assert good_result.normalized_precision_auc > poor_result.normalized_precision_auc

    def test_custom_thresholds(self):
        preds, gts = self._make_boxes()
        custom_thr = np.linspace(0.0, 1.0, 101)
        thr, rates = self.engine.normalized_precision_curve(preds, gts, thresholds=custom_thr)
        assert len(thr) == 101
        assert abs(thr[-1] - 1.0) < 1e-9
