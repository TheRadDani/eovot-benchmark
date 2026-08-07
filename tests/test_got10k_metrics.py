"""Tests for GOT-10k protocol metrics: GOT10kMetrics, MetricsEngine GOT-10k methods,
BenchmarkResult aggregate properties, and round-trip serialisation."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.accuracy import GOT10kMetrics, MetricsEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boxes(n: int, x=10.0, y=10.0, w=50.0, h=50.0) -> np.ndarray:
    """Return n identical boxes (x, y, w, h)."""
    return np.tile([x, y, w, h], (n, 1)).astype(np.float64)


def _engine() -> MetricsEngine:
    return MetricsEngine()


# ===========================================================================
# TestGOT10kMetrics — dataclass behaviour
# ===========================================================================

class TestGOT10kMetrics:
    def test_str_contains_all_fields(self):
        m = GOT10kMetrics(ao=0.5, sr_50=0.6, sr_75=0.3, normalized_precision=0.7)
        s = str(m)
        assert "AO=" in s
        assert "SR@0.5=" in s
        assert "SR@0.75=" in s
        assert "NP@0.20=" in s

    def test_to_dict_keys(self):
        m = GOT10kMetrics(ao=0.5, sr_50=0.6, sr_75=0.3, normalized_precision=0.7)
        d = m.to_dict()
        assert set(d.keys()) == {"ao", "sr_50", "sr_75", "normalized_precision"}

    def test_to_dict_rounding(self):
        m = GOT10kMetrics(ao=0.123456789, sr_50=1.0, sr_75=0.0, normalized_precision=0.5)
        d = m.to_dict()
        assert d["ao"] == round(0.123456789, 4)

    def test_to_dict_values_match_fields(self):
        m = GOT10kMetrics(ao=0.42, sr_50=0.55, sr_75=0.10, normalized_precision=0.80)
        d = m.to_dict()
        assert abs(d["ao"] - 0.42) < 1e-6
        assert abs(d["sr_50"] - 0.55) < 1e-6
        assert abs(d["sr_75"] - 0.10) < 1e-6
        assert abs(d["normalized_precision"] - 0.80) < 1e-6


# ===========================================================================
# TestSuccessRate
# ===========================================================================

class TestSuccessRate:
    def test_empty_returns_zero(self):
        eng = _engine()
        assert eng.success_rate(np.empty(0), 0.5) == 0.0

    def test_all_above_threshold(self):
        eng = _engine()
        ious = np.array([0.6, 0.7, 0.8, 0.9])
        assert eng.success_rate(ious, 0.5) == 1.0

    def test_all_below_threshold(self):
        eng = _engine()
        ious = np.array([0.1, 0.2, 0.3, 0.4])
        assert eng.success_rate(ious, 0.5) == 0.0

    def test_half_above_threshold(self):
        eng = _engine()
        ious = np.array([0.4, 0.4, 0.6, 0.6])
        assert abs(eng.success_rate(ious, 0.5) - 0.5) < 1e-9

    def test_boundary_value_included(self):
        eng = _engine()
        ious = np.array([0.5, 0.5, 0.5])
        assert eng.success_rate(ious, 0.5) == 1.0

    def test_sr50_geq_sr75_always(self):
        eng = _engine()
        ious = np.array([0.3, 0.55, 0.80, 0.90, 0.20])
        assert eng.success_rate(ious, 0.50) >= eng.success_rate(ious, 0.75)


# ===========================================================================
# TestNormalizedCenterDistance
# ===========================================================================

class TestNormalizedCenterDistance:
    def test_empty_returns_empty(self):
        eng = _engine()
        result = eng.normalized_center_distance(np.empty((0, 4)), np.empty((0, 4)))
        assert len(result) == 0

    def test_identical_boxes_zero_distance(self):
        eng = _engine()
        boxes = _boxes(5)
        result = eng.normalized_center_distance(boxes, boxes)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_scale_invariance(self):
        """Larger GT area → smaller normalised distance for the same pixel offset."""
        eng = _engine()
        # Same predicted box and same raw center offset (10 px), different GT sizes.
        # pred center  = (0, 0)   (box: x=-5, y=-5, w=10, h=10)
        # gt_small center = (10, 0), w=10, h=10  → norm = sqrt(100) = 10 → d = 10/10 = 1.0
        # gt_large center = (10, 0), w=100, h=100 → norm = sqrt(10000) = 100 → d = 10/100 = 0.1
        pred       = np.array([[-5.0, -5.0, 10.0,  10.0]])
        gt_small   = np.array([[ 5.0, -5.0, 10.0,  10.0]])
        gt_large   = np.array([[-40.0, -50.0, 100.0, 100.0]])

        d_small = eng.normalized_center_distance(pred, gt_small)[0]
        d_large = eng.normalized_center_distance(pred, gt_large)[0]
        # Same raw offset (10 px); larger GT → 10× smaller normalised distance.
        assert abs(d_small - 1.0) < 1e-9
        assert abs(d_large - 0.1) < 1e-9
        assert d_small > d_large

    def test_output_length_matches_min_input(self):
        eng = _engine()
        preds = _boxes(10)
        gts = _boxes(7)
        result = eng.normalized_center_distance(preds, gts)
        assert len(result) == 7

    def test_known_value(self):
        eng = _engine()
        # GT: x=0, y=0, w=100, h=100  → center (50, 50), area=10000, norm=100
        # pred: x=0, y=0, w=10, h=10  → center (5, 5)
        # raw_dist = sqrt((50-5)^2 + (50-5)^2) = 45*sqrt(2)
        # norm_dist = 45*sqrt(2) / 100
        pred = np.array([[0.0, 0.0, 10.0, 10.0]])
        gt   = np.array([[0.0, 0.0, 100.0, 100.0]])
        expected = 45.0 * np.sqrt(2) / 100.0
        result = eng.normalized_center_distance(pred, gt)
        assert abs(result[0] - expected) < 1e-9


# ===========================================================================
# TestNormalizedPrecisionCurve
# ===========================================================================

class TestNormalizedPrecisionCurve:
    def test_perfect_tracker_rate_one_at_all_thresholds(self):
        eng = _engine()
        boxes = _boxes(20)
        thresholds, rates = eng.normalized_precision_curve(boxes, boxes)
        # threshold 0 → rate should be 0 (strict <); all others should be 1
        np.testing.assert_allclose(rates[1:], 1.0, atol=1e-12)

    def test_output_shapes_match(self):
        eng = _engine()
        boxes = _boxes(10)
        thresholds, rates = eng.normalized_precision_curve(boxes, boxes)
        assert thresholds.shape == rates.shape
        assert len(thresholds) == 21  # default: 21 points in [0, 0.50]

    def test_custom_thresholds_respected(self):
        eng = _engine()
        boxes = _boxes(5)
        custom = np.array([0.1, 0.2, 0.3])
        thresholds, rates = eng.normalized_precision_curve(boxes, boxes, thresholds=custom)
        assert len(thresholds) == 3
        np.testing.assert_array_equal(thresholds, custom)

    def test_rates_monotonically_non_decreasing(self):
        eng = _engine()
        rng = np.random.default_rng(42)
        preds = rng.uniform(0, 100, (50, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1.0
        gts = rng.uniform(0, 100, (50, 4))
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1.0
        _, rates = eng.normalized_precision_curve(preds, gts)
        assert np.all(np.diff(rates) >= -1e-12)


# ===========================================================================
# TestComputeGOT10k
# ===========================================================================

class TestComputeGOT10k:
    def test_perfect_tracker(self):
        eng = _engine()
        boxes = _boxes(30)
        m = eng.compute_got10k(boxes, boxes)
        assert abs(m.ao - 1.0) < 1e-9
        assert abs(m.sr_50 - 1.0) < 1e-9
        assert abs(m.sr_75 - 1.0) < 1e-9
        # For identical boxes, normalised distance = 0 < 0.20 → NP = 1.0
        assert abs(m.normalized_precision - 1.0) < 1e-9

    def test_zero_iou_tracker(self):
        eng = _engine()
        # Predictions far from GTs → IoU 0 everywhere.
        preds = _boxes(10, x=0.0, y=0.0, w=5.0, h=5.0)
        gts   = _boxes(10, x=500.0, y=500.0, w=5.0, h=5.0)
        m = eng.compute_got10k(preds, gts)
        assert abs(m.ao - 0.0) < 1e-9
        assert abs(m.sr_50 - 0.0) < 1e-9
        assert abs(m.sr_75 - 0.0) < 1e-9

    def test_ao_equals_mean_iou(self):
        eng = _engine()
        rng = np.random.default_rng(7)
        preds = rng.uniform(0, 50, (40, 4)); preds[:, 2:] += 10
        gts   = rng.uniform(0, 50, (40, 4)); gts[:, 2:] += 10
        m = eng.compute_got10k(preds, gts)
        ious = eng.batch_iou(preds, gts)
        assert abs(m.ao - float(ious.mean())) < 1e-9

    def test_sr50_geq_sr75(self):
        eng = _engine()
        rng = np.random.default_rng(13)
        preds = rng.uniform(0, 100, (100, 4)); preds[:, 2:] += 5
        gts   = rng.uniform(0, 100, (100, 4)); gts[:, 2:] += 5
        m = eng.compute_got10k(preds, gts)
        assert m.sr_50 >= m.sr_75

    def test_custom_np_threshold(self):
        eng = _engine()
        boxes = _boxes(10)
        # identical boxes → norm_dist = 0 everywhere
        m_default = eng.compute_got10k(boxes, boxes, np_threshold=0.20)
        m_strict  = eng.compute_got10k(boxes, boxes, np_threshold=0.01)
        # Both should be 1.0 since norm_dist = 0 < any positive threshold.
        assert abs(m_default.normalized_precision - 1.0) < 1e-9
        assert abs(m_strict.normalized_precision - 1.0) < 1e-9

    def test_returns_got10k_metrics_type(self):
        eng = _engine()
        boxes = _boxes(5)
        m = eng.compute_got10k(boxes, boxes)
        assert isinstance(m, GOT10kMetrics)


# ===========================================================================
# TestBenchmarkResultGOT10k — aggregate properties on BenchmarkResult
# ===========================================================================

class TestBenchmarkResultGOT10k:
    """Integration tests that go through BenchmarkResult without running the engine."""

    def _make_result(self, got10k_list):
        from eovot.benchmark.engine import BenchmarkResult, SequenceResult
        from eovot.profiling.profiler import ProfilingResult

        def _prof(tracker_name="t"):
            return ProfilingResult(
                tracker_name=tracker_name,
                frame_count=10,
                fps=30.0,
                latency_mean_ms=33.3,
                latency_std_ms=1.0,
                latency_p95_ms=35.0,
                peak_memory_mb=100.0,
            )

        result = BenchmarkResult(tracker_name="TestTracker", dataset_name="TestDataset")
        for i, g in enumerate(got10k_list):
            sr = SequenceResult(
                sequence_name=f"seq{i}",
                ious=np.array([0.5, 0.6]),
                profiling=_prof(),
                got10k_metrics=g,
            )
            result.sequence_results.append(sr)
        return result

    def test_mean_ao(self):
        m1 = GOT10kMetrics(ao=0.4, sr_50=0.5, sr_75=0.2, normalized_precision=0.6)
        m2 = GOT10kMetrics(ao=0.6, sr_50=0.7, sr_75=0.3, normalized_precision=0.8)
        result = self._make_result([m1, m2])
        assert abs(result.mean_ao - 0.5) < 1e-9

    def test_mean_sr_50(self):
        m1 = GOT10kMetrics(ao=0.4, sr_50=0.4, sr_75=0.1, normalized_precision=0.5)
        m2 = GOT10kMetrics(ao=0.6, sr_50=0.8, sr_75=0.4, normalized_precision=0.9)
        result = self._make_result([m1, m2])
        assert abs(result.mean_sr_50 - 0.6) < 1e-9

    def test_mean_sr_75(self):
        m1 = GOT10kMetrics(ao=0.4, sr_50=0.5, sr_75=0.2, normalized_precision=0.6)
        m2 = GOT10kMetrics(ao=0.6, sr_50=0.7, sr_75=0.4, normalized_precision=0.8)
        result = self._make_result([m1, m2])
        assert abs(result.mean_sr_75 - 0.3) < 1e-9

    def test_mean_normalized_precision(self):
        m1 = GOT10kMetrics(ao=0.4, sr_50=0.5, sr_75=0.2, normalized_precision=0.6)
        m2 = GOT10kMetrics(ao=0.6, sr_50=0.7, sr_75=0.4, normalized_precision=0.8)
        result = self._make_result([m1, m2])
        assert abs(result.mean_normalized_precision - 0.7) < 1e-9

    def test_none_when_no_got10k_metrics(self):
        result = self._make_result([None, None])
        assert result.mean_ao is None
        assert result.mean_sr_50 is None
        assert result.mean_sr_75 is None
        assert result.mean_normalized_precision is None

    def test_summary_includes_got10k_keys(self):
        m = GOT10kMetrics(ao=0.5, sr_50=0.6, sr_75=0.3, normalized_precision=0.7)
        result = self._make_result([m])
        s = result.summary()
        assert "mean_ao" in s
        assert "mean_sr_50" in s
        assert "mean_sr_75" in s
        assert "mean_normalized_precision" in s

    def test_summary_excludes_got10k_keys_when_absent(self):
        result = self._make_result([None])
        s = result.summary()
        assert "mean_ao" not in s
        assert "mean_sr_50" not in s


# ===========================================================================
# TestRoundTrip — to_dict / from_dict serialisation
# ===========================================================================

class TestRoundTrip:
    def _make_result_with_got10k(self):
        from eovot.benchmark.engine import BenchmarkResult, SequenceResult
        from eovot.profiling.profiler import ProfilingResult

        prof = ProfilingResult(
            tracker_name="TestTracker",
            frame_count=10,
            fps=25.0,
            latency_mean_ms=40.0,
            latency_std_ms=2.0,
            latency_p95_ms=45.0,
            peak_memory_mb=200.0,
        )
        got10k = GOT10kMetrics(ao=0.55, sr_50=0.65, sr_75=0.30, normalized_precision=0.72)
        sr = SequenceResult(
            sequence_name="walking",
            ious=np.array([0.5, 0.6, 0.7]),
            profiling=prof,
            got10k_metrics=got10k,
        )
        result = BenchmarkResult(tracker_name="TestTracker", dataset_name="OTB100")
        result.sequence_results.append(sr)
        return result

    def test_to_dict_has_got10k_sequence_keys(self):
        result = self._make_result_with_got10k()
        d = result.to_dict()
        seq = d["sequences"][0]
        assert "ao" in seq
        assert "sr_50" in seq
        assert "sr_75" in seq
        assert "normalized_precision" in seq

    def test_from_dict_reconstructs_got10k(self):
        from eovot.benchmark.engine import BenchmarkResult
        result = self._make_result_with_got10k()
        d = result.to_dict()
        restored = BenchmarkResult.from_dict(d)
        g = restored.sequence_results[0].got10k_metrics
        assert g is not None
        assert abs(g.ao - 0.55) < 1e-3
        assert abs(g.sr_50 - 0.65) < 1e-3
        assert abs(g.sr_75 - 0.30) < 1e-3
        assert abs(g.normalized_precision - 0.72) < 1e-3

    def test_backward_compat_no_got10k_keys(self):
        """from_dict must accept legacy dicts that lack GOT-10k keys."""
        from eovot.benchmark.engine import BenchmarkResult
        legacy = {
            "summary": {"tracker": "OldTracker", "dataset": "OTB50"},
            "sequences": [
                {
                    "sequence_name": "car1",
                    "mean_iou": 0.5,
                    "fps": 30.0,
                    "mean_latency_ms": 33.3,
                    "peak_memory_mb": 100.0,
                }
            ],
        }
        restored = BenchmarkResult.from_dict(legacy)
        assert restored.sequence_results[0].got10k_metrics is None
        assert restored.mean_ao is None


# ===========================================================================
# TestComputeGOT10kIntegration — consistency with individual methods
# ===========================================================================

class TestComputeGOT10kIntegration:
    def test_consistency_with_individual_methods(self):
        """compute_got10k scalars must match the output of the individual methods."""
        eng = _engine()
        rng = np.random.default_rng(99)
        preds = rng.uniform(0, 80, (60, 4)); preds[:, 2:] += 5.0
        gts   = rng.uniform(0, 80, (60, 4)); gts[:, 2:] += 5.0

        m = eng.compute_got10k(preds, gts)

        ious = eng.batch_iou(preds, gts)
        assert abs(m.ao - float(ious.mean())) < 1e-9
        assert abs(m.sr_50 - eng.success_rate(ious, 0.50)) < 1e-9
        assert abs(m.sr_75 - eng.success_rate(ious, 0.75)) < 1e-9

        norm_dists = eng.normalized_center_distance(preds, gts)
        expected_np = float((norm_dists < 0.20).mean())
        assert abs(m.normalized_precision - expected_np) < 1e-9
