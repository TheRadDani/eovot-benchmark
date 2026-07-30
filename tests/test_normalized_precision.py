"""Tests for Normalized Precision metric (nPrec / nPrec AUC).

Validates the scale-invariant precision metric introduced in GOT-10k and
adopted by VOT 2020+ and TrackingNet.  The key invariant: normalising by the
GT diagonal makes the metric comparable across targets of different sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.accuracy import (
    AccuracyMetrics,
    MetricsEngine,
    normalized_center_distance,
)


class TestNormalizedCenterDistance:
    """Unit tests for the scalar normalized_center_distance function."""

    def test_perfect_prediction_returns_zero(self):
        box = (10.0, 10.0, 40.0, 30.0)
        assert normalized_center_distance(box, box) == pytest.approx(0.0)

    def test_known_value(self):
        # GT box at (0,0) with size (30, 40) → diagonal = sqrt(900+1600) = 50
        # Predicted box centre at (0+15, 0+20) = (15, 20)
        # GT centre at (15, 20); shift pred by (30, 40) → dist = sqrt(30^2+40^2) = 50
        gt = (0.0, 0.0, 30.0, 40.0)
        pred = (30.0, 40.0, 30.0, 40.0)
        # pixel_dist = 50, diag = 50 → normalised dist = 1.0
        assert normalized_center_distance(pred, gt) == pytest.approx(1.0)

    def test_degenerate_gt_returns_zero(self):
        gt = (5.0, 5.0, 0.0, 0.0)
        pred = (10.0, 10.0, 20.0, 20.0)
        assert normalized_center_distance(pred, gt) == pytest.approx(0.0)

    def test_scale_invariance(self):
        """Same relative error should give same normalised distance regardless of target size."""
        # Small target: 10×10, centred at (5,5); pred centre at (8,5) → dist=3, diag=~14.14
        gt_small = (0.0, 0.0, 10.0, 10.0)
        pred_small = (3.0, 0.0, 10.0, 10.0)  # centre shifted by 3 px

        # Large target: 100×100, centred at (50,50); pred centre at (80,50) → dist=30, diag=~141.4
        gt_large = (0.0, 0.0, 100.0, 100.0)
        pred_large = (30.0, 0.0, 100.0, 100.0)  # centre shifted by 30 px

        small_nd = normalized_center_distance(pred_small, gt_small)
        large_nd = normalized_center_distance(pred_large, gt_large)
        assert small_nd == pytest.approx(large_nd, rel=1e-5)


class TestBatchNormalizedCenterDistance:
    """Tests for the vectorised batch_normalized_center_distance method."""

    def setup_method(self):
        self.engine = MetricsEngine()

    def test_empty_inputs(self):
        preds = np.empty((0, 4))
        gts = np.empty((0, 4))
        result = self.engine.batch_normalized_center_distance(preds, gts)
        assert result.shape == (0,)

    def test_perfect_predictions(self):
        boxes = np.array([[10.0, 10.0, 30.0, 40.0]] * 5)
        result = self.engine.batch_normalized_center_distance(boxes, boxes)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_known_batch(self):
        # GT: size 30×40, diagonal = 50
        # Pred shifted by (3, 4) → pixel dist = 5 → norm dist = 5/50 = 0.1
        gts = np.array([[0.0, 0.0, 30.0, 40.0]] * 3)
        preds = np.array([[3.0, 4.0, 30.0, 40.0]] * 3)
        result = self.engine.batch_normalized_center_distance(preds, gts)
        np.testing.assert_allclose(result, 0.1, rtol=1e-5)

    def test_degenerate_gt_boxes_handled(self):
        gts = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 30.0, 40.0]])
        preds = np.array([[5.0, 5.0, 10.0, 10.0], [3.0, 4.0, 30.0, 40.0]])
        result = self.engine.batch_normalized_center_distance(preds, gts)
        assert result[0] == pytest.approx(0.0)   # degenerate GT → 0
        assert result[1] == pytest.approx(0.1, rel=1e-5)


class TestNormalizedPrecisionCurve:
    """Tests for the normalized_precision_curve method."""

    def setup_method(self):
        self.engine = MetricsEngine()

    def test_perfect_tracker_gives_all_ones(self):
        boxes = np.array([[10.0, 10.0, 30.0, 40.0]] * 20)
        thresholds, rates = self.engine.normalized_precision_curve(boxes, boxes)
        # At every threshold t > 0, all distances are 0 < t, so rate = 1.0
        np.testing.assert_allclose(rates[1:], 1.0, atol=1e-10)

    def test_threshold_range(self):
        boxes = np.zeros((10, 4)) + np.array([0.0, 0.0, 50.0, 50.0])
        thresholds, _ = self.engine.normalized_precision_curve(boxes, boxes)
        assert float(thresholds[0]) == pytest.approx(0.0)
        assert float(thresholds[-1]) == pytest.approx(0.5)

    def test_rate_monotone_in_threshold(self):
        """Precision rate must be non-decreasing as threshold increases."""
        rng = np.random.default_rng(0)
        preds = rng.uniform(0, 100, size=(50, 4))
        preds[:, 2:] = 30.0  # fix width/height
        gts = rng.uniform(0, 100, size=(50, 4))
        gts[:, 2:] = 30.0
        _, rates = self.engine.normalized_precision_curve(preds, gts)
        assert np.all(np.diff(rates) >= -1e-10)

    def test_known_precision_rate(self):
        # All predictions exactly 0.1 normalised units away from GT
        # GT: 30×40 → diagonal = 50
        gts = np.array([[0.0, 0.0, 30.0, 40.0]] * 10)
        preds = np.array([[3.0, 4.0, 30.0, 40.0]] * 10)  # norm_dist = 0.1
        thresholds = np.array([0.05, 0.10, 0.11, 0.20, 0.50])
        _, rates = self.engine.normalized_precision_curve(preds, gts, thresholds)
        # dist = 0.1; strict < threshold
        assert rates[0] == pytest.approx(0.0)   # 0.1 < 0.05 → False
        assert rates[1] == pytest.approx(0.0)   # 0.1 < 0.10 → False (strict)
        assert rates[2] == pytest.approx(1.0)   # 0.1 < 0.11 → True
        assert rates[3] == pytest.approx(1.0)
        assert rates[4] == pytest.approx(1.0)


class TestComputeAll:
    """Integration tests: compute_all should populate normalized_precision_auc."""

    def setup_method(self):
        self.engine = MetricsEngine()

    def test_nprec_auc_present_and_in_range(self):
        rng = np.random.default_rng(1)
        gts = np.column_stack([
            rng.uniform(0, 200, size=(30, 2)),
            rng.uniform(20, 60, size=(30, 2)),
        ])
        preds = gts + rng.normal(0, 5, size=(30, 4))
        result = self.engine.compute_all(preds, gts)

        assert isinstance(result, AccuracyMetrics)
        assert 0.0 <= result.normalized_precision_auc <= 1.0

    def test_perfect_tracker_nprec_auc_is_one(self):
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]] * 50)
        result = self.engine.compute_all(boxes, boxes)
        # At threshold=0 the strict '<' condition yields rate 0 for all-zero distances;
        # the AUC is therefore ≈ 1 − 1/(num_thresholds−1) ≈ 0.98, so check ≥ 0.98.
        assert result.normalized_precision_auc >= 0.98

    def test_nprec_auc_differs_from_prec_auc(self):
        """nPrec AUC uses a different scale than pixel-based precision AUC."""
        rng = np.random.default_rng(42)
        # Small targets (10×10): same pixel error = larger normalised error
        gts_small = np.column_stack([
            rng.uniform(0, 100, size=(40, 2)),
            np.full((40, 2), 10.0),
        ])
        preds_small = gts_small.copy()
        preds_small[:, :2] += 5.0  # 5 px error on a 10px target → norm dist ~0.35

        # Large targets (100×100): same pixel error = smaller normalised error
        gts_large = np.column_stack([
            rng.uniform(0, 100, size=(40, 2)),
            np.full((40, 2), 100.0),
        ])
        preds_large = gts_large.copy()
        preds_large[:, :2] += 5.0  # 5 px error on a 100px target → norm dist ~0.035

        result_small = self.engine.compute_all(preds_small, gts_small)
        result_large = self.engine.compute_all(preds_large, gts_large)

        # nPrec AUC should be higher for large targets (smaller normalised error)
        assert result_large.normalized_precision_auc > result_small.normalized_precision_auc

        # But pixel precision AUC should be identical (same absolute pixel error)
        assert result_small.precision_auc == pytest.approx(
            result_large.precision_auc, rel=0.05
        )


# ---------------------------------------------------------------------------
# BenchmarkResult integration
# ---------------------------------------------------------------------------

class TestBenchmarkResultNPAUC:
    """Aggregate + serialization integration for mean_normalized_precision_auc."""

    def _make_result(self, npauc_values):
        from eovot.benchmark.engine import BenchmarkResult, SequenceResult
        from eovot.profiling.profiler import ProfilingResult

        result = BenchmarkResult(tracker_name="T", dataset_name="D")
        for i, npauc in enumerate(npauc_values):
            pr = ProfilingResult(
                tracker_name="T", frame_count=30, fps=100.0, latency_mean_ms=10.0,
                latency_std_ms=1.0, latency_p95_ms=12.0, peak_memory_mb=50.0,
            )
            am = AccuracyMetrics(
                mean_iou=0.5, success_auc=0.5, precision_auc=0.5,
                normalized_precision_auc=npauc,
            )
            result.sequence_results.append(
                SequenceResult(
                    sequence_name=f"s{i}",
                    ious=np.array([0.5] * 30),
                    profiling=pr,
                    accuracy_metrics=am,
                )
            )
        return result

    def test_mean_npauc_correct(self):
        result = self._make_result([0.2, 0.4, 0.6, 0.8])
        assert result.mean_normalized_precision_auc == pytest.approx(0.5, rel=1e-5)

    def test_mean_npauc_none_when_no_accuracy_metrics(self):
        from eovot.benchmark.engine import BenchmarkResult, SequenceResult
        from eovot.profiling.profiler import ProfilingResult

        result = BenchmarkResult(tracker_name="T", dataset_name="D")
        pr = ProfilingResult(
            tracker_name="T", frame_count=10, fps=50.0, latency_mean_ms=20.0,
            latency_std_ms=1.0, latency_p95_ms=22.0, peak_memory_mb=10.0,
        )
        result.sequence_results.append(
            SequenceResult(
                sequence_name="s0", ious=np.array([0.5] * 10), profiling=pr,
                accuracy_metrics=None,
            )
        )
        assert result.mean_normalized_precision_auc is None

    def test_summary_exposes_npauc(self):
        result = self._make_result([0.3, 0.5])
        s = result.summary()
        assert "normalized_precision_auc" in s
        assert s["normalized_precision_auc"] == pytest.approx(0.4, abs=0.001)

    def test_to_dict_per_sequence_key(self):
        result = self._make_result([0.6, 0.8])
        d = result.to_dict()
        for seq in d["sequences"]:
            assert "normalized_precision_auc" in seq

    def test_to_dict_summary_roundtrip(self):
        result = self._make_result([0.35, 0.65])
        d = result.to_dict()
        # The summary dict should carry the aggregate
        assert "normalized_precision_auc" in d["summary"]
        assert d["summary"]["normalized_precision_auc"] == pytest.approx(0.5, abs=0.001)

    def test_end_to_end_pipeline(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=3, num_frames=25, motion="linear")
        result = BenchmarkEngine(verbose=False).run(MOSSETracker(), dataset, dataset_name="Syn")
        assert result.mean_normalized_precision_auc is not None
        assert 0.0 <= result.mean_normalized_precision_auc <= 1.0 + 1e-9
        for sr in result.sequence_results:
            assert sr.normalized_precision_auc is not None
