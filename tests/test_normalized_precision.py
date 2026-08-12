"""Tests for the Normalized Precision metric (nPrec / nPrec AUC).

Validates the scale-invariant precision metric introduced in GOT-10k and
adopted by VOT 2020+, TrackingNet, and LaSOT.  Centre distance is normalised
by the GT box diagonal (``sqrt(w²+h²)``) before thresholding.

Covers:
- Scalar ``normalized_center_distance`` function correctness
- Vectorised ``batch_normalized_center_distance`` method
- ``normalized_precision_curve`` shape, values, and monotonicity
- ``compute_all`` integrating both ``norm_precision_auc`` and ``normalized_precision_auc``
- Scale-invariance property (the key advantage over OTB precision)
- Backward-compatibility: ``norm_precision_auc`` defaults to 0.0 in AccuracyMetrics
- BenchmarkResult surfaces both keys in summary() and to_dict()
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.accuracy import (
    AccuracyMetrics,
    MetricsEngine,
    normalized_center_distance,
)
from eovot.benchmark.engine import BenchmarkResult, BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(cx: float, cy: float, w: float, h: float) -> tuple:
    """Return (x, y, w, h) from centre and size."""
    return (cx - w / 2, cy - h / 2, w, h)


def _arr(*boxes):
    return np.array(boxes, dtype=np.float64)


ENGINE = MetricsEngine()


# ---------------------------------------------------------------------------
# AccuracyMetrics backward-compatibility
# ---------------------------------------------------------------------------

class TestAccuracyMetricsBackwardCompat:
    def test_default_norm_precision_auc(self):
        m = AccuracyMetrics(mean_iou=0.5, success_auc=0.4, precision_auc=0.6)
        assert m.norm_precision_auc == pytest.approx(0.0)
        assert m.normalized_precision_auc == pytest.approx(0.0)

    def test_explicit_norm_precision_auc(self):
        m = AccuracyMetrics(
            mean_iou=0.5, success_auc=0.4, precision_auc=0.6, norm_precision_auc=0.75
        )
        assert m.norm_precision_auc == pytest.approx(0.75)

    def test_explicit_normalized_precision_auc(self):
        m = AccuracyMetrics(
            mean_iou=0.5, success_auc=0.4, precision_auc=0.6, normalized_precision_auc=0.80
        )
        assert m.normalized_precision_auc == pytest.approx(0.80)

    def test_str_includes_nprec(self):
        m = AccuracyMetrics(
            mean_iou=0.5, success_auc=0.4, precision_auc=0.6, normalized_precision_auc=0.75
        )
        s = str(m)
        assert "nPrec_AUC" in s or "norm_precision" in s.lower()


# ---------------------------------------------------------------------------
# Scalar normalized_center_distance function
# ---------------------------------------------------------------------------

class TestNormalizedCenterDistance:
    """Unit tests for the scalar normalized_center_distance function."""

    def test_perfect_prediction_returns_zero(self):
        box = (10.0, 10.0, 40.0, 30.0)
        assert normalized_center_distance(box, box) == pytest.approx(0.0)

    def test_known_value(self):
        # GT box at (0,0) with size (30, 40) → diagonal = sqrt(900+1600) = 50
        # GT centre = (15, 20); pred shifted by (30, 40) → pred centre = (45, 60)
        # pixel_dist = sqrt(30²+40²) = 50; norm_dist = 50/50 = 1.0
        gt = (0.0, 0.0, 30.0, 40.0)
        pred = (30.0, 40.0, 30.0, 40.0)
        assert normalized_center_distance(pred, gt) == pytest.approx(1.0)

    def test_degenerate_gt_returns_zero(self):
        gt = (5.0, 5.0, 0.0, 0.0)
        pred = (10.0, 10.0, 20.0, 20.0)
        assert normalized_center_distance(pred, gt) == pytest.approx(0.0)

    def test_scale_invariance(self):
        """Same relative error gives same normalised distance regardless of target size."""
        # Small target: 10×10, diagonal ≈ 14.14; pred shifted by 10 % of diagonal
        gt_small = (0.0, 0.0, 10.0, 10.0)
        pred_small = (3.0, 0.0, 10.0, 10.0)

        # Large target: 100×100, diagonal ≈ 141.4; pred shifted by same fraction
        gt_large = (0.0, 0.0, 100.0, 100.0)
        pred_large = (30.0, 0.0, 100.0, 100.0)

        small_nd = normalized_center_distance(pred_small, gt_small)
        large_nd = normalized_center_distance(pred_large, gt_large)
        assert small_nd == pytest.approx(large_nd, rel=1e-5)


# ---------------------------------------------------------------------------
# Vectorised batch_normalized_center_distance
# ---------------------------------------------------------------------------

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
        # GT: size 30×40 → diagonal = 50
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


# ---------------------------------------------------------------------------
# normalized_precision_curve
# ---------------------------------------------------------------------------

class TestNormalizedPrecisionCurve:
    """Tests for normalized_precision_curve (diagonal normalization)."""

    def setup_method(self):
        self.engine = MetricsEngine()

    def test_perfect_tracker_gives_all_ones(self):
        boxes = np.array([[10.0, 10.0, 30.0, 40.0]] * 20)
        thresholds, rates = self.engine.normalized_precision_curve(boxes, boxes)
        # At every threshold t > 0, all distances are 0 < t → rate = 1.0
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
        preds[:, 2:] = 30.0
        gts = rng.uniform(0, 100, size=(50, 4))
        gts[:, 2:] = 30.0
        _, rates = self.engine.normalized_precision_curve(preds, gts)
        assert np.all(np.diff(rates) >= -1e-10)

    def test_known_precision_rate(self):
        # GT: 30×40 → diagonal = 50; pred shifted (3,4) → pixel_dist=5 → norm_dist=0.1
        gts = np.array([[0.0, 0.0, 30.0, 40.0]] * 10)
        preds = np.array([[3.0, 4.0, 30.0, 40.0]] * 10)  # norm_dist = 0.1
        thresholds = np.array([0.05, 0.10, 0.11, 0.20, 0.50])
        _, rates = self.engine.normalized_precision_curve(preds, gts, thresholds)
        assert rates[0] == pytest.approx(0.0)   # 0.1 < 0.05 → False
        assert rates[1] == pytest.approx(0.0)   # 0.1 < 0.10 → False (strict <)
        assert rates[2] == pytest.approx(1.0)   # 0.1 < 0.11 → True
        assert rates[3] == pytest.approx(1.0)
        assert rates[4] == pytest.approx(1.0)

    def test_large_displacement_gives_zero(self):
        """Very large displacement relative to box size → precision = 0."""
        # GT box 10×10 → diagonal ≈ 14.14; pred shifted by 1000 px → norm_dist ≈ 70.7
        preds = np.tile([1000.0, 1000.0, 10.0, 10.0], (10, 1))
        gts = np.tile([0.0, 0.0, 10.0, 10.0], (10, 1))
        _, rates = ENGINE.normalized_precision_curve(preds, gts)
        assert rates[-1] == pytest.approx(0.0)

    def test_scale_invariance(self):
        """Key property: same fractional offset → same precision regardless of absolute scale."""
        # Small: 10×10 box, shift 1 px → norm_dist = 1/√200 ≈ 0.0707
        preds_small = np.tile([1.0, 0.0, 10.0, 10.0], (20, 1))
        gts_small = np.tile([0.0, 0.0, 10.0, 10.0], (20, 1))

        # Large: 100×100 box, shift 10 px → norm_dist = 10/√20000 ≈ 0.0707
        preds_large = np.tile([10.0, 0.0, 100.0, 100.0], (20, 1))
        gts_large = np.tile([0.0, 0.0, 100.0, 100.0], (20, 1))

        thr = np.linspace(0.0, 0.5, 51)
        _, rates_small = self.engine.normalized_precision_curve(preds_small, gts_small, thresholds=thr)
        _, rates_large = self.engine.normalized_precision_curve(preds_large, gts_large, thresholds=thr)
        np.testing.assert_allclose(rates_small, rates_large, atol=1e-9)

    def test_empty_returns_zeros(self):
        thr, rates = ENGINE.normalized_precision_curve(
            np.empty((0, 4)), np.empty((0, 4))
        )
        assert np.all(rates == 0.0)
        assert len(thr) == len(rates)

    def test_default_thresholds_range(self):
        preds = np.tile([0.0, 0.0, 50.0, 50.0], (5, 1))
        gts = np.tile([0.0, 0.0, 50.0, 50.0], (5, 1))
        thr, rates = ENGINE.normalized_precision_curve(preds, gts)
        assert thr[0] == pytest.approx(0.0)
        assert thr[-1] == pytest.approx(0.5)
        assert len(thr) == 51

    def test_curve_is_monotone_non_decreasing(self):
        rng = np.random.default_rng(7)
        preds = rng.uniform(0, 100, (50, 4))
        gts = rng.uniform(0, 100, (50, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 5
        gts[:, 2:] = np.abs(gts[:, 2:]) + 5
        thr, rates = ENGINE.normalized_precision_curve(preds, gts)
        assert np.all(np.diff(rates) >= -1e-9)

    def test_mixed_sequence(self):
        """Half perfect, half drifted → ~0.5 precision at moderate threshold."""
        perfect = np.tile([0.0, 0.0, 50.0, 50.0], (5, 1))
        drifted = np.tile([100.0, 0.0, 50.0, 50.0], (5, 1))
        gt = np.tile([0.0, 0.0, 50.0, 50.0], (10, 1))
        preds = np.vstack([perfect, drifted])
        thr = np.array([0.1])
        _, rates = ENGINE.normalized_precision_curve(preds, gt, thresholds=thr)
        assert rates[0] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# compute_all() integration
# ---------------------------------------------------------------------------

class TestComputeAll:
    """Integration: compute_all populates both norm_precision_auc and normalized_precision_auc."""

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
        # norm_precision_auc is an alias
        assert result.norm_precision_auc == pytest.approx(result.normalized_precision_auc)

    def test_perfect_tracker_nprec_auc_is_high(self):
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]] * 50)
        result = self.engine.compute_all(boxes, boxes)
        # At threshold=0 the strict '<' yields rate 0 for all-zero distances; AUC ≈ 0.98
        assert result.normalized_precision_auc >= 0.98
        assert result.norm_precision_auc >= 0.98

    def test_nprec_auc_differs_for_different_errors(self):
        """nPrec AUC should be higher for large targets (smaller normalised error)."""
        rng = np.random.default_rng(42)
        gts_small = np.column_stack([
            rng.uniform(0, 100, size=(40, 2)),
            np.full((40, 2), 10.0),
        ])
        preds_small = gts_small.copy()
        preds_small[:, :2] += 5.0  # 5 px error on a 10 px target → large norm error

        gts_large = np.column_stack([
            rng.uniform(0, 100, size=(40, 2)),
            np.full((40, 2), 100.0),
        ])
        preds_large = gts_large.copy()
        preds_large[:, :2] += 5.0  # 5 px error on a 100 px target → small norm error

        result_small = self.engine.compute_all(preds_small, gts_small)
        result_large = self.engine.compute_all(preds_large, gts_large)
        assert result_large.normalized_precision_auc > result_small.normalized_precision_auc

    def test_random_boxes_range(self):
        rng = np.random.default_rng(99)
        preds = rng.uniform(0, 100, (50, 4))
        gts = rng.uniform(0, 100, (50, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1
        result = ENGINE.compute_all(preds, gts)
        assert 0.0 <= result.norm_precision_auc <= 1.0

    def test_high_np_auc_for_small_error(self):
        """Small positional error (1% of box size) → NP AUC > 0.95."""
        rng = np.random.default_rng(1)
        size = 60.0
        n = 40
        gts = np.zeros((n, 4), dtype=np.float64)
        gts[:, :2] = rng.uniform(0, 200, (n, 2))
        gts[:, 2:] = size
        noise = rng.uniform(-0.01 * size, 0.01 * size, (n, 2))
        preds = gts.copy()
        preds[:, :2] += noise
        result = ENGINE.compute_all(preds, gts)
        assert result.norm_precision_auc > 0.95, \
            f"Expected NP AUC > 0.95 for 1% error, got {result.norm_precision_auc:.4f}"


# ---------------------------------------------------------------------------
# BenchmarkResult surfaces norm_precision_auc / normalized_precision_auc
# ---------------------------------------------------------------------------

class TestBenchmarkResultNormPrecision:
    def _run_benchmark(self) -> BenchmarkResult:
        dataset = SyntheticDataset(num_sequences=2, num_frames=30, seed=0)
        engine = BenchmarkEngine(verbose=False)
        return engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")

    def test_summary_includes_norm_precision_auc(self):
        result = self._run_benchmark()
        summary = result.summary()
        assert "norm_precision_auc" in summary
        assert "normalized_precision_auc" in summary
        assert 0.0 <= summary["norm_precision_auc"] <= 1.0
        assert 0.0 <= summary["normalized_precision_auc"] <= 1.0

    def test_mean_norm_precision_auc_property(self):
        result = self._run_benchmark()
        assert result.mean_norm_precision_auc is not None
        assert result.mean_normalized_precision_auc is not None
        assert result.mean_norm_precision_auc == pytest.approx(
            result.mean_normalized_precision_auc
        )

    def test_sequence_results_include_norm_precision_auc(self):
        result = self._run_benchmark()
        d = result.to_dict()
        for seq in d["sequences"]:
            assert "norm_precision_auc" in seq
            assert "normalized_precision_auc" in seq
            assert 0.0 <= seq["norm_precision_auc"] <= 1.0

    def test_sequence_result_properties(self):
        result = self._run_benchmark()
        for sr in result.sequence_results:
            assert sr.norm_precision_auc is not None
            assert sr.normalized_precision_auc is not None
            assert 0.0 <= sr.norm_precision_auc <= 1.0

    def test_round_trip_json(self):
        import tempfile, os
        result = self._run_benchmark()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result.save(path)
            loaded = BenchmarkResult.load(path)
            for orig, load in zip(result.sequence_results, loaded.sequence_results):
                if orig.accuracy_metrics is not None:
                    assert load.accuracy_metrics is not None
                    assert load.accuracy_metrics.norm_precision_auc == pytest.approx(
                        orig.accuracy_metrics.norm_precision_auc, abs=1e-4
                    )
        finally:
            os.unlink(path)

    def test_from_dict_backward_compat(self):
        """from_dict() handles old JSON without norm_precision_auc."""
        old_dict = {
            "summary": {"tracker": "MOSSE", "dataset": "OTB"},
            "sequences": [
                {
                    "sequence_name": "seq1",
                    "mean_iou": 0.7,
                    "fps": 100.0,
                    "mean_latency_ms": 10.0,
                    "peak_memory_mb": 50.0,
                    "success_auc": 0.65,
                    "precision_auc": 0.80,
                    # no norm_precision_auc — old format
                }
            ],
        }
        result = BenchmarkResult.from_dict(old_dict)
        assert result.sequence_results[0].accuracy_metrics is not None
        assert result.sequence_results[0].accuracy_metrics.norm_precision_auc == pytest.approx(0.0)
        assert result.sequence_results[0].accuracy_metrics.normalized_precision_auc == pytest.approx(0.0)


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
        assert "normalized_precision_auc" in d["summary"]
        assert d["summary"]["normalized_precision_auc"] == pytest.approx(0.5, abs=0.001)

    def test_end_to_end_pipeline(self):
        dataset = SyntheticDataset(num_sequences=3, num_frames=25, motion="linear")
        result = BenchmarkEngine(verbose=False).run(MOSSETracker(), dataset, dataset_name="Syn")
        assert result.mean_normalized_precision_auc is not None
        assert 0.0 <= result.mean_normalized_precision_auc <= 1.0 + 1e-9
        for sr in result.sequence_results:
            assert sr.normalized_precision_auc is not None
