"""Tests for the LaSOT-standard Normalized Precision metric.

Covers:
- Correctness of the normalized distance computation
- Curve shape and boundary values
- AUC integration via compute_all()
- Scale-invariance property (the key advantage over OTB precision)
- Backward-compatibility: norm_precision_auc defaults to 0.0 in AccuracyMetrics
- BenchmarkResult surfaces norm_precision_auc in summary() and to_dict()
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.accuracy import AccuracyMetrics, MetricsEngine
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

    def test_explicit_norm_precision_auc(self):
        m = AccuracyMetrics(mean_iou=0.5, success_auc=0.4, precision_auc=0.6, norm_precision_auc=0.75)
        assert m.norm_precision_auc == pytest.approx(0.75)

    def test_str_includes_norm_precision(self):
        m = AccuracyMetrics(mean_iou=0.5, success_auc=0.4, precision_auc=0.6, norm_precision_auc=0.75)
        s = str(m)
        assert "norm_precision_AUC" in s


# ---------------------------------------------------------------------------
# Normalized precision curve correctness
# ---------------------------------------------------------------------------

class TestNormalizedPrecisionCurve:
    def test_perfect_prediction_zero_dist(self):
        """Perfect predictions → all norm_dists=0 → precision=1 for all thresholds>0."""
        boxes = _arr([0, 0, 50, 50]) * 5
        preds = np.tile([0.0, 0.0, 50.0, 50.0], (20, 1))
        gts = np.tile([0.0, 0.0, 50.0, 50.0], (20, 1))
        thr, rates = ENGINE.normalized_precision_curve(preds, gts)
        # At threshold = 0: distance is 0, so 0 < 0 is False → precision = 0
        # At all other thresholds: 0 < t is True → precision = 1
        assert rates[-1] == pytest.approx(1.0), "Should be 1.0 at max threshold"
        assert rates[1] == pytest.approx(1.0), "Should be 1.0 already at second threshold"
        assert rates[0] == pytest.approx(0.0), "At threshold=0, 0 < 0 is False"

    def test_large_displacement_gives_zero(self):
        """Very large displacement relative to box size → all norm_dists >> 0.5 → precision=0."""
        # GT box 10x10, prediction shifted by 1000 px → norm_dist = 1000/sqrt(100) = 100
        preds = np.tile([1000.0, 1000.0, 10.0, 10.0], (10, 1))
        gts = np.tile([0.0, 0.0, 10.0, 10.0], (10, 1))
        thr, rates = ENGINE.normalized_precision_curve(preds, gts)
        assert rates[-1] == pytest.approx(0.0), "All precision rates should be 0 for huge displacement"

    def test_known_norm_distance(self):
        """Verify exact normalized distance calculation.

        GT box: (0, 0, 10, 10) → area=100, √area=10
        Pred box: (5, 0, 10, 10) → pred_centre=(10, 5), gt_centre=(5, 5)
        raw_dist = 5.0
        norm_dist = 5.0 / 10.0 = 0.5
        """
        preds = _arr([5.0, 0.0, 10.0, 10.0])
        gts = _arr([0.0, 0.0, 10.0, 10.0])
        thr = np.array([0.0, 0.4, 0.5, 0.6])
        rates_thr, rates = ENGINE.normalized_precision_curve(preds, gts, thresholds=thr)
        # norm_dist = 0.5; threshold < 0.5 should give 0, >= 0.5 is borderline
        assert rates[0] == pytest.approx(0.0)   # 0.5 < 0.0 → False
        assert rates[1] == pytest.approx(0.0)   # 0.5 < 0.4 → False
        # 0.5 < 0.5 is False (strict <)
        assert rates[2] == pytest.approx(0.0)   # 0.5 < 0.5 → False
        assert rates[3] == pytest.approx(1.0)   # 0.5 < 0.6 → True

    def test_scale_invariance(self):
        """Key property: same fractional offset → same precision regardless of absolute scale.

        Small sequence: GT 10×10, pred centre shifted by 1 px → norm_dist = 1/10 = 0.1
        Large sequence: GT 100×100, pred centre shifted by 10 px → norm_dist = 10/100 = 0.1
        Both should give identical precision curves.
        """
        # Small: 10x10 box, shift 1px
        preds_small = np.tile([1.0, 0.0, 10.0, 10.0], (20, 1))
        gts_small = np.tile([0.0, 0.0, 10.0, 10.0], (20, 1))

        # Large: 100x100 box, shift 10px (same fractional shift)
        preds_large = np.tile([10.0, 0.0, 100.0, 100.0], (20, 1))
        gts_large = np.tile([0.0, 0.0, 100.0, 100.0], (20, 1))

        thr = np.linspace(0.0, 0.5, 51)
        _, rates_small = ENGINE.normalized_precision_curve(preds_small, gts_small, thresholds=thr)
        _, rates_large = ENGINE.normalized_precision_curve(preds_large, gts_large, thresholds=thr)

        np.testing.assert_allclose(rates_small, rates_large, atol=1e-9,
                                   err_msg="Normalized precision should be scale-invariant")

    def test_ota_precision_is_not_scale_invariant(self):
        """OTB precision curve DOES change with absolute scale — confirms NP is needed."""
        # Small box
        preds_small = np.tile([1.0, 0.0, 10.0, 10.0], (20, 1))
        gts_small = np.tile([0.0, 0.0, 10.0, 10.0], (20, 1))
        # Large box — same fractional shift
        preds_large = np.tile([10.0, 0.0, 100.0, 100.0], (20, 1))
        gts_large = np.tile([0.0, 0.0, 100.0, 100.0], (20, 1))

        _, rates_s = ENGINE.precision_curve(preds_small, gts_small)
        _, rates_l = ENGINE.precision_curve(preds_large, gts_large)

        # OTB curves should differ (small: dist=0.5px < many thresholds; large: dist=5px < fewer)
        assert not np.allclose(rates_s, rates_l), \
            "OTB precision should differ with scale — this validates NP is needed"

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

    def test_zero_gt_area_handled(self):
        """Degenerate GT box (zero area) should not raise; its norm_dist is inf."""
        preds = _arr([5.0, 5.0, 10.0, 10.0])
        gts = _arr([5.0, 5.0, 0.0, 0.0])  # zero-area GT
        thr, rates = ENGINE.normalized_precision_curve(preds, gts)
        # norm_dist = inf → never < any finite threshold
        assert np.all(rates == 0.0)

    def test_curve_is_monotone_non_increasing(self):
        """Precision curve should be non-increasing (more relaxed threshold → higher rate)."""
        rng = np.random.default_rng(7)
        preds = rng.uniform(0, 100, (50, 4))
        gts = rng.uniform(0, 100, (50, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 5
        gts[:, 2:] = np.abs(gts[:, 2:]) + 5
        thr, rates = ENGINE.normalized_precision_curve(preds, gts)
        assert np.all(np.diff(rates) >= -1e-9), "Curve must be non-increasing"

    def test_mixed_sequence(self):
        """Half frames have zero norm_dist, half have large norm_dist → 0.5 precision."""
        perfect = np.tile([0.0, 0.0, 50.0, 50.0], (5, 1))
        drifted = np.tile([100.0, 0.0, 50.0, 50.0], (5, 1))  # centre shifts 100 px
        gt = np.tile([0.0, 0.0, 50.0, 50.0], (10, 1))
        preds = np.vstack([perfect, drifted])
        thr = np.array([0.1])
        _, rates = ENGINE.normalized_precision_curve(preds, gt, thresholds=thr)
        assert rates[0] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# compute_all() integration
# ---------------------------------------------------------------------------

class TestComputeAllNormPrecision:
    def test_perfect_boxes(self):
        preds = np.tile([10.0, 10.0, 50.0, 50.0], (30, 1))
        gts = np.tile([10.0, 10.0, 50.0, 50.0], (30, 1))
        result = ENGINE.compute_all(preds, gts)
        assert result.norm_precision_auc > 0.9, "Perfect prediction should give high NP AUC"
        assert result.norm_precision_auc <= 1.0

    def test_random_boxes_range(self):
        rng = np.random.default_rng(99)
        preds = rng.uniform(0, 100, (50, 4))
        gts = rng.uniform(0, 100, (50, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1
        result = ENGINE.compute_all(preds, gts)
        assert 0.0 <= result.norm_precision_auc <= 1.0

    def test_high_np_auc_for_small_error(self):
        """Small positional error (< 0.1 box-size) → NP AUC should be high."""
        rng = np.random.default_rng(1)
        size = 60.0
        n = 40
        gts = np.zeros((n, 4), dtype=np.float64)
        gts[:, :2] = rng.uniform(0, 200, (n, 2))
        gts[:, 2:] = size
        # Shift predicted centre by 1% of box size
        noise = rng.uniform(-0.01 * size, 0.01 * size, (n, 2))
        preds = gts.copy()
        preds[:, :2] += noise
        result = ENGINE.compute_all(preds, gts)
        assert result.norm_precision_auc > 0.95, \
            f"Expected NP AUC > 0.95 for 1% error, got {result.norm_precision_auc:.4f}"


# ---------------------------------------------------------------------------
# BenchmarkResult surfaces norm_precision_auc
# ---------------------------------------------------------------------------

class TestBenchmarkResultNormPrecision:
    def _run_benchmark(self) -> BenchmarkResult:
        dataset = SyntheticDataset(num_sequences=2, num_frames=30, seed=0)
        engine = BenchmarkEngine(verbose=False)
        return engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")

    def test_summary_includes_norm_precision_auc(self):
        result = self._run_benchmark()
        summary = result.summary()
        assert "norm_precision_auc" in summary, \
            "summary() must include 'norm_precision_auc'"
        assert 0.0 <= summary["norm_precision_auc"] <= 1.0

    def test_mean_norm_precision_auc_property(self):
        result = self._run_benchmark()
        val = result.mean_norm_precision_auc
        assert val is not None
        assert 0.0 <= val <= 1.0

    def test_sequence_results_include_norm_precision_auc(self):
        result = self._run_benchmark()
        d = result.to_dict()
        for seq in d["sequences"]:
            assert "norm_precision_auc" in seq, \
                f"Sequence {seq['sequence_name']} missing 'norm_precision_auc'"
            assert 0.0 <= seq["norm_precision_auc"] <= 1.0

    def test_sequence_result_property(self):
        result = self._run_benchmark()
        for sr in result.sequence_results:
            val = sr.norm_precision_auc
            assert val is not None
            assert 0.0 <= val <= 1.0

    def test_round_trip_json(self):
        """BenchmarkResult.save() / load() preserves norm_precision_auc."""
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
        """from_dict() should handle old JSON without 'norm_precision_auc'."""
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
                    # no "norm_precision_auc" — old format
                }
            ],
        }
        result = BenchmarkResult.from_dict(old_dict)
        assert result.sequence_results[0].accuracy_metrics is not None
        assert result.sequence_results[0].accuracy_metrics.norm_precision_auc == pytest.approx(0.0)
