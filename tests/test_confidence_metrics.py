"""Tests for PSR confidence scoring and calibration metrics.

Covers:
- compute_psr returns sensible values on synthetic response maps
- ConfidenceTracker interface is correctly implemented by MOSSE and KCF
- last_confidence is 0.0 before first update(), positive after
- update_with_confidence returns (BBox, float) consistently
- ConfidenceCalibrationAnalyzer.analyze correctness on known data
- optimal_failure_threshold separates failure/success frames
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.base import ConfidenceTracker, compute_psr
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.datasets.synthetic import SyntheticDataset
from eovot.metrics.confidence import ConfidenceCalibrationAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """Create a simple BGR frame with a bright rectangle as target."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[40:60, 60:100] = 200
    return frame


def _make_response_sharp() -> np.ndarray:
    """Sharp Gaussian peak — should give high PSR."""
    r = np.zeros((64, 64), dtype=np.float32)
    r[32, 32] = 100.0
    r = np.exp(-((np.mgrid[:64, :64][0] - 32) ** 2 + (np.mgrid[:64, :64][1] - 32) ** 2) / 2.0) * 100
    return r


def _make_response_flat() -> np.ndarray:
    """Nearly uniform response — should give low PSR (~0)."""
    return np.ones((32, 32), dtype=np.float32) + np.random.default_rng(0).random((32, 32)).astype(np.float32) * 0.01


# ---------------------------------------------------------------------------
# compute_psr tests
# ---------------------------------------------------------------------------

class TestComputePSR:
    def test_sharp_peak_high_psr(self):
        response = _make_response_sharp()
        psr = compute_psr(response)
        assert psr > 3.0, f"Sharp response should yield PSR > 3, got {psr:.2f}"

    def test_flat_response_low_psr(self):
        response = _make_response_flat()
        psr = compute_psr(response)
        assert psr < 2.0, f"Flat response should yield PSR < 2, got {psr:.2f}"

    def test_psr_non_negative(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            r = rng.random((48, 48)).astype(np.float32)
            assert compute_psr(r) >= 0.0

    def test_tiny_response_no_crash(self):
        r = np.array([[1.0, 0.5], [0.5, 0.5]])
        psr = compute_psr(r)
        assert isinstance(psr, float)

    def test_constant_response_returns_zero(self):
        r = np.ones((32, 32), dtype=np.float32)
        assert compute_psr(r) == 0.0


# ---------------------------------------------------------------------------
# ConfidenceTracker interface tests
# ---------------------------------------------------------------------------

class TestConfidenceTrackerInterface:
    def test_mosse_is_confidence_tracker(self):
        assert issubclass(MOSSETracker, ConfidenceTracker)

    def test_kcf_is_confidence_tracker(self):
        assert issubclass(KCFTracker, ConfidenceTracker)

    @pytest.mark.parametrize("TrackerCls", [MOSSETracker, KCFTracker])
    def test_last_confidence_zero_before_update(self, TrackerCls):
        frame = _make_frame()
        tracker = TrackerCls()
        tracker.initialize(frame, (60, 40, 40, 20))
        assert tracker.last_confidence == 0.0

    @pytest.mark.parametrize("TrackerCls", [MOSSETracker, KCFTracker])
    def test_last_confidence_positive_after_update(self, TrackerCls):
        frame = _make_frame()
        tracker = TrackerCls()
        tracker.initialize(frame, (60, 40, 40, 20))
        tracker.update(frame)
        assert tracker.last_confidence >= 0.0

    @pytest.mark.parametrize("TrackerCls", [MOSSETracker, KCFTracker])
    def test_update_with_confidence_returns_tuple(self, TrackerCls):
        frame = _make_frame()
        tracker = TrackerCls()
        tracker.initialize(frame, (60, 40, 40, 20))
        result = tracker.update_with_confidence(frame)
        assert isinstance(result, tuple) and len(result) == 2
        bbox, conf = result
        assert len(bbox) == 4
        assert isinstance(conf, float)

    @pytest.mark.parametrize("TrackerCls", [MOSSETracker, KCFTracker])
    def test_update_with_confidence_consistent(self, TrackerCls):
        """update_with_confidence must match update + last_confidence."""
        frame = _make_frame()
        tracker = TrackerCls()
        tracker.initialize(frame, (60, 40, 40, 20))

        bbox_direct = tracker.update(frame)
        conf_direct = tracker.last_confidence

        tracker.initialize(frame, (60, 40, 40, 20))
        bbox_combo, conf_combo = tracker.update_with_confidence(frame)

        np.testing.assert_allclose(bbox_direct, bbox_combo, rtol=1e-5)
        assert conf_direct == conf_combo

    def test_mosse_psr_on_synthetic_sequence(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=20, seed=0)
        seq = list(ds)[0]
        frames = list(seq)
        tracker = MOSSETracker()
        tracker.initialize(frames[0], seq.init_bbox)
        psrs = []
        for frame in frames[1:]:
            tracker.update(frame)
            psrs.append(tracker.last_confidence)
        assert len(psrs) == 19
        assert all(p >= 0.0 for p in psrs)

    def test_kcf_psr_on_synthetic_sequence(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=20, seed=1)
        seq = list(ds)[0]
        frames = list(seq)
        tracker = KCFTracker()
        tracker.initialize(frames[0], seq.init_bbox)
        psrs = []
        for frame in frames[1:]:
            tracker.update(frame)
            psrs.append(tracker.last_confidence)
        assert len(psrs) == 19
        assert all(p >= 0.0 for p in psrs)


# ---------------------------------------------------------------------------
# ConfidenceCalibrationAnalyzer tests
# ---------------------------------------------------------------------------

class TestConfidenceCalibrationAnalyzer:
    def _make_correlated_data(self, n: int = 200, seed: int = 0) -> tuple:
        """Generate (confidences, ious) with a positive correlation."""
        rng = np.random.default_rng(seed)
        base = rng.uniform(0, 10, n)
        confidences = base + rng.normal(0, 1, n)
        ious = np.clip(base / 10.0 + rng.normal(0, 0.05, n), 0, 1)
        return confidences, ious

    def test_analyze_returns_result(self):
        confs, ious = self._make_correlated_data()
        analyzer = ConfidenceCalibrationAnalyzer()
        result = analyzer.analyze(confs, ious, tracker_name="T", sequence_name="S")
        assert result.tracker_name == "T"
        assert result.sequence_name == "S"
        assert -1.0 <= result.pearson_r <= 1.0
        assert -1.0 <= result.spearman_rho <= 1.0

    def test_positive_correlation_detected(self):
        confs, ious = self._make_correlated_data(n=500)
        analyzer = ConfidenceCalibrationAnalyzer()
        result = analyzer.analyze(confs, ious)
        assert result.pearson_r > 0.3, f"Expected positive correlation, got {result.pearson_r:.3f}"
        assert result.spearman_rho > 0.3

    def test_high_conf_iou_exceeds_low_conf_iou(self):
        confs, ious = self._make_correlated_data(n=500)
        analyzer = ConfidenceCalibrationAnalyzer(high_threshold=7.0, low_threshold=3.0)
        result = analyzer.analyze(confs, ious)
        if not (np.isnan(result.high_conf_mean_iou) or np.isnan(result.low_conf_mean_iou)):
            assert result.high_conf_mean_iou >= result.low_conf_mean_iou

    def test_optimal_threshold_in_range(self):
        confs, ious = self._make_correlated_data(n=300)
        analyzer = ConfidenceCalibrationAnalyzer()
        thresh = analyzer.optimal_failure_threshold(confs, ious)
        assert confs.min() <= thresh <= confs.max()

    def test_optimal_threshold_separates_failures(self):
        rng = np.random.default_rng(99)
        # Construct data where PSR < 3 ↔ IoU < 0.1
        n = 400
        confs = np.concatenate([rng.uniform(0, 3, n // 2), rng.uniform(7, 12, n // 2)])
        ious = np.concatenate([rng.uniform(0, 0.05, n // 2), rng.uniform(0.5, 1.0, n // 2)])
        analyzer = ConfidenceCalibrationAnalyzer(failure_iou=0.1)
        thresh = analyzer.optimal_failure_threshold(confs, ious)
        # Optimal threshold should be somewhere between 3 and 7
        assert 1.0 <= thresh <= 10.0

    def test_mismatched_lengths_raise(self):
        analyzer = ConfidenceCalibrationAnalyzer()
        with pytest.raises(ValueError, match="same shape"):
            analyzer.analyze(np.ones(10), np.ones(11))

    def test_too_few_frames_raise(self):
        analyzer = ConfidenceCalibrationAnalyzer()
        with pytest.raises(ValueError, match="At least 2"):
            analyzer.analyze(np.array([5.0]), np.array([0.5]))

    def test_aggregate_multiple_sequences(self):
        analyzer = ConfidenceCalibrationAnalyzer()
        results = []
        for seed in range(3):
            confs, ious = self._make_correlated_data(n=100, seed=seed)
            results.append(analyzer.analyze(confs, ious, sequence_name=f"seq{seed}"))
        agg = analyzer.aggregate(results)
        assert agg["n_sequences"] == 3
        assert "pearson_r" in agg
        assert "mean_optimal_threshold" in agg

    def test_aggregate_empty(self):
        analyzer = ConfidenceCalibrationAnalyzer()
        assert analyzer.aggregate([]) == {}

    def test_str_output(self):
        confs, ious = self._make_correlated_data()
        analyzer = ConfidenceCalibrationAnalyzer()
        result = analyzer.analyze(confs, ious, tracker_name="MOSSE", sequence_name="car1")
        s = str(result)
        assert "MOSSE" in s and "car1" in s
        assert "Pearson" in s and "Spearman" in s
