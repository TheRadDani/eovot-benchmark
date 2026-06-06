"""Tests for PSR-based confidence estimation (trackers/confidence.py)
and confidence-aware robustness analysis (metrics/robustness.py).
"""

import numpy as np
import pytest

from eovot.trackers.confidence import compute_psr, psr_to_confidence
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.metrics.robustness import (
    ConfidenceAwareRobustnessAnalyzer,
    ConfidenceRobustnessResult,
)


# ---------------------------------------------------------------------------
# compute_psr
# ---------------------------------------------------------------------------

class TestComputePSR:
    def test_sharp_peak_gives_high_psr(self):
        response = np.zeros((64, 64), dtype=np.float32)
        response[32, 32] = 100.0
        psr = compute_psr(response)
        assert psr > 7.0, f"Expected PSR > 7 for sharp peak, got {psr:.2f}"

    def test_flat_response_gives_zero_psr(self):
        response = np.ones((32, 32), dtype=np.float32)
        psr = compute_psr(response)
        assert psr == 0.0

    def test_returns_nonnegative(self):
        rng = np.random.default_rng(42)
        response = rng.random((40, 40)).astype(np.float32)
        assert compute_psr(response) >= 0.0

    def test_raises_on_1d_input(self):
        with pytest.raises(ValueError):
            compute_psr(np.ones(10))

    def test_small_response_map(self):
        response = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        # exclusion_size=11 > map size → sidelobe empty → 0.0
        psr = compute_psr(response, exclusion_size=11)
        assert psr == 0.0


# ---------------------------------------------------------------------------
# psr_to_confidence
# ---------------------------------------------------------------------------

class TestPSRToConfidence:
    def test_below_low_gives_zero(self):
        assert psr_to_confidence(0.0) == 0.0
        assert psr_to_confidence(2.9) == 0.0

    def test_above_high_gives_one(self):
        assert psr_to_confidence(7.0) == pytest.approx(1.0)
        assert psr_to_confidence(10.0) == pytest.approx(1.0)

    def test_midpoint(self):
        mid = psr_to_confidence(5.0)  # halfway between 3 and 7
        assert abs(mid - 0.5) < 1e-6

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            psr_to_confidence(5.0, low=7.0, high=3.0)


# ---------------------------------------------------------------------------
# MOSSETracker.update_with_confidence
# ---------------------------------------------------------------------------

def _make_frame(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


class TestMOSSEConfidence:
    def test_returns_bbox_and_confidence(self):
        tracker = MOSSETracker()
        frame0 = _make_frame(seed=1)
        tracker.initialize(frame0, (10, 10, 20, 20))

        frame1 = _make_frame(seed=2)
        bbox, conf = tracker.update_with_confidence(frame1)

        assert len(bbox) == 4
        assert 0.0 <= conf <= 1.0

    def test_confidence_type_is_float(self):
        tracker = MOSSETracker()
        frame0 = _make_frame(seed=3)
        tracker.initialize(frame0, (5, 5, 15, 15))
        frame1 = _make_frame(seed=4)
        _, conf = tracker.update_with_confidence(frame1)
        assert isinstance(conf, float)

    def test_update_and_update_with_confidence_agree_on_bbox(self):
        """Both entry points should return the same bbox for the same frame."""
        rng = np.random.default_rng(99)
        frame0 = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        frame1 = (rng.random((64, 64, 3)) * 255).astype(np.uint8)

        t1 = MOSSETracker()
        t1.initialize(frame0, (10, 10, 20, 20))
        bbox_plain = t1.update(frame1)

        t2 = MOSSETracker()
        t2.initialize(frame0, (10, 10, 20, 20))
        bbox_conf, _ = t2.update_with_confidence(frame1)

        np.testing.assert_allclose(bbox_plain, bbox_conf)


# ---------------------------------------------------------------------------
# KCFTracker.update_with_confidence
# ---------------------------------------------------------------------------

class TestKCFConfidence:
    def test_returns_bbox_and_confidence(self):
        tracker = KCFTracker()
        frame0 = _make_frame(seed=5)
        tracker.initialize(frame0, (8, 8, 24, 24))

        frame1 = _make_frame(seed=6)
        bbox, conf = tracker.update_with_confidence(frame1)

        assert len(bbox) == 4
        assert 0.0 <= conf <= 1.0

    def test_reset_clears_psr(self):
        tracker = KCFTracker()
        frame0 = _make_frame(seed=7)
        tracker.initialize(frame0, (8, 8, 24, 24))
        tracker.update(_make_frame(seed=8))
        assert tracker._last_psr >= 0.0
        tracker.reset()
        assert tracker._last_psr == 0.0


# ---------------------------------------------------------------------------
# ConfidenceAwareRobustnessAnalyzer
# ---------------------------------------------------------------------------

class TestConfidenceAwareRobustnessAnalyzer:
    def _make_ious_and_conf(self, n=50, failure_at=20, conf_drop_at=17):
        ious = np.ones(n, dtype=np.float64)
        ious[failure_at:failure_at + 5] = 0.05  # failure window
        conf = np.ones(n - 1, dtype=np.float64)
        conf[conf_drop_at - 1] = 0.1  # confidence drops before failure
        return ious, conf

    def test_returns_correct_type(self):
        analyzer = ConfidenceAwareRobustnessAnalyzer()
        ious, conf = self._make_ious_and_conf()
        result = analyzer.analyze(ious, conf, tracker_name="MOSSE", sequence_name="seq1")
        assert isinstance(result, ConfidenceRobustnessResult)

    def test_detects_iou_failure(self):
        analyzer = ConfidenceAwareRobustnessAnalyzer()
        ious, conf = self._make_ious_and_conf()
        result = analyzer.analyze(ious, conf)
        assert result.base.num_failures >= 1

    def test_detects_early_warning(self):
        analyzer = ConfidenceAwareRobustnessAnalyzer(
            confidence_threshold=0.3, early_warning_lead=5
        )
        ious, conf = self._make_ious_and_conf(
            n=60, failure_at=22, conf_drop_at=19  # conf drops 3 frames before failure
        )
        result = analyzer.analyze(ious, conf)
        assert len(result.early_warnings) >= 1

    def test_no_early_warning_when_conf_drops_after_failure(self):
        analyzer = ConfidenceAwareRobustnessAnalyzer(
            confidence_threshold=0.3, early_warning_lead=3
        )
        n = 50
        ious = np.ones(n)
        ious[20:25] = 0.05       # IoU failure at 20
        conf = np.ones(n - 1)
        conf[24] = 0.1            # confidence drops *after* failure
        result = analyzer.analyze(ious, conf)
        assert len(result.early_warnings) == 0

    def test_confidence_iou_correlation_perfect(self):
        analyzer = ConfidenceAwareRobustnessAnalyzer()
        n = 30
        ious = np.linspace(0.2, 1.0, n)
        conf = np.linspace(0.2, 1.0, n - 1)  # perfectly correlated
        r = analyzer.confidence_iou_correlation(ious, conf)
        assert r > 0.99

    def test_mean_confidence_in_range(self):
        analyzer = ConfidenceAwareRobustnessAnalyzer()
        ious = np.ones(30)
        conf = np.full(29, 0.6)
        result = analyzer.analyze(ious, conf)
        assert 0.0 <= result.mean_confidence <= 1.0

    def test_all_low_confidence_gives_high_failure_rate(self):
        analyzer = ConfidenceAwareRobustnessAnalyzer(confidence_threshold=0.5)
        ious = np.ones(30)
        conf = np.full(29, 0.1)
        result = analyzer.analyze(ious, conf)
        assert result.confidence_failure_rate > 0.8
