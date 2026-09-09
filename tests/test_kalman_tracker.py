"""Tests for KalmanBoxPredictor and AdaptiveComputeTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.kalman import KalmanBoxPredictor
from eovot.trackers.adaptive_compute import AdaptiveComputeTracker
from eovot.trackers.mosse import MOSSETracker


INIT_BBOX = (30.0, 40.0, 50.0, 60.0)


# -------------------------------------------------------------------------
# KalmanBoxPredictor tests
# -------------------------------------------------------------------------

class TestKalmanBoxPredictor:
    def test_raises_before_init(self):
        kf = KalmanBoxPredictor()
        with pytest.raises(RuntimeError):
            kf.predict()

    def test_correct_requires_predict_first(self):
        kf = KalmanBoxPredictor()
        kf.initialize(INIT_BBOX)
        # predict not called yet — correct should still work (predict was
        # called implicitly at initialize time? No — it should fail).
        # Our impl requires predict() before correct().
        with pytest.raises(RuntimeError):
            kf.correct((31.0, 41.0, 50.0, 60.0))

    def test_prediction_close_to_init_on_first_frame(self):
        kf = KalmanBoxPredictor()
        kf.initialize(INIT_BBOX)
        pred = kf.predict()
        # With zero initial velocity and small process noise the first
        # prediction should be very close to the initial bbox.
        assert abs(pred[0] - INIT_BBOX[0]) < 1.0
        assert abs(pred[1] - INIT_BBOX[1]) < 1.0

    def test_predict_returns_positive_dimensions(self):
        kf = KalmanBoxPredictor()
        kf.initialize(INIT_BBOX)
        for _ in range(20):
            pred = kf.predict()
            kf.correct(pred)   # feed back prediction as measurement
        x, y, w, h = pred
        assert w > 0 and h > 0

    def test_constant_velocity_tracking(self):
        """After several corrections the filter should track constant motion."""
        kf = KalmanBoxPredictor(process_noise=0.01, measurement_noise=0.5)
        kf.initialize(INIT_BBOX)
        vx, vy = 5.0, 3.0
        x, y, w, h = INIT_BBOX
        for _ in range(30):
            x += vx
            y += vy
            kf.predict()
            kf.correct((x, y, w, h))
        # After many frames the filter should be close to the true position
        pred = kf.predict()
        assert abs(pred[0] - (x + vx)) < 5.0
        assert abs(pred[1] - (y + vy)) < 5.0

    def test_innovation_norm_is_small_after_convergence(self):
        kf = KalmanBoxPredictor()
        kf.initialize(INIT_BBOX)
        x, y, w, h = INIT_BBOX
        for _ in range(40):
            kf.predict()
            kf.correct((x, y, w, h))
        assert kf.innovation_norm < 1.0

    def test_is_initialized_flag(self):
        kf = KalmanBoxPredictor()
        assert not kf.is_initialized
        kf.initialize(INIT_BBOX)
        assert kf.is_initialized

    def test_reset_clears_state(self):
        kf = KalmanBoxPredictor()
        kf.initialize(INIT_BBOX)
        kf.predict()
        kf.reset()
        assert not kf.is_initialized
        with pytest.raises(RuntimeError):
            kf.predict()


# -------------------------------------------------------------------------
# AdaptiveComputeTracker tests
# -------------------------------------------------------------------------

def _make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


class TestAdaptiveComputeTracker:
    def _build_tracker(self, budget_ms: float = 1000.0) -> AdaptiveComputeTracker:
        return AdaptiveComputeTracker(
            MOSSETracker(),
            latency_budget_ms=budget_ms,
            warmup_frames=5,
            window_size=10,
        )

    def test_raises_on_non_positive_budget(self):
        with pytest.raises(ValueError):
            AdaptiveComputeTracker(MOSSETracker(), latency_budget_ms=0.0)

    def test_name_has_adaptive_suffix(self):
        t = self._build_tracker()
        assert "adaptive" in t.name

    def test_runs_full_pipeline(self):
        tracker = self._build_tracker(budget_ms=1000.0)
        frame = _make_frame()
        tracker.initialize(frame, INIT_BBOX)
        for _ in range(20):
            bbox = tracker.update(frame)
            assert len(bbox) == 4
            assert bbox[2] > 0 and bbox[3] > 0

    def test_stats_account_for_all_frames(self):
        n_frames = 30
        tracker = self._build_tracker(budget_ms=1000.0)
        frame = _make_frame()
        tracker.initialize(frame, INIT_BBOX)
        for _ in range(n_frames):
            tracker.update(frame)
        stats = tracker.compute_stats()
        total = stats["full_frames"] + stats["kalman_frames"]
        assert total == n_frames

    def test_kalman_fallback_triggered_by_tight_budget(self):
        """With budget=0.001 ms every frame should trigger Kalman mode quickly."""
        tracker = AdaptiveComputeTracker(
            MOSSETracker(),
            latency_budget_ms=0.001,   # impossibly tight
            warmup_frames=3,
            window_size=5,
            safety_sigma=0.0,
        )
        frame = _make_frame()
        tracker.initialize(frame, INIT_BBOX)
        for _ in range(20):
            tracker.update(frame)
        stats = tracker.compute_stats()
        # After warmup, Kalman mode should have kicked in
        assert stats["kalman_frames"] > 0

    def test_kalman_ratio_between_zero_and_one(self):
        tracker = self._build_tracker()
        frame = _make_frame()
        tracker.initialize(frame, INIT_BBOX)
        for _ in range(25):
            tracker.update(frame)
        ratio = tracker.compute_stats()["kalman_ratio"]
        assert 0.0 <= ratio <= 1.0
