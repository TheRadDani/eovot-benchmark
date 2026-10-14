"""Tests for KalmanFilterTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.kalman import KalmanFilterTracker, _KalmanFilter
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def black_frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def init_bbox():
    return (100.0, 80.0, 60.0, 40.0)  # (x, y, w, h)


# ---------------------------------------------------------------------------
# _KalmanFilter unit tests
# ---------------------------------------------------------------------------

class TestKalmanFilterCore:
    def test_predict_no_noise(self):
        """State should evolve deterministically with zero noise."""
        F = np.array([[1, 1], [0, 1]], dtype=np.float64)
        H = np.eye(2, dtype=np.float64)
        Q = np.zeros((2, 2), dtype=np.float64)
        R = np.eye(2, dtype=np.float64) * 1e-10
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2, dtype=np.float64)

        kf = _KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

        state = kf.predict()
        assert state[0] == pytest.approx(1.0, abs=1e-9)  # position = 0 + 1*1
        assert state[1] == pytest.approx(1.0, abs=1e-9)  # velocity unchanged

    def test_update_converges_to_measurement(self):
        """With tiny measurement noise, update should snap state to measurement."""
        F = np.eye(2, dtype=np.float64)
        H = np.eye(2, dtype=np.float64)
        Q = np.zeros((2, 2), dtype=np.float64)
        R = np.eye(2, dtype=np.float64) * 1e-12
        x0 = np.zeros(2)
        P0 = np.eye(2, dtype=np.float64) * 1e6

        kf = _KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
        kf.predict()
        state = kf.update(np.array([5.0, 3.0]))
        assert state[0] == pytest.approx(5.0, abs=1e-4)
        assert state[1] == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# KalmanFilterTracker interface tests
# ---------------------------------------------------------------------------

class TestKalmanFilterTracker:
    def test_initialize_then_update_returns_bbox(self, black_frame, init_bbox):
        tracker = KalmanFilterTracker()
        tracker.initialize(black_frame, init_bbox)
        result = tracker.update(black_frame)
        assert len(result) == 4

    def test_dimensions_non_negative(self, black_frame, init_bbox):
        tracker = KalmanFilterTracker()
        tracker.initialize(black_frame, init_bbox)
        for _ in range(30):
            x, y, w, h = tracker.update(black_frame)
        assert w >= 1.0
        assert h >= 1.0

    def test_update_without_init_raises(self, black_frame):
        tracker = KalmanFilterTracker()
        with pytest.raises(RuntimeError):
            tracker.update(black_frame)

    def test_frame_content_ignored(self, init_bbox):
        """Two different frames should yield identical predictions."""
        frame_a = np.zeros((240, 320, 3), dtype=np.uint8)
        frame_b = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        tracker_a = KalmanFilterTracker()
        tracker_a.initialize(frame_a, init_bbox)
        pred_a = tracker_a.update(frame_a)

        tracker_b = KalmanFilterTracker()
        tracker_b.initialize(frame_b, init_bbox)
        pred_b = tracker_b.update(frame_b)

        np.testing.assert_array_almost_equal(pred_a, pred_b, decimal=10)

    def test_stationary_object_stays_put(self, black_frame, init_bbox):
        """With zero initial velocity, a stationary object should not drift."""
        tracker = KalmanFilterTracker(process_noise=0.0, measurement_noise=1e-8)
        tracker.initialize(black_frame, init_bbox)
        x0, y0, w0, h0 = init_bbox
        cx0, cy0 = x0 + w0 / 2, y0 + h0 / 2

        for _ in range(20):
            x, y, w, h = tracker.update(black_frame)

        cx, cy = x + w / 2, y + h / 2
        # With zero process noise and update feeding prediction back, the
        # filter should stay very close to the initial centre.
        assert abs(cx - cx0) < 1.0
        assert abs(cy - cy0) < 1.0

    def test_constant_velocity_propagation(self, black_frame):
        """Injecting velocity directly into initial state should propagate it.

        The Kalman filter models velocity in state dimensions 4-7.  By
        manually setting those after initialization we verify that the
        constant-velocity model propagates the motion forward correctly.
        """
        tracker = KalmanFilterTracker(process_noise=0.0, measurement_noise=1e-8)
        x, y, w, h = 50.0, 50.0, 40.0, 30.0
        tracker.initialize(black_frame, (x, y, w, h))

        # Inject a known velocity of 5 px/frame in x
        tracker._kf.x[4] = 5.0  # vcx = 5 px/frame

        steps = 5
        for _ in range(steps):
            pred = tracker.update(black_frame)

        # After `steps` frames the centre should have moved ~steps * 5 pixels
        expected_cx = (x + w / 2) + steps * 5.0
        pred_cx = pred[0] + pred[2] / 2
        assert abs(pred_cx - expected_cx) < 2.0

    def test_open_loop_no_update(self, black_frame, init_bbox):
        """update_with_prediction=False should still return valid bboxes."""
        tracker = KalmanFilterTracker(update_with_prediction=False)
        tracker.initialize(black_frame, init_bbox)
        for _ in range(10):
            bbox = tracker.update(black_frame)
            assert len(bbox) == 4
            _, _, w, h = bbox
            assert w >= 1.0
            assert h >= 1.0

    def test_repr(self):
        tracker = KalmanFilterTracker(name="KF-test", process_noise=0.01, measurement_noise=0.1)
        r = repr(tracker)
        assert "KF-test" in r
        assert "0.01" in r

    def test_name_attribute(self):
        tracker = KalmanFilterTracker(name="my-kalman")
        assert tracker.name == "my-kalman"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    def test_registered(self):
        assert "KalmanFilter" in TRACKER_REGISTRY

    def test_build_tracker(self):
        tracker = build_tracker("KalmanFilter")
        assert isinstance(tracker, KalmanFilterTracker)

    def test_build_with_params(self):
        tracker = build_tracker("KalmanFilter", process_noise=5e-3, measurement_noise=5e-2)
        assert tracker._q == pytest.approx(5e-3)
        assert tracker._r == pytest.approx(5e-2)


# ---------------------------------------------------------------------------
# Benchmark engine smoke test
# ---------------------------------------------------------------------------

class TestBenchmarkSmoke:
    def test_benchmark_run(self):
        """Ensure KalmanFilterTracker integrates end-to-end with BenchmarkEngine."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=2, num_frames=20)
        tracker = KalmanFilterTracker()
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")

        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0
        assert len(result.sequence_results) == 2

    def test_fps_exceeds_mosse(self):
        """Kalman filter must be faster than MOSSE — it does no image work."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=2, num_frames=30)
        engine = BenchmarkEngine(verbose=False)

        kalman_result = engine.run(KalmanFilterTracker(), dataset, dataset_name="Synthetic")
        dataset2 = SyntheticDataset(num_sequences=2, num_frames=30)
        mosse_result = engine.run(MOSSETracker(), dataset2, dataset_name="Synthetic")

        assert kalman_result.mean_fps > mosse_result.mean_fps, (
            f"Expected KalmanFilter FPS ({kalman_result.mean_fps:.0f}) > "
            f"MOSSE FPS ({mosse_result.mean_fps:.0f})"
        )
