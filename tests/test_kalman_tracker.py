"""Unit and integration tests for the KalmanFilterTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker
from eovot.trackers.kalman import KalmanFilterTracker
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_frame(h: int = 120, w: int = 160) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


INIT_BBOX = (30.0, 25.0, 50.0, 40.0)  # (x, y, w, h) well inside frame


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class TestKalmanFilterTrackerInterface:
    def test_is_base_tracker(self):
        assert isinstance(KalmanFilterTracker(), BaseTracker)

    def test_default_name(self):
        assert KalmanFilterTracker().name == "KalmanFilter"

    def test_registered_in_registry(self):
        assert "KalmanFilter" in TRACKER_REGISTRY

    def test_build_tracker(self):
        tracker = build_tracker("KalmanFilter")
        assert isinstance(tracker, KalmanFilterTracker)

    def test_build_tracker_with_params(self):
        tracker = build_tracker("KalmanFilter", process_noise=2.0, measurement_noise=5.0)
        assert tracker._q == 2.0
        assert tracker._r == 5.0


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestKalmanFilterTrackerInit:
    def test_initialize_does_not_raise(self):
        tracker = KalmanFilterTracker()
        tracker.initialize(_blank_frame(), INIT_BBOX)

    def test_state_after_init(self):
        tracker = KalmanFilterTracker()
        tracker.initialize(_blank_frame(), INIT_BBOX)
        x, y, w, h = INIT_BBOX
        expected_cx = x + w / 2.0
        expected_cy = y + h / 2.0
        assert tracker._x is not None
        assert abs(tracker._x[0] - expected_cx) < 1e-9
        assert abs(tracker._x[1] - expected_cy) < 1e-9
        # Initial velocity should be zero
        assert abs(tracker._x[2]) < 1e-9
        assert abs(tracker._x[3]) < 1e-9

    def test_scale_stored_correctly(self):
        tracker = KalmanFilterTracker()
        tracker.initialize(_blank_frame(), INIT_BBOX)
        _, _, w, h = INIT_BBOX
        assert abs(tracker._w - w) < 1e-9
        assert abs(tracker._h - h) < 1e-9

    def test_update_before_init_raises(self):
        tracker = KalmanFilterTracker()
        with pytest.raises(RuntimeError):
            tracker.update(_blank_frame())


# ---------------------------------------------------------------------------
# Update / prediction
# ---------------------------------------------------------------------------

class TestKalmanFilterTrackerUpdate:
    def test_update_returns_four_tuple(self):
        tracker = KalmanFilterTracker()
        frames = [_blank_frame() for _ in range(5)]
        tracker.initialize(frames[0], INIT_BBOX)
        for frame in frames[1:]:
            bbox = tracker.update(frame)
            assert len(bbox) == 4

    def test_width_height_preserved(self):
        """Pure-predict tracker keeps w/h constant (no appearance update)."""
        tracker = KalmanFilterTracker()
        frames = [_blank_frame() for _ in range(10)]
        tracker.initialize(frames[0], INIT_BBOX)
        _, _, init_w, init_h = INIT_BBOX
        for frame in frames[1:]:
            _, _, w, h = tracker.update(frame)
            assert abs(w - init_w) < 1e-6
            assert abs(h - init_h) < 1e-6

    def test_static_target_stays_near_initial_center(self):
        """With zero initial velocity the filter predicts the same centre."""
        tracker = KalmanFilterTracker(process_noise=0.0)  # no drift
        frames = [_blank_frame() for _ in range(5)]
        tracker.initialize(frames[0], INIT_BBOX)
        x0, y0, w0, h0 = INIT_BBOX
        cx0 = x0 + w0 / 2.0
        cy0 = y0 + h0 / 2.0
        for frame in frames[1:]:
            bx, by, bw, bh = tracker.update(frame)
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            assert abs(cx - cx0) < 1e-6, f"cx drifted: {cx} vs {cx0}"
            assert abs(cy - cy0) < 1e-6, f"cy drifted: {cy} vs {cy0}"

    def test_linear_motion_predicted_correctly(self):
        """A tracker initialised with velocity via manual state injection."""
        tracker = KalmanFilterTracker(process_noise=0.0)
        frames = [_blank_frame() for _ in range(5)]
        tracker.initialize(frames[0], INIT_BBOX)
        # Inject constant velocity: 5 px/frame rightward
        tracker._x[2] = 5.0
        tracker._x[3] = 0.0
        x0, y0, w0, h0 = INIT_BBOX
        cx0 = x0 + w0 / 2.0
        for i, frame in enumerate(frames[1:], start=1):
            bx, _, bw, _ = tracker.update(frame)
            cx_pred = bx + bw / 2.0
            expected_cx = cx0 + i * 5.0
            assert abs(cx_pred - expected_cx) < 1e-4, f"frame {i}: {cx_pred} vs {expected_cx}"

    def test_frame_content_irrelevant(self):
        """Kalman prediction is purely kinematic; frame content must not matter."""
        tracker1 = KalmanFilterTracker()
        tracker2 = KalmanFilterTracker()
        frame_zeros = np.zeros((120, 160, 3), dtype=np.uint8)
        frame_ones = np.full((120, 160, 3), 255, dtype=np.uint8)
        tracker1.initialize(frame_zeros, INIT_BBOX)
        tracker2.initialize(frame_zeros, INIT_BBOX)
        b1 = tracker1.update(frame_zeros)
        b2 = tracker2.update(frame_ones)
        assert b1 == b2, "Kalman output must be identical regardless of frame content"


# ---------------------------------------------------------------------------
# update_with_measurement
# ---------------------------------------------------------------------------

class TestKalmanFilterUpdateWithMeasurement:
    def test_correction_pulls_toward_measurement(self):
        tracker = KalmanFilterTracker(process_noise=1.0, measurement_noise=1.0)
        tracker.initialize(_blank_frame(), INIT_BBOX)
        x0, y0, w0, h0 = INIT_BBOX
        cx0 = x0 + w0 / 2.0
        cy0 = y0 + h0 / 2.0
        # Apply a measurement 20 px to the right
        bx, _, bw, _ = tracker.update_with_measurement(cx0 + 20.0, cy0)
        cx_corrected = bx + bw / 2.0
        # Corrected centre should be between initial and measurement
        assert cx_corrected > cx0
        assert cx_corrected < cx0 + 20.0

    def test_raises_before_init(self):
        tracker = KalmanFilterTracker()
        with pytest.raises(RuntimeError):
            tracker.update_with_measurement(50.0, 50.0)


# ---------------------------------------------------------------------------
# Benchmark pipeline integration
# ---------------------------------------------------------------------------

class TestKalmanBenchmarkIntegration:
    def test_benchmark_engine_runs_kalman(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear", seed=7)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KalmanFilterTracker(), dataset, dataset_name="Synthetic")

        assert result.tracker_name == "KalmanFilter"
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_high_fps_on_synthetic(self):
        """KalmanFilter should run faster than 1 000 FPS (it is pure NumPy)."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=1, num_frames=50, motion="linear", seed=99)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KalmanFilterTracker(), dataset, dataset_name="Synthetic")
        assert result.mean_fps > 1_000.0, f"Expected >1000 FPS, got {result.mean_fps:.1f}"
