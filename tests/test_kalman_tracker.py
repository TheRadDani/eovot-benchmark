"""Tests for the KalmanFilterTracker.

Covers construction, initialisation, per-frame update, error handling,
benchmark engine integration, and registry availability.
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.kalman import KalmanFilterTracker
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dummy_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a random BGR uint8 frame."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestKalmanFilterTrackerConstruction:
    def test_default_name(self):
        assert KalmanFilterTracker().name == "KalmanFilter"

    def test_custom_params_stored(self):
        t = KalmanFilterTracker(
            process_noise=0.05,
            measurement_noise=5.0,
            search_window_scale=3.0,
            min_correlation=0.5,
        )
        assert t.process_noise == 0.05
        assert t.measurement_noise == 5.0
        assert t.search_window_scale == 3.0
        assert t.min_correlation == 0.5

    def test_repr(self):
        t = KalmanFilterTracker()
        assert "KalmanFilterTracker" in repr(t)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestKalmanInitialize:
    def test_does_not_raise(self):
        t = KalmanFilterTracker()
        t.initialize(_dummy_frame(), (100.0, 80.0, 40.0, 40.0))

    def test_state_set(self):
        t = KalmanFilterTracker()
        t.initialize(_dummy_frame(), (100.0, 80.0, 40.0, 40.0))
        assert t._state is not None
        assert t._template is not None
        assert t._target_size == (40, 40)

    def test_grayscale_frame(self):
        rng = np.random.default_rng(1)
        gray = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        t = KalmanFilterTracker()
        t.initialize(gray, (50.0, 60.0, 30.0, 30.0))

    def test_tiny_bbox_clamps_to_min_2(self):
        t = KalmanFilterTracker()
        t.initialize(_dummy_frame(), (10.0, 10.0, 1.0, 1.0))
        assert t._target_size[0] >= 2
        assert t._target_size[1] >= 2


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

class TestKalmanUpdate:
    def test_returns_four_element_tuple(self):
        t = KalmanFilterTracker()
        frame = _dummy_frame()
        t.initialize(frame, (100.0, 80.0, 40.0, 40.0))
        bbox = t.update(frame)
        assert len(bbox) == 4

    def test_size_unchanged(self):
        t = KalmanFilterTracker()
        frame = _dummy_frame()
        t.initialize(frame, (100.0, 80.0, 40.0, 40.0))
        _, _, w, h = t.update(frame)
        assert w == 40.0
        assert h == 40.0

    def test_multiple_updates_do_not_raise(self):
        t = KalmanFilterTracker()
        frame = _dummy_frame()
        t.initialize(frame, (100.0, 80.0, 40.0, 40.0))
        for _ in range(10):
            t.update(frame)

    def test_update_before_initialize_raises(self):
        t = KalmanFilterTracker()
        with pytest.raises(RuntimeError, match="not initialised"):
            t.update(_dummy_frame())


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestKalmanReset:
    def test_reset_clears_state(self):
        t = KalmanFilterTracker()
        frame = _dummy_frame()
        t.initialize(frame, (100.0, 80.0, 40.0, 40.0))
        t.reset()
        assert t._state is None
        assert t._template is None
        assert t._target_size is None

    def test_update_after_reset_raises(self):
        t = KalmanFilterTracker()
        frame = _dummy_frame()
        t.initialize(frame, (100.0, 80.0, 40.0, 40.0))
        t.reset()
        with pytest.raises(RuntimeError):
            t.update(frame)

    def test_reinitialize_after_reset(self):
        t = KalmanFilterTracker()
        frame = _dummy_frame()
        t.initialize(frame, (100.0, 80.0, 40.0, 40.0))
        t.reset()
        t.initialize(frame, (50.0, 50.0, 30.0, 30.0))
        bbox = t.update(frame)
        assert len(bbox) == 4


# ---------------------------------------------------------------------------
# Benchmark engine integration
# ---------------------------------------------------------------------------

class TestKalmanOnBenchmarkEngine:
    def test_runs_without_error(self):
        dataset = SyntheticDataset(
            num_sequences=2, num_frames=20, motion="linear", seed=7
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(
            KalmanFilterTracker(), dataset, dataset_name="Syn-Kalman"
        )
        assert len(result.sequence_results) == 2

    def test_fps_positive(self):
        dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=7)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KalmanFilterTracker(), dataset, dataset_name="Syn")
        assert result.mean_fps > 0.0

    def test_positive_mean_iou_on_linear(self):
        """On a simple linear sequence the Kalman tracker should achieve IoU > 0."""
        dataset = SyntheticDataset(
            num_sequences=3, num_frames=40,
            motion="linear", seed=8,
            frame_size=(320, 240), bbox_size=(40, 40),
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KalmanFilterTracker(), dataset, dataset_name="Syn")
        assert result.mean_iou > 0.0

    def test_temporal_metrics_populated(self):
        """BenchmarkEngine should compute temporal consistency for Kalman too."""
        dataset = SyntheticDataset(num_sequences=2, num_frames=25, seed=10)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KalmanFilterTracker(), dataset, dataset_name="Syn")
        assert result.mean_smoothness_score is not None
        for seq_res in result.sequence_results:
            assert seq_res.temporal is not None

    def test_on_scale_sequences(self):
        dataset = SyntheticDataset(
            num_sequences=2, num_frames=30, motion="scale", seed=11
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KalmanFilterTracker(), dataset, dataset_name="Syn-Scale")
        assert result.mean_fps > 0.0

    def test_on_occlusion_sequences(self):
        dataset = SyntheticDataset(
            num_sequences=2, num_frames=40, motion="occlusion", seed=12
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KalmanFilterTracker(), dataset, dataset_name="Syn-Occ")
        assert result.mean_fps > 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestKalmanRegistry:
    def test_in_tracker_registry(self):
        assert "KalmanFilter" in TRACKER_REGISTRY

    def test_build_tracker_returns_instance(self):
        tracker = build_tracker("KalmanFilter")
        assert isinstance(tracker, KalmanFilterTracker)

    def test_build_tracker_forwards_params(self):
        tracker = build_tracker("KalmanFilter", process_noise=0.05)
        assert tracker.process_noise == 0.05

    def test_available_trackers_lists_kalman(self):
        from eovot.trackers.registry import available_trackers
        assert "KalmanFilter" in available_trackers()
