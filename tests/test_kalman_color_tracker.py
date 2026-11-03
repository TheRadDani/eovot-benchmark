"""Tests for KalmanColorTracker."""
from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.kalman_color import KalmanColorTracker


def _red_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """BGR frame with a red rectangle at (60,40,40,40)."""
    frame = np.full((h, w, 3), 80, dtype=np.uint8)
    frame[40:80, 60:100] = (0, 0, 200)
    return frame


def _blue_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """BGR frame with a blue rectangle at (60,40,40,40)."""
    frame = np.full((h, w, 3), 80, dtype=np.uint8)
    frame[40:80, 60:100] = (200, 0, 0)
    return frame


# --- construction & interface ---

def test_default_name():
    assert KalmanColorTracker().name == "KalmanColor"


def test_custom_name():
    assert KalmanColorTracker(name="MyKF").name == "MyKF"


def test_initialize_does_not_raise():
    tracker = KalmanColorTracker()
    tracker.initialize(_red_frame(), (60.0, 40.0, 40.0, 40.0))


def test_update_returns_four_floats():
    tracker = KalmanColorTracker()
    frame = _red_frame()
    tracker.initialize(frame, (60.0, 40.0, 40.0, 40.0))
    result = tracker.update(frame)
    assert len(result) == 4
    assert all(isinstance(v, float) for v in result)


def test_update_bbox_size_positive():
    tracker = KalmanColorTracker()
    frame = _red_frame()
    tracker.initialize(frame, (60.0, 40.0, 40.0, 40.0))
    _, _, w, h = tracker.update(frame)
    assert w > 0.0
    assert h > 0.0


# --- multiple consecutive updates ---

def test_sequential_updates_no_crash():
    tracker = KalmanColorTracker()
    frame = _red_frame()
    tracker.initialize(frame, (60.0, 40.0, 40.0, 40.0))
    for _ in range(10):
        result = tracker.update(frame)
        assert len(result) == 4


# --- parameter variants ---

def test_strict_threshold_falls_back_to_kalman():
    """With match_threshold=0, colour search always fails → Kalman-only output."""
    tracker = KalmanColorTracker(match_threshold=0.0)
    frame = _red_frame()
    tracker.initialize(frame, (60.0, 40.0, 40.0, 40.0))
    result = tracker.update(frame)
    assert len(result) == 4


def test_high_n_bins():
    tracker = KalmanColorTracker(n_bins=32)
    frame = _red_frame()
    tracker.initialize(frame, (60.0, 40.0, 40.0, 40.0))
    result = tracker.update(frame)
    assert len(result) == 4


def test_large_search_factor():
    tracker = KalmanColorTracker(search_factor=4.0)
    frame = _red_frame()
    tracker.initialize(frame, (60.0, 40.0, 40.0, 40.0))
    result = tracker.update(frame)
    assert len(result) == 4


# --- edge cases ---

def test_tiny_bbox():
    tracker = KalmanColorTracker()
    frame = _red_frame()
    tracker.initialize(frame, (60.0, 40.0, 3.0, 3.0))
    result = tracker.update(frame)
    assert len(result) == 4


def test_bbox_near_frame_edge():
    tracker = KalmanColorTracker()
    frame = _red_frame()
    tracker.initialize(frame, (0.0, 0.0, 20.0, 20.0))
    result = tracker.update(frame)
    assert len(result) == 4


# --- registry integration ---

def test_in_tracker_registry():
    from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker
    assert "KalmanColor" in TRACKER_REGISTRY
    tracker = build_tracker("KalmanColor")
    assert isinstance(tracker, KalmanColorTracker)


def test_build_tracker_with_params():
    from eovot.trackers.registry import build_tracker
    tracker = build_tracker("KalmanColor", n_bins=8, search_factor=1.5)
    assert tracker.n_bins == 8
    assert tracker.search_factor == 1.5


# --- benchmark engine integration ---

def test_benchmark_engine_integration():
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    dataset = SyntheticDataset(num_sequences=2, num_frames=25, seed=42)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(KalmanColorTracker(), dataset, dataset_name="Synthetic")
    assert result.mean_fps > 0.0
    assert 0.0 <= result.mean_iou <= 1.0
    assert result.peak_memory_mb >= 0.0
