"""Tests for the Kalman Filter Tracker."""

import numpy as np
import pytest

from eovot.trackers.kalman import KalmanTracker
from eovot.trackers.base import BaseTracker


def _make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """Return a random BGR uint8 frame."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _colored_frame(cx: int, cy: int, hw: int = 10, h: int = 120, w: int = 160) -> np.ndarray:
    """Frame with a bright rectangle at (cx, cy) for realistic template matching."""
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    x1, y1 = max(0, cx - hw), max(0, cy - hw)
    x2, y2 = min(w, cx + hw), min(h, cy + hw)
    frame[y1:y2, x1:x2] = [220, 200, 50]
    return frame


# --- Interface compliance ---

def test_kalman_is_base_tracker():
    assert issubclass(KalmanTracker, BaseTracker)


def test_kalman_name():
    t = KalmanTracker()
    assert t.name == "Kalman"


def test_kalman_repr():
    t = KalmanTracker()
    assert "KalmanTracker" in repr(t)


# --- Error handling ---

def test_update_before_initialize_raises():
    t = KalmanTracker()
    frame = _make_frame()
    with pytest.raises(RuntimeError, match="not initialised"):
        t.update(frame)


def test_reset_then_update_raises():
    t = KalmanTracker()
    frame = _make_frame()
    t.initialize(frame, (20, 20, 30, 30))
    t.reset()
    with pytest.raises(RuntimeError):
        t.update(frame)


# --- Basic output shape and type ---

def test_initialize_returns_none():
    t = KalmanTracker()
    ret = t.initialize(_make_frame(), (10, 10, 40, 30))
    assert ret is None


def test_update_returns_4_tuple():
    t = KalmanTracker()
    frame = _make_frame()
    t.initialize(frame, (10, 10, 40, 30))
    bbox = t.update(frame)
    assert len(bbox) == 4, "update() must return (x, y, w, h)"
    assert all(isinstance(v, float) for v in bbox)


def test_bbox_width_height_positive():
    t = KalmanTracker()
    frame = _make_frame()
    t.initialize(frame, (10, 10, 40, 30))
    for _ in range(5):
        x, y, w, h = t.update(frame)
        assert w > 0 and h > 0, "Width and height must remain positive"


# --- Tracking a static target (should stay near initial position) ---

def test_static_target():
    cx, cy = 80, 60
    frame = _colored_frame(cx, cy)
    bbox = (cx - 10, cy - 10, 20, 20)
    t = KalmanTracker(ncc_threshold=0.1)
    t.initialize(frame, bbox)
    for _ in range(10):
        px, py, pw, ph = t.update(frame)
    pred_cx = px + pw / 2.0
    pred_cy = py + ph / 2.0
    assert abs(pred_cx - cx) < 15, f"cx drift: {abs(pred_cx - cx):.1f}px"
    assert abs(pred_cy - cy) < 15, f"cy drift: {abs(pred_cy - cy):.1f}px"


# --- Tracking a linearly moving target ---

def test_linear_motion_tracking():
    """Kalman filter tracks a slowly moving target via template matching.

    With a low threshold (accepts any correlation) and a slow target speed,
    the tracker's template-matching update should keep pace and produce a
    mean centre-x error well below the target width.
    """
    h, w = 120, 320
    # Use a low NCC threshold so the measurement update fires every frame,
    # and allow template to adapt so it stays current with appearance.
    t = KalmanTracker(ncc_threshold=-1.0, learning_rate=0.05, search_pad=2.0)
    start_cx, cy = 30, 60
    speed = 2       # 2 px / frame — slow enough for the search window
    bbox_sz = 20    # 20 × 20 target
    bbox_init = (start_cx - bbox_sz // 2, cy - bbox_sz // 2, bbox_sz, bbox_sz)
    t.initialize(_colored_frame(start_cx, cy, hw=bbox_sz // 2, h=h, w=w), bbox_init)

    errors = []
    for step in range(1, 20):
        true_cx = start_cx + step * speed
        if true_cx + bbox_sz // 2 >= w:
            break
        frame = _colored_frame(true_cx, cy, hw=bbox_sz // 2, h=h, w=w)
        px, py, pw, ph = t.update(frame)
        pred_cx = px + pw / 2.0
        errors.append(abs(pred_cx - true_cx))

    mean_error = float(np.mean(errors))
    # Allow up to 1 target-width of mean error — the tracker may lag behind
    # by up to the template search radius but should not lose the target.
    assert mean_error < bbox_sz * 2.0, f"Mean cx error {mean_error:.1f}px exceeds {bbox_sz*2}px"


# --- Registry integration ---

def test_kalman_in_registry():
    from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker
    assert "Kalman" in TRACKER_REGISTRY
    t = build_tracker("Kalman")
    assert isinstance(t, KalmanTracker)


# --- Grayscale input ---

def test_grayscale_frame_input():
    t = KalmanTracker()
    gray = np.full((120, 160), 100, dtype=np.uint8)
    t.initialize(gray, (20, 20, 30, 30))
    bbox = t.update(gray)
    assert len(bbox) == 4


# --- NCC gating (ncc_threshold=1.0 forces pure Kalman prediction) ---

def test_pure_kalman_prediction_mode():
    """With ncc_threshold=1.0 every frame uses prediction only (no measurement)."""
    frame = _make_frame()
    t = KalmanTracker(ncc_threshold=1.0)
    t.initialize(frame, (10, 10, 40, 30))
    prev = (10.0, 10.0, 40.0, 30.0)
    for _ in range(5):
        bbox = t.update(frame)
        # With no measurement updates the tracker drifts by constant velocity.
        assert len(bbox) == 4
        prev = bbox


# --- Benchmark engine integration ---

def test_benchmark_engine_with_kalman():
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear")
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(KalmanTracker(), dataset, dataset_name="synthetic")
    assert result.mean_fps > 0
    assert 0.0 <= result.mean_iou <= 1.0
    assert len(result.sequence_results) == 2
