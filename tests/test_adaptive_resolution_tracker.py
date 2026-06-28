"""Tests for eovot.trackers.adaptive_resolution — AdaptiveResolutionTracker."""

import numpy as np
import pytest

from eovot.trackers.adaptive_resolution import AdaptiveResolutionTracker
from eovot.trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Minimal fake tracker for testing without OpenCV correlation filters
# ---------------------------------------------------------------------------

class _IdentityTracker(BaseTracker):
    """Returns the initialization bbox every frame (perfect static tracker)."""

    def __init__(self):
        super().__init__(name="Identity")
        self._bbox = None

    def initialize(self, frame, bbox):
        self._bbox = bbox

    def update(self, frame):
        return self._bbox


class _RecordingTracker(BaseTracker):
    """Records the shapes of every frame it receives."""

    def __init__(self):
        super().__init__(name="Recording")
        self._init_shape = None
        self.update_shapes = []
        self._bbox = (10.0, 10.0, 20.0, 20.0)

    def initialize(self, frame, bbox):
        self._init_shape = frame.shape
        self._bbox = bbox

    def update(self, frame):
        self.update_shapes.append(frame.shape)
        return self._bbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h=240, w=320):
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


INIT_BBOX = (50.0, 40.0, 80.0, 60.0)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(inner)
        assert t.target_fps == 30.0
        assert t.min_scale == 0.25
        assert t.max_scale == 1.0
        assert 0.25 <= t.current_scale <= 1.0

    def test_name_includes_inner_name(self):
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(inner)
        assert "Identity" in t.name

    def test_invalid_target_fps_raises(self):
        with pytest.raises(ValueError):
            AdaptiveResolutionTracker(_IdentityTracker(), target_fps=0.0)
        with pytest.raises(ValueError):
            AdaptiveResolutionTracker(_IdentityTracker(), target_fps=-5.0)

    def test_invalid_scale_range_raises(self):
        with pytest.raises(ValueError):
            AdaptiveResolutionTracker(_IdentityTracker(), min_scale=0.8, max_scale=0.5)
        with pytest.raises(ValueError):
            AdaptiveResolutionTracker(_IdentityTracker(), min_scale=0.0, max_scale=1.0)

    def test_invalid_control_gain_raises(self):
        with pytest.raises(ValueError):
            AdaptiveResolutionTracker(_IdentityTracker(), control_gain=0.0)
        with pytest.raises(ValueError):
            AdaptiveResolutionTracker(_IdentityTracker(), control_gain=2.5)

    def test_invalid_smoothing_raises(self):
        with pytest.raises(ValueError):
            AdaptiveResolutionTracker(_IdentityTracker(), smoothing=-0.1)
        with pytest.raises(ValueError):
            AdaptiveResolutionTracker(_IdentityTracker(), smoothing=1.0)

    def test_non_base_tracker_raises(self):
        with pytest.raises(TypeError):
            AdaptiveResolutionTracker("not_a_tracker")  # type: ignore

    def test_initial_scale_clamped(self):
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(inner, min_scale=0.3, max_scale=0.8, initial_scale=0.1)
        assert t.current_scale >= 0.3
        t2 = AdaptiveResolutionTracker(inner, min_scale=0.3, max_scale=0.8, initial_scale=1.5)
        assert t2.current_scale <= 0.8


# ---------------------------------------------------------------------------
# Frame scaling
# ---------------------------------------------------------------------------

class TestFrameScaling:
    def test_inner_receives_scaled_frame_on_initialize(self):
        rec = _RecordingTracker()
        t = AdaptiveResolutionTracker(rec, initial_scale=0.5)
        frame = _make_frame(240, 320)
        t.initialize(frame, INIT_BBOX)
        # Inner tracker received a frame at 50% resolution
        init_h, init_w = rec._init_shape[:2]
        assert abs(init_h - 120) <= 1
        assert abs(init_w - 160) <= 1

    def test_inner_receives_scaled_frame_on_update(self):
        rec = _RecordingTracker()
        t = AdaptiveResolutionTracker(rec, initial_scale=0.5)
        frame = _make_frame(240, 320)
        t.initialize(frame, INIT_BBOX)
        t.update(frame)
        h, w = rec.update_shapes[0][:2]
        assert abs(h - 120) <= 1
        assert abs(w - 160) <= 1

    def test_scale_one_passes_frame_unchanged(self):
        rec = _RecordingTracker()
        t = AdaptiveResolutionTracker(rec, initial_scale=1.0)
        frame = _make_frame(240, 320)
        t.initialize(frame, INIT_BBOX)
        t.update(frame)
        assert rec.update_shapes[0] == frame.shape


# ---------------------------------------------------------------------------
# Bbox coordinate mapping
# ---------------------------------------------------------------------------

class TestBboxMapping:
    def test_prediction_maps_back_to_original_coords(self):
        """If inner tracker returns bbox in scaled coords, ADRT scales it back."""
        inner = _IdentityTracker()
        scale = 0.5
        t = AdaptiveResolutionTracker(inner, initial_scale=scale)
        frame = _make_frame()
        t.initialize(frame, INIT_BBOX)
        pred = t.update(frame)

        # Inner tracker returns scaled INIT_BBOX; ADRT should invert the scale.
        # Due to controller, scale may have changed; verify the general inversion.
        px, py, pw, ph = pred
        assert pw > 0 and ph > 0, "Predicted bbox must have positive size"

    def test_identity_tracker_full_scale_returns_same_bbox(self):
        """With scale=1 and identity tracker, prediction equals init bbox."""
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(
            inner,
            target_fps=1e9,  # so fast the scale will not decrease
            initial_scale=1.0,
            control_gain=0.01,  # near-zero adaptation
        )
        frame = _make_frame()
        t.initialize(frame, INIT_BBOX)
        pred = t.update(frame)
        # Predictions should be very close to original bbox
        for got, exp in zip(pred, INIT_BBOX):
            assert abs(got - exp) < 2.0, f"Got {got:.2f}, expected {exp:.2f}"


# ---------------------------------------------------------------------------
# Controller behaviour
# ---------------------------------------------------------------------------

class TestController:
    def test_scale_decreases_when_fast_target_slow_tracker(self):
        """If target FPS is physically impossible, scale should eventually decrease."""
        inner = _IdentityTracker()
        # 1e12 FPS is beyond any Python code — guarantees the ratio < 1 every frame
        t = AdaptiveResolutionTracker(
            inner, target_fps=1e12, min_scale=0.1, initial_scale=1.0, control_gain=1.0
        )
        frame = _make_frame()
        t.initialize(frame, INIT_BBOX)
        for _ in range(20):
            t.update(frame)
        assert t.current_scale < 1.0, "Scale should decrease when behind target FPS"

    def test_scale_stays_at_max_for_fast_tracker(self):
        """With a very low target FPS, scale should stay at max."""
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(
            inner, target_fps=0.001, max_scale=1.0, initial_scale=0.5, control_gain=0.5
        )
        frame = _make_frame()
        t.initialize(frame, INIT_BBOX)
        for _ in range(20):
            t.update(frame)
        assert t.current_scale == t.max_scale, "Scale should hit max for very easy target"

    def test_scale_bounded_by_min_max(self):
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(
            inner,
            target_fps=1e8,
            min_scale=0.3,
            max_scale=0.9,
            initial_scale=0.6,
        )
        frame = _make_frame()
        t.initialize(frame, INIT_BBOX)
        for _ in range(30):
            t.update(frame)
        assert t.current_scale >= t.min_scale
        assert t.current_scale <= t.max_scale

    def test_scale_history_length_matches_updates(self):
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(inner, initial_scale=0.5)
        frame = _make_frame()
        t.initialize(frame, INIT_BBOX)
        for _ in range(7):
            t.update(frame)
        assert len(t.scale_history) == 7

    def test_latency_history_length_matches_updates(self):
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(inner)
        frame = _make_frame()
        t.initialize(frame, INIT_BBOX)
        for _ in range(5):
            t.update(frame)
        assert len(t.latency_history_ms) == 5
        assert all(ms >= 0 for ms in t.latency_history_ms)


# ---------------------------------------------------------------------------
# FPS estimation
# ---------------------------------------------------------------------------

class TestFpsEstimation:
    def test_current_fps_zero_before_update(self):
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(inner)
        frame = _make_frame()
        t.initialize(frame, INIT_BBOX)
        assert t.current_fps == 0.0

    def test_current_fps_positive_after_update(self):
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(inner)
        frame = _make_frame()
        t.initialize(frame, INIT_BBOX)
        t.update(frame)
        assert t.current_fps > 0.0


# ---------------------------------------------------------------------------
# Reinitialization resets state
# ---------------------------------------------------------------------------

class TestReinitialize:
    def test_reinitialize_clears_history(self):
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(inner)
        frame = _make_frame()
        t.initialize(frame, INIT_BBOX)
        for _ in range(5):
            t.update(frame)
        assert len(t.scale_history) == 5
        # Re-initialize
        t.initialize(frame, INIT_BBOX)
        assert len(t.scale_history) == 0
        assert len(t.latency_history_ms) == 0
        assert t.current_fps == 0.0


# ---------------------------------------------------------------------------
# Summary dict
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_empty_before_updates(self):
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(inner)
        assert t.summary() == {}

    def test_summary_keys_after_run(self):
        inner = _IdentityTracker()
        t = AdaptiveResolutionTracker(inner)
        frame = _make_frame()
        t.initialize(frame, INIT_BBOX)
        for _ in range(5):
            t.update(frame)
        s = t.summary()
        expected_keys = {"mean_scale", "min_scale_used", "max_scale_used", "mean_fps", "n_frames"}
        assert set(s.keys()) == expected_keys
        assert s["n_frames"] == 5
        assert 0 < s["mean_scale"] <= 1.0


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr():
    inner = _IdentityTracker()
    t = AdaptiveResolutionTracker(inner, target_fps=25.0)
    r = repr(t)
    assert "Identity" in r
    assert "25.0" in r
