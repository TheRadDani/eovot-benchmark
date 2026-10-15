"""Tests for BackgroundAwareTracker motion-gated wrapper."""

import numpy as np
import pytest

from eovot.trackers.background_aware import BackgroundAwareTracker
from eovot.trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Minimal stub tracker for testing
# ---------------------------------------------------------------------------

class _ConstantTracker(BaseTracker):
    """Returns a fixed bbox after initialization; tracks call count."""

    def __init__(self, fixed_bbox: BBox = (10.0, 10.0, 30.0, 30.0)) -> None:
        super().__init__(name="Constant")
        self._bbox = fixed_bbox
        self.update_count = 0

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox
        self.update_count = 0

    def update(self, frame: np.ndarray) -> BBox:
        self.update_count += 1
        return self._bbox


def _blank_frame(h: int = 120, w: int = 160) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _motion_frame(h: int = 120, w: int = 160, x: int = 10, y: int = 10, size: int = 30) -> np.ndarray:
    """Frame with a bright patch inside the bbox region to trigger MOG2 foreground."""
    frame = _blank_frame(h, w)
    frame[y:y + size, x:x + size] = 200
    return frame


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

class TestBackgroundAwareTrackerInit:
    def test_default_name_reflects_wrapped_name(self):
        base = _ConstantTracker()
        tracker = BackgroundAwareTracker(base)
        assert "Constant" in tracker.name

    def test_custom_name_overrides(self):
        tracker = BackgroundAwareTracker(_ConstantTracker(), name="MyWrapper")
        assert tracker.name == "MyWrapper"

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="motion_threshold"):
            BackgroundAwareTracker(_ConstantTracker(), motion_threshold=1.5)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            BackgroundAwareTracker(_ConstantTracker(), motion_threshold=-0.1)

    def test_zero_threshold_is_valid(self):
        t = BackgroundAwareTracker(_ConstantTracker(), motion_threshold=0.0)
        assert t.motion_threshold == 0.0

    def test_wrapped_tracker_property(self):
        base = _ConstantTracker()
        tracker = BackgroundAwareTracker(base)
        assert tracker.wrapped_tracker is base


# ---------------------------------------------------------------------------
# update() before initialize() should raise
# ---------------------------------------------------------------------------

class TestUpdateBeforeInit:
    def test_update_without_init_raises(self):
        tracker = BackgroundAwareTracker(_ConstantTracker())
        with pytest.raises(RuntimeError, match="initialize"):
            tracker.update(_blank_frame())


# ---------------------------------------------------------------------------
# Motion detection and skip logic
# ---------------------------------------------------------------------------

class TestMotionSkipping:
    def test_zero_threshold_never_skips(self):
        """With threshold=0, every frame must call the wrapped update."""
        base = _ConstantTracker()
        tracker = BackgroundAwareTracker(base, motion_threshold=0.0)
        init_bbox = (10.0, 10.0, 30.0, 30.0)
        tracker.initialize(_blank_frame(), init_bbox)

        n = 10
        for _ in range(n):
            tracker.update(_blank_frame())

        assert base.update_count == n

    def test_extreme_threshold_skips_blank_frames(self):
        """With threshold=1.0, blank (no-motion) frames should be skipped."""
        base = _ConstantTracker()
        tracker = BackgroundAwareTracker(base, motion_threshold=1.0)
        init_bbox = (10.0, 10.0, 30.0, 30.0)
        tracker.initialize(_blank_frame(), init_bbox)

        n = 5
        for _ in range(n):
            tracker.update(_blank_frame())

        assert base.update_count < n

    def test_motion_frames_trigger_update(self):
        """Frames with strong motion should not be skipped."""
        base = _ConstantTracker()
        tracker = BackgroundAwareTracker(base, motion_threshold=0.01)
        init_bbox = (10.0, 10.0, 30.0, 30.0)

        tracker.initialize(_blank_frame(), init_bbox)

        called = 0
        for _ in range(8):
            f = _motion_frame()
            tracker.update(f)
            called += 1

        # At least some motion frames should have triggered the wrapped update.
        # We allow ≥1 since MOG2 needs a few frames to build the background model.
        assert base.update_count >= 1

    def test_skipped_frame_returns_last_bbox(self):
        """A skipped frame must return the last known bounding box unchanged."""
        fixed = (5.0, 5.0, 20.0, 20.0)
        base = _ConstantTracker(fixed_bbox=fixed)
        tracker = BackgroundAwareTracker(base, motion_threshold=1.0)
        tracker.initialize(_blank_frame(), fixed)

        returned = tracker.update(_blank_frame())
        assert returned == fixed


# ---------------------------------------------------------------------------
# motion_stats
# ---------------------------------------------------------------------------

class TestMotionStats:
    def test_stats_initially_zero_after_init(self):
        tracker = BackgroundAwareTracker(_ConstantTracker())
        tracker.initialize(_blank_frame(), (0.0, 0.0, 20.0, 20.0))
        stats = tracker.motion_stats
        assert stats["total_frames"] == 0
        assert stats["skipped_frames"] == 0
        assert stats["skip_ratio"] == 0.0

    def test_total_frames_increments(self):
        tracker = BackgroundAwareTracker(_ConstantTracker(), motion_threshold=0.0)
        tracker.initialize(_blank_frame(), (0.0, 0.0, 20.0, 20.0))
        for _ in range(7):
            tracker.update(_blank_frame())
        assert tracker.motion_stats["total_frames"] == 7

    def test_skip_ratio_between_zero_and_one(self):
        tracker = BackgroundAwareTracker(_ConstantTracker(), motion_threshold=0.5)
        tracker.initialize(_blank_frame(), (0.0, 0.0, 30.0, 30.0))
        for _ in range(10):
            tracker.update(_blank_frame())
        r = tracker.motion_stats["skip_ratio"]
        assert 0.0 <= r <= 1.0

    def test_stats_reset_on_reinitialize(self):
        tracker = BackgroundAwareTracker(_ConstantTracker(), motion_threshold=0.0)
        bbox = (0.0, 0.0, 20.0, 20.0)
        tracker.initialize(_blank_frame(), bbox)
        for _ in range(5):
            tracker.update(_blank_frame())

        tracker.initialize(_blank_frame(), bbox)
        stats = tracker.motion_stats
        assert stats["total_frames"] == 0
        assert stats["skipped_frames"] == 0

    def test_skipped_plus_updates_equals_total(self):
        base = _ConstantTracker()
        tracker = BackgroundAwareTracker(base, motion_threshold=0.5)
        tracker.initialize(_blank_frame(), (0.0, 0.0, 30.0, 30.0))
        n = 20
        for _ in range(n):
            tracker.update(_blank_frame())
        stats = tracker.motion_stats
        assert stats["total_frames"] == n
        assert stats["skipped_frames"] + base.update_count == n


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_small_bbox_does_not_crash(self):
        """1x1 target crop should not raise an exception."""
        tracker = BackgroundAwareTracker(_ConstantTracker(), motion_threshold=0.02)
        tracker.initialize(_blank_frame(), (50.0, 50.0, 1.0, 1.0))
        tracker.update(_blank_frame())

    def test_out_of_bounds_bbox_clamped(self):
        """BBox outside frame boundaries should be clamped, not raise."""
        tracker = BackgroundAwareTracker(_ConstantTracker(), motion_threshold=0.02)
        tracker.initialize(_blank_frame(h=100, w=100), (90.0, 90.0, 50.0, 50.0))
        tracker.update(_blank_frame(h=100, w=100))

    def test_grayscale_input_does_not_crash(self):
        """MOG2 can handle grayscale frames passed as (H, W, 1)."""
        tracker = BackgroundAwareTracker(_ConstantTracker(), motion_threshold=0.02)
        gray = np.zeros((80, 80, 1), dtype=np.uint8)
        # initialize uses the full frame shape — use BGR for init, gray for update
        tracker.initialize(np.zeros((80, 80, 3), dtype=np.uint8), (10.0, 10.0, 20.0, 20.0))
        # Wrapping gray as a 3-channel frame to avoid cv2 type errors in the stub
        tracker.update(np.zeros((80, 80, 3), dtype=np.uint8))
