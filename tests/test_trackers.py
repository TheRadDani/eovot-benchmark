"""Unit tests for MOSSE and KCF trackers."""

import numpy as np
import pytest

from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.base import BaseTracker


def _make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """Create a random BGR uint8 frame."""
    rng = np.random.default_rng(0)
    return (rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def _make_sequence(n_frames: int = 10, h: int = 120, w: int = 160):
    """Return a list of identical frames (static scene)."""
    frame = _make_frame(h, w)
    return [frame.copy() for _ in range(n_frames)]


INIT_BBOX = (20.0, 20.0, 40.0, 40.0)  # x, y, w, h — well inside the 120×160 frame


class TestBaseTrackerInterface:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseTracker()  # type: ignore[abstract]

    def test_mosse_is_base_tracker(self):
        assert isinstance(MOSSETracker(), BaseTracker)

    def test_kcf_is_base_tracker(self):
        assert isinstance(KCFTracker(), BaseTracker)


class TestMOSSETracker:
    def setup_method(self):
        self.tracker = MOSSETracker(learning_rate=0.125, sigma=2.0)

    def test_name(self):
        assert self.tracker.name == "MOSSE"

    def test_initialize_does_not_raise(self):
        frame = _make_frame()
        self.tracker.initialize(frame, INIT_BBOX)

    def test_update_returns_bbox(self):
        frames = _make_sequence(5)
        self.tracker.initialize(frames[0], INIT_BBOX)
        for frame in frames[1:]:
            bbox = self.tracker.update(frame)
            assert len(bbox) == 4, "update() must return a 4-tuple (x, y, w, h)"
            x, y, w, h = bbox
            assert w > 0 and h > 0, "Width and height must be positive"

    def test_update_without_init_raises(self):
        tracker = MOSSETracker()
        frame = _make_frame()
        with pytest.raises(RuntimeError):
            tracker.update(frame)

    def test_reinitialize(self):
        frames = _make_sequence(4)
        self.tracker.initialize(frames[0], INIT_BBOX)
        self.tracker.update(frames[1])
        # Re-initialize on a different frame — should not raise
        self.tracker.initialize(frames[2], INIT_BBOX)
        bbox = self.tracker.update(frames[3])
        assert len(bbox) == 4

    def test_bbox_within_frame_bounds(self):
        """Tracker should not produce boxes entirely outside the frame."""
        h, w = 120, 160
        frames = _make_sequence(6, h=h, w=w)
        self.tracker.initialize(frames[0], INIT_BBOX)
        for frame in frames[1:]:
            x, y, bw, bh = self.tracker.update(frame)
            # Bounding box top-left should be within a reasonable range
            assert x > -w and y > -h, "Box drifted far outside frame"


class TestKCFTracker:
    def setup_method(self):
        self.tracker = KCFTracker()

    def test_name(self):
        assert self.tracker.name == "KCF"

    def test_initialize_does_not_raise(self):
        frame = _make_frame()
        self.tracker.initialize(frame, INIT_BBOX)

    def test_update_returns_bbox(self):
        frames = _make_sequence(5)
        self.tracker.initialize(frames[0], INIT_BBOX)
        for frame in frames[1:]:
            bbox = self.tracker.update(frame)
            assert len(bbox) == 4
            x, y, w, h = bbox
            assert w > 0 and h > 0

    def test_update_without_init_raises(self):
        tracker = KCFTracker()
        frame = _make_frame()
        with pytest.raises(RuntimeError):
            tracker.update(frame)

    def test_custom_params(self):
        tracker = KCFTracker(learning_rate=0.075, kernel_sigma=0.5, lambda_=1e-3)
        frames = _make_sequence(3)
        tracker.initialize(frames[0], INIT_BBOX)
        bbox = tracker.update(frames[1])
        assert len(bbox) == 4


class TestMOSSEOOBRobustness:
    """Regression tests for issue #171 — tracker crash on out-of-frame bbox."""

    def test_update_does_not_crash_when_target_fully_off_right(self):
        """Target drifted completely off the right edge must not raise."""
        tracker = MOSSETracker()
        frame = _make_frame(h=60, w=80)
        # Initialize with a valid bbox
        tracker.initialize(frame, (10.0, 10.0, 20.0, 20.0))
        # Manually force the bbox far off the right edge
        tracker._bbox = [200, 10, 20, 20]
        # update() must not raise; it should return a bbox tuple
        bbox = tracker.update(frame)
        assert len(bbox) == 4

    def test_update_does_not_crash_when_target_fully_off_left(self):
        """Target drifted completely off the left edge must not raise."""
        tracker = MOSSETracker()
        frame = _make_frame(h=60, w=80)
        tracker.initialize(frame, (10.0, 10.0, 20.0, 20.0))
        tracker._bbox = [-50, 10, 20, 20]
        bbox = tracker.update(frame)
        assert len(bbox) == 4

    def test_update_does_not_crash_when_target_fully_off_top(self):
        """Target drifted completely above the frame must not raise."""
        tracker = MOSSETracker()
        frame = _make_frame(h=60, w=80)
        tracker.initialize(frame, (10.0, 10.0, 20.0, 20.0))
        tracker._bbox = [10, -50, 20, 20]
        bbox = tracker.update(frame)
        assert len(bbox) == 4

    def test_update_does_not_crash_when_target_fully_off_bottom(self):
        """Target drifted completely below the frame must not raise."""
        tracker = MOSSETracker()
        frame = _make_frame(h=60, w=80)
        tracker.initialize(frame, (10.0, 10.0, 20.0, 20.0))
        tracker._bbox = [10, 200, 20, 20]
        bbox = tracker.update(frame)
        assert len(bbox) == 4

    def test_extract_patch_oob_returns_correct_shape(self):
        """_extract_patch must always return shape (h, w) even when OOB."""
        import cv2
        tracker = MOSSETracker()
        gray = np.zeros((60, 80), dtype=np.uint8)
        # Fully OOB positions
        for x, y in [(-50, 0), (200, 0), (0, -50), (0, 200), (-50, -50)]:
            patch = tracker._extract_patch(gray, x, y, 20, 20)
            assert patch.shape == (20, 20), (
                f"OOB patch at ({x},{y}) returned shape {patch.shape}, expected (20, 20)"
            )

    def test_tracker_recovers_after_drift_back_into_frame(self):
        """Tracker resumes normal updates after the target returns in frame."""
        tracker = MOSSETracker()
        frame = _make_frame(h=60, w=80)
        tracker.initialize(frame, (10.0, 10.0, 20.0, 20.0))
        # Drive the tracker OOB
        tracker._bbox = [200, 10, 20, 20]
        tracker.update(frame)
        # Now put it back in frame — should not raise
        tracker._bbox = [10, 10, 20, 20]
        bbox = tracker.update(frame)
        assert len(bbox) == 4
