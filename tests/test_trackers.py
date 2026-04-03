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
