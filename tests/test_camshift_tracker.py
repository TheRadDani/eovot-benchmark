"""Unit tests for CamShiftTracker."""

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker
from eovot.trackers.camshift import CamShiftTracker

INIT_BBOX = (40.0, 30.0, 40.0, 40.0)  # x, y, w, h


def _make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """Random BGR background with a saturated red square at INIT_BBOX.

    A flat-colour patch with high saturation gives CamShift's hue
    histogram something distinctive to lock onto, unlike pure random
    noise where every patch has an arbitrary, unstable hue.
    """
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    x, y, bw, bh = (int(v) for v in INIT_BBOX)
    frame[y:y + bh, x:x + bw] = (0, 0, 220)  # saturated red (BGR)
    return frame


def _make_sequence(n_frames: int = 5, h: int = 120, w: int = 160):
    """A static scene: the coloured target never moves."""
    frame = _make_frame(h, w)
    return [frame.copy() for _ in range(n_frames)]


class TestCamShiftTracker:
    def setup_method(self):
        self.tracker = CamShiftTracker()

    def test_is_base_tracker(self):
        assert isinstance(self.tracker, BaseTracker)

    def test_name(self):
        assert self.tracker.name == "CamShift"

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

    def test_tracks_static_colour_target(self):
        """On a static scene the window should stay near the target."""
        frames = _make_sequence(5)
        self.tracker.initialize(frames[0], INIT_BBOX)
        x0, y0, w0, h0 = INIT_BBOX
        for frame in frames[1:]:
            x, y, w, h = self.tracker.update(frame)
            assert abs((x + w / 2) - (x0 + w0 / 2)) < w0
            assert abs((y + h / 2) - (y0 + h0 / 2)) < h0

    def test_update_without_init_raises(self):
        tracker = CamShiftTracker()
        frame = _make_frame()
        with pytest.raises(RuntimeError):
            tracker.update(frame)

    def test_invalid_bbox_raises(self):
        tracker = CamShiftTracker()
        frame = _make_frame()
        with pytest.raises(ValueError):
            tracker.initialize(frame, (10.0, 10.0, 0.0, 0.0))

    def test_custom_params(self):
        tracker = CamShiftTracker(hist_bins=8, sat_min=30, val_min=20, max_iter=5)
        frames = _make_sequence(3)
        tracker.initialize(frames[0], INIT_BBOX)
        bbox = tracker.update(frames[1])
        assert len(bbox) == 4

    def test_falls_back_when_window_collapses(self):
        """A frame with no matching colour anywhere should not crash and
        should keep returning a valid, positive-area bounding box."""
        frame0 = _make_frame()
        self.tracker.initialize(frame0, INIT_BBOX)

        # Uniform grey frame: back-projection is ~0 everywhere, so the
        # CamShift window can collapse — tracker must fall back gracefully.
        blank = np.full_like(frame0, 128)
        x, y, w, h = self.tracker.update(blank)
        assert w > 0 and h > 0
