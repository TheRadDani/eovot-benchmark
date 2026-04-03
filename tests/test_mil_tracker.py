"""Tests for the MILTracker wrapper.

Uses fully in-memory synthetic frames — no dataset download required.
Validates interface compliance, initialisation, update behaviour, and
fallback when update is called before initialise.
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker
from eovot.trackers.mil import MILTracker

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_H, FRAME_W = 240, 320
INIT_BBOX = (20.0, 15.0, 80.0, 60.0)  # x, y, w, h — target region


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker() -> MILTracker:
    return MILTracker()


@pytest.fixture
def green_frame() -> np.ndarray:
    """BGR frame with a bright green rectangle as the tracking target."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    x, y, w, h = (int(v) for v in INIT_BBOX)
    frame[y : y + h, x : x + w] = (0, 255, 0)
    return frame


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------

class TestMILTrackerInterface:
    def test_is_base_tracker(self):
        assert isinstance(MILTracker(), BaseTracker)

    def test_default_name(self):
        assert MILTracker().name == "MIL"

    def test_custom_name(self):
        t = MILTracker(name="MyMIL")
        assert t.name == "MyMIL"

    def test_repr_contains_name(self):
        r = repr(MILTracker())
        assert "MIL" in r

    def test_custom_feature_count_accepted(self):
        """MILTracker should accept feature_count >= 250 (OpenCV constraint)."""
        t = MILTracker(feature_count=300)
        assert t._feature_count == 300

    def test_feature_count_below_minimum_raises(self):
        """feature_count < 250 should raise ValueError (OpenCV internal limit)."""
        with pytest.raises(ValueError, match="feature_count"):
            MILTracker(feature_count=100)


# ---------------------------------------------------------------------------
# Behaviour tests
# ---------------------------------------------------------------------------

class TestMILTrackerBehaviour:
    def test_initialize_does_not_raise(self, tracker, green_frame):
        tracker.initialize(green_frame, INIT_BBOX)

    def test_update_returns_4_tuple(self, tracker, green_frame):
        tracker.initialize(green_frame, INIT_BBOX)
        pred = tracker.update(green_frame)
        assert len(pred) == 4, "update() must return a 4-element bbox"

    def test_update_returns_floats(self, tracker, green_frame):
        tracker.initialize(green_frame, INIT_BBOX)
        pred = tracker.update(green_frame)
        assert all(isinstance(v, float) for v in pred), (
            "bbox values should be floats"
        )

    def test_update_before_init_returns_fallback(self, tracker, green_frame):
        """Calling update() before initialize() must not raise."""
        pred = tracker.update(green_frame)
        assert len(pred) == 4

    def test_width_height_positive(self, tracker, green_frame):
        tracker.initialize(green_frame, INIT_BBOX)
        pred = tracker.update(green_frame)
        _x, _y, w, h = pred
        assert w > 0, "predicted width must be positive"
        assert h > 0, "predicted height must be positive"

    def test_multiple_updates_do_not_raise(self, tracker, green_frame):
        tracker.initialize(green_frame, INIT_BBOX)
        for _ in range(10):
            pred = tracker.update(green_frame)
            assert len(pred) == 4

    def test_stationary_target_centre_stays_close(self, tracker, green_frame):
        """On a static scene the predicted centre should stay near the init box."""
        tracker.initialize(green_frame, INIT_BBOX)
        for _ in range(5):
            pred = tracker.update(green_frame)

        px, py, pw, ph = pred
        ix, iy, iw, ih = INIT_BBOX
        cx_pred, cy_pred = px + pw / 2, py + ph / 2
        cx_init, cy_init = ix + iw / 2, iy + ih / 2

        # Allow up to 40% of the target width/height as centre drift
        assert abs(cx_pred - cx_init) < iw * 0.4, (
            f"centre-x drifted too far: {cx_pred:.1f} vs {cx_init:.1f}"
        )
        assert abs(cy_pred - cy_init) < ih * 0.4, (
            f"centre-y drifted too far: {cy_pred:.1f} vs {cy_init:.1f}"
        )

    def test_reinitialize_works(self, tracker, green_frame):
        """Tracker should be re-initialisable without errors."""
        tracker.initialize(green_frame, INIT_BBOX)
        tracker.update(green_frame)
        # Re-initialise on same frame with different bbox
        new_bbox = (50.0, 40.0, 60.0, 50.0)
        tracker.initialize(green_frame, new_bbox)
        pred = tracker.update(green_frame)
        assert len(pred) == 4


# ---------------------------------------------------------------------------
# Custom feature count tests
# ---------------------------------------------------------------------------

class TestMILTrackerFeatureCount:
    def test_custom_feature_count_is_used(self, green_frame):
        """feature_count >= 250 should initialise and track without errors."""
        tracker = MILTracker(feature_count=300)
        tracker.initialize(green_frame, INIT_BBOX)
        pred = tracker.update(green_frame)
        assert len(pred) == 4
