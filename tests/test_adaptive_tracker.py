"""Tests for AdaptiveResolutionTracker.

All tests use synthetic in-memory frames and a ConstantTracker to avoid
dependency on real datasets or trained models.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pytest

from eovot.trackers.adaptive import AdaptiveResolutionTracker
from eovot.trackers.base import BaseTracker

BBox = Tuple[float, float, float, float]

FRAME_H, FRAME_W = 240, 320
FIXED_BOX: BBox = (20.0, 20.0, 60.0, 60.0)


# ---------------------------------------------------------------------------
# Minimal fake tracker for testing
# ---------------------------------------------------------------------------

class ConstantTracker(BaseTracker):
    """Always returns a fixed bbox; useful for controlled unit testing."""

    def __init__(self, box: BBox = FIXED_BOX) -> None:
        self._box = box

    @property
    def name(self) -> str:
        return "ConstantTracker"

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        pass

    def update(self, frame: np.ndarray) -> BBox:
        # Returns a bbox in the *input frame's* coordinate space.
        # If the frame is scaled, we return a proportional box.
        h, w = frame.shape[:2]
        fx = w / FRAME_W
        fy = h / FRAME_H
        x, y, bw, bh = self._box
        return (x * fx, y * fy, bw * fx, bh * fy)


def make_frame() -> np.ndarray:
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

class TestAdaptiveTrackerInterface:
    def test_is_base_tracker_subclass(self):
        tracker = AdaptiveResolutionTracker(ConstantTracker())
        assert isinstance(tracker, BaseTracker)

    def test_name_wraps_base_name(self):
        base = ConstantTracker()
        tracker = AdaptiveResolutionTracker(base)
        assert "ConstantTracker" in tracker.name
        assert "Adaptive" in tracker.name

    def test_base_tracker_accessible(self):
        base = ConstantTracker()
        tracker = AdaptiveResolutionTracker(base)
        assert tracker.base_tracker is base

    def test_update_before_initialize_raises(self):
        tracker = AdaptiveResolutionTracker(ConstantTracker())
        with pytest.raises(RuntimeError, match="initialize"):
            tracker.update(make_frame())


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestAdaptiveTrackerValidation:
    def test_invalid_scale_order_raises(self):
        with pytest.raises(ValueError, match="scale"):
            AdaptiveResolutionTracker(
                ConstantTracker(),
                scale_low=0.5,   # low < high: invalid
                scale_high=0.8,
            )

    def test_scale_high_zero_raises(self):
        with pytest.raises(ValueError, match="scale"):
            AdaptiveResolutionTracker(
                ConstantTracker(),
                scale_high=0.0,
            )

    def test_invalid_threshold_order_raises(self):
        with pytest.raises(ValueError, match="[Tt]hreshold"):
            AdaptiveResolutionTracker(
                ConstantTracker(),
                conf_high_thresh=0.3,
                conf_low_thresh=0.7,  # low > high: invalid
            )

    def test_equal_thresholds_raise(self):
        with pytest.raises(ValueError, match="[Tt]hreshold"):
            AdaptiveResolutionTracker(
                ConstantTracker(),
                conf_high_thresh=0.5,
                conf_low_thresh=0.5,
            )

    def test_hysteresis_zero_raises(self):
        with pytest.raises(ValueError, match="hysteresis"):
            AdaptiveResolutionTracker(ConstantTracker(), hysteresis_frames=0)

    def test_history_window_zero_raises(self):
        with pytest.raises(ValueError, match="history_window"):
            AdaptiveResolutionTracker(ConstantTracker(), history_window=0)

    def test_valid_default_construction(self):
        tracker = AdaptiveResolutionTracker(ConstantTracker())
        assert tracker.current_scale == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scale adaptation logic
# ---------------------------------------------------------------------------

class TestScaleAdaptation:
    def _make_tracker(self, **kwargs) -> AdaptiveResolutionTracker:
        defaults = dict(
            scale_low=1.0,
            scale_medium=0.75,
            scale_high=0.5,
            conf_high_thresh=0.6,
            conf_low_thresh=0.3,
            hysteresis_frames=2,
            history_window=3,
        )
        defaults.update(kwargs)
        return AdaptiveResolutionTracker(ConstantTracker(FIXED_BOX), **defaults)

    def test_starts_at_scale_low(self):
        tracker = self._make_tracker()
        tracker.initialize(make_frame(), FIXED_BOX)
        assert tracker.current_scale == pytest.approx(1.0)

    def test_scale_converges_to_high_after_stable_tracking(self):
        """After enough high-confidence frames the scale should reach scale_high."""
        tracker = self._make_tracker(hysteresis_frames=2, history_window=2)
        tracker.initialize(make_frame(), FIXED_BOX)
        for _ in range(20):
            tracker.update(make_frame())
        # A constant tracker with the same box every frame => high consecutive IoU
        assert tracker.current_scale == pytest.approx(0.5)

    def test_scale_falls_back_under_sudden_drift(self):
        """A drastically shifted prediction should reduce confidence and scale."""
        drifted_box: BBox = (200.0, 150.0, 30.0, 30.0)  # far from FIXED_BOX
        drift_tracker = ConstantTracker(drifted_box)
        tracker = AdaptiveResolutionTracker(
            drift_tracker,
            scale_low=1.0, scale_medium=0.75, scale_high=0.5,
            conf_high_thresh=0.6, conf_low_thresh=0.3,
            hysteresis_frames=2, history_window=3,
        )
        tracker.initialize(make_frame(), FIXED_BOX)
        for _ in range(10):
            tracker.update(make_frame())
        # Drifted predictions have zero consecutive IoU -> scale should stay low
        assert tracker.current_scale == pytest.approx(1.0)

    def test_current_scale_is_one_of_configured_levels(self):
        tracker = self._make_tracker()
        tracker.initialize(make_frame(), FIXED_BOX)
        valid_scales = {1.0, 0.75, 0.5}
        for _ in range(30):
            tracker.update(make_frame())
            assert tracker.current_scale in valid_scales


# ---------------------------------------------------------------------------
# Coordinate remapping
# ---------------------------------------------------------------------------

class TestCoordinateRemapping:
    def test_prediction_in_original_space(self):
        """Returned bbox must be in the original (unscaled) frame coordinates."""
        tracker = AdaptiveResolutionTracker(
            ConstantTracker(FIXED_BOX),
            scale_high=0.5,
            conf_high_thresh=0.01,  # force scale_high immediately
            hysteresis_frames=1,
            history_window=1,
        )
        tracker.initialize(make_frame(), FIXED_BOX)
        # First update forces high scale immediately.
        pred = tracker.update(make_frame())
        x, y, w, h = pred
        # Bbox must fit within the original frame dimensions
        assert 0.0 <= x < FRAME_W
        assert 0.0 <= y < FRAME_H
        assert w >= 1.0
        assert h >= 1.0
        assert x + w <= FRAME_W
        assert y + h <= FRAME_H

    def test_bbox_clamped_within_frame(self):
        """Out-of-bounds predictions must be clamped to frame limits."""
        # Tracker returns a box that extends beyond the frame boundary
        out_of_bounds_box: BBox = (300.0, 220.0, 100.0, 100.0)  # extends OOB
        tracker = AdaptiveResolutionTracker(ConstantTracker(out_of_bounds_box))
        tracker.initialize(make_frame(), FIXED_BOX)
        pred = tracker.update(make_frame())
        x, y, w, h = pred
        assert x + w <= FRAME_W
        assert y + h <= FRAME_H


# ---------------------------------------------------------------------------
# Confidence history
# ---------------------------------------------------------------------------

class TestConfidenceHistory:
    def test_history_empty_before_any_update(self):
        tracker = AdaptiveResolutionTracker(ConstantTracker())
        tracker.initialize(make_frame(), FIXED_BOX)
        assert tracker.confidence_history == []

    def test_history_grows_after_updates(self):
        tracker = AdaptiveResolutionTracker(ConstantTracker(), history_window=5)
        tracker.initialize(make_frame(), FIXED_BOX)
        for i in range(4):
            tracker.update(make_frame())
            assert len(tracker.confidence_history) == i + 1

    def test_history_capped_at_window(self):
        window = 4
        tracker = AdaptiveResolutionTracker(ConstantTracker(), history_window=window)
        tracker.initialize(make_frame(), FIXED_BOX)
        for _ in range(20):
            tracker.update(make_frame())
        assert len(tracker.confidence_history) <= window

    def test_confidence_values_in_unit_interval(self):
        tracker = AdaptiveResolutionTracker(ConstantTracker())
        tracker.initialize(make_frame(), FIXED_BOX)
        for _ in range(15):
            tracker.update(make_frame())
        for c in tracker.confidence_history:
            assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_sequence_produces_correct_bbox_count(self):
        """update() called N times must produce N predictions."""
        tracker = AdaptiveResolutionTracker(ConstantTracker(FIXED_BOX))
        tracker.initialize(make_frame(), FIXED_BOX)
        preds = [tracker.update(make_frame()) for _ in range(20)]
        assert len(preds) == 20

    def test_predictions_are_tuples_of_four_floats(self):
        tracker = AdaptiveResolutionTracker(ConstantTracker(FIXED_BOX))
        tracker.initialize(make_frame(), FIXED_BOX)
        pred = tracker.update(make_frame())
        assert len(pred) == 4
        for v in pred:
            assert isinstance(v, float)

    def test_reinitialize_resets_scale_to_low(self):
        """A second initialize() must reset the adaptive state."""
        tracker = AdaptiveResolutionTracker(
            ConstantTracker(FIXED_BOX),
            scale_high=0.5, conf_high_thresh=0.01, hysteresis_frames=1,
        )
        tracker.initialize(make_frame(), FIXED_BOX)
        for _ in range(10):
            tracker.update(make_frame())
        # After re-init, scale should reset to scale_low.
        tracker.initialize(make_frame(), FIXED_BOX)
        assert tracker.current_scale == pytest.approx(1.0)
        assert tracker.confidence_history == []
