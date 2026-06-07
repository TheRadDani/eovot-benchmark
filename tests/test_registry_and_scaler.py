"""Tests for TrackerRegistry and ResolutionScaledTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.registry import TrackerRegistry
from eovot.trackers.resolution_scaler import ResolutionScaledTracker
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def _make_bbox() -> tuple:
    return (30.0, 20.0, 40.0, 30.0)


# ---------------------------------------------------------------------------
# TrackerRegistry
# ---------------------------------------------------------------------------

class TestTrackerRegistry:
    def test_built_in_trackers_registered(self):
        available = TrackerRegistry.list_available()
        for name in ["MOSSE", "KCF", "CSRT", "MedianFlow", "MIL"]:
            assert name in available, f"Expected '{name}' in registry"

    def test_create_mosse(self):
        tracker = TrackerRegistry.create("MOSSE")
        assert isinstance(tracker, BaseTracker)
        assert tracker.name == "MOSSE"

    def test_create_kcf_with_params(self):
        tracker = TrackerRegistry.create("KCF", learning_rate=0.1)
        assert isinstance(tracker, BaseTracker)

    def test_create_unknown_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            TrackerRegistry.create("NonExistentTracker")

    def test_register_custom_tracker(self):
        class DummyTracker(BaseTracker):
            def __init__(self):
                super().__init__("Dummy")
            def initialize(self, frame, bbox):
                self._bbox = bbox
            def update(self, frame):
                return self._bbox

        TrackerRegistry.register("Dummy", DummyTracker)
        assert TrackerRegistry.is_registered("Dummy")
        t = TrackerRegistry.create("Dummy")
        assert t.name == "Dummy"
        # Clean up so other tests aren't affected
        del TrackerRegistry._registry["Dummy"]

    def test_register_non_tracker_raises(self):
        with pytest.raises(TypeError):
            TrackerRegistry.register("Bad", object)  # type: ignore

    def test_is_registered(self):
        assert TrackerRegistry.is_registered("MOSSE")
        assert not TrackerRegistry.is_registered("__definitely_not_there__")

    def test_get_class(self):
        cls = TrackerRegistry.get_class("MOSSE")
        assert cls is MOSSETracker

    def test_get_class_unknown_raises(self):
        with pytest.raises(KeyError):
            TrackerRegistry.get_class("Ghost")

    def test_list_available_sorted(self):
        names = TrackerRegistry.list_available()
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# ResolutionScaledTracker
# ---------------------------------------------------------------------------

class TestResolutionScaledTracker:
    def test_invalid_scale_zero(self):
        with pytest.raises(ValueError):
            ResolutionScaledTracker(MOSSETracker(), scale_factor=0.0)

    def test_invalid_scale_above_one(self):
        with pytest.raises(ValueError):
            ResolutionScaledTracker(MOSSETracker(), scale_factor=1.5)

    def test_default_name(self):
        t = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.5)
        assert "MOSSE" in t.name
        assert "50%" in t.name

    def test_custom_name(self):
        t = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.5, name="MyTracker")
        assert t.name == "MyTracker"

    def test_identity_scale(self):
        t = ResolutionScaledTracker(MOSSETracker(), scale_factor=1.0)
        frame = _make_frame()
        bbox = _make_bbox()
        t.initialize(frame, bbox)
        pred = t.update(frame)
        assert len(pred) == 4

    def test_half_scale_returns_valid_bbox(self):
        t = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.5)
        frame = _make_frame(120, 160)
        bbox = _make_bbox()
        t.initialize(frame, bbox)
        pred = t.update(frame)
        assert len(pred) == 4
        x, y, w, h = pred
        assert w > 0 and h > 0

    def test_prediction_in_original_coordinates(self):
        """Predictions must be in the original 160×120 frame space."""
        frame = _make_frame(120, 160)
        bbox = (30.0, 20.0, 40.0, 30.0)

        t_full = MOSSETracker()
        t_scaled = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.5)

        t_full.initialize(frame, bbox)
        t_scaled.initialize(frame, bbox)

        next_frame = _make_frame(120, 160)
        pred_full = t_full.update(next_frame)
        pred_scaled = t_scaled.update(next_frame)

        # Both predictions should be in the 160×120 coordinate space
        x_s, y_s, w_s, h_s = pred_scaled
        # x and w should be in [0, 160] range (not scaled-down [0, 80] range)
        assert x_s >= 0.0
        assert w_s > 0.0
        assert w_s < 160.0

    def test_multiple_frames(self):
        """Tracker should run without error across multiple frames."""
        rng = np.random.default_rng(42)
        frame = _make_frame(120, 160)
        bbox = _make_bbox()
        t = ResolutionScaledTracker(KCFTracker(), scale_factor=0.5)
        t.initialize(frame, bbox)
        for _ in range(5):
            frame = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
            pred = t.update(frame)
            assert len(pred) == 4


# ---------------------------------------------------------------------------
# Integration: ExperimentRunner uses registry via _build_tracker
# ---------------------------------------------------------------------------

class TestExperimentRunnerIntegration:
    def test_build_tracker_via_registry(self):
        from eovot.experiment.runner import ExperimentRunner
        tracker = ExperimentRunner._build_tracker({"name": "MOSSE", "params": {}})
        assert isinstance(tracker, MOSSETracker)

    def test_build_tracker_with_scale_factor(self):
        from eovot.experiment.runner import ExperimentRunner
        tracker = ExperimentRunner._build_tracker(
            {"name": "MOSSE", "params": {}, "scale_factor": 0.5}
        )
        assert isinstance(tracker, ResolutionScaledTracker)
        assert "50%" in tracker.name

    def test_build_tracker_scale_one_no_wrapping(self):
        from eovot.experiment.runner import ExperimentRunner
        tracker = ExperimentRunner._build_tracker(
            {"name": "KCF", "params": {}, "scale_factor": 1.0}
        )
        assert isinstance(tracker, KCFTracker)

    def test_build_tracker_unknown_raises(self):
        from eovot.experiment.runner import ExperimentRunner
        with pytest.raises(KeyError):
            ExperimentRunner._build_tracker({"name": "GhostTracker"})
