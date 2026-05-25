"""Tests for TemplatePatchTracker and the unified TRACKER_REGISTRY."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers import TRACKER_REGISTRY, TemplatePatchTracker
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_frame(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _gray_frame(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# TRACKER_REGISTRY
# ---------------------------------------------------------------------------

class TestTrackerRegistry:
    def test_all_expected_keys_present(self):
        expected = {"MOSSE", "KCF", "CSRT", "MedianFlow", "MIL", "TemplateMatch"}
        assert expected == set(TRACKER_REGISTRY)

    def test_all_values_are_base_tracker_subclasses(self):
        for name, cls in TRACKER_REGISTRY.items():
            assert issubclass(cls, BaseTracker), (
                f"TRACKER_REGISTRY['{name}'] is not a BaseTracker subclass"
            )

    def test_template_match_in_registry(self):
        assert "TemplateMatch" in TRACKER_REGISTRY
        assert TRACKER_REGISTRY["TemplateMatch"] is TemplatePatchTracker

    def test_mil_in_registry(self):
        from eovot.trackers import MILTracker
        assert "MIL" in TRACKER_REGISTRY
        assert TRACKER_REGISTRY["MIL"] is MILTracker

    def test_instantiation_from_registry(self):
        for name, cls in TRACKER_REGISTRY.items():
            try:
                tracker = cls()
            except ImportError:
                # Optional contrib trackers (CSRT, MIL) may not be available
                # in every OpenCV build — that is expected and not a bug.
                pytest.skip(f"{name} requires opencv-contrib-python")
                continue
            assert isinstance(tracker, BaseTracker)


# ---------------------------------------------------------------------------
# TemplatePatchTracker — construction
# ---------------------------------------------------------------------------

class TestTemplatePatchTrackerInit:
    def test_default_name(self):
        t = TemplatePatchTracker()
        assert t.name == "TemplateMatch"

    def test_custom_name(self):
        t = TemplatePatchTracker(name="MyTemplate")
        assert t.name == "MyTemplate"

    def test_invalid_search_factor_raises(self):
        with pytest.raises(ValueError, match="search_factor"):
            TemplatePatchTracker(search_factor=0.5)

    def test_invalid_update_rate_low_raises(self):
        with pytest.raises(ValueError, match="update_rate"):
            TemplatePatchTracker(update_rate=-0.1)

    def test_invalid_update_rate_high_raises(self):
        with pytest.raises(ValueError, match="update_rate"):
            TemplatePatchTracker(update_rate=1.5)

    def test_valid_boundary_update_rates(self):
        TemplatePatchTracker(update_rate=0.0)
        TemplatePatchTracker(update_rate=1.0)

    def test_repr_contains_class_and_name(self):
        t = TemplatePatchTracker()
        r = repr(t)
        assert "TemplatePatchTracker" in r
        assert "TemplateMatch" in r


# ---------------------------------------------------------------------------
# TemplatePatchTracker — initialize
# ---------------------------------------------------------------------------

class TestTemplatePatchTrackerInitialize:
    def test_initialize_does_not_raise_bgr(self):
        t = TemplatePatchTracker()
        frame = _bgr_frame()
        t.initialize(frame, (20, 15, 40, 30))

    def test_initialize_does_not_raise_gray(self):
        t = TemplatePatchTracker()
        frame = _gray_frame()
        t.initialize(frame, (20, 15, 40, 30))

    def test_initialize_clips_small_bbox(self):
        # Very small bbox — should not raise
        t = TemplatePatchTracker()
        t.initialize(_bgr_frame(), (0, 0, 1, 1))

    def test_initialize_at_boundary(self):
        t = TemplatePatchTracker()
        frame = _bgr_frame(h=80, w=80)
        t.initialize(frame, (70, 70, 20, 20))  # partly outside frame

    def test_template_stored_after_init(self):
        t = TemplatePatchTracker()
        t.initialize(_bgr_frame(), (20, 20, 30, 30))
        assert t._template is not None
        assert t._template.ndim == 2  # grayscale


# ---------------------------------------------------------------------------
# TemplatePatchTracker — update
# ---------------------------------------------------------------------------

class TestTemplatePatchTrackerUpdate:
    def _track(
        self,
        n_frames: int = 10,
        h: int = 120,
        w: int = 160,
        bbox=(20, 15, 40, 30),
        update_rate: float = 0.06,
    ):
        tracker = TemplatePatchTracker(update_rate=update_rate)
        rng = np.random.default_rng(42)
        frames = [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n_frames)]
        tracker.initialize(frames[0], bbox)
        preds = [bbox]
        for frame in frames[1:]:
            pred = tracker.update(frame)
            preds.append(pred)
        return preds

    def test_update_returns_4_tuple(self):
        preds = self._track()
        for p in preds:
            assert len(p) == 4

    def test_update_returns_positive_width_height(self):
        preds = self._track()
        for p in preds:
            _, _, w, h = p
            assert w > 0
            assert h > 0

    def test_update_without_initialize_returns_fallback(self):
        t = TemplatePatchTracker()
        result = t.update(_bgr_frame())
        assert len(result) == 4

    def test_update_returns_numeric_values(self):
        preds = self._track()
        for p in preds:
            assert all(isinstance(v, (int, float, np.floating)) for v in p)

    def test_static_template_bounded_drift(self):
        # With update_rate=0 the template is frozen, so position should stay
        # within ±1 pixel of the settled location across many frames.
        tracker = TemplatePatchTracker(update_rate=0.0)
        frame = _bgr_frame()
        tracker.initialize(frame, (20, 20, 30, 30))
        tracker.update(frame)  # warm-up frame
        ref = tracker.update(frame)
        for _ in range(10):
            p = tracker.update(frame)
            assert abs(p[0] - ref[0]) <= 1.0
            assert abs(p[1] - ref[1]) <= 1.0

    def test_grayscale_input(self):
        tracker = TemplatePatchTracker()
        gray = _gray_frame()
        tracker.initialize(gray, (10, 10, 20, 20))
        pred = tracker.update(gray)
        assert len(pred) == 4

    def test_multi_scale_does_not_raise(self):
        tracker = TemplatePatchTracker(scale_factors=(0.8, 1.0, 1.2))
        frame = _bgr_frame()
        tracker.initialize(frame, (30, 30, 40, 40))
        for _ in range(5):
            pred = tracker.update(frame)
            assert len(pred) == 4

    def test_large_search_factor(self):
        tracker = TemplatePatchTracker(search_factor=5.0)
        frame = _bgr_frame()
        tracker.initialize(frame, (30, 30, 20, 20))
        pred = tracker.update(frame)
        assert len(pred) == 4

    def test_single_scale(self):
        tracker = TemplatePatchTracker(scale_factors=(1.0,))
        frame = _bgr_frame()
        tracker.initialize(frame, (40, 40, 30, 30))
        pred = tracker.update(frame)
        assert len(pred) == 4

    def test_output_within_reasonable_range(self):
        # On a static frame the tracker should stay close to the init bbox
        tracker = TemplatePatchTracker(update_rate=0.0)
        h, w = 120, 160
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw a distinct rectangle so the template has signal
        frame[40:70, 50:90] = [200, 100, 50]
        init_bbox = (50.0, 40.0, 40.0, 30.0)
        tracker.initialize(frame, init_bbox)
        px, py, pw, ph = tracker.update(frame)
        # Centre should stay within the frame
        assert 0 <= px < w
        assert 0 <= py < h
        assert pw > 0
        assert ph > 0
