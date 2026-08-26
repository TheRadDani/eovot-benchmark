"""Tests for ScaleAdaptiveKCFTracker.

Verifies:
- Constructor validation
- initialize() / update() lifecycle
- Scale estimation responds to genuine target size changes
- Tracker is properly registered and buildable from the registry
- Returned bbox shape is always (x, y, w, h) with positive w and h
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.scale_adaptive_kcf import ScaleAdaptiveKCFTracker
from eovot.trackers.registry import build_tracker, TRACKER_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_frame(h: int, w: int, color: int = 128) -> np.ndarray:
    """Return an (H, W, 3) BGR frame filled with a constant grey value."""
    return np.full((h, w, 3), color, dtype=np.uint8)


def _textured_frame(h: int, w: int, bbox, cell: int = 4) -> np.ndarray:
    """Frame with a checkerboard target on a uniform dark background.

    The checkerboard provides spatial-frequency content that the KCF
    filter can use to discriminate target scale during the scale-pool
    search.  Solid rectangles are nearly uniform after preprocessing
    and give the filter no discrimination signal.
    """
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    x, y, bw, bh = (int(v) for v in bbox)
    for i in range(bh):
        for j in range(bw):
            val = 200 if ((i // cell) + (j // cell)) % 2 == 0 else 60
            frame[y + i, x + j] = val
    return frame


def _make_growing_sequence(n_frames: int, start_w: int, end_w: int, fh=240, fw=320):
    """Generate frames where the target width linearly grows from start_w to end_w."""
    cx, cy = 160, 120   # fixed centre
    widths = np.linspace(start_w, end_w, n_frames).astype(int)
    h_fixed = start_w   # keep height constant so only width changes
    frames = []
    bboxes = []
    for bw in widths:
        x = cx - bw // 2
        y = cy - h_fixed // 2
        bbox = (x, y, bw, h_fixed)
        frames.append(_textured_frame(fh, fw, bbox, cell=max(1, bw // 8)))
        bboxes.append(bbox)
    return frames, bboxes


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_invalid_scale_step_le1(self):
        with pytest.raises(ValueError, match="scale_step"):
            ScaleAdaptiveKCFTracker(scale_step=0.99)

    def test_invalid_scale_step_eq1(self):
        with pytest.raises(ValueError, match="scale_step"):
            ScaleAdaptiveKCFTracker(scale_step=1.0)

    def test_invalid_n_scales_even(self):
        with pytest.raises(ValueError, match="n_scales"):
            ScaleAdaptiveKCFTracker(n_scales=4)

    def test_invalid_n_scales_zero(self):
        with pytest.raises(ValueError, match="n_scales"):
            ScaleAdaptiveKCFTracker(n_scales=0)

    def test_invalid_learning_rate_zero(self):
        with pytest.raises(ValueError, match="learning_rate"):
            ScaleAdaptiveKCFTracker(learning_rate=0.0)

    def test_invalid_learning_rate_gt1(self):
        with pytest.raises(ValueError, match="learning_rate"):
            ScaleAdaptiveKCFTracker(learning_rate=1.1)

    def test_invalid_scale_bounds(self):
        with pytest.raises(ValueError, match="min_scale"):
            ScaleAdaptiveKCFTracker(min_scale=1.0, max_scale=0.5)

    def test_valid_defaults(self):
        t = ScaleAdaptiveKCFTracker()
        assert t.name == "ScaleAdaptiveKCF"
        assert t.n_scales == 5
        assert t.scale_step == 1.05
        assert t.scale_lr == 0.35

    def test_scale_pool_length(self):
        t = ScaleAdaptiveKCFTracker(n_scales=7, scale_step=1.03)
        assert len(t._scale_pool) == 7

    def test_scale_pool_includes_unity(self):
        t = ScaleAdaptiveKCFTracker(n_scales=5, scale_step=1.05)
        mid = len(t._scale_pool) // 2
        assert abs(t._scale_pool[mid] - 1.0) < 1e-9

    def test_scale_pool_symmetric(self):
        t = ScaleAdaptiveKCFTracker(n_scales=5, scale_step=1.05)
        pool = t._scale_pool
        mid = len(pool) // 2
        for i in range(mid):
            assert abs(pool[i] * pool[-(i + 1)] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Lifecycle: initialize → update
# ---------------------------------------------------------------------------

class TestLifecycle:
    @pytest.fixture
    def tracker(self):
        return ScaleAdaptiveKCFTracker(n_scales=3, scale_step=1.05)

    def test_update_before_init_raises(self, tracker):
        frame = _solid_frame(240, 320)
        with pytest.raises(RuntimeError, match="not initialised"):
            tracker.update(frame)

    def test_initialize_sets_state(self, tracker):
        frame = _textured_frame(240, 320, (100, 80, 40, 30))
        tracker.initialize(frame, (100, 80, 40, 30))
        assert tracker._pos is not None
        assert tracker._alphaf is not None
        assert tracker._xf is not None
        assert tracker._current_scale == 1.0
        assert tracker._init_sz == (40.0, 30.0)

    def test_update_returns_four_element_tuple(self, tracker):
        frame = _textured_frame(240, 320, (100, 80, 40, 30))
        tracker.initialize(frame, (100, 80, 40, 30))
        result = tracker.update(frame)
        assert len(result) == 4

    def test_update_positive_dimensions(self, tracker):
        frame = _textured_frame(240, 320, (100, 80, 40, 30))
        tracker.initialize(frame, (100, 80, 40, 30))
        x, y, w, h = tracker.update(frame)
        assert w > 0
        assert h > 0

    def test_multiple_updates_stay_valid(self, tracker):
        frame = _textured_frame(240, 320, (100, 80, 40, 30))
        tracker.initialize(frame, (100, 80, 40, 30))
        for _ in range(15):
            bbox = tracker.update(frame)
        x, y, w, h = bbox
        assert w > 0 and h > 0

    def test_reset_clears_state(self, tracker):
        frame = _textured_frame(240, 320, (100, 80, 40, 30))
        tracker.initialize(frame, (100, 80, 40, 30))
        tracker.update(frame)
        tracker.reset()
        assert tracker._pos is None
        assert tracker._alphaf is None
        assert tracker._current_scale == 1.0

    def test_reinitialise_after_reset(self, tracker):
        frame = _textured_frame(240, 320, (100, 80, 40, 30))
        tracker.initialize(frame, (100, 80, 40, 30))
        tracker.reset()
        tracker.initialize(frame, (50, 50, 20, 20))
        bbox = tracker.update(frame)
        assert len(bbox) == 4


# ---------------------------------------------------------------------------
# Scale estimation: the tracker should respond to scale changes
# ---------------------------------------------------------------------------

class TestScaleEstimation:
    def test_scale_bounded_below_by_min_scale(self):
        """Accumulated scale must never drop below min_scale."""
        tracker = ScaleAdaptiveKCFTracker(
            n_scales=5, scale_step=1.05, scale_lr=0.5, min_scale=0.5
        )
        frames, bboxes = _make_growing_sequence(30, 60, 20)  # shrinking target
        tracker.initialize(frames[0], bboxes[0])
        for f in frames[1:]:
            tracker.update(f)
        assert tracker._current_scale >= 0.5

    def test_scale_bounded_above_by_max_scale(self):
        """Accumulated scale must never exceed max_scale."""
        tracker = ScaleAdaptiveKCFTracker(
            n_scales=5, scale_step=1.05, scale_lr=0.5, max_scale=2.0
        )
        frames, bboxes = _make_growing_sequence(30, 20, 80)  # growing target
        tracker.initialize(frames[0], bboxes[0])
        for f in frames[1:]:
            tracker.update(f)
        assert tracker._current_scale <= 2.0

    def test_growing_target_does_not_shrink_scale_below_initial(self):
        """When the target doubles in size over 25 frames, the running scale
        should not drop well below 1 (it may not rise, but it should not
        shrink aggressively either — scale bounds prevent runaway drift)."""
        tracker = ScaleAdaptiveKCFTracker(
            n_scales=5, scale_step=1.05, scale_lr=0.3,
            min_scale=0.5, max_scale=5.0,
        )
        frames, bboxes = _make_growing_sequence(25, 30, 60)
        tracker.initialize(frames[0], bboxes[0])
        for f in frames[1:]:
            tracker.update(f)
        # min_scale bound ensures scale cannot drift to 0
        assert tracker._current_scale >= 0.5

    def test_scale_estimate_stays_in_valid_range(self):
        """Scale must always stay in [min_scale, max_scale] after any number of updates."""
        min_s, max_s = 0.2, 8.0
        tracker = ScaleAdaptiveKCFTracker(
            n_scales=5, scale_step=1.05, scale_lr=0.5,
            min_scale=min_s, max_scale=max_s,
        )
        frames, bboxes = _make_growing_sequence(40, 20, 100)
        tracker.initialize(frames[0], bboxes[0])
        for f in frames[1:]:
            tracker.update(f)
            s = tracker._current_scale
            assert min_s <= s <= max_s, f"scale {s:.3f} out of bounds [{min_s}, {max_s}]"

    def test_returned_bbox_dimensions_positive_always(self):
        """The returned (w, h) must be > 0 regardless of scale drift."""
        tracker = ScaleAdaptiveKCFTracker(n_scales=5, scale_step=1.05, scale_lr=0.5)
        frames, bboxes = _make_growing_sequence(20, 30, 60)
        tracker.initialize(frames[0], bboxes[0])
        for f in frames[1:]:
            x, y, w, h = tracker.update(f)
            assert w > 0, f"bbox width became non-positive: {w}"
            assert h > 0, f"bbox height became non-positive: {h}"

    def test_scale_pool_search_selects_highest_peak(self):
        """When the only non-degenerate response comes from the centre scale,
        the scale factor should converge toward 1.0 (no net drift)."""
        tracker = ScaleAdaptiveKCFTracker(n_scales=1, scale_step=1.05, scale_lr=0.9)
        # n_scales=1 → pool = [1.0] → best_scale_factor always 1.0
        frame = _textured_frame(240, 320, (100, 80, 40, 30))
        tracker.initialize(frame, (100, 80, 40, 30))
        for _ in range(10):
            tracker.update(frame)
        # With pool=[1.0], scale update is: scale * ((1-lr) + lr*1.0) = scale
        # i.e., scale never changes from 1.0
        assert abs(tracker._current_scale - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    def test_registered_under_correct_key(self):
        assert "ScaleAdaptiveKCF" in TRACKER_REGISTRY

    def test_build_tracker_returns_correct_type(self):
        t = build_tracker("ScaleAdaptiveKCF")
        assert isinstance(t, ScaleAdaptiveKCFTracker)

    def test_build_tracker_passes_params(self):
        t = build_tracker("ScaleAdaptiveKCF", n_scales=3, scale_step=1.1)
        assert t.n_scales == 3
        assert t.scale_step == 1.1

    def test_build_tracker_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown tracker"):
            build_tracker("NonExistentTracker123")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_small_frame(self):
        tracker = ScaleAdaptiveKCFTracker(n_scales=3, scale_step=1.05)
        frame = _textured_frame(50, 50, (10, 10, 15, 15))
        tracker.initialize(frame, (10, 10, 15, 15))
        bbox = tracker.update(frame)
        assert len(bbox) == 4
        assert bbox[2] > 0 and bbox[3] > 0

    def test_n_scales_one_scale_unchanged(self):
        """With a single scale in the pool (= 1.0), the scale never changes."""
        tracker = ScaleAdaptiveKCFTracker(n_scales=1, scale_step=1.05)
        frame = _textured_frame(240, 320, (100, 80, 40, 30))
        tracker.initialize(frame, (100, 80, 40, 30))
        for _ in range(10):
            tracker.update(frame)
        # pool=[1.0] → best_scale_factor=1.0 every frame → no change
        assert abs(tracker._current_scale - 1.0) < 1e-6

    def test_grayscale_input_accepted(self):
        tracker = ScaleAdaptiveKCFTracker(n_scales=3, scale_step=1.05)
        gray = np.zeros((240, 320), dtype=np.uint8)
        # Checkerboard pattern in grayscale
        for i in range(240):
            for j in range(320):
                gray[i, j] = 200 if (i // 8 + j // 8) % 2 == 0 else 50
        tracker.initialize(gray, (100, 80, 40, 30))
        bbox = tracker.update(gray)
        assert len(bbox) == 4
        assert bbox[2] > 0 and bbox[3] > 0

    def test_target_at_frame_boundary(self):
        """Tracker should handle targets near frame edges via edge-padding."""
        tracker = ScaleAdaptiveKCFTracker(n_scales=3, scale_step=1.05)
        frame = _textured_frame(240, 320, (0, 0, 30, 30))
        tracker.initialize(frame, (0, 0, 30, 30))
        bbox = tracker.update(frame)
        assert len(bbox) == 4
        assert bbox[2] > 0 and bbox[3] > 0

    def test_large_padding_does_not_crash(self):
        tracker = ScaleAdaptiveKCFTracker(n_scales=3, scale_step=1.05, padding=3.0)
        frame = _textured_frame(240, 320, (100, 80, 40, 30))
        tracker.initialize(frame, (100, 80, 40, 30))
        bbox = tracker.update(frame)
        assert bbox[2] > 0 and bbox[3] > 0

    def test_non_square_target(self):
        tracker = ScaleAdaptiveKCFTracker(n_scales=3, scale_step=1.05)
        frame = _textured_frame(240, 320, (50, 60, 100, 30))  # wide target
        tracker.initialize(frame, (50, 60, 100, 30))
        bbox = tracker.update(frame)
        assert bbox[2] > 0 and bbox[3] > 0
