"""Tests for the Scale-Adaptive MOSSE tracker."""

import numpy as np
import pytest

from eovot.trackers.scale_adaptive_mosse import ScaleAdaptiveMOSSETracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a random BGR frame."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _solid_frame(h: int = 240, w: int = 320, val: int = 128) -> np.ndarray:
    """Return a uniform gray BGR frame."""
    return np.full((h, w, 3), val, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_name(self):
        t = ScaleAdaptiveMOSSETracker()
        assert t.name == "ScaleAdaptiveMOSSE"

    def test_repr(self):
        t = ScaleAdaptiveMOSSETracker()
        assert "ScaleAdaptiveMOSSE" in repr(t)

    def test_invalid_n_scales(self):
        with pytest.raises(ValueError):
            ScaleAdaptiveMOSSETracker(n_scales=0)

    def test_invalid_scale_step(self):
        with pytest.raises(ValueError):
            ScaleAdaptiveMOSSETracker(scale_step=1.0)

    def test_invalid_learning_rate_zero(self):
        with pytest.raises(ValueError):
            ScaleAdaptiveMOSSETracker(learning_rate=0.0)

    def test_invalid_scale_lr_zero(self):
        with pytest.raises(ValueError):
            ScaleAdaptiveMOSSETracker(scale_lr=0.0)

    def test_scale_pool_length(self):
        t = ScaleAdaptiveMOSSETracker(n_scales=5, scale_step=1.05)
        assert len(t._scale_pool) == 5

    def test_scale_pool_symmetric(self):
        t = ScaleAdaptiveMOSSETracker(n_scales=7, scale_step=1.03)
        pool = t._scale_pool
        # Pool should be symmetric: pool[k] ≈ 1/pool[-(k+1)]
        assert abs(pool[0] * pool[-1] - 1.0) < 1e-9

    def test_scale_pool_centre_is_one(self):
        t = ScaleAdaptiveMOSSETracker(n_scales=7)
        centre = len(t._scale_pool) // 2
        assert abs(t._scale_pool[centre] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_state_after_init(self):
        t = ScaleAdaptiveMOSSETracker()
        frame = _synthetic_frame()
        t.initialize(frame, (50, 40, 60, 50))
        assert t._H_conj is not None
        assert t._window is not None
        assert t._bbox is not None
        assert t._template_size == (50, 60)

    def test_init_bbox_too_small(self):
        t = ScaleAdaptiveMOSSETracker()
        with pytest.raises(ValueError):
            t.initialize(_synthetic_frame(), (10, 10, 2, 2))

    def test_init_scale_reset(self):
        t = ScaleAdaptiveMOSSETracker()
        t.initialize(_synthetic_frame(), (30, 30, 40, 40))
        assert t._current_scale == 1.0

    def test_window_shape(self):
        t = ScaleAdaptiveMOSSETracker()
        t.initialize(_synthetic_frame(), (10, 10, 50, 40))
        assert t._window.shape == (40, 50)

    def test_grayscale_input(self):
        t = ScaleAdaptiveMOSSETracker()
        gray = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        t.initialize(gray, (50, 50, 60, 60))
        assert t._H_conj is not None


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_before_init_raises(self):
        t = ScaleAdaptiveMOSSETracker()
        with pytest.raises(RuntimeError):
            t.update(_synthetic_frame())

    def test_update_returns_four_tuple(self):
        t = ScaleAdaptiveMOSSETracker()
        frame = _synthetic_frame()
        t.initialize(frame, (50, 40, 60, 50))
        bbox = t.update(_synthetic_frame())
        assert len(bbox) == 4

    def test_update_returns_floats(self):
        t = ScaleAdaptiveMOSSETracker()
        frame = _synthetic_frame()
        t.initialize(frame, (50, 40, 60, 50))
        bbox = t.update(_synthetic_frame())
        assert all(isinstance(v, float) for v in bbox)

    def test_update_positive_dimensions(self):
        t = ScaleAdaptiveMOSSETracker()
        frame = _synthetic_frame()
        t.initialize(frame, (50, 40, 60, 50))
        for _ in range(5):
            _, _, w, h = t.update(_synthetic_frame())
            assert w > 0 and h > 0

    def test_multi_step_update(self):
        t = ScaleAdaptiveMOSSETracker()
        frame = _synthetic_frame()
        t.initialize(frame, (50, 40, 60, 50))
        for _ in range(10):
            bbox = t.update(_synthetic_frame())
        assert len(bbox) == 4

    def test_scale_changes_over_sequence(self):
        """Scale should remain close to 1 on a static-looking scene."""
        t = ScaleAdaptiveMOSSETracker()
        frame = _solid_frame()
        t.initialize(frame, (100, 80, 60, 50))
        for _ in range(5):
            t.update(_solid_frame())
        # On a flat scene, accumulated scale should stay near 1
        assert 0.5 < t._current_scale < 2.0


# ---------------------------------------------------------------------------
# Scale pool behaviour
# ---------------------------------------------------------------------------

class TestScalePool:
    def test_n_scales_one(self):
        t = ScaleAdaptiveMOSSETracker(n_scales=1)
        frame = _synthetic_frame()
        t.initialize(frame, (50, 50, 60, 60))
        bbox = t.update(_synthetic_frame())
        assert len(bbox) == 4

    def test_large_scale_pool(self):
        t = ScaleAdaptiveMOSSETracker(n_scales=9, scale_step=1.02)
        frame = _synthetic_frame()
        t.initialize(frame, (50, 50, 60, 60))
        bbox = t.update(_synthetic_frame())
        assert len(bbox) == 4


# ---------------------------------------------------------------------------
# Benchmark interface compatibility
# ---------------------------------------------------------------------------

class TestInterface:
    def test_base_tracker_subclass(self):
        from eovot.trackers.base import BaseTracker
        assert issubclass(ScaleAdaptiveMOSSETracker, BaseTracker)

    def test_registered_in_experiment_runner(self):
        from eovot.experiment.runner import ExperimentRunner
        cfg = {"name": "ScaleAdaptiveMOSSE", "params": {}}
        tracker = ExperimentRunner._build_tracker(cfg)
        assert tracker.name == "ScaleAdaptiveMOSSE"

    def test_importable_from_package(self):
        from eovot.trackers import ScaleAdaptiveMOSSETracker as T
        assert T is ScaleAdaptiveMOSSETracker
