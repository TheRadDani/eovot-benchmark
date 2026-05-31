"""Tests for the ScaleAdaptiveMOSSETracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.scale_adaptive_mosse import ScaleAdaptiveMOSSETracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    """Create a synthetic BGR frame with a coloured rectangle target."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(20, 80, (h, w, 3), dtype=np.uint8)
    return frame


def _draw_target(frame: np.ndarray, x: int, y: int, tw: int, th: int) -> np.ndarray:
    """Draw a bright rectangle at (x, y, tw, th) on a copy of frame."""
    f = frame.copy()
    y1, y2 = max(0, y), min(frame.shape[0], y + th)
    x1, x2 = max(0, x), min(frame.shape[1], x + tw)
    f[y1:y2, x1:x2] = [220, 180, 50]
    return f


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_construction(self):
        t = ScaleAdaptiveMOSSETracker()
        assert t.name == "ScaleAdaptiveMOSSE"
        assert t.n_scales == 5
        assert t.scale_step == 1.05
        assert len(t._scale_factors) == 5

    def test_scale_factors_symmetric(self):
        t = ScaleAdaptiveMOSSETracker(n_scales=5, scale_step=1.10)
        factors = t._scale_factors
        assert len(factors) == 5
        # Middle factor must equal 1.0 (step^0)
        assert abs(factors[2] - 1.0) < 1e-9
        # Factors must be strictly increasing
        for i in range(len(factors) - 1):
            assert factors[i] < factors[i + 1]

    def test_invalid_n_scales_even(self):
        with pytest.raises(ValueError, match="odd"):
            ScaleAdaptiveMOSSETracker(n_scales=4)

    def test_invalid_n_scales_zero(self):
        with pytest.raises(ValueError, match="odd"):
            ScaleAdaptiveMOSSETracker(n_scales=0)

    def test_invalid_scale_step(self):
        with pytest.raises(ValueError, match="scale_step"):
            ScaleAdaptiveMOSSETracker(scale_step=0.95)

    def test_initialize_sets_internal_state(self):
        t = ScaleAdaptiveMOSSETracker()
        frame = _make_frame()
        frame = _draw_target(frame, 50, 40, 20, 20)
        t.initialize(frame, (50, 40, 20, 20))
        assert t._H_conj is not None
        assert t._pos == (60.0, 50.0)  # cx, cy
        assert t._target_w == 20.0
        assert t._target_h == 20.0

    def test_invalid_bbox_raises(self):
        t = ScaleAdaptiveMOSSETracker()
        frame = _make_frame()
        with pytest.raises(ValueError):
            t.initialize(frame, (10, 10, 0, 20))

    def test_initialize_on_grayscale(self):
        t = ScaleAdaptiveMOSSETracker()
        gray = np.ones((80, 80), dtype=np.uint8) * 128
        gray[20:40, 20:40] = 200
        t.initialize(gray, (20, 20, 20, 20))
        assert t._H_conj is not None


# ---------------------------------------------------------------------------
# Update / tracking tests
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_returns_bbox_tuple(self):
        t = ScaleAdaptiveMOSSETracker()
        bg = _make_frame()
        frame0 = _draw_target(bg, 60, 50, 20, 20)
        t.initialize(frame0, (60, 50, 20, 20))
        frame1 = _draw_target(bg, 62, 52, 20, 20)
        bbox = t.update(frame1)
        assert len(bbox) == 4
        assert all(isinstance(v, float) for v in bbox)

    def test_update_before_initialize_raises(self):
        t = ScaleAdaptiveMOSSETracker()
        frame = _make_frame()
        with pytest.raises(RuntimeError, match="not initialised"):
            t.update(frame)

    def test_tracks_linear_motion(self):
        """Tracker should stay close to a slowly-moving target."""
        rng = np.random.default_rng(7)
        bg = rng.integers(20, 80, (120, 160, 3), dtype=np.uint8)

        t = ScaleAdaptiveMOSSETracker(n_scales=3, psr_threshold=0.0)
        tw, th = 24, 24
        x0, y0 = 60, 50
        frame0 = _draw_target(bg, x0, y0, tw, th)
        t.initialize(frame0, (x0, y0, tw, th))

        errors = []
        for step in range(1, 8):
            x = x0 + step * 2
            y = y0 + step
            frame = _draw_target(bg, x, y, tw, th)
            pred = t.update(frame)
            cx_gt = x + tw / 2.0
            cy_gt = y + th / 2.0
            cx_pred = pred[0] + pred[2] / 2.0
            cy_pred = pred[1] + pred[3] / 2.0
            err = np.sqrt((cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2)
            errors.append(err)

        mean_err = np.mean(errors)
        assert mean_err < 30.0, f"Mean tracking error {mean_err:.1f}px too large"

    def test_bbox_w_h_positive(self):
        """Predicted bounding boxes must have positive width and height."""
        rng = np.random.default_rng(3)
        bg = rng.integers(20, 80, (120, 160, 3), dtype=np.uint8)
        t = ScaleAdaptiveMOSSETracker(n_scales=3)
        frame0 = _draw_target(bg, 60, 50, 24, 24)
        t.initialize(frame0, (60, 50, 24, 24))
        for i in range(5):
            frame = _draw_target(bg, 60 + i, 50 + i, 24, 24)
            x, y, w, h = t.update(frame)
            assert w > 0, "Width must be positive"
            assert h > 0, "Height must be positive"


# ---------------------------------------------------------------------------
# Scale adaptation tests
# ---------------------------------------------------------------------------

class TestScaleAdaptation:
    def test_scale_increases_on_growing_target(self):
        """Running target size should increase when the object grows."""
        rng = np.random.default_rng(11)
        bg = rng.integers(20, 80, (200, 200, 3), dtype=np.uint8)

        t = ScaleAdaptiveMOSSETracker(n_scales=5, scale_step=1.10, scale_lr=0.5)
        tw0, th0 = 20, 20
        cx, cy = 100, 100
        frame0 = _draw_target(bg, int(cx - tw0 / 2), int(cy - th0 / 2), tw0, th0)
        t.initialize(frame0, (cx - tw0 / 2, cy - th0 / 2, tw0, th0))

        initial_w = t._target_w

        # Grow the target over 10 frames.
        for step in range(1, 11):
            tw = tw0 + step * 3
            th = th0 + step * 3
            frame = _draw_target(bg, int(cx - tw / 2), int(cy - th / 2), tw, th)
            t.update(frame)

        final_w = t._target_w
        # The tracker's internal size should have grown at least a little.
        assert final_w >= initial_w, (
            f"Expected scale adaptation (w: {initial_w:.1f} → {final_w:.1f})"
        )

    def test_n_scales_one_disables_pyramid(self):
        """n_scales=1 means only the current scale is tested (no pyramid)."""
        t = ScaleAdaptiveMOSSETracker(n_scales=1)
        assert len(t._scale_factors) == 1
        assert abs(t._scale_factors[0] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Reset test
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        t = ScaleAdaptiveMOSSETracker()
        frame = _make_frame()
        frame = _draw_target(frame, 50, 40, 20, 20)
        t.initialize(frame, (50, 40, 20, 20))
        assert t._H_conj is not None
        t.reset()
        assert t._H_conj is None
        assert t._pos is None

    def test_reinitialize_after_reset(self):
        t = ScaleAdaptiveMOSSETracker()
        frame = _make_frame()
        frame = _draw_target(frame, 50, 40, 20, 20)
        t.initialize(frame, (50, 40, 20, 20))
        t.reset()
        t.initialize(frame, (30, 30, 16, 16))
        assert t._pos == (38.0, 38.0)


# ---------------------------------------------------------------------------
# Benchmark engine integration
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:
    def test_runs_with_benchmark_engine(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear", seed=1)
        engine = BenchmarkEngine(verbose=False)
        tracker = ScaleAdaptiveMOSSETracker(n_scales=3)
        result = engine.run(tracker, ds, dataset_name="Synthetic")

        assert result.tracker_name == "ScaleAdaptiveMOSSE"
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_repr(self):
        t = ScaleAdaptiveMOSSETracker()
        assert "ScaleAdaptiveMOSSETracker" in repr(t)
        assert "ScaleAdaptiveMOSSE" in t.name
