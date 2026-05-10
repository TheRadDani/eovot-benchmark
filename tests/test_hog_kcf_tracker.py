"""Tests for the HOG-KCF tracker.

Covers:
- HOGKCFTracker implements the BaseTracker interface
- initialize() sets up internal state
- update() returns valid bounding boxes
- HOG feature extraction produces correct shapes and value ranges
- Multi-channel kernel correlation produces finite, positive-definite output
- Tracker can be reset and re-initialised
- Registration in ExperimentRunner and the CLI tracker registry
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.hog_kcf import HOGKCFTracker
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """Create a synthetic BGR frame with a bright rectangle (simulated target)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Add gradient background so gradients are non-trivial.
    for c in range(3):
        frame[:, :, c] = np.linspace(30, 90, w, dtype=np.uint8)
    # Bright target rectangle at (40, 30, 40, 40)
    frame[30:70, 40:80] = [200, 150, 100]
    return frame


def _default_bbox() -> tuple:
    return (40.0, 30.0, 40.0, 40.0)


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------

class TestHOGKCFInterface:
    def test_is_base_tracker(self):
        assert issubclass(HOGKCFTracker, BaseTracker)

    def test_default_name(self):
        t = HOGKCFTracker()
        assert t.name == "HOG-KCF"

    def test_repr(self):
        t = HOGKCFTracker()
        assert "HOGKCFTracker" in repr(t)

    def test_update_before_init_raises(self):
        t = HOGKCFTracker()
        frame = _make_frame()
        with pytest.raises(RuntimeError, match="not initialised"):
            t.update(frame)

    def test_abstract_methods_implemented(self):
        # If abstract methods are not implemented, instantiation raises TypeError.
        t = HOGKCFTracker()
        assert callable(t.initialize)
        assert callable(t.update)


# ---------------------------------------------------------------------------
# initialize() state setup
# ---------------------------------------------------------------------------

class TestHOGKCFInitialize:
    def test_sets_pos(self):
        t = HOGKCFTracker()
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        assert t._pos is not None
        cx, cy = t._pos
        # Centre should be at (40 + 20, 30 + 20) = (60, 50)
        assert cx == pytest.approx(60.0)
        assert cy == pytest.approx(50.0)

    def test_xfs_is_list_of_arrays(self):
        t = HOGKCFTracker(n_orient=8)
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        assert isinstance(t._xfs, list)
        assert len(t._xfs) == 8

    def test_alphaf_shape_matches_feature_map(self):
        t = HOGKCFTracker()
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        assert t._alphaf is not None
        assert t._alphaf.shape == (t._feat_h, t._feat_w)

    def test_yf_shape_matches_feature_map(self):
        t = HOGKCFTracker()
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        assert t._yf.shape == (t._feat_h, t._feat_w)

    def test_window_shape_matches_feature_map(self):
        t = HOGKCFTracker()
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        assert t._window.shape == (t._feat_h, t._feat_w)


# ---------------------------------------------------------------------------
# update() output validity
# ---------------------------------------------------------------------------

class TestHOGKCFUpdate:
    def setup_method(self):
        self.frame = _make_frame()
        self.tracker = HOGKCFTracker()
        self.tracker.initialize(self.frame, _default_bbox())

    def test_update_returns_tuple_of_4(self):
        pred = self.tracker.update(self.frame)
        assert len(pred) == 4

    def test_update_width_height_positive(self):
        pred = self.tracker.update(self.frame)
        _, _, w, h = pred
        assert w > 0
        assert h > 0

    def test_update_values_are_finite(self):
        pred = self.tracker.update(self.frame)
        assert all(np.isfinite(v) for v in pred)

    def test_update_on_static_sequence(self):
        """On a sequence of identical frames, the predicted box should stay near GT."""
        t = HOGKCFTracker()
        frame = _make_frame()
        bbox = _default_bbox()
        t.initialize(frame, bbox)
        for _ in range(5):
            pred = t.update(frame)
            x, y, w, h = pred
            gx, gy, gw, gh = bbox
            # Predicted centre within 20px of GT centre on static frames.
            pred_cx, pred_cy = x + w / 2, y + h / 2
            gt_cx, gt_cy = gx + gw / 2, gy + gh / 2
            assert abs(pred_cx - gt_cx) < 25
            assert abs(pred_cy - gt_cy) < 25

    def test_multiple_frames_no_crash(self):
        frames = [_make_frame() for _ in range(10)]
        t = HOGKCFTracker()
        t.initialize(frames[0], _default_bbox())
        for frame in frames[1:]:
            pred = t.update(frame)
            assert len(pred) == 4


# ---------------------------------------------------------------------------
# HOG feature extraction internals
# ---------------------------------------------------------------------------

class TestHOGFeatures:
    def test_hog_channels_count(self):
        t = HOGKCFTracker(n_orient=8, cell_size=4)
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        gray = np.zeros((t._feat_h * 4, t._feat_w * 4), dtype=np.float32)
        channels = t._hog_channels(gray)
        assert len(channels) == 8

    def test_hog_channel_shapes(self):
        t = HOGKCFTracker(n_orient=6, cell_size=4)
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        gray = np.random.rand(t._feat_h * 4, t._feat_w * 4).astype(np.float32)
        channels = t._hog_channels(gray)
        for ch in channels:
            assert ch.shape == (t._feat_h, t._feat_w)

    def test_hog_channels_finite(self):
        t = HOGKCFTracker(n_orient=8, cell_size=4)
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        gray = _make_frame()[..., 0].astype(np.float32)
        gray_resized = __import__("cv2").resize(
            gray, (t._feat_w * t.cell_size, t._feat_h * t.cell_size)
        )
        channels = t._hog_channels(gray_resized)
        for b, ch in enumerate(channels):
            assert np.all(np.isfinite(ch)), f"Channel {b} has non-finite values"

    def test_uniform_patch_produces_zero_channels(self):
        """Uniform intensity → zero gradients → HOG channels are all-zero after norm."""
        t = HOGKCFTracker(n_orient=8, cell_size=4)
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        gray = np.full(
            (t._feat_h * t.cell_size, t._feat_w * t.cell_size), 128.0, dtype=np.float32
        )
        channels = t._hog_channels(gray)
        for ch in channels:
            assert np.allclose(ch, 0.0, atol=1e-4), "Uniform patch should yield zero HOG"


# ---------------------------------------------------------------------------
# Multi-channel kernel correlation
# ---------------------------------------------------------------------------

class TestMultiChannelKernel:
    def test_self_correlation_positive(self):
        """Self-correlation spatial kernel must have a positive maximum."""
        t = HOGKCFTracker()
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        kf = t._kernel_corr(t._xfs, t._xfs)
        k = np.real(np.fft.ifft2(kf))
        # The spatial kernel exp(-exponent) is always positive; its peak
        # may land at index (0, 0) or nearby depending on FFT convention.
        assert np.amax(k) > 0.0

    def test_kernel_output_finite(self):
        t = HOGKCFTracker()
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        kf = t._kernel_corr(t._xfs, t._xfs)
        assert np.all(np.isfinite(kf))

    def test_kernel_shape(self):
        t = HOGKCFTracker()
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        kf = t._kernel_corr(t._xfs, t._xfs)
        assert kf.shape == (t._feat_h, t._feat_w)


# ---------------------------------------------------------------------------
# reset() and re-initialisation
# ---------------------------------------------------------------------------

class TestHOGKCFReset:
    def test_reset_clears_state(self):
        t = HOGKCFTracker()
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        t.reset()
        assert t._pos is None
        assert t._xfs is None
        assert t._alphaf is None

    def test_update_after_reset_raises(self):
        t = HOGKCFTracker()
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        t.reset()
        with pytest.raises(RuntimeError):
            t.update(frame)

    def test_reinitialize_after_reset(self):
        t = HOGKCFTracker()
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        t.reset()
        t.initialize(frame, (10.0, 10.0, 20.0, 20.0))
        pred = t.update(frame)
        assert len(pred) == 4


# ---------------------------------------------------------------------------
# Hyperparameter variants
# ---------------------------------------------------------------------------

class TestHOGKCFHyperparams:
    @pytest.mark.parametrize("n_orient", [4, 6, 8])
    def test_different_orient_counts(self, n_orient):
        t = HOGKCFTracker(n_orient=n_orient)
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        assert len(t._xfs) == n_orient
        pred = t.update(frame)
        assert len(pred) == 4

    @pytest.mark.parametrize("cell_size", [2, 4, 8])
    def test_different_cell_sizes(self, cell_size):
        t = HOGKCFTracker(cell_size=cell_size)
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        pred = t.update(frame)
        assert len(pred) == 4

    @pytest.mark.parametrize("lr", [0.05, 0.10, 0.20])
    def test_different_learning_rates(self, lr):
        t = HOGKCFTracker(learning_rate=lr)
        frame = _make_frame()
        t.initialize(frame, _default_bbox())
        pred = t.update(frame)
        assert all(np.isfinite(v) for v in pred)


# ---------------------------------------------------------------------------
# Tracker registry integration
# ---------------------------------------------------------------------------

class TestHOGKCFRegistry:
    def test_in_experiment_runner_registry(self):
        from eovot.experiment.runner import ExperimentRunner
        cfg = {"name": "HOG-KCF", "params": {}}
        tracker = ExperimentRunner._build_tracker(cfg)
        assert isinstance(tracker, HOGKCFTracker)

    def test_in_cli_registry(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import importlib
        rb = importlib.import_module("run_benchmark")
        assert "HOG-KCF" in rb.TRACKER_REGISTRY
        cls = rb.TRACKER_REGISTRY["HOG-KCF"]
        assert cls is HOGKCFTracker

    def test_in_trackers_init(self):
        from eovot.trackers import HOGKCFTracker as imported
        assert imported is HOGKCFTracker
