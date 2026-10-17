"""Tests for the DCF (Discriminative Correlation Filter) tracker with HOG features."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.dcf import DCFTracker
from eovot.trackers.registry import available_trackers, build_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = (rng.uniform(40, 120, (h, w, 3))).astype(np.uint8)
    return frame


def _frame_with_target(
    h: int = 120, w: int = 160, cx: int = 80, cy: int = 60,
    tw: int = 30, th: int = 20, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = (rng.uniform(40, 100, (h, w, 3))).astype(np.uint8)
    x1 = max(0, cx - tw // 2)
    y1 = max(0, cy - th // 2)
    x2 = min(w, cx + tw // 2)
    y2 = min(h, cy + th // 2)
    frame[y1:y2, x1:x2] = [200, 200, 200]  # bright target
    return frame


def _default_tracker() -> DCFTracker:
    return DCFTracker(learning_rate=0.1, padding=1.0, num_hog_bins=9, cell_size=4)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestDCFConstruction:
    def test_default_name(self):
        t = DCFTracker()
        assert t.name == "DCF"

    def test_custom_params(self):
        t = DCFTracker(learning_rate=0.05, padding=2.0, lambda_=1e-3,
                       num_hog_bins=6, cell_size=8)
        assert t.learning_rate == 0.05
        assert t.padding == 2.0
        assert t.lambda_ == 1e-3
        assert t.num_hog_bins == 6
        assert t.cell_size == 8

    def test_registry_contains_dcf(self):
        assert "DCF" in available_trackers()

    def test_build_tracker_dcf(self):
        t = build_tracker("DCF")
        assert isinstance(t, DCFTracker)

    def test_build_tracker_with_params(self):
        t = build_tracker("DCF", learning_rate=0.05)
        assert t.learning_rate == 0.05


# ---------------------------------------------------------------------------
# Update before initialize raises RuntimeError
# ---------------------------------------------------------------------------

class TestDCFInitCheck:
    def test_update_before_init_raises(self):
        t = _default_tracker()
        frame = _make_frame()
        with pytest.raises(RuntimeError, match="not initialised"):
            t.update(frame)


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------

class TestDCFInitialize:
    def test_initialize_sets_internal_state(self):
        t = _default_tracker()
        frame = _make_frame()
        t.initialize(frame, (50, 40, 30, 20))
        assert t._pos is not None
        assert t._target_sz is not None
        assert t._patch_sz is not None
        assert t._hf_num is not None
        assert t._hf_den is not None
        assert t._window is not None
        assert t._yf is not None

    def test_initialize_hf_num_channels(self):
        t = DCFTracker(num_hog_bins=9, cell_size=4)
        t.initialize(_make_frame(), (50, 40, 30, 20))
        assert len(t._hf_num) == 9

    def test_initialize_hf_den_shape(self):
        t = DCFTracker(num_hog_bins=9, cell_size=4)
        t.initialize(_make_frame(), (50, 40, 30, 20))
        h_c = t._patch_sz[1] // t.cell_size
        w_c = t._patch_sz[0] // t.cell_size
        assert t._hf_den.shape == (h_c, w_c)

    def test_initialize_window_shape_matches_hf_den(self):
        t = _default_tracker()
        t.initialize(_make_frame(), (50, 40, 30, 20))
        assert t._window.shape == t._hf_den.shape

    def test_patch_aligned_to_cell_size(self):
        t = DCFTracker(cell_size=4)
        t.initialize(_make_frame(), (50, 40, 30, 20))
        pw, ph = t._patch_sz
        assert pw % t.cell_size == 0
        assert ph % t.cell_size == 0

    def test_grayscale_frame_accepted(self):
        t = _default_tracker()
        gray = np.zeros((120, 160), dtype=np.uint8)
        t.initialize(gray, (50, 40, 30, 20))
        assert t._pos is not None

    def test_reinitialize_resets_state(self):
        t = _default_tracker()
        frame = _make_frame()
        t.initialize(frame, (50, 40, 30, 20))
        t.initialize(frame, (10, 10, 20, 15))
        cx, cy = t._pos
        assert abs(cx - 20.0) < 1.0  # center of new bbox
        assert abs(cy - 17.5) < 1.0


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

class TestDCFUpdate:
    def test_update_returns_4_tuple(self):
        t = _default_tracker()
        frame = _make_frame()
        t.initialize(frame, (50, 40, 30, 20))
        bbox = t.update(frame)
        assert len(bbox) == 4

    def test_update_returns_floats(self):
        t = _default_tracker()
        frame = _make_frame()
        t.initialize(frame, (50, 40, 30, 20))
        x, y, w, h = t.update(frame)
        assert all(isinstance(v, float) for v in (x, y, w, h))

    def test_target_size_preserved(self):
        """update() must not change target size (DCF is scale-fixed)."""
        t = _default_tracker()
        frame = _make_frame()
        t.initialize(frame, (50, 40, 30, 20))
        for _ in range(3):
            _, _, w, h = t.update(frame)
        assert abs(w - 30.0) < 1e-6
        assert abs(h - 20.0) < 1e-6

    def test_multiple_updates(self):
        """Tracker should update consistently across many frames."""
        t = _default_tracker()
        frame = _make_frame()
        t.initialize(frame, (50, 40, 30, 20))
        for _ in range(10):
            bbox = t.update(_make_frame(seed=_))
            assert len(bbox) == 4

    def test_tracks_moving_target(self):
        """DCF should roughly follow a bright patch moving linearly."""
        t = DCFTracker(learning_rate=0.2, padding=1.5, cell_size=4)
        cx0, cy0 = 40, 40
        tw, th = 24, 24

        frame0 = _frame_with_target(cx=cx0, cy=cy0, tw=tw, th=th, seed=0)
        t.initialize(frame0, (cx0 - tw // 2, cy0 - th // 2, tw, th))

        errors = []
        for step in range(1, 6):
            cx_true = cx0 + step * 4
            cy_true = cy0 + step * 2
            frame = _frame_with_target(cx=cx_true, cy=cy_true, tw=tw, th=th, seed=step)
            x, y, w, h = t.update(frame)
            pred_cx = x + w / 2
            pred_cy = y + h / 2
            errors.append(((pred_cx - cx_true) ** 2 + (pred_cy - cy_true) ** 2) ** 0.5)

        # Tracker should stay within 20 px of truth on a simple linear motion
        assert min(errors) < 20.0, f"min error = {min(errors):.1f} px; tracker did not follow target"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestDCFReset:
    def test_reset_clears_state(self):
        t = _default_tracker()
        t.initialize(_make_frame(), (50, 40, 30, 20))
        t.reset()
        assert t._pos is None
        assert t._hf_num is None
        assert t._hf_den is None

    def test_update_after_reset_raises(self):
        t = _default_tracker()
        t.initialize(_make_frame(), (50, 40, 30, 20))
        t.reset()
        with pytest.raises(RuntimeError):
            t.update(_make_frame())


# ---------------------------------------------------------------------------
# HOG feature extraction internals
# ---------------------------------------------------------------------------

class TestDCFHOG:
    def test_hog_shape(self):
        t = DCFTracker(num_hog_bins=9, cell_size=4)
        patch = np.random.rand(40, 40).astype(np.float32)
        hog = t._hog_features(patch)
        assert hog.shape == (10, 10, 9)

    def test_hog_cells_are_normalized(self):
        t = DCFTracker(num_hog_bins=9, cell_size=4)
        patch = (np.random.rand(40, 40) * 200).astype(np.float32)
        hog = t._hog_features(patch)
        cell_norms = np.sqrt((hog ** 2).sum(axis=2))
        # All norms should be ≤ 1 + small eps
        assert np.all(cell_norms <= 1.0 + 1e-3)

    def test_hog_output_non_negative(self):
        t = DCFTracker(num_hog_bins=9, cell_size=4)
        patch = np.random.rand(80, 80).astype(np.float32) * 200
        hog = t._hog_features(patch)
        assert np.all(hog >= 0.0)

    def test_hog_different_cell_sizes(self):
        for cs in (4, 8):
            t = DCFTracker(cell_size=cs)
            patch = np.random.rand(64, 64).astype(np.float32)
            hog = t._hog_features(patch)
            assert hog.shape[0] == 64 // cs
            assert hog.shape[1] == 64 // cs

    def test_gaussian_labels_peak_at_origin(self):
        labels = DCFTracker._gaussian_labels(20, 20)
        assert labels[0, 0] == pytest.approx(1.0, abs=1e-4)

    def test_gaussian_labels_shape(self):
        labels = DCFTracker._gaussian_labels(30, 40)
        assert labels.shape == (30, 40)

    def test_hann2d_shape(self):
        win = DCFTracker._hann2d(10, 15)
        assert win.shape == (10, 15)

    def test_hann2d_edges_near_zero(self):
        win = DCFTracker._hann2d(20, 20)
        assert win[0, 0] < 0.01
        assert win[-1, -1] < 0.01


# ---------------------------------------------------------------------------
# Integration: full benchmark loop with synthetic data
# ---------------------------------------------------------------------------

class TestDCFBenchmarkIntegration:
    def test_full_sequence(self):
        """DCF should run a complete sequence without error."""
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(
            num_sequences=1, num_frames=20,
            frame_size=(160, 120), bbox_size=(30, 20),
            motion="linear", seed=42,
        )
        t = DCFTracker(learning_rate=0.1, padding=1.0, cell_size=4)
        seq = ds[0]
        frames = list(seq)
        t.initialize(frames[0], seq.init_bbox)
        for frame in frames[1:]:
            bbox = t.update(frame)
            assert len(bbox) == 4

    def test_ious_are_valid(self):
        """IoUs should be in [0, 1] when tracking a slow-moving synthetic target."""
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.metrics.accuracy import MetricsEngine

        ds = SyntheticDataset(
            num_sequences=1, num_frames=30,
            frame_size=(160, 120), bbox_size=(30, 20),
            motion="linear", seed=7,
        )
        t = DCFTracker(learning_rate=0.15, padding=1.5, cell_size=4)
        seq = ds[0]
        frames = list(seq)
        gt = seq.ground_truth
        preds = [seq.init_bbox]

        t.initialize(frames[0], seq.init_bbox)
        for frame in frames[1:]:
            preds.append(t.update(frame))

        metrics = MetricsEngine()
        ious = metrics.batch_iou(np.array(preds), gt)
        assert np.all(ious >= 0.0)
        assert np.all(ious <= 1.0)
