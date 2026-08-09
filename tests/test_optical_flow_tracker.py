"""Tests for eovot.trackers.optical_flow.OpticalFlowTracker.

Uses synthetic frames (NumPy-generated) so no video files are needed.
Tests cover: initialization, update, bbox propagation, re-detection,
registry integration, and the module-level helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.optical_flow import (
    OpticalFlowTracker,
    _estimate_scale,
    _to_gray,
    _update_bbox,
)
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


# ---------------------------------------------------------------------------
# Synthetic frame helpers
# ---------------------------------------------------------------------------

def _make_frame(
    height: int = 240,
    width: int = 320,
    seed: int = 0,
) -> np.ndarray:
    """Return a random-noise BGR frame with reproducible content."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _shifted_frame(frame: np.ndarray, dx: int = 5, dy: int = 3) -> np.ndarray:
    """Return a frame shifted right by dx and down by dy (nearest-pixel)."""
    H, W = frame.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    import cv2
    return cv2.warpAffine(frame, M, (W, H))


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestOpticalFlowTrackerInit:
    def test_default_name(self):
        t = OpticalFlowTracker()
        assert t.name == "OpticalFlow"

    def test_custom_name(self):
        t = OpticalFlowTracker(name="MyFlow")
        assert t.name == "MyFlow"

    def test_invalid_max_corners(self):
        with pytest.raises(ValueError, match="max_corners"):
            OpticalFlowTracker(max_corners=0)

    def test_invalid_min_corners(self):
        with pytest.raises(ValueError, match="min_corners"):
            OpticalFlowTracker(min_corners=0)

    def test_invalid_quality_level_low(self):
        with pytest.raises(ValueError, match="quality_level"):
            OpticalFlowTracker(quality_level=0.0)

    def test_invalid_quality_level_high(self):
        with pytest.raises(ValueError, match="quality_level"):
            OpticalFlowTracker(quality_level=1.0)

    def test_invalid_fb_threshold(self):
        with pytest.raises(ValueError, match="fb_threshold"):
            OpticalFlowTracker(fb_threshold=0.0)

    def test_repr(self):
        t = OpticalFlowTracker(name="Flow")
        r = repr(t)
        assert "OpticalFlowTracker" in r
        assert "Flow" in r


# ---------------------------------------------------------------------------
# initialize / update interface
# ---------------------------------------------------------------------------

class TestOpticalFlowTrackerInterface:
    @pytest.fixture
    def tracker(self):
        return OpticalFlowTracker(name="Test", max_corners=50, redetect_interval=5)

    @pytest.fixture
    def frame0(self):
        return _make_frame(seed=1)

    @pytest.fixture
    def bbox(self):
        return (60.0, 50.0, 80.0, 60.0)

    def test_update_before_init_raises(self, tracker):
        frame = _make_frame()
        with pytest.raises(RuntimeError, match="initialize"):
            tracker.update(frame)

    def test_initialize_no_error(self, tracker, frame0, bbox):
        tracker.initialize(frame0, bbox)

    def test_update_returns_tuple_of_4(self, tracker, frame0, bbox):
        tracker.initialize(frame0, bbox)
        frame1 = _make_frame(seed=2)
        result = tracker.update(frame1)
        assert len(result) == 4

    def test_update_returns_finite_values(self, tracker, frame0, bbox):
        tracker.initialize(frame0, bbox)
        frame1 = _make_frame(seed=2)
        x, y, w, h = tracker.update(frame1)
        assert all(np.isfinite(v) for v in (x, y, w, h))

    def test_multiple_updates(self, tracker, frame0, bbox):
        tracker.initialize(frame0, bbox)
        for i in range(1, 10):
            frame = _make_frame(seed=i + 1)
            bbox_pred = tracker.update(frame)
            assert len(bbox_pred) == 4

    def test_reinitialize_resets_state(self, tracker, frame0, bbox):
        tracker.initialize(frame0, bbox)
        for _ in range(5):
            tracker.update(_make_frame())
        # Re-initialize on the same tracker should not error
        tracker.initialize(frame0, (10.0, 10.0, 40.0, 40.0))
        result = tracker.update(_make_frame(seed=99))
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Tracking accuracy on synthetic translating target
# ---------------------------------------------------------------------------

class TestOpticalFlowTrackerAccuracy:
    """Verify that the tracker follows a small, clean translation."""

    def test_tracks_horizontal_shift(self):
        rng = np.random.default_rng(7)
        # Structured frame: gradient so corners are detectable
        frame0 = np.zeros((200, 200, 3), dtype=np.uint8)
        frame0[:, :] = rng.integers(50, 200, (200, 200, 3), dtype=np.uint8)

        bbox = (60.0, 60.0, 60.0, 60.0)
        tracker = OpticalFlowTracker(
            max_corners=150,
            min_corners=5,
            redetect_interval=0,
        )
        tracker.initialize(frame0, bbox)

        dx, dy = 4, 2
        frame1 = _shifted_frame(frame0, dx=dx, dy=dy)
        x, y, w, h = tracker.update(frame1)

        # Expected centre after shift: (90 + dx, 90 + dy) = (94, 92)
        expected_cx = 90.0 + dx
        expected_cy = 90.0 + dy
        pred_cx = x + w / 2
        pred_cy = y + h / 2

        # Allow ±10 px tolerance (LK flow on pure random noise is noisy)
        assert abs(pred_cx - expected_cx) < 15, (
            f"CX off by {abs(pred_cx - expected_cx):.1f} px"
        )
        assert abs(pred_cy - expected_cy) < 15, (
            f"CY off by {abs(pred_cy - expected_cy):.1f} px"
        )

    def test_width_roughly_preserved_on_translation(self):
        """Scale should stay close to 1 for a pure translation."""
        frame0 = _make_frame(seed=42)
        bbox = (80.0, 60.0, 80.0, 60.0)
        tracker = OpticalFlowTracker(max_corners=100, min_corners=5)
        tracker.initialize(frame0, bbox)

        frame1 = _shifted_frame(frame0, dx=3, dy=3)
        _, _, w, h = tracker.update(frame1)

        # Width and height should be within 50% of original (pure translation)
        assert 40.0 < w < 120.0, f"width {w:.1f} far from original 80"
        assert 30.0 < h < 90.0, f"height {h:.1f} far from original 60"


# ---------------------------------------------------------------------------
# Integration with benchmark engine (SyntheticDataset)
# ---------------------------------------------------------------------------

class TestOpticalFlowBenchmarkIntegration:
    def test_full_benchmark_run(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(
            num_sequences=2, num_frames=20,
            frame_size=(160, 120), bbox_size=(25, 25),
            motion="linear", seed=0,
        )
        tracker = OpticalFlowTracker(max_corners=80, min_corners=5)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")

        assert result.tracker_name == "OpticalFlow"
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_predictions_stored(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(
            num_sequences=1, num_frames=15,
            frame_size=(120, 90), bbox_size=(20, 20),
            motion="linear", seed=5,
        )
        tracker = OpticalFlowTracker(max_corners=60, min_corners=3)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")

        seq = result.sequence_results[0]
        assert seq.predictions is not None
        assert seq.predictions.shape[1] == 4


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    def test_opticalflow_in_registry(self):
        assert "OpticalFlow" in TRACKER_REGISTRY

    def test_build_tracker_default(self):
        t = build_tracker("OpticalFlow")
        assert isinstance(t, OpticalFlowTracker)
        assert t.name == "OpticalFlow"

    def test_build_tracker_with_params(self):
        t = build_tracker("OpticalFlow", max_corners=50, lk_window=15)
        assert t.max_corners == 50
        assert t.lk_window == 15


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestToGray:
    def test_bgr_to_gray(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        g = _to_gray(frame)
        assert g.ndim == 2

    def test_already_gray_passthrough(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        out = _to_gray(gray)
        assert out.ndim == 2
        assert out is gray

    def test_single_channel_3d(self):
        frame = np.zeros((50, 50, 1), dtype=np.uint8)
        out = _to_gray(frame)
        assert out.ndim == 2


class TestEstimateScale:
    def test_identical_points_returns_one(self):
        pts = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
        assert _estimate_scale(pts, pts) == pytest.approx(1.0, abs=1e-6)

    def test_double_scale(self):
        pts = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
        pts2 = pts * 2.0
        s = _estimate_scale(pts, pts2)
        assert s == pytest.approx(2.0, abs=0.01)

    def test_half_scale(self):
        pts = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
        pts2 = pts * 0.5
        s = _estimate_scale(pts, pts2)
        # Clamped at 0.5
        assert s == pytest.approx(0.5, abs=0.01)

    def test_single_point_returns_one(self):
        pts = np.array([[5, 5]], dtype=np.float32)
        assert _estimate_scale(pts, pts) == pytest.approx(1.0)

    def test_clamped_above(self):
        pts = np.array([[0, 0], [10, 0]], dtype=np.float32)
        pts2 = pts * 5.0  # would be 5× without clamp
        s = _estimate_scale(pts, pts2)
        assert s <= 2.0

    def test_clamped_below(self):
        pts = np.array([[0, 0], [10, 0]], dtype=np.float32)
        pts2 = pts * 0.1  # would be 0.1× without clamp
        s = _estimate_scale(pts, pts2)
        assert s >= 0.5


class TestUpdateBbox:
    def test_pure_translation(self):
        bbox = (10.0, 20.0, 40.0, 30.0)
        # bbox centre = (10 + 40/2, 20 + 30/2) = (30, 35)
        prev = np.array([[30.0, 35.0], [50.0, 35.0]], dtype=np.float32)
        curr = prev + np.array([[5.0, 3.0]], dtype=np.float32)
        x, y, w, h = _update_bbox(bbox, prev, curr)
        # Centre should shift by (5, 3): expected (35, 38)
        cx = x + w / 2
        cy = y + h / 2
        assert cx == pytest.approx(30.0 + 5.0, abs=0.5)  # 35.0
        assert cy == pytest.approx(35.0 + 3.0, abs=0.5)  # 38.0

    def test_scale_preserved_on_translation(self):
        bbox = (10.0, 10.0, 40.0, 30.0)
        prev = np.array([[20.0, 20.0], [40.0, 20.0], [20.0, 35.0]], dtype=np.float32)
        curr = prev + 5.0  # pure translation
        _, _, w, h = _update_bbox(bbox, prev, curr)
        # Scale = 1 → width and height should be unchanged
        assert w == pytest.approx(40.0, abs=1.0)
        assert h == pytest.approx(30.0, abs=1.0)

    def test_returns_4_elements(self):
        bbox = (0.0, 0.0, 50.0, 50.0)
        pts = np.array([[10.0, 10.0], [30.0, 10.0]], dtype=np.float32)
        result = _update_bbox(bbox, pts, pts)
        assert len(result) == 4
