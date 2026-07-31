"""Unit tests for LKFlowTracker.

Tests cover:
- Basic initialize/update contract
- Bounding-box coordinate inversion (scale back to original resolution)
- Forward-backward filtering logic
- RANSAC enable/disable
- Failure detection when target disappears
- Registry integration
- Full benchmark engine integration via SyntheticDataset
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.lk_flow import LKFlowTracker, _median_scale, _to_gray
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 240, w: int = 320, color: tuple = (60, 120, 60)) -> np.ndarray:
    """Return a solid-color BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _make_textured_frame(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    """Return a frame with enough texture for corner detection."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(30, 220, size=(h, w, 3), dtype=np.uint8)
    return frame


def _make_sequence(n_frames: int = 20, h: int = 240, w: int = 320, dx: int = 2) -> list:
    """Generate a sequence of textured frames with a moving bright patch (target)."""
    frames = []
    for i in range(n_frames):
        rng = np.random.default_rng(i)
        frame = rng.integers(20, 80, size=(h, w, 3), dtype=np.uint8)
        # Draw a bright rectangle that shifts right by dx each frame
        x = 50 + i * dx
        frame[60:100, x : x + 40] = [200, 220, 200]
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLKFlowTrackerConstruction:
    def test_default_name(self):
        t = LKFlowTracker()
        assert t.name == "LKFlow"

    def test_custom_name(self):
        t = LKFlowTracker(name="MyLK")
        assert t.name == "MyLK"

    def test_has_failure_detection(self):
        assert LKFlowTracker().has_failure_detection is True

    def test_lost_initially_false(self):
        assert LKFlowTracker().lost is False

    def test_repr(self):
        t = LKFlowTracker()
        assert "LKFlow" in repr(t)


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------

class TestLKFlowInitialize:
    def test_initialize_sets_state(self):
        t = LKFlowTracker()
        frames = _make_sequence()
        t.initialize(frames[0], (50, 60, 40, 40))
        assert t._prev_gray is not None
        assert t._bbox is not None
        assert not t.lost

    def test_initialize_invalid_bbox_raises(self):
        t = LKFlowTracker()
        frame = _make_textured_frame()
        with pytest.raises(ValueError):
            t.initialize(frame, (10, 10, 0, 40))
        with pytest.raises(ValueError):
            t.initialize(frame, (10, 10, 40, -5))

    def test_initialize_resets_lost_flag(self):
        t = LKFlowTracker(min_points=1)
        frames = _make_sequence()
        t.initialize(frames[0], (50, 60, 40, 40))
        t.lost = True
        t.initialize(frames[0], (50, 60, 40, 40))
        assert not t.lost

    def test_initialize_grayscale_frame(self):
        t = LKFlowTracker()
        gray = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        # Should not raise
        t.initialize(gray, (50, 60, 40, 40))


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestLKFlowUpdate:
    def test_update_without_initialize_raises(self):
        t = LKFlowTracker()
        frame = _make_textured_frame()
        with pytest.raises(RuntimeError):
            t.update(frame)

    def test_update_returns_four_tuple(self):
        t = LKFlowTracker()
        frames = _make_sequence()
        t.initialize(frames[0], (50, 60, 40, 40))
        bbox = t.update(frames[1])
        assert len(bbox) == 4
        assert all(isinstance(v, float) for v in bbox)

    def test_update_tracks_moving_target(self):
        """Track a bright patch drifting right; predicted x should increase."""
        dx = 3
        frames = _make_sequence(n_frames=15, dx=dx)
        t = LKFlowTracker(fb_thresh=2.0, min_points=3)
        t.initialize(frames[0], (50, 60, 40, 40))
        x_prev = 50.0
        for frame in frames[1:]:
            bx, by, bw, bh = t.update(frame)
            # x should generally move rightward (not necessarily every frame,
            # but the final position should be ahead of the initial)
        assert bx > 50.0  # patch has drifted right

    def test_update_no_ransac(self):
        """Tracker with use_ransac=False should still run without errors."""
        t = LKFlowTracker(use_ransac=False, min_points=3)
        frames = _make_sequence()
        t.initialize(frames[0], (50, 60, 40, 40))
        for f in frames[1:6]:
            bbox = t.update(f)
            assert len(bbox) == 4

    def test_update_preserves_approximate_size(self):
        """On a static frame the bounding box should not grow or shrink wildly."""
        t = LKFlowTracker(min_points=3)
        frames = _make_sequence(dx=1)
        t.initialize(frames[0], (50, 60, 40, 40))
        for f in frames[1:5]:
            bx, by, bw, bh = t.update(f)
        # Width should remain roughly 40 ± 30 pixels
        assert 10 < bw < 120


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------

class TestLKFlowFailureDetection:
    def test_failure_on_blank_frame(self):
        """A completely uniform frame has no trackable corners — should flag lost."""
        t = LKFlowTracker(min_points=5)
        textured = _make_textured_frame()
        t.initialize(textured, (50, 60, 40, 40))

        blank = _make_frame()  # uniform color — no corners
        _ = t.update(blank)    # may or may not lose here
        _ = t.update(blank)    # second blank frame should definitely lose
        assert t.lost

    def test_returns_last_valid_bbox_on_failure(self):
        t = LKFlowTracker(min_points=5)
        textured = _make_textured_frame()
        t.initialize(textured, (50, 60, 40, 40))

        blank = _make_frame()
        t.update(blank)
        t.update(blank)

        assert t.lost
        bbox = t.update(blank)
        assert len(bbox) == 4  # still returns a 4-tuple


# ---------------------------------------------------------------------------
# _to_gray helper
# ---------------------------------------------------------------------------

class TestToGray:
    def test_bgr_to_gray(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        g = _to_gray(bgr)
        assert g.ndim == 2

    def test_bgra_to_gray(self):
        bgra = np.zeros((10, 10, 4), dtype=np.uint8)
        g = _to_gray(bgra)
        assert g.ndim == 2

    def test_already_gray(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        g = _to_gray(gray)
        assert g is gray


# ---------------------------------------------------------------------------
# _median_scale helper
# ---------------------------------------------------------------------------

class TestMedianScale:
    def test_scale_one_when_no_motion(self):
        pts = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
        assert abs(_median_scale(pts, pts) - 1.0) < 0.05

    def test_scale_two_when_doubled(self):
        old = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
        new = old * 2.0
        s = _median_scale(old, new)
        assert abs(s - 2.0) < 0.1

    def test_single_point_fallback(self):
        pts = np.array([[5, 5]], dtype=np.float32)
        assert _median_scale(pts, pts) == 1.0

    def test_empty_fallback(self):
        pts = np.empty((0, 2), dtype=np.float32)
        assert _median_scale(pts, pts) == 1.0

    def test_clamped_to_bounds(self):
        old = np.array([[0, 0], [1, 0]], dtype=np.float32)
        new = np.array([[0, 0], [1000, 0]], dtype=np.float32)  # extreme scale
        s = _median_scale(old, new)
        assert s <= 10.0


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    def test_lkflow_in_registry(self):
        assert "LKFlow" in TRACKER_REGISTRY

    def test_build_tracker_lkflow(self):
        t = build_tracker("LKFlow")
        assert isinstance(t, LKFlowTracker)
        assert t.name == "LKFlow"

    def test_build_tracker_with_params(self):
        t = build_tracker("LKFlow", fb_thresh=2.0, use_ransac=False)
        assert t.fb_thresh == 2.0
        assert t.use_ransac is False


# ---------------------------------------------------------------------------
# Benchmark engine integration
# ---------------------------------------------------------------------------

class TestBenchmarkEngineIntegration:
    def test_full_benchmark_run(self):
        """LKFlowTracker completes a full benchmark run with SyntheticDataset."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=2, num_frames=30, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(LKFlowTracker(min_points=3), dataset, dataset_name="Synthetic")

        assert result.tracker_name == "LKFlow"
        assert result.dataset_name == "Synthetic"
        assert len(result.sequence_results) == 2
        assert 0.0 <= result.mean_iou <= 1.0
        assert result.mean_fps > 0.0

    def test_benchmark_produces_accuracy_metrics(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=1, num_frames=20, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(LKFlowTracker(min_points=3), dataset, dataset_name="Syn")

        seq = result.sequence_results[0]
        assert seq.accuracy_metrics is not None
        assert 0.0 <= seq.accuracy_metrics.success_auc <= 1.0
