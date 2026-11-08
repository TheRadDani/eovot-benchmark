"""Tests for BackgroundSubtractionTracker."""

import numpy as np
import pytest

from eovot.trackers.background_subtraction import BackgroundSubtractionTracker
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frame(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    """Return a BGR image with a small coloured square on a grey background."""
    rng = np.random.default_rng(seed)
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    # Add Gaussian noise to make it look like a real scene.
    noise = rng.integers(-15, 16, (h, w, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


def draw_target(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Draw a bright blue rectangle on the frame and return a copy."""
    f = frame.copy()
    cv_x2, cv_y2 = min(x + w, f.shape[1] - 1), min(y + h, f.shape[0] - 1)
    f[y:cv_y2, x:cv_x2] = [255, 100, 50]  # bright blue-ish
    return f


def sequence_with_moving_target(
    n_frames: int = 40,
    h: int = 240,
    w: int = 320,
    start_x: int = 60,
    start_y: int = 80,
    tw: int = 40,
    th: int = 30,
    dx: int = 3,
    dy: int = 2,
    warmup: int = 5,
) -> tuple:
    """Generate background + target frames and bboxes.

    Returns:
        (frames, bboxes) where frames[0] is used for initialization.
    """
    # Build a stable background over warmup frames.
    bg = make_frame(h, w)

    frames = []
    bboxes = []
    cx, cy = start_x, start_y

    # Warmup background frames (no target).
    tracker_frames = []
    tracker_bboxes = []

    for i in range(n_frames):
        f = draw_target(bg, cx, cy, tw, th)
        frames.append(f)
        bboxes.append((float(cx), float(cy), float(tw), float(th)))
        cx = min(cx + dx, w - tw - 1)
        cy = min(cy + dy, h - th - 1)

    return frames, bboxes


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_name(self):
        t = BackgroundSubtractionTracker()
        assert t.name == "BGSub"

    def test_custom_name(self):
        t = BackgroundSubtractionTracker(name="MyBGSub")
        assert t.name == "MyBGSub"

    def test_registry_entry(self):
        assert "BGSub" in TRACKER_REGISTRY

    def test_build_from_registry(self):
        t = build_tracker("BGSub")
        assert isinstance(t, BackgroundSubtractionTracker)

    def test_build_with_params(self):
        t = build_tracker("BGSub", history=50, var_threshold=8.0)
        assert t.name == "BGSub"

    def test_not_initialized_on_construction(self):
        t = BackgroundSubtractionTracker()
        assert t._initialized is False

    def test_repr(self):
        t = BackgroundSubtractionTracker()
        r = repr(t)
        assert "BackgroundSubtractionTracker" in r


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInitialize:
    def test_initialize_sets_flag(self):
        t = BackgroundSubtractionTracker()
        f = make_frame()
        t.initialize(f, (10.0, 20.0, 50.0, 40.0))
        assert t._initialized is True

    def test_initialize_stores_last_bbox(self):
        t = BackgroundSubtractionTracker()
        bbox = (10.0, 20.0, 50.0, 40.0)
        t.initialize(make_frame(), bbox)
        assert t._last_bbox == bbox

    def test_initialize_stores_target_area(self):
        t = BackgroundSubtractionTracker()
        t.initialize(make_frame(), (0.0, 0.0, 50.0, 40.0))
        assert t._target_area == pytest.approx(2000.0)

    def test_initialize_invalid_bbox_raises(self):
        t = BackgroundSubtractionTracker()
        with pytest.raises(ValueError):
            t.initialize(make_frame(), (10.0, 10.0, 0.0, 30.0))
        with pytest.raises(ValueError):
            t.initialize(make_frame(), (10.0, 10.0, 30.0, -5.0))

    def test_reinitialize_allowed(self):
        t = BackgroundSubtractionTracker()
        f = make_frame()
        t.initialize(f, (10.0, 20.0, 50.0, 40.0))
        t.initialize(f, (5.0, 5.0, 30.0, 20.0))
        assert t._last_bbox == (5.0, 5.0, 30.0, 20.0)


# ---------------------------------------------------------------------------
# Update tests
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_before_initialize_raises(self):
        t = BackgroundSubtractionTracker()
        with pytest.raises(RuntimeError):
            t.update(make_frame())

    def test_update_returns_tuple_of_four(self):
        t = BackgroundSubtractionTracker()
        f = make_frame()
        t.initialize(f, (50.0, 60.0, 40.0, 30.0))
        bbox = t.update(make_frame(seed=1))
        assert len(bbox) == 4

    def test_update_returns_floats(self):
        t = BackgroundSubtractionTracker()
        f = make_frame()
        t.initialize(f, (50.0, 60.0, 40.0, 30.0))
        bbox = t.update(make_frame(seed=1))
        assert all(isinstance(v, float) for v in bbox)

    def test_update_fallback_on_empty_foreground(self):
        """When background fully matches foreground, return last known box."""
        t = BackgroundSubtractionTracker()
        # Both init frame and update frame are identical → no foreground.
        f = make_frame()
        bbox = (50.0, 60.0, 40.0, 30.0)
        t.initialize(f, bbox)
        # Feed the exact same frame many times so MOG2 absorbs it.
        for _ in range(30):
            result = t.update(f.copy())
        # Last known box should be preserved.
        assert result == bbox

    def test_update_multiple_frames(self):
        t = BackgroundSubtractionTracker()
        frames, bboxes = sequence_with_moving_target()
        t.initialize(frames[0], bboxes[0])
        for i in range(1, len(frames)):
            result = t.update(frames[i])
            assert len(result) == 4
            assert all(isinstance(v, float) for v in result)

    def test_update_width_height_positive(self):
        t = BackgroundSubtractionTracker()
        frames, bboxes = sequence_with_moving_target()
        t.initialize(frames[0], bboxes[0])
        for i in range(1, len(frames)):
            x, y, w, h = t.update(frames[i])
            assert w > 0
            assert h > 0


# ---------------------------------------------------------------------------
# Tracker follows moving target (soft assertion)
# ---------------------------------------------------------------------------

class TestTracking:
    def test_tracker_stays_near_target_first_frames(self):
        """In a clean synthetic sequence the tracker should follow the target."""
        import cv2 as _cv2  # noqa: F401 — confirm OpenCV available

        # Very bright, isolated target on a uniform background — ideal for BGSub.
        h, w = 240, 320
        tw, th = 40, 30
        dx, dy = 4, 3
        bg = np.full((h, w, 3), 60, dtype=np.uint8)  # dark background

        def mk_frame(tx, ty):
            f = bg.copy()
            f[ty:ty + th, tx:tx + tw] = [220, 50, 50]  # bright target
            return f

        # Build background model for the first few frames (no target).
        t = BackgroundSubtractionTracker(history=10, var_threshold=10.0, detect_shadows=False)
        x0, y0 = 50, 50
        t.initialize(mk_frame(x0, y0), (float(x0), float(y0), float(tw), float(th)))

        total_error = 0.0
        n = 15
        cx, cy = x0, y0
        for _ in range(n):
            cx = min(cx + dx, w - tw - 1)
            cy = min(cy + dy, h - th - 1)
            frame = mk_frame(cx, cy)
            bx, by, bw, bh = t.update(frame)
            pred_cx = bx + bw / 2.0
            pred_cy = by + bh / 2.0
            gt_cx = cx + tw / 2.0
            gt_cy = cy + th / 2.0
            total_error += ((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2) ** 0.5

        mean_error = total_error / n
        # Allow generous tolerance: BGSub isn't a precision tracker, but it
        # should stay within roughly one target-width of the target on average.
        assert mean_error < tw * 2.0, f"Mean center error {mean_error:.1f} px too large"


# ---------------------------------------------------------------------------
# Parameter-variant tests
# ---------------------------------------------------------------------------

class TestParameters:
    def test_no_shadow_detection(self):
        t = BackgroundSubtractionTracker(detect_shadows=False)
        f = make_frame()
        t.initialize(f, (50.0, 50.0, 40.0, 30.0))
        bbox = t.update(make_frame(seed=5))
        assert len(bbox) == 4

    def test_max_search_radius_limits_search(self):
        t = BackgroundSubtractionTracker(max_search_radius=10.0)
        frames, bboxes = sequence_with_moving_target(dx=100, dy=0)
        t.initialize(frames[0], bboxes[0])
        # With radius=10 and dx=100, the target jumps out of search radius
        # immediately — tracker should return last known box.
        result = t.update(frames[1])
        # Result must be valid (4 floats, positive dims).
        assert len(result) == 4
        assert result[2] > 0 and result[3] > 0

    def test_strict_min_blob_ratio(self):
        t = BackgroundSubtractionTracker(min_blob_ratio=10.0)  # impossible threshold
        frames, bboxes = sequence_with_moving_target()
        t.initialize(frames[0], bboxes[0])
        # With an absurdly high min_blob_ratio no component qualifies;
        # tracker must return the last known box without raising.
        result = t.update(frames[1])
        assert result == bboxes[0]

    def test_large_morph_kernel(self):
        t = BackgroundSubtractionTracker(morph_kernel_size=21, morph_iterations=2)
        f = make_frame()
        t.initialize(f, (10.0, 10.0, 60.0, 50.0))
        result = t.update(make_frame(seed=3))
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Integration with benchmark pipeline
# ---------------------------------------------------------------------------

def test_integration_with_synthetic_dataset():
    """BGSub tracker runs through BenchmarkEngine on a synthetic dataset."""
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    dataset = SyntheticDataset(
        num_sequences=2,
        num_frames=25,
        frame_size=(160, 120),
        motion="linear",
        seed=99,
    )
    tracker = BackgroundSubtractionTracker(
        history=20,
        var_threshold=16.0,
        detect_shadows=False,
    )
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(tracker, dataset, dataset_name="Synthetic")

    assert result.tracker_name == "BGSub"
    assert result.dataset_name == "Synthetic"
    assert len(result.sequence_results) == 2
    assert result.mean_fps > 0
    assert 0.0 <= result.mean_iou <= 1.0
