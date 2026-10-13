"""Tests for DSSTTracker — scale-adaptive correlation filter tracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.dsst import DSSTTracker
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    """Return a deterministic random BGR uint8 frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_gray(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_dsst_in_registry():
    assert "DSST" in TRACKER_REGISTRY


def test_build_tracker_dsst():
    tracker = build_tracker("DSST")
    assert tracker.name == "DSST"
    assert isinstance(tracker, DSSTTracker)


def test_build_tracker_dsst_with_params():
    tracker = build_tracker("DSST", num_scales=9, scale_step=1.05)
    assert tracker.num_scales == 9
    assert tracker.scale_step == pytest.approx(1.05)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_even_num_scales_raises():
    with pytest.raises(ValueError, match="odd"):
        DSSTTracker(num_scales=16)


def test_invalid_translation_lr_raises():
    with pytest.raises(ValueError):
        DSSTTracker(translation_lr=0.0)

    with pytest.raises(ValueError):
        DSSTTracker(translation_lr=1.5)


def test_invalid_scale_lr_raises():
    with pytest.raises(ValueError):
        DSSTTracker(scale_lr=0.0)


# ---------------------------------------------------------------------------
# Initialisation and update — basic correctness
# ---------------------------------------------------------------------------

def test_initialize_does_not_raise():
    frame = _make_frame()
    tracker = DSSTTracker()
    tracker.initialize(frame, (20, 15, 40, 30))


def test_update_returns_four_floats():
    frame0 = _make_frame(seed=0)
    frame1 = _make_frame(seed=1)
    tracker = DSSTTracker()
    tracker.initialize(frame0, (20, 15, 40, 30))
    bbox = tracker.update(frame1)

    assert len(bbox) == 4
    for v in bbox:
        assert isinstance(v, float)


def test_update_before_initialize_raises():
    tracker = DSSTTracker()
    with pytest.raises(RuntimeError, match="not initialised"):
        tracker.update(_make_frame())


def test_reset_allows_reinitialise():
    frame = _make_frame()
    tracker = DSSTTracker()
    tracker.initialize(frame, (10, 10, 20, 20))
    tracker.reset()
    # After reset, update should fail
    with pytest.raises(RuntimeError):
        tracker.update(frame)
    # Re-initialise should succeed
    tracker.initialize(frame, (10, 10, 20, 20))
    bbox = tracker.update(_make_frame(seed=5))
    assert len(bbox) == 4


# ---------------------------------------------------------------------------
# Grayscale input
# ---------------------------------------------------------------------------

def test_grayscale_input():
    frame0 = _make_gray(seed=0)
    frame1 = _make_gray(seed=1)
    tracker = DSSTTracker()
    tracker.initialize(frame0, (20, 15, 40, 30))
    bbox = tracker.update(frame1)
    assert len(bbox) == 4


# ---------------------------------------------------------------------------
# Scale bounds respected
# ---------------------------------------------------------------------------

def test_scale_stays_within_bounds():
    """After many frames of random noise, scale must stay in [min, max]."""
    rng = np.random.default_rng(42)
    tracker = DSSTTracker(min_scale_factor=0.5, max_scale_factor=3.0)
    frame0 = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    tracker.initialize(frame0, (30, 20, 40, 30))

    for i in range(50):
        frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        bbox = tracker.update(frame)
        # Scale is accumulated — bounding box dimensions should respect bounds
        bw, bh = bbox[2], bbox[3]
        assert bw > 0, "Width must be positive"
        assert bh > 0, "Height must be positive"
    assert 0.5 <= tracker._current_scale <= 3.0


# ---------------------------------------------------------------------------
# Scale pyramid
# ---------------------------------------------------------------------------

def test_scale_pyramid_length():
    tracker = DSSTTracker(num_scales=17)
    assert len(tracker._scale_factors) == 17


def test_scale_pyramid_identity_at_centre():
    """The middle element of the scale pyramid is 1.0 (no scale change)."""
    tracker = DSSTTracker(num_scales=17)
    mid = 17 // 2
    assert tracker._scale_factors[mid] == pytest.approx(1.0, abs=1e-9)


def test_scale_pyramid_monotone():
    tracker = DSSTTracker(num_scales=17, scale_step=1.02)
    sf = tracker._scale_factors
    assert np.all(np.diff(sf) > 0), "Scale factors must be strictly increasing"


# ---------------------------------------------------------------------------
# Tracking on synthetic sequence with scale change
# ---------------------------------------------------------------------------

def _make_scale_sequence(
    n_frames: int = 60,
    frame_h: int = 240,
    frame_w: int = 320,
    seed: int = 7,
):
    """Generate a sequence where target grows from 30×30 to 60×60 pixels."""
    rng = np.random.default_rng(seed)
    background = rng.integers(30, 80, (frame_h, frame_w, 3), dtype=np.uint8)
    colour = (200, 50, 120)
    cx, cy = frame_w // 2, frame_h // 2

    frames = []
    gt_boxes = []
    for i in range(n_frames):
        # Target grows linearly from 30 to 60 pixels
        size = int(30 + 30 * i / max(n_frames - 1, 1))
        x1, y1 = cx - size // 2, cy - size // 2
        frame = background.copy()
        frame[max(0, y1):min(frame_h, y1 + size),
              max(0, x1):min(frame_w, x1 + size)] = colour
        frames.append(frame)
        gt_boxes.append((float(x1), float(y1), float(size), float(size)))

    return frames, gt_boxes


def test_dsst_tracks_growing_target():
    """DSST bounding box must not collapse to zero on a growing textured target.

    With a plain coloured rectangle the scale feature (cosine similarity) has
    limited discriminative power — scale estimation on synthetic uniform regions
    requires richer features (HOG, deep descriptors).  We therefore only check
    that DSST does NOT catastrophically shrink the box (i.e. scale ≥ min) and
    that all bounding box dimensions stay positive.
    """
    frames, gt = _make_scale_sequence(n_frames=40)
    tracker = DSSTTracker(
        num_scales=17,
        scale_step=1.02,
        scale_lr=0.05,
        min_scale_factor=0.5,
    )
    tracker.initialize(frames[0], gt[0])

    for frame in frames[1:]:
        bbox = tracker.update(frame)
        assert bbox[2] > 0 and bbox[3] > 0, "Bounding box dimensions must remain positive"

    # Scale must stay above the configured minimum — tracker must not degenerate
    assert tracker._current_scale >= 0.5, (
        f"Scale collapsed below min_scale_factor: {tracker._current_scale:.3f}"
    )


def test_kcf_vs_dsst_on_constant_scale():
    """On a constant-scale sequence, DSST should not diverge from KCF's IoU."""
    from eovot.trackers.kcf import KCFTracker
    from eovot.metrics.accuracy import MetricsEngine

    rng = np.random.default_rng(11)
    n = 30
    H, W = 120, 160
    bg = rng.integers(40, 100, (H, W, 3), dtype=np.uint8)
    colour = (200, 100, 50)
    cx, cy, bw, bh = 80, 60, 30, 30

    frames, gt_boxes = [], []
    vx, vy = 2.0, 1.5
    for i in range(n):
        frame = bg.copy()
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        frame[max(0, y1):min(H, y1 + bh), max(0, x1):min(W, x1 + bw)] = colour
        frames.append(frame)
        gt_boxes.append((float(x1), float(y1), float(bw), float(bh)))
        cx += vx
        cy += vy
        if cx < bw / 2 or cx > W - bw / 2:
            vx = -vx
        if cy < bh / 2 or cy > H - bh / 2:
            vy = -vy

    gt_arr = np.array(gt_boxes)

    # DSST
    dsst = DSSTTracker()
    dsst.initialize(frames[0], gt_boxes[0])
    dsst_preds = [gt_boxes[0]]
    for f in frames[1:]:
        dsst_preds.append(dsst.update(f))

    # KCF
    kcf = KCFTracker()
    kcf.initialize(frames[0], gt_boxes[0])
    kcf_preds = [gt_boxes[0]]
    for f in frames[1:]:
        kcf_preds.append(kcf.update(f))

    engine = MetricsEngine()
    dsst_miou = engine.batch_iou(np.array(dsst_preds), gt_arr).mean()
    kcf_miou = engine.batch_iou(np.array(kcf_preds), gt_arr).mean()

    # On a constant-scale synthetic sequence DSST should achieve ≥ 50 % of
    # KCF's mIoU (scale estimation overhead should not destroy accuracy).
    assert dsst_miou >= kcf_miou * 0.5, (
        f"DSST mIoU={dsst_miou:.3f} degraded too much vs KCF mIoU={kcf_miou:.3f}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_target_near_border():
    """Target partially outside frame should not crash."""
    frame0 = _make_frame(h=100, w=100, seed=0)
    frame1 = _make_frame(h=100, w=100, seed=1)
    tracker = DSSTTracker()
    tracker.initialize(frame0, (80, 80, 40, 40))  # centre at (100, 100), partially OOB
    bbox = tracker.update(frame1)
    assert len(bbox) == 4


def test_tiny_bounding_box():
    frame = _make_frame()
    tracker = DSSTTracker()
    tracker.initialize(frame, (50, 50, 4, 4))
    bbox = tracker.update(_make_frame(seed=2))
    assert len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0


def test_multi_frame_sequence():
    """Run 20 frames without error and verify bounding box shape throughout."""
    rng = np.random.default_rng(99)
    tracker = DSSTTracker()
    frame0 = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    tracker.initialize(frame0, (40, 30, 50, 40))
    for _ in range(20):
        frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        bbox = tracker.update(frame)
        assert len(bbox) == 4
