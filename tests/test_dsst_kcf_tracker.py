"""Tests for the Scale-Adaptive KCF (DSST-KCF) tracker."""

import numpy as np
import pytest

from eovot.trackers.dsst_kcf import DSSKCFTracker
from eovot.trackers.base import BaseTracker
from eovot.trackers.registry import build_tracker, TRACKER_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _make_sequence(n_frames: int = 10, seed: int = 42):
    """Return (frames, init_bbox) with a slowly drifting target."""
    rng = np.random.default_rng(seed)
    frames = [_make_frame(seed=i) for i in range(n_frames)]
    init_bbox = (40.0, 40.0, 60.0, 50.0)
    return frames, init_bbox


# ---------------------------------------------------------------------------
# Interface / structural tests
# ---------------------------------------------------------------------------

def test_dsst_kcf_is_base_tracker():
    tracker = DSSKCFTracker()
    assert isinstance(tracker, BaseTracker)


def test_dsst_kcf_name():
    tracker = DSSKCFTracker()
    assert tracker.name == "DSST-KCF"


def test_dsst_kcf_registered():
    assert "DSST-KCF" in TRACKER_REGISTRY


def test_build_tracker_dsst_kcf():
    tracker = build_tracker("DSST-KCF")
    assert isinstance(tracker, DSSKCFTracker)


def test_build_tracker_with_params():
    tracker = build_tracker("DSST-KCF", n_scales=17, scale_step=1.03)
    assert tracker.n_scales == 17
    assert tracker.scale_step == pytest.approx(1.03)


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

def test_initialize_and_update_returns_bbox():
    tracker = DSSKCFTracker()
    frames, bbox = _make_sequence()
    tracker.initialize(frames[0], bbox)
    pred = tracker.update(frames[1])
    assert len(pred) == 4
    assert all(isinstance(v, float) for v in pred)


def test_update_before_init_raises():
    tracker = DSSKCFTracker()
    frame = _make_frame()
    with pytest.raises(RuntimeError):
        tracker.update(frame)


def test_reset_allows_reinit():
    tracker = DSSKCFTracker()
    frames, bbox = _make_sequence()
    tracker.initialize(frames[0], bbox)
    tracker.update(frames[1])
    tracker.reset()
    # After reset, update should raise
    with pytest.raises(RuntimeError):
        tracker.update(frames[2])
    # Re-initialize should work fine
    tracker.initialize(frames[0], bbox)
    pred = tracker.update(frames[1])
    assert len(pred) == 4


def test_predicted_bbox_finite():
    tracker = DSSKCFTracker()
    frames, bbox = _make_sequence(n_frames=15)
    tracker.initialize(frames[0], bbox)
    for frame in frames[1:]:
        pred = tracker.update(frame)
        assert all(np.isfinite(v) for v in pred), f"Non-finite bbox: {pred}"


def test_predicted_bbox_width_height_positive():
    tracker = DSSKCFTracker()
    frames, bbox = _make_sequence(n_frames=10)
    tracker.initialize(frames[0], bbox)
    for frame in frames[1:]:
        x, y, w, h = tracker.update(frame)
        assert w > 0, f"Width must be positive, got {w}"
        assert h > 0, f"Height must be positive, got {h}"


def test_scale_adapts_on_synthetic_zoom():
    """Scale filter should produce varying w/h when target size changes.

    We test this indirectly: the tracker should not crash and should produce
    bounding boxes with positive, finite dimensions when the apparent target
    size in a noisy sequence varies frame-to-frame.
    """
    tracker = DSSKCFTracker(n_scales=17)
    frames, bbox = _make_sequence(n_frames=20)
    tracker.initialize(frames[0], bbox)
    sizes = []
    for frame in frames[1:]:
        _, _, w, h = tracker.update(frame)
        sizes.append(w * h)
    # All sizes should be positive and finite
    assert all(s > 0 and np.isfinite(s) for s in sizes)


def test_multiple_sequences():
    """Tracker can be re-initialised across multiple sequences."""
    tracker = DSSKCFTracker()
    for seq_idx in range(3):
        frames, bbox = _make_sequence(seed=seq_idx * 10)
        tracker.initialize(frames[0], bbox)
        for frame in frames[1:5]:
            pred = tracker.update(frame)
            assert len(pred) == 4


# ---------------------------------------------------------------------------
# Configuration / parameter tests
# ---------------------------------------------------------------------------

def test_odd_n_scales():
    tracker = DSSKCFTracker(n_scales=11)
    frames, bbox = _make_sequence()
    tracker.initialize(frames[0], bbox)
    pred = tracker.update(frames[1])
    assert len(pred) == 4


def test_even_n_scales():
    tracker = DSSKCFTracker(n_scales=10)
    frames, bbox = _make_sequence()
    tracker.initialize(frames[0], bbox)
    pred = tracker.update(frames[1])
    assert len(pred) == 4


def test_small_scale_model_area():
    tracker = DSSKCFTracker(scale_model_max_area=64)
    frames, bbox = _make_sequence()
    tracker.initialize(frames[0], bbox)
    pred = tracker.update(frames[1])
    assert len(pred) == 4


def test_grayscale_input():
    tracker = DSSKCFTracker()
    frames, bbox = _make_sequence()
    gray_frames = [f[:, :, 0] for f in frames]  # (H, W) grayscale
    tracker.initialize(gray_frames[0], bbox)
    pred = tracker.update(gray_frames[1])
    assert len(pred) == 4


def test_target_near_frame_boundary():
    """Tracker should handle edge-of-frame initialisation gracefully."""
    tracker = DSSKCFTracker()
    frame = _make_frame()
    bbox = (1.0, 1.0, 20.0, 20.0)   # top-left corner
    tracker.initialize(frame, bbox)
    pred = tracker.update(_make_frame(seed=1))
    assert len(pred) == 4
    assert all(np.isfinite(v) for v in pred)


# ---------------------------------------------------------------------------
# Integration with benchmark engine
# ---------------------------------------------------------------------------

def test_integration_with_benchmark_engine():
    """DSST-KCF runs end-to-end through BenchmarkEngine on a synthetic dataset."""
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=0)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(DSSKCFTracker(), dataset, dataset_name="Synthetic")

    assert result.tracker_name == "DSST-KCF"
    assert result.dataset_name == "Synthetic"
    assert len(result.sequence_results) == 2
    assert 0.0 <= result.mean_iou <= 1.0
    assert result.mean_fps > 0.0
