"""Tests for AdaptiveKCFTracker.

Covers the core contract (initialize / update / reset), confidence
properties, multi-scale behaviour, and re-detection mechanics.
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.adaptive_kcf import AdaptiveKCFTracker, _PSR_HIGH, _PSR_LOW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(h: int = 120, w: int = 160, seed: int = 42) -> np.ndarray:
    """Synthetic BGR frame for testing (no external data required)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _moving_frame(
    h: int = 120, w: int = 160, dx: int = 0, dy: int = 0, seed: int = 0
) -> np.ndarray:
    """Frame with a bright rectangle shifted by (dx, dy)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    x0, y0 = 40 + dx, 30 + dy
    frame[max(0, y0):min(h, y0 + 20), max(0, x0):min(w, x0 + 30)] = 200
    return frame


INIT_BBOX = (40.0, 30.0, 30.0, 20.0)


# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------

def test_name():
    assert AdaptiveKCFTracker().name == "AdaptiveKCF"


def test_initialize_does_not_raise():
    tracker = AdaptiveKCFTracker()
    tracker.initialize(_frame(), INIT_BBOX)


def test_update_returns_four_element_tuple():
    tracker = AdaptiveKCFTracker()
    tracker.initialize(_frame(), INIT_BBOX)
    bbox = tracker.update(_frame(seed=1))
    assert len(bbox) == 4


def test_output_bbox_positive_size():
    tracker = AdaptiveKCFTracker()
    tracker.initialize(_frame(), INIT_BBOX)
    _, _, w, h = tracker.update(_frame(seed=2))
    assert w > 0
    assert h > 0


def test_update_before_init_raises():
    tracker = AdaptiveKCFTracker()
    with pytest.raises(RuntimeError, match="not initialised"):
        tracker.update(_frame())


# ---------------------------------------------------------------------------
# Confidence / PSR
# ---------------------------------------------------------------------------

def test_psr_nonnegative_after_update():
    tracker = AdaptiveKCFTracker()
    tracker.initialize(_frame(), INIT_BBOX)
    tracker.update(_frame())
    assert tracker.confidence >= 0.0


def test_psr_history_grows():
    tracker = AdaptiveKCFTracker()
    tracker.initialize(_frame(), INIT_BBOX)
    for i in range(7):
        tracker.update(_frame(seed=i))
    assert len(tracker.psr_history) == 7


def test_psr_history_returns_copy():
    tracker = AdaptiveKCFTracker()
    tracker.initialize(_frame(), INIT_BBOX)
    tracker.update(_frame())
    h1 = tracker.psr_history
    h1.append(999.0)
    assert len(tracker.psr_history) == 1  # internal list unchanged


def test_is_tracking_reliable_is_bool():
    tracker = AdaptiveKCFTracker()
    tracker.initialize(_frame(), INIT_BBOX)
    tracker.update(_frame())
    assert isinstance(tracker.is_tracking_reliable, bool)


def test_compute_psr_static_nonnegative():
    """PSR must be ≥ 0 for any non-degenerate response map."""
    rng = np.random.default_rng(0)
    response = rng.random((32, 32)).astype(np.float32)
    psr = AdaptiveKCFTracker._compute_psr(response)
    assert psr >= 0.0


def test_compute_psr_clear_gaussian_peak():
    """A narrow Gaussian peak on a noisy background should yield a high PSR."""
    rng = np.random.default_rng(7)
    response = rng.random((32, 32)).astype(np.float32) * 0.1  # low-level noise
    # Place a sharp Gaussian peak at (16, 16)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            val = np.exp(-(dy ** 2 + dx ** 2) / 2.0)
            response[16 + dy, 16 + dx] += val * 10.0
    psr = AdaptiveKCFTracker._compute_psr(response)
    assert psr > 5.0


def test_compute_psr_flat_returns_zero():
    """A flat response map has no sidelobe variance — PSR should be 0."""
    response = np.ones((32, 32), dtype=np.float32)
    psr = AdaptiveKCFTracker._compute_psr(response)
    assert psr == 0.0


# ---------------------------------------------------------------------------
# Multi-scale tracking
# ---------------------------------------------------------------------------

def test_single_scale_equivalent_to_num_scales_1():
    """num_scales=1 disables multi-scale search (only the current scale tested)."""
    tracker = AdaptiveKCFTracker(num_scales=1)
    tracker.initialize(_frame(), INIT_BBOX)
    bbox = tracker.update(_frame(seed=3))
    assert len(bbox) == 4


def test_scale_levels_count():
    tracker = AdaptiveKCFTracker(num_scales=5, scale_step=1.05)
    tracker.initialize(_frame(), INIT_BBOX)
    levels = tracker._scale_levels()
    assert len(levels) == 5


def test_scale_levels_centre_is_current():
    tracker = AdaptiveKCFTracker(num_scales=3, scale_step=1.05)
    tracker.initialize(_frame(), INIT_BBOX)
    levels = tracker._scale_levels()
    # Middle element should equal current_scale (1.0 at init time)
    assert abs(levels[1] - tracker._current_scale) < 1e-9


def test_output_bbox_size_positive_across_frames():
    """Width and height must remain positive across frames (scale must not collapse)."""
    tracker = AdaptiveKCFTracker(scale_step=1.05, num_scales=3)
    tracker.initialize(_frame(), INIT_BBOX)
    for i in range(10):
        _, _, w, h = tracker.update(_frame(seed=i))
        assert w > 0 and h > 0


# ---------------------------------------------------------------------------
# Adaptive learning rate
# ---------------------------------------------------------------------------

def test_adaptive_lr_high_psr():
    tracker = AdaptiveKCFTracker(learning_rate=0.1)
    assert tracker._adaptive_lr(_PSR_HIGH + 1.0) == pytest.approx(0.1)


def test_adaptive_lr_low_psr():
    tracker = AdaptiveKCFTracker(learning_rate=0.1)
    assert tracker._adaptive_lr(_PSR_LOW - 1.0) == pytest.approx(0.0)


def test_adaptive_lr_mid_psr():
    tracker = AdaptiveKCFTracker(learning_rate=0.1)
    mid = (_PSR_HIGH + _PSR_LOW) / 2.0
    lr = tracker._adaptive_lr(mid)
    assert 0.0 < lr < 0.1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    tracker = AdaptiveKCFTracker()
    tracker.initialize(_frame(), INIT_BBOX)
    tracker.update(_frame())
    tracker.reset()
    assert tracker._pos is None
    assert tracker._alphaf is None
    assert tracker.psr_history == []
    assert tracker._low_conf_count == 0


def test_reset_allows_reinit():
    tracker = AdaptiveKCFTracker()
    tracker.initialize(_frame(), INIT_BBOX)
    tracker.update(_frame())
    tracker.reset()
    tracker.initialize(_frame(seed=99), (20.0, 15.0, 25.0, 15.0))
    bbox = tracker.update(_frame(seed=100))
    assert len(bbox) == 4


def test_update_after_reset_raises():
    tracker = AdaptiveKCFTracker()
    tracker.initialize(_frame(), INIT_BBOX)
    tracker.reset()
    with pytest.raises(RuntimeError):
        tracker.update(_frame())


# ---------------------------------------------------------------------------
# Re-detection smoke test
# ---------------------------------------------------------------------------

def test_redetect_returns_valid_position():
    """Sustained low confidence should trigger re-detection without error."""
    tracker = AdaptiveKCFTracker(psr_threshold=1000.0, redetect_after=2)
    tracker.initialize(_frame(), INIT_BBOX)
    # psr_threshold is impossibly high so every frame triggers failure counter
    for i in range(4):
        bbox = tracker.update(_frame(seed=i))
        assert len(bbox) == 4
        _, _, w, h = bbox
        assert w > 0 and h > 0


# ---------------------------------------------------------------------------
# Multi-frame stress test
# ---------------------------------------------------------------------------

def test_multiple_frames_no_exception():
    tracker = AdaptiveKCFTracker()
    frames = [_moving_frame(dx=i * 2) for i in range(20)]
    tracker.initialize(frames[0], INIT_BBOX)
    for f in frames[1:]:
        bbox = tracker.update(f)
        assert len(bbox) == 4


def test_grayscale_input_accepted():
    """Tracker should handle 2-D (H, W) grayscale frames."""
    tracker = AdaptiveKCFTracker()
    gray = np.random.randint(0, 256, (120, 160), dtype=np.uint8)
    tracker.initialize(gray, INIT_BBOX)
    bbox = tracker.update(gray)
    assert len(bbox) == 4
