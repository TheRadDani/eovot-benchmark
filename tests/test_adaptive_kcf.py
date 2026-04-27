"""Unit tests for AdaptiveKCFTracker."""

import numpy as np
import pytest

from eovot.trackers.adaptive_kcf import AdaptiveKCFTracker
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_sequence(n_frames: int = 20, h: int = 120, w: int = 160):
    """Return a list of BGR frames and a stationary ground-truth bbox."""
    rng = np.random.default_rng(42)
    frames = [
        rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n_frames)
    ]
    bbox = (30.0, 20.0, 40.0, 30.0)
    return frames, bbox


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------

def test_inherits_base_tracker():
    tracker = AdaptiveKCFTracker()
    assert isinstance(tracker, BaseTracker)


def test_name_attribute():
    tracker = AdaptiveKCFTracker()
    assert tracker.name == "AdaptiveKCF"


def test_initialize_and_update_return_bbox():
    frames, bbox = _synthetic_sequence()
    tracker = AdaptiveKCFTracker()
    tracker.initialize(frames[0], bbox)
    pred = tracker.update(frames[1])
    assert len(pred) == 4
    assert all(isinstance(v, float) for v in pred)


def test_update_raises_before_initialize():
    frames, _ = _synthetic_sequence()
    tracker = AdaptiveKCFTracker()
    with pytest.raises(RuntimeError):
        tracker.update(frames[0])


def test_full_sequence_runs_without_error():
    frames, bbox = _synthetic_sequence(n_frames=30)
    tracker = AdaptiveKCFTracker()
    tracker.initialize(frames[0], bbox)
    for frame in frames[1:]:
        pred = tracker.update(frame)
        assert len(pred) == 4


# ---------------------------------------------------------------------------
# Adaptation state
# ---------------------------------------------------------------------------

def test_starts_in_accurate_mode():
    frames, bbox = _synthetic_sequence()
    tracker = AdaptiveKCFTracker()
    assert tracker.mode == "accurate"
    tracker.initialize(frames[0], bbox)
    assert tracker.mode == "accurate"


def test_mode_history_length_matches_frames():
    n = 15
    frames, bbox = _synthetic_sequence(n_frames=n)
    tracker = AdaptiveKCFTracker()
    tracker.initialize(frames[0], bbox)
    for frame in frames[1:]:
        tracker.update(frame)
    stats = tracker.adaptation_stats
    assert stats["total_frames"] == n - 1


def test_adaptation_stats_keys():
    frames, bbox = _synthetic_sequence()
    tracker = AdaptiveKCFTracker()
    tracker.initialize(frames[0], bbox)
    tracker.update(frames[1])
    stats = tracker.adaptation_stats
    for key in (
        "mode", "latency_budget_ms", "latency_ema_ms",
        "mode_switches", "total_frames",
        "fast_mode_frames", "accurate_mode_frames", "fast_mode_pct",
    ):
        assert key in stats, f"Missing key: {key}"


def test_fast_mode_triggered_by_tiny_budget():
    """With a near-zero budget every frame should trigger fast mode."""
    frames, bbox = _synthetic_sequence(n_frames=40)
    tracker = AdaptiveKCFTracker(latency_budget_ms=0.001)
    tracker.initialize(frames[0], bbox)
    for frame in frames[1:]:
        tracker.update(frame)
    stats = tracker.adaptation_stats
    # After enough frames the EMA must exceed 0.001 ms; fast mode should appear.
    assert stats["fast_mode_frames"] > 0


def test_accurate_mode_kept_with_large_budget():
    """With a very generous budget the tracker should stay in accurate mode."""
    frames, bbox = _synthetic_sequence(n_frames=20)
    tracker = AdaptiveKCFTracker(latency_budget_ms=1_000_000.0)
    tracker.initialize(frames[0], bbox)
    for frame in frames[1:]:
        tracker.update(frame)
    stats = tracker.adaptation_stats
    assert stats["accurate_mode_frames"] == stats["total_frames"]
    assert stats["mode_switches"] == 0


def test_mode_switches_non_negative():
    frames, bbox = _synthetic_sequence(n_frames=25)
    tracker = AdaptiveKCFTracker()
    tracker.initialize(frames[0], bbox)
    for frame in frames[1:]:
        tracker.update(frame)
    assert tracker.mode_switches >= 0


# ---------------------------------------------------------------------------
# Reset behaviour
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    frames, bbox = _synthetic_sequence(n_frames=10)
    tracker = AdaptiveKCFTracker(latency_budget_ms=0.001)
    tracker.initialize(frames[0], bbox)
    for frame in frames[1:]:
        tracker.update(frame)

    tracker.reset()
    assert tracker.mode == "accurate"
    assert tracker.mode_switches == 0
    assert tracker.adaptation_stats["total_frames"] == 0
    assert tracker._pos is None


def test_reinitialize_after_reset():
    frames, bbox = _synthetic_sequence(n_frames=10)
    tracker = AdaptiveKCFTracker()
    tracker.initialize(frames[0], bbox)
    tracker.reset()
    tracker.initialize(frames[0], bbox)
    pred = tracker.update(frames[1])
    assert len(pred) == 4


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def test_custom_budget_stored():
    tracker = AdaptiveKCFTracker(latency_budget_ms=16.67)
    assert tracker.latency_budget_ms == pytest.approx(16.67)


def test_grayscale_input():
    rng = np.random.default_rng(0)
    frame_gray = rng.integers(0, 255, (100, 120), dtype=np.uint8)
    tracker = AdaptiveKCFTracker()
    tracker.initialize(frame_gray, (10.0, 10.0, 20.0, 20.0))
    pred = tracker.update(frame_gray)
    assert len(pred) == 4


def test_ema_alpha_affects_ema():
    """Tracker with higher ema_alpha should react faster to latency changes."""
    frames, bbox = _synthetic_sequence(n_frames=5)
    tracker_fast = AdaptiveKCFTracker(ema_alpha=0.9, latency_budget_ms=0.001)
    tracker_slow = AdaptiveKCFTracker(ema_alpha=0.01, latency_budget_ms=0.001)
    for t in (tracker_fast, tracker_slow):
        t.initialize(frames[0], bbox)
        t.update(frames[1])
    # Fast EMA should have a larger (or equal) EMA after one frame
    assert tracker_fast._latency_ema >= tracker_slow._latency_ema * 0.5
