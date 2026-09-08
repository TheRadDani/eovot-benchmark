"""Tests for AdaptiveBudgetTracker — closed-loop latency-adaptive resolution control."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.trackers.adaptive_budget import AdaptationEvent, AdaptiveBudgetTracker
from eovot.trackers.base import BaseTracker, BBox
from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FixedTracker(BaseTracker):
    """Deterministic tracker that always returns the init bbox."""

    def __init__(self, name: str = "Fixed") -> None:
        super().__init__(name=name)
        self._bbox: BBox = (0, 0, 10, 10)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        return self._bbox


class _SlowTracker(BaseTracker):
    """Tracker with configurable artificial delay (simulates heavy processing)."""

    def __init__(self, delay_ms: float = 50.0) -> None:
        super().__init__(name="SlowTracker")
        self._delay = delay_ms / 1000.0
        self._bbox: BBox = (0, 0, 10, 10)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        time.sleep(self._delay)
        return self._bbox


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

def test_invalid_budget_raises():
    with pytest.raises(ValueError, match="budget_ms must be positive"):
        AdaptiveBudgetTracker(_FixedTracker(), budget_ms=-5.0)


def test_invalid_min_scale_raises():
    with pytest.raises(ValueError, match="min_scale must be in"):
        AdaptiveBudgetTracker(_FixedTracker(), budget_ms=30.0, min_scale=0.0)


def test_invalid_scale_step_raises():
    with pytest.raises(ValueError, match="scale_step must be in"):
        AdaptiveBudgetTracker(_FixedTracker(), budget_ms=30.0, scale_step=1.5)


def test_invalid_restore_headroom_raises():
    with pytest.raises(ValueError, match="restore_headroom must be in"):
        AdaptiveBudgetTracker(_FixedTracker(), budget_ms=30.0, restore_headroom=1.1)


def test_name_includes_budget():
    tracker = AdaptiveBudgetTracker(_FixedTracker(name="KCF"), budget_ms=20.0)
    assert "adaptive" in tracker.name
    assert "20ms" in tracker.name
    assert "KCF" in tracker.name


def test_repr_shows_config():
    tracker = AdaptiveBudgetTracker(_FixedTracker(), budget_ms=25.0, min_scale=0.3)
    r = repr(tracker)
    assert "25.0" in r
    assert "0.3" in r


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_initialize_resets_state():
    tracker = AdaptiveBudgetTracker(_FixedTracker(), budget_ms=30.0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (10.0, 10.0, 20.0, 20.0)
    tracker.initialize(frame, bbox)
    assert tracker.current_scale == 1.0
    assert tracker.adaptation_events == []


def test_initialize_twice_resets_events():
    tracker = AdaptiveBudgetTracker(_FixedTracker(), budget_ms=30.0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (5.0, 5.0, 15.0, 15.0)
    tracker.initialize(frame, bbox)
    # Manually append a fake event to confirm reset
    tracker.adaptation_events.append(
        AdaptationEvent(1, 1.0, 0.8, 40.0, 30.0, "down")
    )
    tracker.initialize(frame, bbox)
    assert tracker.adaptation_events == []


def test_update_before_initialize_raises():
    tracker = AdaptiveBudgetTracker(_FixedTracker(), budget_ms=30.0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="initialize"):
        tracker.update(frame)


# ---------------------------------------------------------------------------
# Bbox coordinate round-trip
# ---------------------------------------------------------------------------

def test_bbox_roundtrip_at_full_scale():
    """With scale=1.0 the returned bbox should equal what the inner tracker returns."""
    inner = _FixedTracker()
    tracker = AdaptiveBudgetTracker(inner, budget_ms=1000.0)  # huge budget → no adaptation
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = (20.0, 30.0, 40.0, 50.0)
    tracker.initialize(frame, bbox)
    result = tracker.update(frame)
    assert len(result) == 4
    x, y, w, h = result
    assert x == pytest.approx(20.0, abs=1.0)
    assert y == pytest.approx(30.0, abs=1.0)
    assert w == pytest.approx(40.0, abs=1.0)
    assert h == pytest.approx(50.0, abs=1.0)


def test_bbox_unscaled_back_to_original():
    """Bbox returned is always in original-resolution coordinates."""
    tracker = AdaptiveBudgetTracker(_FixedTracker(), budget_ms=1000.0, scale_step=0.5)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (10.0, 10.0, 30.0, 30.0)
    tracker.initialize(frame, bbox)
    # Force a lower scale by overriding _scale to test unscaling
    tracker._scale = 0.5
    tracker._initialized = True
    # Re-init inner tracker at new scale
    tracker._tracker.initialize(
        tracker._scale_frame(frame, 0.5),
        tracker._scale_bbox(bbox, 0.5),
    )
    result = tracker.update(frame)
    x, y, w, h = result
    # The inner tracker returns the init bbox in scaled space; unscaling should
    # return coordinates close to original (within float rounding)
    assert x == pytest.approx(10.0, abs=2.0)
    assert y == pytest.approx(10.0, abs=2.0)


# ---------------------------------------------------------------------------
# Adaptation logic
# ---------------------------------------------------------------------------

def test_no_adaptation_when_budget_is_huge():
    """With a very large budget no scale adjustments should occur."""
    tracker = AdaptiveBudgetTracker(_FixedTracker(), budget_ms=1e6, window_size=3)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracker.initialize(frame, (5.0, 5.0, 20.0, 20.0))
    for _ in range(20):
        tracker.update(frame)
    assert tracker.adaptation_events == []
    assert tracker.current_scale == 1.0


def test_adaptation_event_fields():
    """An AdaptationEvent carries correct metadata."""
    event = AdaptationEvent(
        frame_idx=12,
        old_scale=1.0,
        new_scale=0.7,
        trigger_latency_ms=45.0,
        budget_ms=30.0,
        direction="down",
    )
    assert event.frame_idx == 12
    assert event.old_scale == 1.0
    assert event.new_scale == 0.7
    assert event.direction == "down"
    s = str(event)
    assert "↓" in s
    assert "12" in s


def test_adaptation_event_up_arrow():
    event = AdaptationEvent(1, 0.5, 0.65, 20.0, 30.0, "up")
    assert "↑" in str(event)


# ---------------------------------------------------------------------------
# AdaptationEvent __str__ smoke
# ---------------------------------------------------------------------------

def test_adaptation_summary_structure():
    tracker = AdaptiveBudgetTracker(_FixedTracker(), budget_ms=30.0)
    summary = tracker.adaptation_summary()
    assert "n_adaptations" in summary
    assert "n_down" in summary
    assert "n_up" in summary
    assert "min_scale_reached" in summary
    assert "final_scale" in summary
    assert "budget_ms" in summary
    assert summary["budget_ms"] == 30.0
    assert summary["n_adaptations"] == 0


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine via SyntheticDataset
# ---------------------------------------------------------------------------

def test_benchmark_engine_integration():
    """AdaptiveBudgetTracker works end-to-end with BenchmarkEngine."""
    from eovot.trackers.mosse import MOSSETracker

    tracker = AdaptiveBudgetTracker(MOSSETracker(), budget_ms=200.0)
    dataset = SyntheticDataset(num_sequences=2, num_frames=30, seed=99)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(tracker, dataset, dataset_name="Synthetic")

    assert result.mean_iou >= 0.0
    assert result.mean_fps > 0.0
    assert len(result.sequence_results) == 2


# ---------------------------------------------------------------------------
# Scale helper statics
# ---------------------------------------------------------------------------

def test_scale_frame_reduces_size():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    scaled = AdaptiveBudgetTracker._scale_frame(frame, 0.5)
    assert scaled.shape == (100, 150, 3)


def test_scale_frame_identity_at_one():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = AdaptiveBudgetTracker._scale_frame(frame, 1.0)
    assert result is frame  # identity — no copy


def test_scale_bbox_and_unscale_roundtrip():
    bbox = (10.0, 20.0, 30.0, 40.0)
    scaled = AdaptiveBudgetTracker._scale_bbox(bbox, 0.5)
    assert scaled == pytest.approx((5.0, 10.0, 15.0, 20.0))
    restored = AdaptiveBudgetTracker._unscale_bbox(scaled, 0.5)
    assert restored == pytest.approx(bbox)


def test_unscale_bbox_identity_at_one():
    bbox = (1.0, 2.0, 3.0, 4.0)
    result = AdaptiveBudgetTracker._unscale_bbox(bbox, 1.0)
    assert result is bbox  # identity — no copy
