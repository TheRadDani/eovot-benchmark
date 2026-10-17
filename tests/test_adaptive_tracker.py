"""Tests for LatencyBudgetTracker (adaptive.py)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from eovot.trackers.adaptive import (
    AdaptationEvent,
    AdaptiveConfig,
    AdaptiveState,
    LatencyBudgetTracker,
)
from eovot.trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_BBOX: BBox = (10.0, 10.0, 40.0, 40.0)
_FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


class _InstantTracker(BaseTracker):
    """Tracker that returns in zero time."""

    def __init__(self):
        super().__init__("Instant")
        self.update_calls = 0

    def initialize(self, frame, bbox):
        self.update_calls = 0

    def update(self, frame) -> BBox:
        self.update_calls += 1
        return _BBOX


class _SlowTracker(BaseTracker):
    """Tracker that sleeps for ``sleep_ms`` milliseconds on each update."""

    def __init__(self, sleep_ms: float = 50.0):
        super().__init__("Slow")
        self._sleep_s = sleep_ms / 1_000.0
        self.update_calls = 0

    def initialize(self, frame, bbox):
        self.update_calls = 0

    def update(self, frame) -> BBox:
        time.sleep(self._sleep_s)
        self.update_calls += 1
        return _BBOX


def _run(tracker: BaseTracker, n: int = 20):
    """Initialize and run *n* update calls; return list of bboxes."""
    tracker.initialize(_FRAME, _BBOX)
    return [tracker.update(_FRAME) for _ in range(n)]


# ---------------------------------------------------------------------------
# AdaptiveConfig validation
# ---------------------------------------------------------------------------

class TestAdaptiveConfig:
    def test_defaults_valid(self):
        cfg = AdaptiveConfig(target_fps=30.0)
        assert cfg.min_skip == 1
        assert cfg.max_skip == 8
        assert cfg.cooldown_frames == 10

    def test_zero_fps_raises(self):
        with pytest.raises(ValueError):
            AdaptiveConfig(target_fps=0.0)

    def test_negative_fps_raises(self):
        with pytest.raises(ValueError):
            AdaptiveConfig(target_fps=-10.0)

    def test_min_skip_zero_raises(self):
        with pytest.raises(ValueError):
            AdaptiveConfig(target_fps=30.0, min_skip=0)

    def test_max_skip_less_than_min_raises(self):
        with pytest.raises(ValueError):
            AdaptiveConfig(target_fps=30.0, min_skip=4, max_skip=2)

    def test_bad_ema_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            AdaptiveConfig(target_fps=30.0, ema_alpha=0.0)

    def test_ema_alpha_one_is_valid(self):
        cfg = AdaptiveConfig(target_fps=30.0, ema_alpha=1.0)
        assert cfg.ema_alpha == 1.0

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError):
            AdaptiveConfig(target_fps=30.0, tolerance=-0.1)

    def test_zero_tolerance_valid(self):
        cfg = AdaptiveConfig(target_fps=30.0, tolerance=0.0)
        assert cfg.tolerance == 0.0


# ---------------------------------------------------------------------------
# LatencyBudgetTracker construction
# ---------------------------------------------------------------------------

class TestLatencyBudgetTrackerConstruction:
    def test_default_config_applied(self):
        t = LatencyBudgetTracker(_InstantTracker())
        assert t._cfg.target_fps == 30.0

    def test_name_wraps_inner_name(self):
        t = LatencyBudgetTracker(_InstantTracker(), AdaptiveConfig(target_fps=25.0))
        assert "Adaptive" in t.name
        assert "Instant" in t.name

    def test_underlying_tracker_accessible(self):
        inner = _InstantTracker()
        t = LatencyBudgetTracker(inner, AdaptiveConfig(target_fps=25.0))
        assert t.underlying_tracker is inner

    def test_initial_skip_set_to_min_after_initialize(self):
        cfg = AdaptiveConfig(target_fps=30.0, min_skip=2, max_skip=6)
        t = LatencyBudgetTracker(_InstantTracker(), cfg)
        t.initialize(_FRAME, _BBOX)
        assert t._skip == 2


# ---------------------------------------------------------------------------
# Behavior: initialize / update
# ---------------------------------------------------------------------------

class TestLatencyBudgetTrackerBehavior:
    def test_uninitialized_raises(self):
        t = LatencyBudgetTracker(_InstantTracker(), AdaptiveConfig(target_fps=30.0))
        with pytest.raises(RuntimeError):
            t.update(_FRAME)

    def test_update_returns_four_tuple(self):
        cfg = AdaptiveConfig(target_fps=30.0, min_skip=1, max_skip=1)
        t = LatencyBudgetTracker(_InstantTracker(), cfg)
        t.initialize(_FRAME, _BBOX)
        bbox = t.update(_FRAME)
        assert len(bbox) == 4

    def test_skip_1_calls_inner_every_frame(self):
        inner = _InstantTracker()
        cfg = AdaptiveConfig(target_fps=30.0, min_skip=1, max_skip=1)
        t = LatencyBudgetTracker(inner, cfg)
        t.initialize(_FRAME, _BBOX)
        for _ in range(10):
            t.update(_FRAME)
        assert inner.update_calls == 10

    def test_skip_3_calls_inner_subset(self):
        """With fixed skip=3 (min=max=3) only every 3rd frame calls inner."""
        inner = _InstantTracker()
        cfg = AdaptiveConfig(target_fps=30.0, min_skip=3, max_skip=3)
        t = LatencyBudgetTracker(inner, cfg)
        t.initialize(_FRAME, _BBOX)
        for _ in range(9):
            t.update(_FRAME)
        # Frames 3, 6, 9 are active → exactly 3 calls
        assert inner.update_calls == 3

    def test_passive_frames_return_last_active_bbox(self):
        """Skipped frames must return the cached bbox, not a stale one."""
        inner = _InstantTracker()
        cfg = AdaptiveConfig(target_fps=30.0, min_skip=3, max_skip=3)
        t = LatencyBudgetTracker(inner, cfg)
        t.initialize(_FRAME, _BBOX)
        bboxes = [t.update(_FRAME) for _ in range(3)]
        # Frames 1 and 2 are passive; frame 3 is active — all should return _BBOX
        assert all(b == _BBOX for b in bboxes)

    def test_reinitialize_resets_counters(self):
        inner = _InstantTracker()
        cfg = AdaptiveConfig(target_fps=30.0)
        t = LatencyBudgetTracker(inner, cfg)
        t.initialize(_FRAME, _BBOX)
        for _ in range(10):
            t.update(_FRAME)
        t.initialize(_FRAME, _BBOX)
        assert t._frame_idx == 0
        assert t.active_frame_count == 0

    def test_active_frame_count_with_skip_2(self):
        cfg = AdaptiveConfig(target_fps=30.0, min_skip=2, max_skip=2)
        t = LatencyBudgetTracker(_InstantTracker(), cfg)
        t.initialize(_FRAME, _BBOX)
        for _ in range(10):
            t.update(_FRAME)
        assert t.active_frame_count == 5


# ---------------------------------------------------------------------------
# Adaptation logic
# ---------------------------------------------------------------------------

class TestAdaptationLogic:
    def test_slow_tracker_increases_skip(self):
        """A tracker slower than the target should cause skip rate to rise."""
        inner = _SlowTracker(sleep_ms=50)
        cfg = AdaptiveConfig(
            target_fps=10_000.0,   # impossible target
            min_skip=1,
            max_skip=4,
            ema_alpha=1.0,         # immediate EMA reaction
            tolerance=0.01,
            cooldown_frames=1,
        )
        t = LatencyBudgetTracker(inner, cfg)
        t.initialize(_FRAME, _BBOX)
        for _ in range(5):
            t.update(_FRAME)
        assert t._skip > 1, "Skip rate should increase when tracker is too slow"

    def test_fast_tracker_at_high_skip_decreases_skip(self):
        """A tracker faster than the target should pull skip rate down."""
        inner = _InstantTracker()
        cfg = AdaptiveConfig(
            target_fps=0.0001,   # near-zero target — instant tracker always faster
            min_skip=1,
            max_skip=4,
            ema_alpha=1.0,
            tolerance=0.01,
            cooldown_frames=1,
        )
        t = LatencyBudgetTracker(inner, cfg)
        t.initialize(_FRAME, _BBOX)
        t._skip = 4              # force start at max
        for _ in range(5):
            t.update(_FRAME)
        assert t._skip < 4, "Skip rate should decrease when tracker is fast"

    def test_adaptation_history_recorded(self):
        inner = _SlowTracker(sleep_ms=50)
        cfg = AdaptiveConfig(
            target_fps=100_000.0,
            min_skip=1,
            max_skip=4,
            ema_alpha=1.0,
            tolerance=0.0,
            cooldown_frames=1,
        )
        t = LatencyBudgetTracker(inner, cfg)
        t.initialize(_FRAME, _BBOX)
        for _ in range(5):
            t.update(_FRAME)
        assert t.state.total_adaptations > 0
        assert len(t.state.history) > 0

    def test_adaptation_event_fields(self):
        inner = _SlowTracker(sleep_ms=50)
        cfg = AdaptiveConfig(
            target_fps=100_000.0,
            min_skip=1,
            max_skip=4,
            ema_alpha=1.0,
            tolerance=0.0,
            cooldown_frames=1,
        )
        t = LatencyBudgetTracker(inner, cfg)
        t.initialize(_FRAME, _BBOX)
        for _ in range(4):
            t.update(_FRAME)
        ev = t.state.history[0]
        assert isinstance(ev, AdaptationEvent)
        assert ev.new_skip > ev.old_skip    # skip rate went up (slow tracker)
        assert ev.ema_latency_ms > 0.0
        assert ev.frame_idx > 0

    def test_cooldown_limits_total_adaptations(self):
        """Long cooldown should suppress most adaptation events."""
        inner = _SlowTracker(sleep_ms=50)
        cfg = AdaptiveConfig(
            target_fps=100_000.0,
            min_skip=1,
            max_skip=8,
            ema_alpha=1.0,
            tolerance=0.0,
            cooldown_frames=100,   # very long cooldown
        )
        t = LatencyBudgetTracker(inner, cfg)
        t.initialize(_FRAME, _BBOX)
        for _ in range(5):
            t.update(_FRAME)
        # With 100-frame cooldown and only 5 active frames, at most 1 event
        assert t.state.total_adaptations <= 1

    def test_skip_never_exceeds_max(self):
        inner = _SlowTracker(sleep_ms=50)
        cfg = AdaptiveConfig(
            target_fps=100_000.0,
            min_skip=1,
            max_skip=3,
            ema_alpha=1.0,
            tolerance=0.0,
            cooldown_frames=1,
        )
        t = LatencyBudgetTracker(inner, cfg)
        t.initialize(_FRAME, _BBOX)
        for _ in range(20):
            t.update(_FRAME)
        assert t._skip <= 3

    def test_skip_never_below_min(self):
        inner = _InstantTracker()
        cfg = AdaptiveConfig(
            target_fps=0.0001,
            min_skip=2,
            max_skip=6,
            ema_alpha=1.0,
            tolerance=0.0,
            cooldown_frames=1,
        )
        t = LatencyBudgetTracker(inner, cfg)
        t.initialize(_FRAME, _BBOX)
        t._skip = 6
        for _ in range(20):
            t.update(_FRAME)
        assert t._skip >= 2


# ---------------------------------------------------------------------------
# State inspection
# ---------------------------------------------------------------------------

class TestAdaptiveState:
    def test_state_returns_adaptive_state(self):
        t = LatencyBudgetTracker(_InstantTracker(), AdaptiveConfig(target_fps=30.0))
        t.initialize(_FRAME, _BBOX)
        assert isinstance(t.state, AdaptiveState)

    def test_projected_fps_positive_after_update(self):
        t = LatencyBudgetTracker(_InstantTracker(), AdaptiveConfig(target_fps=30.0))
        t.initialize(_FRAME, _BBOX)
        t.update(_FRAME)
        assert t.projected_fps > 0.0

    def test_state_history_is_independent_copy(self):
        """Mutating the returned history list must not affect internal state."""
        t = LatencyBudgetTracker(_InstantTracker(), AdaptiveConfig(target_fps=30.0))
        t.initialize(_FRAME, _BBOX)
        history = t.state.history
        original_len = len(t._history)
        history.append(None)  # type: ignore[arg-type]
        assert len(t._history) == original_len

    def test_state_current_skip_matches_internal(self):
        cfg = AdaptiveConfig(target_fps=30.0, min_skip=2, max_skip=2)
        t = LatencyBudgetTracker(_InstantTracker(), cfg)
        t.initialize(_FRAME, _BBOX)
        assert t.state.current_skip == 2
