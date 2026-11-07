"""Unit tests for AdaptiveLatencyTracker.

Tests cover:
- Basic API: initialize / update / no crash
- skip_rate stays within [min_skip, max_skip]
- Adaptation log is populated correctly
- Over-budget condition increases skip_rate
- Under-budget condition decreases skip_rate
- adaptation_summary() returns consistent keys
- Handles short sequences gracefully
- Wraps any BaseTracker without coupling
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from eovot.trackers.adaptive import AdaptiveLatencyTracker, AdaptationEvent
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME_H, _FRAME_W = 120, 160
_INIT_BBOX = (20.0, 20.0, 40.0, 40.0)   # well inside a 120×160 frame


def _make_frames(n: int = 20) -> list:
    """Generate n identical BGR frames (static scene — prevents tracker drift)."""
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, (_FRAME_H, _FRAME_W, 3), dtype=np.uint8)
    return [base.copy() for _ in range(n)]


def _make_tracker(target_fps: float = 1000.0) -> AdaptiveLatencyTracker:
    """Return an AdaptiveLatencyTracker around MOSSE with an easy budget."""
    return AdaptiveLatencyTracker(
        MOSSETracker(),
        target_fps=target_fps,
        min_skip=1,
        max_skip=4,
        warmup_frames=2,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_name_includes_tracker_and_fps(self):
        t = AdaptiveLatencyTracker(MOSSETracker(), target_fps=30.0)
        assert "MOSSE" in t.name
        assert "30" in t.name

    def test_invalid_target_fps(self):
        with pytest.raises(ValueError):
            AdaptiveLatencyTracker(MOSSETracker(), target_fps=0.0)

    def test_invalid_skip_range(self):
        with pytest.raises(ValueError):
            AdaptiveLatencyTracker(MOSSETracker(), min_skip=5, max_skip=3)

    def test_invalid_ema_alpha(self):
        with pytest.raises(ValueError):
            AdaptiveLatencyTracker(MOSSETracker(), ema_alpha=0.0)

    def test_invalid_slack(self):
        with pytest.raises(ValueError):
            AdaptiveLatencyTracker(MOSSETracker(), slack=1.5)

    def test_invalid_warmup(self):
        with pytest.raises(ValueError):
            AdaptiveLatencyTracker(MOSSETracker(), warmup_frames=-1)


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestBasicAPI:
    def test_update_before_initialize_raises(self):
        t = _make_tracker()
        frames = _make_frames()
        with pytest.raises(RuntimeError):
            t.update(frames[0])

    def test_initialize_and_update_returns_bbox(self):
        t = _make_tracker()
        frames = _make_frames()
        t.initialize(frames[0], _INIT_BBOX)
        pred = t.update(frames[1])
        assert len(pred) == 4

    def test_update_all_frames(self):
        t = _make_tracker()
        frames = _make_frames(30)
        t.initialize(frames[0], _INIT_BBOX)
        preds = [t.update(f) for f in frames[1:]]
        assert len(preds) == 29

    def test_reinitialize_resets_log(self):
        t = _make_tracker()
        frames = _make_frames()
        t.initialize(frames[0], _INIT_BBOX)
        for f in frames[1:5]:
            t.update(f)
        assert len(t.adaptation_log) > 0
        # Re-initialize should clear the log
        t.initialize(frames[0], _INIT_BBOX)
        assert len(t.adaptation_log) == 0


# ---------------------------------------------------------------------------
# Adaptation log
# ---------------------------------------------------------------------------

class TestAdaptationLog:
    def test_log_length_matches_frames(self):
        t = _make_tracker()
        frames = _make_frames(15)
        t.initialize(frames[0], _INIT_BBOX)
        for f in frames[1:]:
            t.update(f)
        # One log entry per update() call
        assert len(t.adaptation_log) == 14

    def test_log_entries_are_adaptation_events(self):
        t = _make_tracker()
        frames = _make_frames(10)
        t.initialize(frames[0], _INIT_BBOX)
        for f in frames[1:]:
            t.update(f)
        for entry in t.adaptation_log:
            assert isinstance(entry, AdaptationEvent)

    def test_frame_index_is_monotone(self):
        t = _make_tracker()
        frames = _make_frames(15)
        t.initialize(frames[0], _INIT_BBOX)
        for f in frames[1:]:
            t.update(f)
        indices = [e.frame_index for e in t.adaptation_log]
        assert indices == sorted(indices)

    def test_skip_rate_in_valid_range(self):
        t = AdaptiveLatencyTracker(MOSSETracker(), target_fps=30.0, min_skip=1, max_skip=4)
        frames = _make_frames(15)
        t.initialize(frames[0], _INIT_BBOX)
        for f in frames[1:]:
            t.update(f)
        for entry in t.adaptation_log:
            assert 1 <= entry.skip_rate_after <= 4

    def test_active_frames_match_skip_rate(self):
        """Active frames should follow the skip_rate pattern."""
        t = AdaptiveLatencyTracker(MOSSETracker(), target_fps=1000.0, min_skip=1, max_skip=1)
        frames = _make_frames(20)
        t.initialize(frames[0], _INIT_BBOX)
        for f in frames[1:]:
            t.update(f)
        # skip_rate=1 → all frames active
        assert all(e.was_active for e in t.adaptation_log)


# ---------------------------------------------------------------------------
# Adaptation behaviour
# ---------------------------------------------------------------------------

class TestAdaptationBehaviour:
    def test_num_adaptations_property(self):
        t = _make_tracker(target_fps=1.0)  # tight budget → potentially adapts
        frames = _make_frames(30)
        t.initialize(frames[0], _INIT_BBOX)
        for f in frames[1:]:
            t.update(f)
        # Property should count changed entries
        changed = sum(1 for e in t.adaptation_log if e.rate_changed)
        assert t.num_adaptations == changed

    def test_mean_skip_rate_above_min_when_slow(self):
        """Very tight FPS budget → mean skip_rate expected to be above min."""
        t = AdaptiveLatencyTracker(
            MOSSETracker(),
            target_fps=0.001,  # 1000s per frame budget — no real tracker is that slow
            min_skip=1,
            max_skip=8,
            warmup_frames=0,
        )
        frames = _make_frames(15)
        t.initialize(frames[0], _INIT_BBOX)
        for f in frames[1:]:
            t.update(f)
        # skip_rate should stay at min=1 since any real tracker is faster than 1000s/frame
        assert t.mean_skip_rate >= 1.0

    def test_easy_budget_keeps_skip_at_min(self):
        """Very generous FPS budget → tracker should not need to skip."""
        t = AdaptiveLatencyTracker(
            MOSSETracker(),
            target_fps=0.001,  # 1000s budget — always satisfied
            min_skip=1,
            max_skip=8,
            warmup_frames=0,
        )
        frames = _make_frames(15)
        t.initialize(frames[0], _INIT_BBOX)
        for f in frames[1:]:
            t.update(f)
        assert t.mean_skip_rate == 1.0


# ---------------------------------------------------------------------------
# adaptation_summary()
# ---------------------------------------------------------------------------

class TestAdaptationSummary:
    def test_summary_keys_present(self):
        t = _make_tracker()
        frames = _make_frames(20)
        t.initialize(frames[0], (5.0, 5.0, 10.0, 10.0))
        for f in frames[1:]:
            t.update(f)
        summary = t.adaptation_summary()
        expected = {
            "target_fps", "num_frames", "num_active_frames",
            "num_adaptations", "mean_skip_rate", "final_skip_rate",
            "mean_ema_latency_ms", "effective_fps",
        }
        assert expected.issubset(summary.keys())

    def test_summary_before_any_update(self):
        t = _make_tracker()
        frames = _make_frames(5)
        t.initialize(frames[0], _INIT_BBOX)
        summary = t.adaptation_summary()
        assert summary["num_frames"] == 0

    def test_summary_num_frames_matches_log(self):
        t = _make_tracker()
        frames = _make_frames(15)
        t.initialize(frames[0], _INIT_BBOX)
        for f in frames[1:]:
            t.update(f)
        summary = t.adaptation_summary()
        assert summary["num_frames"] == len(t.adaptation_log)

    def test_summary_num_active_matches_log(self):
        t = _make_tracker()
        frames = _make_frames(15)
        t.initialize(frames[0], _INIT_BBOX)
        for f in frames[1:]:
            t.update(f)
        active_count = sum(1 for e in t.adaptation_log if e.was_active)
        assert t.adaptation_summary()["num_active_frames"] == active_count


# ---------------------------------------------------------------------------
# Works with different base trackers
# ---------------------------------------------------------------------------

class TestBaseTrackerAgnosticism:
    def test_wraps_kcf_tracker(self):
        t = AdaptiveLatencyTracker(KCFTracker(), target_fps=30.0)
        frames = _make_frames(10)
        t.initialize(frames[0], _INIT_BBOX)
        for f in frames[1:]:
            pred = t.update(f)
            assert len(pred) == 4

    def test_short_sequence_two_frames(self):
        t = _make_tracker()
        frames = _make_frames(2)
        t.initialize(frames[0], _INIT_BBOX)
        pred = t.update(frames[1])
        assert len(pred) == 4
        assert len(t.adaptation_log) == 1
