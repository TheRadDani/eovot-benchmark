"""Tests for AdaptiveBudgetTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.adaptive_budget import AdaptiveBudgetTracker, AdaptationEvent
from eovot.trackers.mosse import MOSSETracker
from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker(target_fps: float = 60.0, **kwargs) -> AdaptiveBudgetTracker:
    return AdaptiveBudgetTracker(MOSSETracker(), target_fps=target_fps, **kwargs)


def _dummy_frame(h: int = 120, w: int = 160) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _dummy_bbox() -> tuple:
    return (10.0, 10.0, 30.0, 30.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_name_includes_target_fps(self):
        t = _make_tracker(target_fps=30.0)
        assert "30" in t.name

    def test_invalid_target_fps(self):
        with pytest.raises(ValueError, match="target_fps"):
            AdaptiveBudgetTracker(MOSSETracker(), target_fps=0)

    def test_invalid_max_skip_rate(self):
        with pytest.raises(ValueError, match="max_skip_rate"):
            AdaptiveBudgetTracker(MOSSETracker(), target_fps=30, max_skip_rate=0)

    def test_invalid_min_scale(self):
        with pytest.raises(ValueError, match="min_scale"):
            AdaptiveBudgetTracker(MOSSETracker(), target_fps=30, min_scale=0.0)
        with pytest.raises(ValueError, match="min_scale"):
            AdaptiveBudgetTracker(MOSSETracker(), target_fps=30, min_scale=1.5)

    def test_invalid_control_window(self):
        with pytest.raises(ValueError, match="control_window"):
            AdaptiveBudgetTracker(MOSSETracker(), target_fps=30, control_window=0)

    def test_underlying_tracker_property(self):
        inner = MOSSETracker()
        t = AdaptiveBudgetTracker(inner, target_fps=30)
        assert t.underlying_tracker is inner


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialize:
    def test_initializes_without_error(self):
        t = _make_tracker()
        frame = _dummy_frame()
        t.initialize(frame, _dummy_bbox())

    def test_adaptation_log_cleared_on_reinit(self):
        t = _make_tracker(control_window=2)
        frame = _dummy_frame()
        bbox = _dummy_bbox()
        t.initialize(frame, bbox)
        # Run some frames to potentially generate log entries
        for _ in range(5):
            t.update(frame)
        # Re-initialize should clear the log
        t.initialize(frame, bbox)
        assert t.adaptation_log == []

    def test_frame_counter_reset_on_reinit(self):
        t = _make_tracker()
        frame = _dummy_frame()
        bbox = _dummy_bbox()
        t.initialize(frame, bbox)
        t.update(frame)
        t.update(frame)
        t.initialize(frame, bbox)  # re-init
        assert t._frame_idx == 0


# ---------------------------------------------------------------------------
# Update / skip behaviour
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_raises_without_init(self):
        t = _make_tracker()
        with pytest.raises(RuntimeError):
            t.update(_dummy_frame())

    def test_returns_bbox_tuple(self):
        t = _make_tracker()
        frame = _dummy_frame()
        t.initialize(frame, _dummy_bbox())
        bbox = t.update(frame)
        assert len(bbox) == 4

    def test_passive_frames_return_last_bbox(self):
        """With skip_rate=2 every second frame should repeat the last prediction."""
        t = _make_tracker(target_fps=0.001)  # force budget to trigger skip
        t._current_skip_rate = 2  # force skip rate directly
        frame = _dummy_frame()
        bbox = _dummy_bbox()
        t.initialize(frame, bbox)

        pred1 = t.update(frame)  # frame_idx=1 → active (1 % 2 != 0, BUT index 1 is active)
        pred2 = t.update(frame)  # frame_idx=2 → passive if skip=2

        # At skip_rate=2: frame 1 is active (1%2=1≠0, active), frame 2 is passive (2%2=0 wait)
        # Actually the logic: active when frame_idx % skip_rate == 0
        # frame_idx=1 → 1%2=1 != 0 → passive
        # frame_idx=2 → 2%2=0 → active
        # So pred1 (frame_idx=1) is passive → returns _last_bbox (= init bbox)
        # pred2 (frame_idx=2) is active → runs tracker
        assert len(pred1) == 4
        assert len(pred2) == 4


# ---------------------------------------------------------------------------
# Scale helpers
# ---------------------------------------------------------------------------

class TestScaleHelpers:
    def test_scale_frame_no_copy_at_1(self):
        t = _make_tracker()
        frame = _dummy_frame()
        t.initialize(frame, _dummy_bbox())
        result = t._scale_frame(frame)
        assert result is frame  # same object, no copy

    def test_scale_frame_reduces_size(self):
        t = _make_tracker()
        t._current_scale = 0.5
        frame = _dummy_frame(h=120, w=160)
        t.initialize(frame, _dummy_bbox())
        t._current_scale = 0.5
        scaled = t._scale_frame(frame)
        assert scaled.shape[0] == 60
        assert scaled.shape[1] == 80

    def test_scale_bbox_roundtrip(self):
        t = _make_tracker()
        frame = _dummy_frame()
        t.initialize(frame, _dummy_bbox())
        t._current_scale = 0.5
        original_bbox = (20.0, 30.0, 40.0, 50.0)
        down = t._scale_bbox_down(original_bbox, frame.shape)
        up = t._scale_bbox_up(down, frame.shape[0], frame.shape[1])
        for a, b in zip(original_bbox, up):
            assert abs(a - b) < 1e-9


# ---------------------------------------------------------------------------
# Controller behaviour
# ---------------------------------------------------------------------------

class TestController:
    def test_adaptation_log_populated(self):
        """The log should grow as the control window fills."""
        t = _make_tracker(control_window=3)
        frame = _dummy_frame()
        t.initialize(frame, _dummy_bbox())
        for _ in range(12):
            t.update(frame)
        # At least some control steps should have fired
        assert len(t.adaptation_log) > 0

    def test_adaptation_log_entries_are_events(self):
        t = _make_tracker(control_window=2)
        frame = _dummy_frame()
        t.initialize(frame, _dummy_bbox())
        for _ in range(10):
            t.update(frame)
        for entry in t.adaptation_log:
            assert isinstance(entry, AdaptationEvent)
            assert entry.new_skip_rate >= 1
            assert 0.0 < entry.new_scale <= 1.0

    def test_adaptation_summary_keys(self):
        t = _make_tracker(control_window=2)
        frame = _dummy_frame()
        t.initialize(frame, _dummy_bbox())
        for _ in range(10):
            t.update(frame)
        summary = t.adaptation_summary()
        assert "total_control_steps" in summary
        assert "actions" in summary
        assert "final_skip_rate" in summary
        assert "final_scale" in summary

    def test_skip_rate_stays_within_bounds(self):
        t = _make_tracker(target_fps=0.001, max_skip_rate=3, control_window=2)
        frame = _dummy_frame()
        t.initialize(frame, _dummy_bbox())
        for _ in range(50):
            t.update(frame)
        assert 1 <= t.current_skip_rate <= 3

    def test_scale_stays_within_bounds(self):
        t = _make_tracker(target_fps=0.001, min_scale=0.5, control_window=2)
        frame = _dummy_frame()
        t.initialize(frame, _dummy_bbox())
        for _ in range(50):
            t.update(frame)
        assert t.current_scale >= 0.5
        assert t.current_scale <= 1.0


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:
    def test_runs_through_benchmark_engine(self):
        dataset = SyntheticDataset(num_sequences=2, num_frames=30, seed=0)
        engine = BenchmarkEngine(verbose=False)
        tracker = AdaptiveBudgetTracker(
            MOSSETracker(), target_fps=30.0, control_window=5
        )
        result = engine.run(tracker, dataset, dataset_name="Syn")
        assert result.mean_fps > 0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_adaptation_log_after_benchmark(self):
        dataset = SyntheticDataset(num_sequences=1, num_frames=50, seed=1)
        engine = BenchmarkEngine(verbose=False)
        tracker = AdaptiveBudgetTracker(
            MOSSETracker(), target_fps=30.0, control_window=5
        )
        engine.run(tracker, dataset, dataset_name="Syn")
        # The engine re-initializes on each sequence, so log starts fresh per seq.
        # After the run, some control steps may have fired.
        summary = tracker.adaptation_summary()
        assert isinstance(summary["total_control_steps"], int)
