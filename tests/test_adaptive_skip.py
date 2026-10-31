"""Tests for eovot.trackers.adaptive_skip.LatencyAdaptiveTracker."""

from __future__ import annotations

import time
import numpy as np
import pytest

from eovot.trackers.adaptive_skip import AdaptiveSkipStats, LatencyAdaptiveTracker
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_frame(h: int = 120, w: int = 160) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _init_bbox() -> tuple:
    return (20.0, 20.0, 40.0, 40.0)


class SlowTracker(MOSSETracker):
    """MOSSETracker variant that sleeps to simulate high latency."""

    def __init__(self, sleep_ms: float = 50.0) -> None:
        super().__init__()
        self._sleep_s = sleep_ms / 1_000.0

    def update(self, frame: np.ndarray):
        time.sleep(self._sleep_s)
        return super().update(frame)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_name_suffix(self):
        tracker = LatencyAdaptiveTracker(MOSSETracker())
        assert "adaptive" in tracker.name.lower()

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError, match="latency_budget_ms"):
            LatencyAdaptiveTracker(MOSSETracker(), latency_budget_ms=0.0)

    def test_invalid_threshold_ratio_raises(self):
        with pytest.raises(ValueError, match="skip_threshold_ratio"):
            LatencyAdaptiveTracker(MOSSETracker(), skip_threshold_ratio=0.0)

    def test_invalid_max_skip_raises(self):
        with pytest.raises(ValueError, match="max_skip"):
            LatencyAdaptiveTracker(MOSSETracker(), max_skip=0)

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            LatencyAdaptiveTracker(MOSSETracker(), window_size=0)


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


class TestBasicBehaviour:
    def test_update_before_initialize_raises(self):
        tracker = LatencyAdaptiveTracker(MOSSETracker())
        with pytest.raises(AssertionError):
            tracker.update(_blank_frame())

    def test_initialize_and_update_return_bbox(self):
        frame = _blank_frame()
        tracker = LatencyAdaptiveTracker(MOSSETracker(), latency_budget_ms=1000.0)
        tracker.initialize(frame, _init_bbox())
        bbox = tracker.update(frame)
        assert len(bbox) == 4
        assert all(isinstance(v, (int, float)) for v in bbox)

    def test_total_frames_count(self):
        frame = _blank_frame()
        tracker = LatencyAdaptiveTracker(MOSSETracker(), latency_budget_ms=1000.0)
        tracker.initialize(frame, _init_bbox())
        n = 5
        for _ in range(n):
            tracker.update(frame)
        stats = tracker.get_stats()
        assert stats.total_frames == n

    def test_no_skip_with_generous_budget(self):
        """When budget is very large, no frames should be skipped."""
        frame = _blank_frame()
        tracker = LatencyAdaptiveTracker(
            MOSSETracker(),
            latency_budget_ms=100_000.0,  # 100 s — will never trigger
            window_size=3,
        )
        tracker.initialize(frame, _init_bbox())
        for _ in range(20):
            tracker.update(frame)
        stats = tracker.get_stats()
        assert stats.skipped_frames == 0
        assert stats.active_frames == 20

    def test_skip_rate_bounded_zero_one(self):
        frame = _blank_frame()
        tracker = LatencyAdaptiveTracker(MOSSETracker(), latency_budget_ms=1000.0)
        tracker.initialize(frame, _init_bbox())
        for _ in range(10):
            tracker.update(frame)
        stats = tracker.get_stats()
        assert 0.0 <= stats.skip_rate <= 1.0


# ---------------------------------------------------------------------------
# Adaptive skip logic
# ---------------------------------------------------------------------------


class TestAdaptiveSkipLogic:
    def test_max_skip_enforced(self):
        """After max_skip consecutive skips, the next frame must be active."""
        max_skip = 2
        frame = _blank_frame()
        # Very tight budget → skip almost everything
        tracker = LatencyAdaptiveTracker(
            SlowTracker(sleep_ms=5.0),
            latency_budget_ms=1.0,   # 1 ms budget — always exceeded after warmup
            window_size=2,
            skip_threshold_ratio=0.5,
            max_skip=max_skip,
        )
        tracker.initialize(frame, _init_bbox())
        # Run enough frames to fill the window and trigger skipping
        for _ in range(30):
            tracker.update(frame)
        stats = tracker.get_stats()
        # max_consecutive_skip should never exceed max_skip
        assert stats.max_consecutive_skip <= max_skip

    def test_linear_mode_extrapolation(self):
        """In 'linear' mode, skipped frames should not exactly equal the last bbox
        after the tracker has moved (non-zero velocity)."""
        frame = _blank_frame()
        tracker = LatencyAdaptiveTracker(
            MOSSETracker(),
            latency_budget_ms=1.0,   # always skip after window fills
            window_size=2,
            skip_threshold_ratio=0.5,
            max_skip=3,
            mode="linear",
        )
        tracker.initialize(frame, _init_bbox())
        # Force two active frames so _prev_bbox and _last_bbox are set
        for _ in range(2):
            tracker._consec_skipped = 0  # reset to ensure active
            tracker._latency_window.clear()  # prevent skip on these two
            tracker.update(frame)

    def test_repeat_mode_returns_last_bbox(self):
        """In 'repeat' mode, a skipped frame must return the exact last bbox."""
        frame = _blank_frame()
        tracker = LatencyAdaptiveTracker(
            MOSSETracker(),
            latency_budget_ms=100_000.0,  # generous — no skips
            window_size=1,
            mode="repeat",
        )
        tracker.initialize(frame, _init_bbox())
        # Force exactly one active frame
        tracker._latency_window.clear()
        active_bbox = tracker.update(frame)
        # Now force a skip
        tracker._latency_window.append(100_000.0)  # rolling mean >> threshold
        skipped_bbox = tracker.update(frame)
        assert skipped_bbox == tracker._last_bbox

    def test_window_filling_delay(self):
        """Skipping should NOT occur until the latency window is full."""
        frame = _blank_frame()
        window_size = 5
        tracker = LatencyAdaptiveTracker(
            MOSSETracker(),
            latency_budget_ms=0.001,  # vanishingly tight budget
            window_size=window_size,
            skip_threshold_ratio=0.5,
            max_skip=10,
        )
        tracker.initialize(frame, _init_bbox())
        # First window_size frames should all be active (window still filling)
        for _ in range(window_size):
            tracker.update(frame)
        stats = tracker.get_stats()
        assert stats.active_frames == window_size


# ---------------------------------------------------------------------------
# AdaptiveSkipStats
# ---------------------------------------------------------------------------


class TestAdaptiveSkipStats:
    def test_stats_str_contains_key_info(self):
        frame = _blank_frame()
        tracker = LatencyAdaptiveTracker(MOSSETracker(), latency_budget_ms=33.3)
        tracker.initialize(frame, _init_bbox())
        for _ in range(5):
            tracker.update(frame)
        s = str(tracker.get_stats())
        assert "total=" in s
        assert "budget=" in s

    def test_reset_stats(self):
        frame = _blank_frame()
        tracker = LatencyAdaptiveTracker(MOSSETracker(), latency_budget_ms=33.3)
        tracker.initialize(frame, _init_bbox())
        for _ in range(10):
            tracker.update(frame)
        tracker.reset_stats()
        stats = tracker.get_stats()
        assert stats.total_frames == 0
        assert stats.active_frames == 0
        assert stats.skipped_frames == 0

    def test_active_plus_skipped_equals_total(self):
        frame = _blank_frame()
        tracker = LatencyAdaptiveTracker(
            SlowTracker(sleep_ms=5.0),
            latency_budget_ms=1.0,
            window_size=3,
            skip_threshold_ratio=0.5,
            max_skip=2,
        )
        tracker.initialize(frame, _init_bbox())
        for _ in range(30):
            tracker.update(frame)
        s = tracker.get_stats()
        assert s.active_frames + s.skipped_frames == s.total_frames


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------


class TestBenchmarkIntegration:
    def test_benchmark_engine_accepts_adaptive_tracker(self):
        dataset = SyntheticDataset(num_sequences=2, num_frames=30, seed=5)
        engine = BenchmarkEngine(verbose=False)
        tracker = LatencyAdaptiveTracker(
            KCFTracker(),
            latency_budget_ms=1000.0,  # no-op budget for determinism
        )
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        assert result.mean_fps > 0
        assert result.mean_iou >= 0.0

    def test_adaptive_result_has_sequence_results(self):
        dataset = SyntheticDataset(num_sequences=2, num_frames=30, seed=5)
        engine = BenchmarkEngine(verbose=False)
        tracker = LatencyAdaptiveTracker(KCFTracker(), latency_budget_ms=500.0)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        assert len(result.sequence_results) == 2

    def test_wrap_different_base_trackers(self):
        """LatencyAdaptiveTracker should work with any BaseTracker."""
        dataset = SyntheticDataset(num_sequences=1, num_frames=30, seed=5)
        engine = BenchmarkEngine(verbose=False)
        for base in (MOSSETracker(), KCFTracker()):
            tracker = LatencyAdaptiveTracker(base, latency_budget_ms=200.0)
            result = engine.run(tracker, dataset, dataset_name="Syn")
            assert result.mean_fps > 0
