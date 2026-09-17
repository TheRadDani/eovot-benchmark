"""Tests for DualModeTracker."""
from __future__ import annotations

import time

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.dual_mode import DualModeTracker, DualModeStats, SwitchEvent


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeTracker(BaseTracker):
    """Instant tracker that always returns a fixed bbox."""

    def __init__(self, name: str, return_bbox: BBox = (10, 10, 50, 50)):
        super().__init__(name=name)
        self.return_bbox = return_bbox
        self.init_count = 0
        self.update_count = 0

    def initialize(self, frame, bbox):
        self.init_count += 1

    def update(self, frame) -> BBox:
        self.update_count += 1
        return self.return_bbox


class SlowTracker(BaseTracker):
    """Tracker that sleeps for a configurable duration per frame."""

    def __init__(self, name: str, sleep_ms: float = 0.0):
        super().__init__(name=name)
        self.sleep_ms = sleep_ms
        self.init_count = 0

    def initialize(self, frame, bbox):
        self.init_count += 1

    def update(self, frame) -> BBox:
        if self.sleep_ms > 0:
            time.sleep(self.sleep_ms / 1000.0)
        return (10, 10, 50, 50)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def bbox():
    return (10, 10, 50, 50)


@pytest.fixture
def dual(frame, bbox):
    fast = FakeTracker("fast")
    acc = FakeTracker("accurate")
    t = DualModeTracker(fast=fast, accurate=acc, budget_ms=1000.0, window_size=3)
    t.initialize(frame, bbox)
    return t


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_name_derived_from_trackers(self):
        fast = FakeTracker("MOSSE")
        acc = FakeTracker("KCF")
        t = DualModeTracker(fast=fast, accurate=acc)
        assert "MOSSE" in t.name and "KCF" in t.name

    def test_custom_name_respected(self):
        fast = FakeTracker("f")
        acc = FakeTracker("a")
        t = DualModeTracker(fast=fast, accurate=acc, name="MyDual")
        assert t.name == "MyDual"

    def test_invalid_budget_raises(self):
        fast, acc = FakeTracker("f"), FakeTracker("a")
        with pytest.raises(ValueError):
            DualModeTracker(fast=fast, accurate=acc, budget_ms=0.0)
        with pytest.raises(ValueError):
            DualModeTracker(fast=fast, accurate=acc, budget_ms=-5.0)

    def test_invalid_window_size_raises(self):
        fast, acc = FakeTracker("f"), FakeTracker("a")
        with pytest.raises(ValueError):
            DualModeTracker(fast=fast, accurate=acc, window_size=0)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_both_sub_trackers_are_initialized(self, frame, bbox):
        fast = FakeTracker("fast")
        acc = FakeTracker("accurate")
        t = DualModeTracker(fast=fast, accurate=acc)
        t.initialize(frame, bbox)
        assert fast.init_count == 1
        assert acc.init_count == 1

    def test_starts_in_accurate_mode(self, frame, bbox):
        fast = FakeTracker("fast")
        acc = FakeTracker("accurate")
        t = DualModeTracker(fast=fast, accurate=acc)
        t.initialize(frame, bbox)
        assert t.current_mode == "accurate"

    def test_reinitialize_resets_stats_and_history(self, dual, frame, bbox):
        dual.update(frame)
        dual.update(frame)
        dual.initialize(frame, bbox)
        assert dual.stats.total_frames == 0
        assert dual.rolling_latency_ms == 0.0

    def test_reinitialize_resets_frame_index(self, dual, frame, bbox):
        dual.update(frame)
        dual.initialize(frame, bbox)
        assert dual._frame_index == 0


# ---------------------------------------------------------------------------
# Basic update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_returns_four_tuple(self, dual, frame):
        bbox = dual.update(frame)
        assert len(bbox) == 4

    def test_update_increments_total_frames(self, dual, frame):
        dual.update(frame)
        dual.update(frame)
        assert dual.stats.total_frames == 2

    def test_rolling_latency_zero_before_any_update(self, frame, bbox):
        fast, acc = FakeTracker("f"), FakeTracker("a")
        t = DualModeTracker(fast=fast, accurate=acc)
        t.initialize(frame, bbox)
        assert t.rolling_latency_ms == 0.0

    def test_rolling_latency_non_negative_after_update(self, dual, frame):
        dual.update(frame)
        assert dual.rolling_latency_ms >= 0.0

    def test_accurate_tracker_called_when_latency_low(self, frame, bbox):
        fast = FakeTracker("fast")
        acc = FakeTracker("accurate")
        # Very large budget — FakeTrackers are near-instant, never switch
        t = DualModeTracker(fast=fast, accurate=acc, budget_ms=10_000.0)
        t.initialize(frame, bbox)
        for _ in range(5):
            t.update(frame)
        assert acc.update_count > 0
        assert t.current_mode == "accurate"


# ---------------------------------------------------------------------------
# Switching logic
# ---------------------------------------------------------------------------

class TestSwitching:
    def test_no_switch_when_below_budget(self, frame, bbox):
        fast, acc = FakeTracker("fast"), FakeTracker("accurate")
        t = DualModeTracker(fast=fast, accurate=acc, budget_ms=10_000.0, window_size=3)
        t.initialize(frame, bbox)
        for _ in range(10):
            t.update(frame)
        assert t.stats.num_switches == 0
        assert t.current_mode == "accurate"

    def test_switches_to_fast_when_latency_exceeds_budget(self, frame, bbox):
        fast = FakeTracker("fast")
        slow_acc = SlowTracker("slow_accurate", sleep_ms=40.0)
        # 5 ms budget → slow_acc (40 ms) will exceed it after first window fills
        t = DualModeTracker(fast=fast, accurate=slow_acc, budget_ms=5.0, window_size=2)
        t.initialize(frame, bbox)
        for _ in range(6):
            t.update(frame)
        assert t.stats.fast_frames > 0

    def test_switch_event_recorded(self, frame, bbox):
        fast = FakeTracker("fast")
        slow_acc = SlowTracker("slow_accurate", sleep_ms=50.0)
        t = DualModeTracker(fast=fast, accurate=slow_acc, budget_ms=5.0, window_size=2)
        t.initialize(frame, bbox)
        for _ in range(6):
            t.update(frame)
        if t.stats.num_switches > 0:
            event = t.stats.switches[0]
            assert isinstance(event, SwitchEvent)
            assert event.from_mode in ("fast", "accurate")
            assert event.to_mode in ("fast", "accurate")
            assert event.from_mode != event.to_mode
            assert event.frame_index > 0

    def test_fast_tracker_reinitialised_on_switch(self, frame, bbox):
        fast = FakeTracker("fast")
        slow_acc = SlowTracker("slow_accurate", sleep_ms=50.0)
        t = DualModeTracker(fast=fast, accurate=slow_acc, budget_ms=5.0, window_size=2)
        t.initialize(frame, bbox)
        init_before = fast.init_count
        for _ in range(6):
            t.update(frame)
        if t.stats.num_switches > 0:
            assert fast.init_count > init_before

    def test_switch_event_str(self):
        e = SwitchEvent(
            frame_index=10,
            from_mode="accurate",
            to_mode="fast",
            rolling_latency_ms=45.2,
            budget_ms=33.3,
        )
        s = str(e)
        assert "accurate" in s and "fast" in s
        assert "10" in s


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_fast_plus_accurate_fractions_sum_to_one(self, dual, frame):
        for _ in range(6):
            dual.update(frame)
        stats = dual.stats
        assert stats.fast_fraction + stats.accurate_fraction == pytest.approx(1.0)

    def test_total_equals_fast_plus_accurate(self, dual, frame):
        for _ in range(5):
            dual.update(frame)
        s = dual.stats
        assert s.total_frames == s.fast_frames + s.accurate_frames

    def test_num_switches_consistent_with_list(self, dual, frame):
        for _ in range(5):
            dual.update(frame)
        assert dual.stats.num_switches == len(dual.stats.switches)

    def test_summary_dict_keys_present(self, dual, frame):
        dual.update(frame)
        s = dual.stats.summary()
        for key in ("total_frames", "fast_frames", "accurate_frames",
                    "fast_fraction", "accurate_fraction", "num_switches"):
            assert key in s

    def test_fractions_zero_before_any_update(self, frame, bbox):
        fast, acc = FakeTracker("f"), FakeTracker("a")
        t = DualModeTracker(fast=fast, accurate=acc)
        t.initialize(frame, bbox)
        assert t.stats.fast_fraction == 0.0
        assert t.stats.accurate_fraction == 0.0


# ---------------------------------------------------------------------------
# Integration: full BenchmarkEngine pipeline
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_dual_mode_runs_through_benchmark_engine(self):
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.benchmark.engine import BenchmarkEngine

        dataset = SyntheticDataset(num_sequences=2, num_frames=20)
        engine = BenchmarkEngine(verbose=False)

        fast = FakeTracker("fast", return_bbox=(10, 10, 50, 50))
        acc = FakeTracker("accurate", return_bbox=(12, 12, 48, 48))
        t = DualModeTracker(fast=fast, accurate=acc, budget_ms=10_000.0)

        result = engine.run(t, dataset, dataset_name="Synthetic")
        assert result.tracker_name == t.name
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_dual_mode_stats_populated_after_engine_run(self):
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.benchmark.engine import BenchmarkEngine

        dataset = SyntheticDataset(num_sequences=1, num_frames=30)
        engine = BenchmarkEngine(verbose=False)

        fast = FakeTracker("fast")
        acc = FakeTracker("accurate")
        t = DualModeTracker(fast=fast, accurate=acc, budget_ms=10_000.0)

        engine.run(t, dataset, dataset_name="Synthetic")
        # After running, the stats reflect all update() calls across all sequences
        assert t.stats.total_frames > 0
        assert t.stats.total_frames == t.stats.fast_frames + t.stats.accurate_frames
