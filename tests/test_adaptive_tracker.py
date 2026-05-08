"""Tests for AdaptiveTracker and BudgetMonitor."""

import time

import numpy as np
import pytest

from eovot.profiling.budget import BudgetMonitor, ComputeBudget, RoutingDecision
from eovot.trackers.adaptive import AdaptiveTracker
from eovot.trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Minimal stub trackers
# ---------------------------------------------------------------------------


class ConstantTracker(BaseTracker):
    """Returns a fixed bbox; optionally sleeps to simulate latency."""

    def __init__(
        self,
        bbox: BBox = (10.0, 10.0, 50.0, 50.0),
        delay_ms: float = 0.0,
        name: str = "const",
    ) -> None:
        super().__init__(name)
        self._bbox = bbox
        self._delay_s = delay_ms / 1_000.0

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        if self._delay_s > 0:
            time.sleep(self._delay_s)
        return self._bbox


class ShiftingTracker(BaseTracker):
    """Each update shifts the bbox by (+1, +1, 0, 0)."""

    def __init__(self, name: str = "shifting") -> None:
        super().__init__(name)
        self._bbox: BBox = (0.0, 0.0, 50.0, 50.0)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        x, y, w, h = self._bbox
        self._bbox = (x + 1, y + 1, w, h)
        return self._bbox


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def frame() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


@pytest.fixture()
def init_bbox() -> BBox:
    return (10.0, 10.0, 50.0, 50.0)


@pytest.fixture()
def simple_budget() -> ComputeBudget:
    return ComputeBudget(target_fps=30.0)


# ---------------------------------------------------------------------------
# ComputeBudget
# ---------------------------------------------------------------------------


class TestComputeBudget:
    def test_budget_ms_30fps(self, simple_budget):
        assert abs(simple_budget.budget_ms - 33.333) < 0.01

    def test_budget_ms_60fps(self):
        b = ComputeBudget(target_fps=60.0)
        assert abs(b.budget_ms - 16.667) < 0.01

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError, match="target_fps"):
            ComputeBudget(target_fps=0.0)

    def test_negative_fps_raises(self):
        with pytest.raises(ValueError, match="target_fps"):
            ComputeBudget(target_fps=-10.0)

    def test_invalid_switch_margin_raises(self):
        with pytest.raises(ValueError, match="switch_margin"):
            ComputeBudget(target_fps=30.0, switch_margin=1.0)

    def test_switch_margin_zero_is_valid(self):
        b = ComputeBudget(target_fps=30.0, switch_margin=0.0)
        assert b.switch_margin == 0.0


# ---------------------------------------------------------------------------
# BudgetMonitor
# ---------------------------------------------------------------------------


class TestBudgetMonitor:
    def test_warm_up_always_recommends_primary(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        # Before any frames are recorded, should recommend primary
        assert monitor.should_use_primary()

    def test_switch_to_fallback_when_over_budget(self):
        budget = ComputeBudget(target_fps=30.0, window=4, switch_margin=0.0)
        monitor = BudgetMonitor(budget)
        for _ in range(4):
            monitor.record_frame(elapsed_ms=50.0, used_primary=True)  # > 33 ms
        assert not monitor.should_use_primary()

    def test_stay_primary_when_under_budget(self):
        budget = ComputeBudget(target_fps=30.0, window=4, switch_margin=0.0)
        monitor = BudgetMonitor(budget)
        for _ in range(4):
            monitor.record_frame(elapsed_ms=10.0, used_primary=True)  # well under 33 ms
        assert monitor.should_use_primary()

    def test_primary_ratio_all_primary(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        for _ in range(5):
            monitor.record_frame(10.0, used_primary=True)
        assert monitor.primary_ratio == 1.0

    def test_primary_ratio_all_fallback(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        for _ in range(5):
            monitor.record_frame(10.0, used_primary=False)
        assert monitor.primary_ratio == 0.0

    def test_primary_ratio_half(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        for i in range(10):
            monitor.record_frame(10.0, used_primary=(i % 2 == 0))
        assert abs(monitor.primary_ratio - 0.5) < 1e-9

    def test_budget_violation_rate(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        monitor.record_frame(10.0, used_primary=True)    # within budget
        monitor.record_frame(50.0, used_primary=False)   # over budget
        assert abs(monitor.budget_violation_rate - 0.5) < 1e-9

    def test_no_violations_when_all_within(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        for _ in range(5):
            monitor.record_frame(5.0, used_primary=True)
        assert monitor.budget_violation_rate == 0.0

    def test_mean_utilization_nonzero(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        monitor.record_frame(10.0, used_primary=True)
        assert monitor.mean_utilization > 0.0

    def test_summary_has_required_keys(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        monitor.record_frame(10.0, used_primary=True)
        keys = monitor.summary()
        assert "target_fps" in keys
        assert "budget_ms" in keys
        assert "total_frames" in keys
        assert "primary_ratio" in keys
        assert "budget_violation_rate" in keys
        assert "mean_utilization" in keys

    def test_reset_clears_all_state(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        for _ in range(5):
            monitor.record_frame(10.0, used_primary=True)
        monitor.reset()
        assert monitor.primary_ratio == 0.0
        assert monitor.budget_violation_rate == 0.0
        assert monitor.mean_utilization == 0.0

    def test_frame_index_increments(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        d1 = monitor.record_frame(5.0, used_primary=True)
        d2 = monitor.record_frame(5.0, used_primary=True)
        assert d2.frame_index == d1.frame_index + 1

    def test_routing_decision_over_budget_flag(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        decision = monitor.record_frame(
            elapsed_ms=simple_budget.budget_ms + 1.0, used_primary=True
        )
        assert decision.over_budget

    def test_routing_decision_within_budget_flag(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        decision = monitor.record_frame(elapsed_ms=1.0, used_primary=True)
        assert not decision.over_budget

    def test_utilization_property(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        decision = monitor.record_frame(
            elapsed_ms=simple_budget.budget_ms / 2, used_primary=True
        )
        assert abs(decision.utilization - 0.5) < 1e-6

    def test_empty_monitor_stats(self, simple_budget):
        monitor = BudgetMonitor(simple_budget)
        assert monitor.primary_ratio == 0.0
        assert monitor.budget_violation_rate == 0.0
        assert monitor.mean_utilization == 0.0


# ---------------------------------------------------------------------------
# AdaptiveTracker
# ---------------------------------------------------------------------------


class TestAdaptiveTrackerInit:
    def test_initialize_does_not_raise(self, frame, init_bbox, simple_budget):
        tracker = AdaptiveTracker(
            primary=ConstantTracker(name="primary"),
            fallback=ConstantTracker(name="fallback"),
            budget=simple_budget,
        )
        tracker.initialize(frame, init_bbox)

    def test_default_name_includes_sub_tracker_names(self, simple_budget):
        tracker = AdaptiveTracker(
            primary=ConstantTracker(name="primary"),
            fallback=ConstantTracker(name="fallback"),
            budget=simple_budget,
        )
        assert "primary" in tracker.name
        assert "fallback" in tracker.name

    def test_custom_name(self, simple_budget):
        tracker = AdaptiveTracker(
            primary=ConstantTracker(name="p"),
            fallback=ConstantTracker(name="f"),
            budget=simple_budget,
            name="MyAdaptive",
        )
        assert tracker.name == "MyAdaptive"


class TestAdaptiveTrackerUpdate:
    def test_update_returns_four_element_tuple(self, frame, init_bbox, simple_budget):
        tracker = AdaptiveTracker(
            primary=ConstantTracker(name="p"),
            fallback=ConstantTracker(name="f"),
            budget=simple_budget,
        )
        tracker.initialize(frame, init_bbox)
        bbox = tracker.update(frame)
        assert len(bbox) == 4

    def test_decisions_recorded_after_update(self, frame, init_bbox, simple_budget):
        tracker = AdaptiveTracker(
            primary=ConstantTracker(name="p"),
            fallback=ConstantTracker(name="f"),
            budget=simple_budget,
        )
        tracker.initialize(frame, init_bbox)
        n = 5
        for _ in range(n):
            tracker.update(frame)
        assert len(tracker.decisions) == n

    def test_initialize_resets_decisions(self, frame, init_bbox, simple_budget):
        tracker = AdaptiveTracker(
            primary=ConstantTracker(name="p"),
            fallback=ConstantTracker(name="f"),
            budget=simple_budget,
        )
        tracker.initialize(frame, init_bbox)
        for _ in range(3):
            tracker.update(frame)
        tracker.initialize(frame, init_bbox)
        assert len(tracker.decisions) == 0

    def test_primary_decision_during_warm_up(self, frame, init_bbox, simple_budget):
        """During warm-up the routing decision should always select primary."""
        tracker = AdaptiveTracker(
            primary=ConstantTracker(name="p"),
            fallback=ConstantTracker(name="f"),
            budget=ComputeBudget(target_fps=30.0, window=20),
        )
        tracker.initialize(frame, init_bbox)
        tracker.update(frame)
        # First frame is always warm-up → decision.used_primary must be True
        assert tracker.decisions[-1].used_primary

    def test_falls_back_when_over_budget(self, frame, init_bbox):
        """Fill the window with over-budget latencies; fallback should be selected."""
        budget = ComputeBudget(target_fps=30.0, window=4, switch_margin=0.0)
        tracker = AdaptiveTracker(
            primary=ConstantTracker(name="p"),
            fallback=ConstantTracker(name="f"),
            budget=budget,
        )
        tracker.initialize(frame, init_bbox)
        # Manually saturate the monitor with over-budget latencies
        for _ in range(4):
            tracker.monitor.record_frame(elapsed_ms=50.0, used_primary=True)
        # Monitor should now recommend fallback
        assert not tracker.monitor.should_use_primary()
        tracker.update(frame)
        assert not tracker.decisions[-1].used_primary

    def test_routing_summary_has_expected_keys(self, frame, init_bbox, simple_budget):
        tracker = AdaptiveTracker(
            primary=ConstantTracker(name="p"),
            fallback=ConstantTracker(name="f"),
            budget=simple_budget,
        )
        tracker.initialize(frame, init_bbox)
        tracker.update(frame)
        s = tracker.routing_summary
        assert "target_fps" in s
        assert "primary_ratio" in s
        assert "budget_violation_rate" in s

    def test_decisions_list_is_copy(self, frame, init_bbox, simple_budget):
        """Modifying the returned list should not affect internal state."""
        tracker = AdaptiveTracker(
            primary=ConstantTracker(name="p"),
            fallback=ConstantTracker(name="f"),
            budget=simple_budget,
        )
        tracker.initialize(frame, init_bbox)
        tracker.update(frame)
        dec_copy = tracker.decisions
        dec_copy.clear()
        assert len(tracker.decisions) == 1

    def test_both_sub_trackers_are_updated(self, frame, init_bbox, simple_budget):
        """ShiftingTracker increments its internal state each update."""
        primary = ShiftingTracker(name="p")
        fallback = ShiftingTracker(name="f")
        tracker = AdaptiveTracker(primary=primary, fallback=fallback, budget=simple_budget)
        tracker.initialize(frame, init_bbox)
        for _ in range(3):
            tracker.update(frame)
        # Both sub-trackers should have been called 3 times
        px, py, *_ = primary._bbox
        fx, fy, *_ = fallback._bbox
        # Each was initialised at (10, 10) and incremented 3 times
        assert px == pytest.approx(13.0)
        assert fx == pytest.approx(13.0)
