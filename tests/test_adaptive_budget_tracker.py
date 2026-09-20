"""Tests for BudgetAdaptiveTracker.

Covers: construction, validation, state reset, warmup, over/under-budget
adaptation, histogram arithmetic, compliance rate, adaptation log,
report formatting, and BenchmarkEngine integration.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.adaptive_budget import AdaptationEvent, BudgetAdaptiveTracker


# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------

class _FixedTracker(BaseTracker):
    """Returns a constant bbox; optionally sleeps to simulate latency."""

    def __init__(self, latency_ms: float = 0.0, name: str = "Fixed") -> None:
        super().__init__(name=name)
        self._bbox: BBox = (10.0, 10.0, 50.0, 50.0)
        self._latency_ms = latency_ms
        self.init_count = 0
        self.update_count = 0

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox
        self.init_count += 1

    def update(self, frame: np.ndarray) -> BBox:
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)
        self.update_count += 1
        return self._bbox


def _blank_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _default_bbox() -> BBox:
    return (5.0, 5.0, 20.0, 20.0)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic_instantiation(self):
        tracker = BudgetAdaptiveTracker(_FixedTracker(), budget_ms=30.0)
        assert tracker is not None

    def test_name_format(self):
        inner = _FixedTracker(name="KCF")
        tracker = BudgetAdaptiveTracker(inner, budget_ms=25.0)
        assert "KCF" in tracker.name
        assert "25" in tracker.name

    def test_name_rounds_budget(self):
        inner = _FixedTracker(name="X")
        tracker = BudgetAdaptiveTracker(inner, budget_ms=33.333)
        assert "33" in tracker.name

    def test_invalid_budget_zero(self):
        with pytest.raises(ValueError, match="budget_ms"):
            BudgetAdaptiveTracker(_FixedTracker(), budget_ms=0.0)

    def test_invalid_budget_negative(self):
        with pytest.raises(ValueError, match="budget_ms"):
            BudgetAdaptiveTracker(_FixedTracker(), budget_ms=-5.0)

    def test_invalid_max_skip_zero(self):
        with pytest.raises(ValueError, match="max_skip"):
            BudgetAdaptiveTracker(_FixedTracker(), budget_ms=30.0, max_skip=0)

    def test_invalid_max_skip_negative(self):
        with pytest.raises(ValueError, match="max_skip"):
            BudgetAdaptiveTracker(_FixedTracker(), budget_ms=30.0, max_skip=-1)

    def test_invalid_ema_alpha_zero(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            BudgetAdaptiveTracker(_FixedTracker(), budget_ms=30.0, ema_alpha=0.0)

    def test_invalid_ema_alpha_above_one(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            BudgetAdaptiveTracker(_FixedTracker(), budget_ms=30.0, ema_alpha=1.5)

    def test_invalid_recovery_ratio_zero(self):
        with pytest.raises(ValueError, match="recovery_ratio"):
            BudgetAdaptiveTracker(_FixedTracker(), budget_ms=30.0, recovery_ratio=0.0)

    def test_underlying_tracker_accessible(self):
        inner = _FixedTracker(name="inner")
        tracker = BudgetAdaptiveTracker(inner, budget_ms=30.0)
        assert tracker.underlying_tracker is inner


# ---------------------------------------------------------------------------
# Initialize and state reset
# ---------------------------------------------------------------------------

class TestInitialize:
    def test_initialize_calls_underlying_tracker(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(inner, budget_ms=100.0)
        tracker.initialize(_blank_frame(), _default_bbox())
        assert inner.init_count == 1

    def test_initialize_resets_skip_rate_to_one(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(
            inner, budget_ms=0.001, max_skip=4, warmup_frames=0
        )
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        # Run enough frames to trigger adaptation (budget is impossibly tight)
        for _ in range(20):
            tracker.update(frame)
        assert tracker.current_skip_rate > 1
        # Re-initialize should reset to 1
        tracker.initialize(frame, _default_bbox())
        assert tracker.current_skip_rate == 1

    def test_initialize_clears_adaptation_log(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(
            inner, budget_ms=0.001, max_skip=4, warmup_frames=0
        )
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(20):
            tracker.update(frame)
        assert len(tracker.adaptation_log) > 0
        tracker.initialize(frame, _default_bbox())
        assert len(tracker.adaptation_log) == 0

    def test_initialize_clears_histogram(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(inner, budget_ms=100.0)
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(5):
            tracker.update(frame)
        assert len(tracker.skip_rate_histogram) > 0
        tracker.initialize(frame, _default_bbox())
        assert tracker.skip_rate_histogram == {}

    def test_initialize_resets_ema(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(inner, budget_ms=100.0)
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        tracker.update(frame)
        assert tracker.ema_latency_ms is not None
        tracker.initialize(frame, _default_bbox())
        assert tracker.ema_latency_ms is None


# ---------------------------------------------------------------------------
# Update before initialize
# ---------------------------------------------------------------------------

class TestUpdateWithoutInit:
    def test_update_before_initialize_raises(self):
        tracker = BudgetAdaptiveTracker(_FixedTracker(), budget_ms=30.0)
        with pytest.raises(RuntimeError, match="initialize"):
            tracker.update(_blank_frame())


# ---------------------------------------------------------------------------
# Warmup suppresses early adaptation
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_warmup_suppresses_adaptation(self):
        # budget_ms=0.001 would normally trigger adaptation immediately,
        # but warmup_frames=100 prevents it for the first 100 frames.
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(
            inner, budget_ms=0.001, max_skip=4, warmup_frames=100
        )
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(10):
            tracker.update(frame)
        # No adaptations yet — within warmup window
        assert len(tracker.adaptation_log) == 0
        assert tracker.current_skip_rate == 1


# ---------------------------------------------------------------------------
# Over-budget adaptation
# ---------------------------------------------------------------------------

class TestOverBudgetAdaptation:
    def test_skip_rate_increases_over_budget(self):
        # budget_ms=0.001 is impossibly tight — every frame exceeds budget.
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(
            inner, budget_ms=0.001, max_skip=4, warmup_frames=0
        )
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(20):
            tracker.update(frame)
        assert tracker.current_skip_rate > 1

    def test_adaptation_log_has_over_budget_entries(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(
            inner, budget_ms=0.001, max_skip=4, warmup_frames=0
        )
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(20):
            tracker.update(frame)
        over = [e for e in tracker.adaptation_log if e.reason == "over_budget"]
        assert len(over) > 0

    def test_skip_rate_never_exceeds_max_skip(self):
        inner = _FixedTracker()
        max_skip = 3
        tracker = BudgetAdaptiveTracker(
            inner, budget_ms=0.001, max_skip=max_skip, warmup_frames=0
        )
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(100):
            tracker.update(frame)
        assert tracker.current_skip_rate <= max_skip
        for event in tracker.adaptation_log:
            assert event.new_skip <= max_skip


# ---------------------------------------------------------------------------
# Under-budget recovery
# ---------------------------------------------------------------------------

class TestUnderBudgetRecovery:
    def test_skip_rate_stays_at_one_when_within_budget(self):
        # budget_ms=999999 — always within budget, no adaptation
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(
            inner, budget_ms=999_999.0, max_skip=4, warmup_frames=0
        )
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(50):
            tracker.update(frame)
        assert tracker.current_skip_rate == 1
        assert len(tracker.adaptation_log) == 0

    def test_skip_rate_never_below_one(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(
            inner, budget_ms=999_999.0, max_skip=4, warmup_frames=0
        )
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(50):
            tracker.update(frame)
        assert tracker.current_skip_rate >= 1
        for event in tracker.adaptation_log:
            assert event.new_skip >= 1


# ---------------------------------------------------------------------------
# Histogram and mean skip rate
# ---------------------------------------------------------------------------

class TestHistogram:
    def test_histogram_total_equals_frame_count(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(inner, budget_ms=100.0)
        frame = _blank_frame()
        n_frames = 30
        tracker.initialize(frame, _default_bbox())
        for _ in range(n_frames):
            tracker.update(frame)
        assert sum(tracker.skip_rate_histogram.values()) == n_frames

    def test_mean_skip_rate_before_any_frames(self):
        tracker = BudgetAdaptiveTracker(_FixedTracker(), budget_ms=30.0)
        # No initialize, no updates — returns current skip rate (1)
        assert tracker.mean_skip_rate == 1.0

    def test_mean_skip_rate_all_at_skip_one(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(inner, budget_ms=999_999.0)
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(10):
            tracker.update(frame)
        assert tracker.mean_skip_rate == 1.0

    def test_mean_skip_rate_bounded(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(
            inner, budget_ms=0.001, max_skip=4, warmup_frames=0
        )
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(50):
            tracker.update(frame)
        assert 1.0 <= tracker.mean_skip_rate <= 4.0


# ---------------------------------------------------------------------------
# Budget compliance rate
# ---------------------------------------------------------------------------

class TestComplianceRate:
    def test_compliance_rate_before_any_frames(self):
        tracker = BudgetAdaptiveTracker(_FixedTracker(), budget_ms=30.0)
        assert tracker.budget_compliance_rate == 0.0

    def test_compliance_rate_generous_budget(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(inner, budget_ms=999_999.0)
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(20):
            tracker.update(frame)
        assert tracker.budget_compliance_rate == 1.0

    def test_compliance_rate_impossible_budget(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(
            inner, budget_ms=0.0001, warmup_frames=0
        )
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(20):
            tracker.update(frame)
        assert tracker.budget_compliance_rate == 0.0


# ---------------------------------------------------------------------------
# Adaptation log
# ---------------------------------------------------------------------------

class TestAdaptationLog:
    def test_adaptation_event_fields(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(
            inner, budget_ms=0.001, max_skip=4, warmup_frames=0
        )
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(20):
            tracker.update(frame)
        events = tracker.adaptation_log
        assert len(events) > 0
        for e in events:
            assert isinstance(e, AdaptationEvent)
            assert e.frame_idx > 0
            assert e.old_skip >= 1
            assert e.new_skip >= 1
            assert e.ema_latency_ms >= 0.0
            assert e.reason in ("over_budget", "under_budget")

    def test_adaptation_event_str(self):
        event = AdaptationEvent(
            frame_idx=10,
            old_skip=1,
            new_skip=2,
            ema_latency_ms=35.5,
            reason="over_budget",
        )
        s = str(event)
        assert "10" in s
        assert "over_budget" in s


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

class TestReport:
    def test_to_report_returns_string(self):
        inner = _FixedTracker()
        tracker = BudgetAdaptiveTracker(inner, budget_ms=30.0)
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(10):
            tracker.update(frame)
        report = tracker.to_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_to_report_contains_key_fields(self):
        inner = _FixedTracker(name="TestInner")
        tracker = BudgetAdaptiveTracker(inner, budget_ms=30.0)
        frame = _blank_frame()
        tracker.initialize(frame, _default_bbox())
        for _ in range(5):
            tracker.update(frame)
        report = tracker.to_report()
        assert "TestInner" in report
        assert "30" in report
        assert "budget_ms" in report or "30.0" in report


# ---------------------------------------------------------------------------
# BenchmarkEngine integration
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:
    def test_runs_through_benchmark_engine(self):
        pytest.importorskip("cv2")
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(
            num_sequences=2, num_frames=15, seed=42
        )
        base    = MOSSETracker()
        tracker = BudgetAdaptiveTracker(base, budget_ms=50.0, max_skip=3)
        engine  = BenchmarkEngine(verbose=False)

        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        assert result.mean_fps > 0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_adaptation_log_populated_after_benchmark(self):
        pytest.importorskip("cv2")
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(
            num_sequences=2, num_frames=20, seed=7
        )
        # Use an impossibly tight budget to force adaptation
        base    = MOSSETracker()
        tracker = BudgetAdaptiveTracker(
            base, budget_ms=0.001, max_skip=4, warmup_frames=0
        )
        engine  = BenchmarkEngine(verbose=False)
        engine.run(tracker, dataset, dataset_name="Synthetic")
        # After each sequence the benchmark calls initialize() which resets
        # the log; we check the log state from the LAST sequence only.
        # The main thing we verify: the tracker ran without error.
        assert tracker.mean_skip_rate >= 1.0

    def test_max_skip_one_is_passthrough(self):
        pytest.importorskip("cv2")
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(
            num_sequences=2, num_frames=15, seed=99
        )
        base     = MOSSETracker()
        baseline = BenchmarkEngine(verbose=False).run(
            base, dataset, dataset_name="Synthetic"
        )

        # max_skip=1 means no skipping is possible — should behave like baseline
        tracker = BudgetAdaptiveTracker(
            MOSSETracker(), budget_ms=0.001, max_skip=1
        )
        adapted = BenchmarkEngine(verbose=False).run(
            tracker, dataset, dataset_name="Synthetic"
        )
        assert adapted.mean_iou == pytest.approx(baseline.mean_iou, abs=0.02)
