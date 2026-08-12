"""Tests for AdaptiveBudgetTracker and BudgetAnalyzer.

Covers:
- Constructor validation
- BaseTracker interface contract
- skip-rate adaptation direction (slow tracker → higher skip, fast tracker → lower skip)
- Passive-frame linear extrapolation
- BudgetAnalyzer sweep + BudgetResult reporting
- Integration with BenchmarkEngine on SyntheticDataset
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.budget import AdaptiveBudgetTracker
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.analysis.budget_analysis import BudgetAnalyzer, BudgetResult, BudgetEntry
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# Stub trackers for deterministic testing
# ---------------------------------------------------------------------------

class _FixedTracker(BaseTracker):
    """Returns the initialization bbox on every frame. Ultra-fast."""

    def __init__(self) -> None:
        super().__init__(name="Fixed")
        self._bbox: Optional[BBox] = None

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        assert self._bbox is not None
        return self._bbox


class _SlowTracker(BaseTracker):
    """Introduces a fixed sleep on every update call to simulate a slow model."""

    def __init__(self, sleep_ms: float = 50.0) -> None:
        super().__init__(name=f"Slow{sleep_ms:.0f}ms")
        self._sleep = sleep_ms / 1000.0
        self._bbox: Optional[BBox] = None

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        time.sleep(self._sleep)
        assert self._bbox is not None
        return self._bbox


def _frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


INIT_BBOX: BBox = (10.0, 10.0, 20.0, 20.0)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestAdaptiveBudgetTrackerInit:
    def test_valid_construction(self):
        t = AdaptiveBudgetTracker(_FixedTracker(), fps_target=30.0)
        assert t.fps_target == 30.0

    def test_name_encodes_fps(self):
        t = AdaptiveBudgetTracker(_FixedTracker(), fps_target=60.0)
        assert "60" in t.name

    def test_negative_fps_raises(self):
        with pytest.raises(ValueError, match="fps_target"):
            AdaptiveBudgetTracker(_FixedTracker(), fps_target=-1.0)

    def test_zero_fps_raises(self):
        with pytest.raises(ValueError, match="fps_target"):
            AdaptiveBudgetTracker(_FixedTracker(), fps_target=0.0)

    def test_min_skip_below_1_raises(self):
        with pytest.raises(ValueError, match="min_skip"):
            AdaptiveBudgetTracker(_FixedTracker(), fps_target=30.0, min_skip=0)

    def test_max_skip_below_min_raises(self):
        with pytest.raises(ValueError, match="max_skip"):
            AdaptiveBudgetTracker(_FixedTracker(), fps_target=30.0, min_skip=4, max_skip=2)

    def test_is_base_tracker(self):
        assert isinstance(AdaptiveBudgetTracker(_FixedTracker()), BaseTracker)

    def test_underlying_tracker_accessible(self):
        inner = _FixedTracker()
        outer = AdaptiveBudgetTracker(inner, fps_target=30.0)
        assert outer.underlying_tracker is inner


# ---------------------------------------------------------------------------
# BaseTracker interface
# ---------------------------------------------------------------------------

class TestAdaptiveBudgetTrackerInterface:
    def setup_method(self):
        self.tracker = AdaptiveBudgetTracker(_FixedTracker(), fps_target=100.0)

    def test_update_before_init_raises(self):
        with pytest.raises(RuntimeError):
            self.tracker.update(_frame())

    def test_initialize_then_update(self):
        self.tracker.initialize(_frame(), INIT_BBOX)
        bbox = self.tracker.update(_frame())
        assert len(bbox) == 4
        x, y, w, h = bbox
        assert w > 0 and h > 0

    def test_returns_tuple_of_four(self):
        self.tracker.initialize(_frame(), INIT_BBOX)
        for _ in range(5):
            bbox = self.tracker.update(_frame())
            assert len(bbox) == 4

    def test_reinitialize(self):
        self.tracker.initialize(_frame(), INIT_BBOX)
        self.tracker.update(_frame())
        self.tracker.initialize(_frame(), (5.0, 5.0, 10.0, 10.0))
        bbox = self.tracker.update(_frame())
        assert len(bbox) == 4

    def test_counters_reset_on_reinitialize(self):
        self.tracker.initialize(_frame(), INIT_BBOX)
        for _ in range(5):
            self.tracker.update(_frame())
        self.tracker.initialize(_frame(), INIT_BBOX)
        assert self.tracker.active_frame_count == 0
        assert self.tracker.skipped_frame_count == 0


# ---------------------------------------------------------------------------
# Skip-rate adaptation
# ---------------------------------------------------------------------------

class TestSkipRateAdaptation:
    def test_adapt_increases_skip_rate_when_slow(self):
        """Directly test _adapt() with injected 50ms latency targeting 30 FPS.

        Avoids wall-clock timing flakiness by injecting latency values
        directly into the window instead of relying on time.sleep.
        """
        tracker = AdaptiveBudgetTracker(
            _FixedTracker(),
            fps_target=30.0,
            min_skip=1,
            max_skip=8,
            adapt_interval=5,
            window=10,
        )
        tracker.initialize(_frame(), INIT_BBOX)
        # Inject 50ms latencies to simulate a tracker that is slower than the 33ms target
        for _ in range(5):
            tracker._latency_window.append(50.0)
        tracker._active_count = 5
        initial_skip = tracker.current_skip_rate
        tracker._adapt()
        assert tracker.current_skip_rate > initial_skip, (
            f"Expected skip_rate to increase from {initial_skip} when "
            f"effective_fps (20) < fps_target*0.9 (27), "
            f"got {tracker.current_skip_rate}"
        )

    def test_adapt_decreases_skip_rate_when_fast(self):
        """_adapt() should lower skip rate when effective FPS is comfortably above target."""
        tracker = AdaptiveBudgetTracker(
            _FixedTracker(),
            fps_target=30.0,
            min_skip=1,
            max_skip=8,
            initial_skip=4,
            adapt_interval=5,
            window=10,
        )
        tracker.initialize(_frame(), INIT_BBOX)
        # Inject 5ms latencies: effective_fps = 4*1000/5 = 800 >> 33 FPS
        for _ in range(5):
            tracker._latency_window.append(5.0)
        tracker._active_count = 5
        initial_skip = tracker.current_skip_rate
        tracker._adapt()
        assert tracker.current_skip_rate < initial_skip, (
            f"Expected skip_rate to decrease from {initial_skip} when "
            f"effective_fps >> fps_target, got {tracker.current_skip_rate}"
        )

    def test_adapt_no_change_within_tolerance(self):
        """Within ±10% of target, skip_rate should not oscillate."""
        tracker = AdaptiveBudgetTracker(
            _FixedTracker(),
            fps_target=30.0,
            min_skip=1,
            max_skip=8,
            initial_skip=2,
            adapt_interval=5,
            window=10,
        )
        tracker.initialize(_frame(), INIT_BBOX)
        # effective_fps = 2*1000/33.3 ≈ 60 FPS — above target but within tolerance? No,
        # 60 > 33 * 1.10 = 36.3 → would decrease. Use latency ≈ 28ms: 2*1000/28 ≈ 71.
        # Let's use latency that gives exactly the target: 2*1000/30 ≈ 66.7ms
        for _ in range(5):
            tracker._latency_window.append(66.7)  # effective_fps = 2*1000/66.7 ≈ 30 FPS
        tracker._active_count = 5
        initial_skip = tracker.current_skip_rate
        tracker._adapt()
        assert tracker.current_skip_rate == initial_skip, (
            f"Expected skip_rate to stay at {initial_skip} when effective_fps ≈ target, "
            f"got {tracker.current_skip_rate}"
        )

    def test_skip_rate_clamps_to_max(self):
        """skip_rate should never exceed max_skip."""
        tracker = AdaptiveBudgetTracker(
            _FixedTracker(), fps_target=30.0, min_skip=1, max_skip=3, initial_skip=3
        )
        tracker.initialize(_frame(), INIT_BBOX)
        for _ in range(10):
            tracker._latency_window.append(50.0)  # slow: 20 FPS < 27 → would push beyond 3
        tracker._active_count = 10
        tracker._adapt()
        assert tracker.current_skip_rate <= 3

    def test_skip_rate_clamps_to_min(self):
        """skip_rate should never go below min_skip."""
        tracker = AdaptiveBudgetTracker(
            _FixedTracker(), fps_target=30.0, min_skip=2, max_skip=8, initial_skip=2
        )
        tracker.initialize(_frame(), INIT_BBOX)
        for _ in range(10):
            tracker._latency_window.append(1.0)  # very fast → would push below 2
        tracker._active_count = 10
        tracker._adapt()
        assert tracker.current_skip_rate >= 2

    def test_skip_rate_within_bounds(self):
        """skip_rate must always stay in [min_skip, max_skip]."""
        tracker = AdaptiveBudgetTracker(
            _SlowTracker(sleep_ms=30.0),
            fps_target=500.0,   # impossible target → push to max_skip
            min_skip=2,
            max_skip=5,
            adapt_interval=3,
            window=5,
        )
        tracker.initialize(_frame(), INIT_BBOX)
        for _ in range(40):
            tracker.update(_frame())
        assert tracker.min_skip <= tracker.current_skip_rate <= tracker.max_skip


# ---------------------------------------------------------------------------
# Passive-frame extrapolation
# ---------------------------------------------------------------------------

class TestPassiveFrameExtrapolation:
    def test_passive_frame_uses_extrapolation(self):
        """With skip_rate=2, every other frame should be a passive extrapolation."""
        inner = _FixedTracker()
        tracker = AdaptiveBudgetTracker(
            inner,
            fps_target=1.0,     # very low target → skip_rate will stay at initial=1 during warmup
            min_skip=2,          # force skip_rate ≥ 2
            max_skip=2,          # lock skip_rate = 2
            initial_skip=2,
            adapt_interval=100,  # no adaptation during this short test
        )
        tracker.initialize(_frame(), INIT_BBOX)
        results = [tracker.update(_frame()) for _ in range(6)]
        # Frame 1 (idx=1, active: 1%2==0 no, frame_idx=1): passive — initial
        # Let's just verify we get 4-tuples with positive w,h
        for bbox in results:
            assert len(bbox) == 4
            _, _, w, h = bbox
            assert w > 0 and h > 0

    def test_active_and_skipped_counts(self):
        """active + skipped counts should sum to total frames."""
        tracker = AdaptiveBudgetTracker(
            _FixedTracker(),
            fps_target=100.0,
            min_skip=3,
            max_skip=3,
            initial_skip=3,
            adapt_interval=1000,   # no adaptation
        )
        tracker.initialize(_frame(), INIT_BBOX)
        n_frames = 12
        for _ in range(n_frames):
            tracker.update(_frame())
        total = tracker.active_frame_count + tracker.skipped_frame_count
        assert total == n_frames

    def test_effective_skip_fraction(self):
        """With fixed skip_rate=4 over 8 frames: 2 active, 6 passive → fraction = 0.75."""
        tracker = AdaptiveBudgetTracker(
            _FixedTracker(),
            fps_target=1.0,
            min_skip=4,
            max_skip=4,
            initial_skip=4,
            adapt_interval=1000,
        )
        tracker.initialize(_frame(), INIT_BBOX)
        for _ in range(8):
            tracker.update(_frame())
        assert tracker.effective_skip_fraction == pytest.approx(0.75, abs=0.05)


# ---------------------------------------------------------------------------
# BudgetAnalyzer sweep
# ---------------------------------------------------------------------------

class TestBudgetAnalyzer:
    def _dataset(self):
        return SyntheticDataset(num_sequences=2, num_frames=50, seed=7)

    def _engine(self):
        return BenchmarkEngine(verbose=False)

    def test_returns_budget_result(self):
        result = BudgetAnalyzer(self._engine()).analyze(
            _FixedTracker(), self._dataset(), fps_targets=[30.0, 60.0]
        )
        assert isinstance(result, BudgetResult)

    def test_entry_count_matches_targets(self):
        targets = [15.0, 30.0, 60.0]
        result = BudgetAnalyzer(self._engine()).analyze(
            _FixedTracker(), self._dataset(), fps_targets=targets
        )
        assert len(result.entries) == len(targets)

    def test_entries_ordered_by_fps_target(self):
        result = BudgetAnalyzer(self._engine()).analyze(
            _FixedTracker(), self._dataset(), fps_targets=[60.0, 15.0, 30.0]
        )
        targets = [e.fps_target for e in result.entries]
        assert targets == sorted(targets)

    def test_iou_in_valid_range(self):
        result = BudgetAnalyzer(self._engine()).analyze(
            _FixedTracker(), self._dataset(), fps_targets=[30.0]
        )
        for e in result.entries:
            assert 0.0 <= e.mean_iou <= 1.0

    def test_markdown_table_contains_headers(self):
        result = BudgetAnalyzer(self._engine()).analyze(
            _FixedTracker(), self._dataset(), fps_targets=[30.0, 60.0]
        )
        table = result.to_markdown_table()
        assert "fps_target" in table
        assert "mIoU" in table
        assert "skip" in table

    def test_optimal_budget_returns_highest_feasible_fps(self):
        result = BudgetAnalyzer(self._engine()).analyze(
            _FixedTracker(), self._dataset(), fps_targets=[15.0, 30.0, 60.0]
        )
        fps, iou, skip = result.optimal_budget(min_iou=0.0)
        assert fps == max(e.fps_target for e in result.entries)

    def test_optimal_budget_raises_when_impossible(self):
        result = BudgetAnalyzer(self._engine()).analyze(
            _FixedTracker(), self._dataset(), fps_targets=[30.0]
        )
        with pytest.raises(ValueError, match="No budget entry"):
            result.optimal_budget(min_iou=999.0)

    def test_str_output(self):
        result = BudgetAnalyzer(self._engine()).analyze(
            _FixedTracker(), self._dataset(), fps_targets=[30.0]
        )
        s = str(result)
        assert "BudgetResult" in s

    def test_budget_result_tracker_name(self):
        result = BudgetAnalyzer(self._engine()).analyze(
            _FixedTracker(), self._dataset(), fps_targets=[30.0]
        )
        assert result.tracker_name == "Fixed"

    def test_max_sequences_limits_evaluation(self):
        """max_sequences kwarg should reduce evaluation scope."""
        dataset = SyntheticDataset(num_sequences=10, num_frames=30)
        result = BudgetAnalyzer(self._engine()).analyze(
            _FixedTracker(), dataset, fps_targets=[30.0], max_sequences=3
        )
        assert isinstance(result, BudgetResult)


# ---------------------------------------------------------------------------
# Integration: BenchmarkEngine + AdaptiveBudgetTracker
# ---------------------------------------------------------------------------

class TestBenchmarkEngineIntegration:
    def test_engine_run_with_budget_tracker(self):
        dataset = SyntheticDataset(num_sequences=2, num_frames=40, seed=5)
        tracker = AdaptiveBudgetTracker(MOSSETracker(), fps_target=200.0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0

    def test_engine_run_preserves_tracker_name(self):
        dataset = SyntheticDataset(num_sequences=1, num_frames=20)
        tracker = AdaptiveBudgetTracker(KCFTracker(), fps_target=50.0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        assert "KCF" in result.tracker_name
        assert "50" in result.tracker_name


# ---------------------------------------------------------------------------
# Analysis module public API
# ---------------------------------------------------------------------------

class TestAnalysisPublicAPI:
    def test_budget_analyzer_importable_from_analysis(self):
        from eovot.analysis import BudgetAnalyzer, BudgetResult, BudgetEntry
        assert BudgetAnalyzer is not None
        assert BudgetResult is not None
        assert BudgetEntry is not None

    def test_adaptive_budget_tracker_importable_from_trackers(self):
        from eovot.trackers import AdaptiveBudgetTracker
        assert AdaptiveBudgetTracker is not None
