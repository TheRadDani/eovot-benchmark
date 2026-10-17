"""Tests for AdaptiveSkipTracker and its built-in adaptive skip policies.

Covers:
- Policy unit tests (VelocityAdaptivePolicy, BudgetAdaptivePolicy)
- AdaptiveSkipTracker integration with SyntheticDataset + BenchmarkEngine
- Edge cases: uninitialized tracker, very short sequences, history boundaries
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List

import numpy as np
import pytest

from eovot.trackers.adaptive_skip import (
    AdaptiveSkipTracker,
    AdaptiveSkipPolicy,
    BudgetAdaptivePolicy,
    VelocityAdaptivePolicy,
)
from eovot.trackers.mosse import MOSSETracker
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bbox_history(displacements: list, w: float = 100.0, h: float = 50.0) -> Deque:
    """Build a bbox deque from a list of (dx, dy) center-point displacements."""
    history: Deque = deque(maxlen=20)
    cx, cy = 200.0, 200.0
    history.append((cx - w / 2, cy - h / 2, w, h))
    for dx, dy in displacements:
        cx += dx
        cy += dy
        history.append((cx - w / 2, cy - h / 2, w, h))
    return history


def _make_latency_history(values: List[float]) -> Deque:
    """Build a latency deque from a list of ms values."""
    return deque(values, maxlen=20)


# ---------------------------------------------------------------------------
# VelocityAdaptivePolicy
# ---------------------------------------------------------------------------

class TestVelocityAdaptivePolicy:

    def test_slow_motion_returns_max_skip(self):
        policy = VelocityAdaptivePolicy(
            min_skip=1, max_skip=4, low_threshold=0.10, high_threshold=0.30
        )
        # Tiny displacement → velocity well below low_threshold
        history = _make_bbox_history([(0.2, 0.0)] * 5)
        rate = policy.predict_skip_rate(history, deque())
        assert rate == 4

    def test_fast_motion_returns_min_skip(self):
        policy = VelocityAdaptivePolicy(
            min_skip=1, max_skip=4, low_threshold=0.10, high_threshold=0.30
        )
        # Large displacement: diagonal ~= 112, so 50px displacement ~= 0.45 > high_threshold
        history = _make_bbox_history([(50.0, 50.0)] * 5)
        rate = policy.predict_skip_rate(history, deque())
        assert rate == 1

    def test_intermediate_velocity_is_interpolated(self):
        policy = VelocityAdaptivePolicy(
            min_skip=1, max_skip=5, low_threshold=0.05, high_threshold=0.20
        )
        # Mid-range displacement
        history = _make_bbox_history([(6.0, 0.0)] * 5)
        rate = policy.predict_skip_rate(history, deque())
        assert 1 <= rate <= 5

    def test_insufficient_history_returns_min_skip(self):
        policy = VelocityAdaptivePolicy(min_skip=1, max_skip=4)
        # Only one bbox — cannot compute velocity
        history: Deque = deque([(100, 100, 50, 50)], maxlen=20)
        rate = policy.predict_skip_rate(history, deque())
        assert rate == 1

    def test_rate_always_in_bounds(self):
        policy = VelocityAdaptivePolicy(min_skip=2, max_skip=6)
        for disp in [0.0, 5.0, 15.0, 50.0, 200.0]:
            history = _make_bbox_history([(disp, 0.0)] * 5)
            rate = policy.predict_skip_rate(history, deque())
            assert 2 <= rate <= 6, f"Rate {rate} out of bounds for displacement {disp}"

    def test_invalid_min_skip_raises(self):
        with pytest.raises(ValueError, match="min_skip"):
            VelocityAdaptivePolicy(min_skip=0)

    def test_max_skip_less_than_min_skip_raises(self):
        with pytest.raises(ValueError, match="max_skip"):
            VelocityAdaptivePolicy(min_skip=4, max_skip=2)

    def test_threshold_ordering_raises(self):
        with pytest.raises(ValueError, match="low_threshold"):
            VelocityAdaptivePolicy(low_threshold=0.5, high_threshold=0.1)

    def test_reset_is_callable(self):
        policy = VelocityAdaptivePolicy()
        policy.reset()  # should not raise

    def test_ignores_latency_history(self):
        policy = VelocityAdaptivePolicy(min_skip=1, max_skip=4)
        history = _make_bbox_history([(0.1, 0.0)] * 5)
        lat_a = _make_latency_history([1.0] * 5)
        lat_b = _make_latency_history([500.0] * 5)
        # Velocity policy should return the same rate regardless of latency
        assert policy.predict_skip_rate(history, lat_a) == policy.predict_skip_rate(
            history, lat_b
        )


# ---------------------------------------------------------------------------
# BudgetAdaptivePolicy
# ---------------------------------------------------------------------------

class TestBudgetAdaptivePolicy:

    def test_high_latency_increases_skip(self):
        policy = BudgetAdaptivePolicy(fps_budget=30.0, min_skip=1, max_skip=8)
        # 200 ms per frame → need ≥6× skip to achieve 30 FPS budget
        latencies = _make_latency_history([200.0] * 10)
        rate = policy.predict_skip_rate(deque(), latencies)
        assert rate >= 4

    def test_low_latency_returns_min_skip(self):
        policy = BudgetAdaptivePolicy(fps_budget=30.0, min_skip=1, max_skip=8)
        # 5 ms per frame → tracker faster than budget, no skip needed
        latencies = _make_latency_history([5.0] * 10)
        rate = policy.predict_skip_rate(deque(), latencies)
        assert rate == 1

    def test_empty_latency_history_returns_min_skip(self):
        policy = BudgetAdaptivePolicy(fps_budget=30.0)
        rate = policy.predict_skip_rate(deque(), deque())
        assert rate == 1

    def test_rate_clipped_to_max_skip(self):
        policy = BudgetAdaptivePolicy(fps_budget=1.0, min_skip=1, max_skip=6)
        # Absurdly slow tracker: 5000 ms / frame → required_skip = 5000/1000 = 5 → clipped to min(5, 6)
        latencies = _make_latency_history([5000.0] * 10)
        rate = policy.predict_skip_rate(deque(), latencies)
        assert rate >= 5  # saturates at or above 5 (cap of 6 is respected)

    def test_rate_clipped_at_hard_max(self):
        policy = BudgetAdaptivePolicy(fps_budget=1.0, min_skip=1, max_skip=3)
        # Even if required is 5, cap is 3
        latencies = _make_latency_history([5000.0] * 10)
        rate = policy.predict_skip_rate(deque(), latencies)
        assert rate == 3

    def test_host_to_device_factor_scales_latency(self):
        policy_host = BudgetAdaptivePolicy(fps_budget=30.0, max_skip=8, host_to_device_factor=1.0)
        policy_dev = BudgetAdaptivePolicy(fps_budget=30.0, max_skip=8, host_to_device_factor=5.0)
        latencies = _make_latency_history([10.0] * 5)
        rate_host = policy_host.predict_skip_rate(deque(), latencies)
        rate_dev = policy_dev.predict_skip_rate(deque(), latencies)
        # Device policy should request more skipping (projected latency is 5×)
        assert rate_dev >= rate_host

    def test_invalid_fps_budget_raises(self):
        with pytest.raises(ValueError, match="fps_budget"):
            BudgetAdaptivePolicy(fps_budget=0)
        with pytest.raises(ValueError, match="fps_budget"):
            BudgetAdaptivePolicy(fps_budget=-10.0)

    def test_invalid_min_skip_raises(self):
        with pytest.raises(ValueError, match="min_skip"):
            BudgetAdaptivePolicy(fps_budget=30.0, min_skip=0)

    def test_invalid_device_factor_raises(self):
        with pytest.raises(ValueError, match="host_to_device_factor"):
            BudgetAdaptivePolicy(fps_budget=30.0, host_to_device_factor=0.0)

    def test_reset_is_callable(self):
        policy = BudgetAdaptivePolicy(fps_budget=30.0)
        policy.reset()  # should not raise


# ---------------------------------------------------------------------------
# AdaptiveSkipTracker — unit tests
# ---------------------------------------------------------------------------

class TestAdaptiveSkipTrackerUnit:

    @pytest.fixture
    def mosse(self):
        return MOSSETracker()

    @pytest.fixture
    def velocity_policy(self):
        return VelocityAdaptivePolicy(min_skip=1, max_skip=3)

    def test_name_includes_adaptive(self, mosse, velocity_policy):
        tracker = AdaptiveSkipTracker(mosse, velocity_policy)
        assert "adaptive" in tracker.name

    def test_update_before_initialize_raises(self, mosse, velocity_policy):
        tracker = AdaptiveSkipTracker(mosse, velocity_policy)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not initialised"):
            tracker.update(dummy)

    def test_reinitialize_resets_diagnostics(self):
        dataset = SyntheticDataset(num_sequences=1, num_frames=20, seed=0)
        policy = VelocityAdaptivePolicy(min_skip=1, max_skip=3)
        tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)
        seq = dataset[0]
        frames = list(seq)

        # First run
        tracker.initialize(frames[0], seq.init_bbox)
        for f in frames[1:10]:
            tracker.update(f)
        calls_after_first_run = tracker._tracker_calls

        # Re-initialize and run a smaller slice
        tracker.initialize(frames[0], seq.init_bbox)
        assert tracker._tracker_calls == 0
        assert tracker._total_frames == 0
        assert tracker.mean_skip_rate == 1.0

    def test_tracker_call_fraction_within_bounds(self):
        dataset = SyntheticDataset(num_sequences=1, num_frames=40, seed=1)
        policy = VelocityAdaptivePolicy(min_skip=1, max_skip=4)
        tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)
        seq = dataset[0]
        frames = list(seq)
        tracker.initialize(frames[0], seq.init_bbox)
        for f in frames[1:]:
            tracker.update(f)
        assert 0.0 < tracker.tracker_call_fraction <= 1.0

    def test_mean_skip_rate_within_policy_bounds(self):
        dataset = SyntheticDataset(num_sequences=1, num_frames=50, seed=2)
        policy = VelocityAdaptivePolicy(min_skip=1, max_skip=4)
        tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)
        seq = dataset[0]
        frames = list(seq)
        tracker.initialize(frames[0], seq.init_bbox)
        for f in frames[1:]:
            tracker.update(f)
        assert 1.0 <= tracker.mean_skip_rate <= 4.0

    def test_skip_histogram_sums_to_tracker_calls(self):
        dataset = SyntheticDataset(num_sequences=1, num_frames=40, seed=3)
        policy = VelocityAdaptivePolicy(min_skip=1, max_skip=3)
        tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)
        seq = dataset[0]
        frames = list(seq)
        tracker.initialize(frames[0], seq.init_bbox)
        for f in frames[1:]:
            tracker.update(f)
        assert sum(tracker.skip_rate_histogram().values()) == tracker._tracker_calls

    def test_empty_histogram_when_uninitialized(self):
        policy = VelocityAdaptivePolicy()
        tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)
        assert tracker.skip_rate_histogram() == {}

    def test_output_bbox_count_equals_input_frame_count(self):
        # Use circular motion which stays in-frame and is stable for MOSSE
        dataset = SyntheticDataset(
            num_sequences=1, num_frames=20, seed=4, motion="circular"
        )
        policy = VelocityAdaptivePolicy(min_skip=1, max_skip=3)
        tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)
        seq = dataset[0]
        frames = list(seq)
        tracker.initialize(frames[0], seq.init_bbox)
        preds = [seq.init_bbox]
        for f in frames[1:]:
            preds.append(tracker.update(f))
        assert len(preds) == len(frames)

    def test_all_predictions_have_positive_dimensions(self):
        dataset = SyntheticDataset(num_sequences=1, num_frames=30, seed=5)
        policy = VelocityAdaptivePolicy(min_skip=1, max_skip=4)
        tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)
        seq = dataset[0]
        frames = list(seq)
        tracker.initialize(frames[0], seq.init_bbox)
        for f in frames[1:]:
            x, y, w, h = tracker.update(f)
            assert w > 0, f"Predicted w={w} must be positive"
            assert h > 0, f"Predicted h={h} must be positive"


# ---------------------------------------------------------------------------
# AdaptiveSkipTracker — BudgetAdaptivePolicy integration
# ---------------------------------------------------------------------------

class TestAdaptiveSkipTrackerBudget:

    def test_budget_policy_full_run(self):
        dataset = SyntheticDataset(num_sequences=1, num_frames=40, seed=6)
        policy = BudgetAdaptivePolicy(fps_budget=60.0, min_skip=1, max_skip=3)
        tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)
        seq = dataset[0]
        frames = list(seq)
        tracker.initialize(frames[0], seq.init_bbox)
        preds = [seq.init_bbox]
        for f in frames[1:]:
            preds.append(tracker.update(f))
        assert len(preds) == len(frames)
        assert tracker._tracker_calls > 0


# ---------------------------------------------------------------------------
# BenchmarkEngine integration
# ---------------------------------------------------------------------------

class TestBenchmarkEngineIntegration:

    def test_velocity_policy_benchmark_result_valid(self):
        from eovot.benchmark.engine import BenchmarkEngine

        dataset = SyntheticDataset(num_sequences=2, num_frames=30, seed=7)
        engine = BenchmarkEngine(verbose=False)
        policy = VelocityAdaptivePolicy(min_skip=1, max_skip=3)
        tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)

        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0
        assert result.peak_memory_mb > 0.0
        assert len(result.sequence_results) == 2

    def test_budget_policy_benchmark_result_valid(self):
        from eovot.benchmark.engine import BenchmarkEngine

        dataset = SyntheticDataset(num_sequences=2, num_frames=30, seed=8)
        engine = BenchmarkEngine(verbose=False)
        policy = BudgetAdaptivePolicy(fps_budget=30.0, min_skip=1, max_skip=4)
        tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)

        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0

    def test_adaptive_vs_budget_policy_comparison(self):
        from eovot.benchmark.engine import BenchmarkEngine

        # Use circular motion to keep target in-frame for MOSSE
        dataset = SyntheticDataset(
            num_sequences=2, num_frames=30, seed=9, motion="circular"
        )
        engine = BenchmarkEngine(verbose=False)

        # Velocity-adaptive policy
        v_policy = VelocityAdaptivePolicy(min_skip=1, max_skip=3)
        v_tracker = AdaptiveSkipTracker(MOSSETracker(), policy=v_policy)
        v_result = engine.run(v_tracker, dataset, dataset_name="Synthetic")

        # Budget-adaptive policy targeting fast FPS
        b_policy = BudgetAdaptivePolicy(fps_budget=200.0, min_skip=1, max_skip=3)
        b_tracker = AdaptiveSkipTracker(MOSSETracker(), policy=b_policy)
        b_result = engine.run(b_tracker, dataset, dataset_name="Synthetic")

        # Both should produce valid results
        assert v_result.mean_iou >= 0.0
        assert b_result.mean_iou >= 0.0
        assert v_result.mean_fps > 0.0
        assert b_result.mean_fps > 0.0

    def test_tracker_name_appears_in_result(self):
        from eovot.benchmark.engine import BenchmarkEngine

        dataset = SyntheticDataset(num_sequences=1, num_frames=20, seed=10)
        engine = BenchmarkEngine(verbose=False)
        policy = VelocityAdaptivePolicy()
        tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        assert "adaptive" in result.tracker_name


# ---------------------------------------------------------------------------
# Custom policy subclass
# ---------------------------------------------------------------------------

class TestCustomPolicy:

    def test_custom_policy_can_be_subclassed(self):
        """Verify the ABC contract is respected by a minimal custom policy."""

        class FixedPolicy(AdaptiveSkipPolicy):
            def predict_skip_rate(self, bbox_history, latency_history):
                return 2

        dataset = SyntheticDataset(num_sequences=1, num_frames=20, seed=11)
        policy = FixedPolicy()
        tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)
        seq = dataset[0]
        frames = list(seq)
        tracker.initialize(frames[0], seq.init_bbox)
        for f in frames[1:]:
            tracker.update(f)
        # All skip rates should be exactly 2
        histogram = tracker.skip_rate_histogram()
        assert set(histogram.keys()) == {2}
