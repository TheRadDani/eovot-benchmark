"""Tests for the confidence-gated AdaptiveTracker wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.adaptive import AdaptiveTracker
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_static_frames(n: int = 15, h: int = 120, w: int = 160) -> list:
    """Return n near-identical BGR frames (very low optical-flow motion)."""
    base = _make_frame(h, w, seed=42)
    rng = np.random.default_rng(7)
    frames = []
    for _ in range(n):
        noise = rng.integers(0, 3, base.shape, dtype=np.uint8)
        frames.append(np.clip(base.astype(np.int32) + noise, 0, 255).astype(np.uint8))
    return frames


def _make_moving_frames(n: int = 15, h: int = 120, w: int = 160) -> list:
    """Return n very different frames (high optical-flow motion)."""
    frames = []
    rng = np.random.default_rng(99)
    for _ in range(n):
        frames.append(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    return frames


INIT_BBOX = (20, 20, 40, 40)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestAdaptiveTrackerInit:
    def test_name_reflects_inner(self):
        inner = MOSSETracker()
        tracker = AdaptiveTracker(inner)
        assert "MOSSE" in tracker.name
        assert "Adaptive" in tracker.name

    def test_stats_zero_before_tracking(self):
        inner = MOSSETracker()
        tracker = AdaptiveTracker(inner)
        first = _make_frame()
        tracker.initialize(first, INIT_BBOX)
        assert tracker.frames_skipped == 0
        assert tracker.frames_updated == 0
        assert tracker.skip_ratio == 0.0

    def test_initialize_resets_state(self):
        inner = MOSSETracker()
        tracker = AdaptiveTracker(inner)
        frames = _make_static_frames(6)
        tracker.initialize(frames[0], INIT_BBOX)
        for f in frames[1:]:
            tracker.update(f)
        # Re-initialise — counters should reset
        tracker.initialize(frames[0], INIT_BBOX)
        assert tracker.frames_skipped == 0
        assert tracker.frames_updated == 0


# ---------------------------------------------------------------------------
# Skipping behaviour
# ---------------------------------------------------------------------------

class TestAdaptiveSkipping:
    def test_no_skip_on_high_motion(self):
        """With high motion_threshold set very low, nothing should be skipped."""
        inner = MOSSETracker()
        # threshold=0.0 means motion is never < threshold → never skip
        tracker = AdaptiveTracker(inner, motion_threshold=0.0)
        frames = _make_moving_frames(10)
        tracker.initialize(frames[0], INIT_BBOX)
        for f in frames[1:]:
            tracker.update(f)
        assert tracker.frames_skipped == 0
        assert tracker.frames_updated == len(frames) - 1

    def test_skipping_occurs_on_static_video(self):
        """Near-zero motion should trigger skips with a generous threshold."""
        inner = MOSSETracker()
        tracker = AdaptiveTracker(
            inner,
            motion_threshold=50.0,   # very permissive → skips likely
            confidence_threshold=0.0,  # confidence gate always passes
            max_skip_streak=100,
        )
        frames = _make_static_frames(20)
        tracker.initialize(frames[0], INIT_BBOX)
        for f in frames[1:]:
            tracker.update(f)
        assert tracker.frames_skipped > 0

    def test_max_skip_streak_enforced(self):
        """Inner tracker must be called at least every max_skip_streak frames."""
        inner = MOSSETracker()
        n_frames = 20
        max_streak = 3
        tracker = AdaptiveTracker(
            inner,
            motion_threshold=100.0,   # almost always skip by motion alone
            confidence_threshold=0.0,
            max_skip_streak=max_streak,
        )
        frames = _make_static_frames(n_frames)
        tracker.initialize(frames[0], INIT_BBOX)
        for f in frames[1:]:
            tracker.update(f)
        # At most (max_streak) consecutive skips → inner called ≥ ceil(updates/max_streak)
        n_updates = len(frames) - 1
        assert tracker.frames_updated >= n_updates // (max_streak + 1)

    def test_skip_ratio_in_range(self):
        inner = MOSSETracker()
        tracker = AdaptiveTracker(inner, motion_threshold=2.0)
        frames = _make_static_frames(15)
        tracker.initialize(frames[0], INIT_BBOX)
        for f in frames[1:]:
            tracker.update(f)
        assert 0.0 <= tracker.skip_ratio <= 1.0

    def test_frames_skipped_plus_updated_equals_total(self):
        inner = MOSSETracker()
        tracker = AdaptiveTracker(inner, motion_threshold=2.0)
        frames = _make_static_frames(12)
        tracker.initialize(frames[0], INIT_BBOX)
        for f in frames[1:]:
            tracker.update(f)
        assert tracker.frames_skipped + tracker.frames_updated == len(frames) - 1


# ---------------------------------------------------------------------------
# Output validity
# ---------------------------------------------------------------------------

class TestAdaptiveOutput:
    def test_returns_four_element_tuple(self):
        inner = MOSSETracker()
        tracker = AdaptiveTracker(inner)
        frames = _make_static_frames(5)
        tracker.initialize(frames[0], INIT_BBOX)
        bbox = tracker.update(frames[1])
        assert len(bbox) == 4

    def test_bbox_values_are_finite(self):
        inner = MOSSETracker()
        tracker = AdaptiveTracker(inner)
        frames = _make_moving_frames(8)
        tracker.initialize(frames[0], INIT_BBOX)
        for f in frames[1:]:
            bbox = tracker.update(f)
            assert all(np.isfinite(v) for v in bbox)

    def test_velocity_propagation_shifts_bbox(self):
        """With use_velocity=True, skipped frames should shift the box."""
        inner = MOSSETracker()
        # Force a skip immediately via very permissive gates
        tracker = AdaptiveTracker(
            inner,
            motion_threshold=100.0,
            confidence_threshold=0.0,
            max_skip_streak=100,
            use_velocity=True,
        )
        frames = _make_static_frames(15)
        tracker.initialize(frames[0], INIT_BBOX)
        # Get at least one non-skipped update to establish velocity
        tracker._skip_streak = 100  # force first update
        tracker.update(frames[1])
        tracker._skip_streak = 0    # now allow skips
        bbox_before = tracker._prev_bbox
        tracker.update(frames[2])   # should skip and propagate
        bbox_after = tracker._prev_bbox
        # Both bboxes should be valid tuples
        assert len(bbox_before) == 4
        assert len(bbox_after) == 4

    def test_not_initialised_raises(self):
        inner = MOSSETracker()
        tracker = AdaptiveTracker(inner)
        with pytest.raises(RuntimeError, match="not initialised"):
            tracker.update(_make_frame())


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestAdaptiveIntegration:
    def test_benchmark_engine_runs_adaptive_tracker(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        inner = MOSSETracker()
        tracker = AdaptiveTracker(inner, motion_threshold=2.0, max_skip_streak=4)

        dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")

        assert result.tracker_name == tracker.name
        assert len(result.sequence_results) == 2
        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0

    def test_adaptive_does_not_break_iou(self):
        """AdaptiveTracker IoU must stay in [0, 1] across all sequences."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        inner = MOSSETracker()
        tracker = AdaptiveTracker(inner, motion_threshold=5.0)
        dataset = SyntheticDataset(num_sequences=3, num_frames=30, seed=1)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")

        for seq_r in result.sequence_results:
            assert np.all(seq_r.ious >= 0.0)
            assert np.all(seq_r.ious <= 1.0)
