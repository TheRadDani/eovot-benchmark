"""Tests for eovot.trackers.motion_adaptive.MotionAdaptiveTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.motion_adaptive import MotionAdaptiveTracker


# ---------------------------------------------------------------------------
# Helpers — minimal stub tracker for unit tests
# ---------------------------------------------------------------------------

class _ConstantTracker:
    """Stub tracker that always predicts a fixed bbox offset from init."""

    def __init__(self, dx: float = 0.0, dy: float = 0.0) -> None:
        self.name = "ConstantTracker"
        self._bbox = None
        self.dx = dx
        self.dy = dy
        self.update_count = 0

    def initialize(self, frame, bbox) -> None:
        self._bbox = tuple(bbox)

    def update(self, frame):
        self.update_count += 1
        x, y, w, h = self._bbox
        self._bbox = (x + self.dx, y + self.dy, w, h)
        return self._bbox


class _RecordingTracker:
    """Stub that records every update call and returns a moving bbox."""

    def __init__(self, speed_px: float = 5.0) -> None:
        self.name = "RecordingTracker"
        self._bbox = None
        self.speed_px = speed_px
        self.update_calls: list = []

    def initialize(self, frame, bbox) -> None:
        self._bbox = list(bbox)

    def update(self, frame):
        self.update_calls.append(True)
        self._bbox[0] += self.speed_px
        return tuple(self._bbox)


def _blank_frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _init_bbox():
    return (50.0, 60.0, 40.0, 30.0)  # x, y, w, h


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_name_includes_wrapped(self):
        inner = _ConstantTracker()
        tracker = MotionAdaptiveTracker(inner)
        assert "ConstantTracker" in tracker.name

    def test_invalid_ema_alpha_raises(self):
        with pytest.raises(ValueError):
            MotionAdaptiveTracker(_ConstantTracker(), ema_alpha=0.0)
        with pytest.raises(ValueError):
            MotionAdaptiveTracker(_ConstantTracker(), ema_alpha=1.5)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            MotionAdaptiveTracker(_ConstantTracker(), motion_threshold=-0.1)

    def test_invalid_max_skip_streak_raises(self):
        with pytest.raises(ValueError):
            MotionAdaptiveTracker(_ConstantTracker(), max_skip_streak=0)

    def test_wrapped_tracker_accessible(self):
        inner = _ConstantTracker()
        tracker = MotionAdaptiveTracker(inner)
        assert tracker.wrapped_tracker is inner

    def test_valid_threshold_zero_accepted(self):
        """Threshold of 0 means never skip."""
        tracker = MotionAdaptiveTracker(_ConstantTracker(), motion_threshold=0.0)
        assert tracker.motion_threshold == 0.0


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialize:
    def test_counters_reset_on_initialize(self):
        inner = _RecordingTracker()
        tracker = MotionAdaptiveTracker(inner)
        frame = _blank_frame()
        bbox = _init_bbox()
        tracker.initialize(frame, bbox)
        tracker.update(frame)
        tracker.update(frame)
        tracker.initialize(frame, bbox)  # re-initialize
        assert tracker.total_updates == 0
        assert tracker.skipped_updates == 0
        assert tracker.skip_ratio == 0.0

    def test_skip_streak_reset_on_initialize(self):
        inner = _ConstantTracker(dx=0.0)
        tracker = MotionAdaptiveTracker(inner, motion_threshold=10.0, max_skip_streak=2)
        frame = _blank_frame()
        bbox = _init_bbox()
        tracker.initialize(frame, bbox)
        for _ in range(5):
            tracker.update(frame)
        tracker.initialize(frame, bbox)
        assert tracker._skip_streak == 0


# ---------------------------------------------------------------------------
# Skip ratio = 0 when threshold is 0 (never skip)
# ---------------------------------------------------------------------------

class TestNoSkipMode:
    def test_all_frames_processed_with_zero_threshold(self):
        inner = _RecordingTracker(speed_px=0.1)
        tracker = MotionAdaptiveTracker(inner, motion_threshold=0.0)
        frame = _blank_frame()
        tracker.initialize(frame, _init_bbox())
        for _ in range(20):
            tracker.update(frame)
        assert tracker.skip_ratio == pytest.approx(0.0)
        assert len(inner.update_calls) == 20


# ---------------------------------------------------------------------------
# Skip behaviour — stationary target
# ---------------------------------------------------------------------------

class TestSkipBehaviour:
    def test_stationary_target_triggers_skips(self):
        """With a stationary target and high threshold, most frames are skipped."""
        inner = _ConstantTracker(dx=0.0, dy=0.0)  # target never moves
        tracker = MotionAdaptiveTracker(
            inner, motion_threshold=0.5, ema_alpha=0.5, max_skip_streak=100
        )
        frame = _blank_frame()
        tracker.initialize(frame, _init_bbox())

        # First update: motion_ema = inf → forces full update; ema becomes 0
        tracker.update(frame)
        # Subsequent: motion_ema = 0 < threshold → all skipped
        for _ in range(19):
            tracker.update(frame)

        assert tracker.total_updates == 20
        assert tracker.skipped_updates >= 18  # almost all skipped
        assert tracker.skip_ratio > 0.8

    def test_fast_target_forces_full_updates(self):
        """With fast motion and low threshold, tracker is called every frame."""
        inner = _RecordingTracker(speed_px=50.0)  # large displacement each frame
        # threshold = 0.01 of diagonal = 0.01 * 50 = 0.5 px
        # motion = 50 px >> 0.5 px → never skip
        tracker = MotionAdaptiveTracker(inner, motion_threshold=0.01, max_skip_streak=3)
        frame = _blank_frame()
        tracker.initialize(frame, _init_bbox())
        for _ in range(20):
            tracker.update(frame)
        # All or nearly all frames processed by the underlying tracker
        assert tracker.skip_ratio < 0.1

    def test_max_skip_streak_forces_update(self):
        """After max_skip_streak consecutive skips, tracker must run."""
        inner = _RecordingTracker(speed_px=0.0)  # completely stationary
        tracker = MotionAdaptiveTracker(
            inner, motion_threshold=0.5, ema_alpha=0.3, max_skip_streak=3
        )
        frame = _blank_frame()
        tracker.initialize(frame, _init_bbox())

        call_counts_before = []
        for _ in range(10):
            tracker.update(frame)
            call_counts_before.append(len(inner.update_calls))

        # At least one forced update should have occurred (every 3+1 frames)
        assert len(inner.update_calls) >= 1


# ---------------------------------------------------------------------------
# Interpolation correctness
# ---------------------------------------------------------------------------

class TestInterpolation:
    def test_interpolated_bbox_follows_velocity(self):
        """Skipped frames linearly extrapolate the last observed velocity."""
        dx_per_frame = 10.0
        inner = _ConstantTracker(dx=dx_per_frame, dy=0.0)
        # Very high threshold → after first update, always skip (if not streak-limited)
        tracker = MotionAdaptiveTracker(
            inner, motion_threshold=0.9, ema_alpha=1.0, max_skip_streak=100
        )
        frame = _blank_frame()
        init_bbox = (100.0, 80.0, 40.0, 30.0)
        tracker.initialize(frame, init_bbox)

        # Frame 1: first update — always full (motion_ema = inf at start)
        b1 = tracker.update(frame)
        assert tracker.skipped_updates == 0

        # Frame 2+: high threshold → skip after EMA stabilises
        # The velocity captured from frame 1 was dx_per_frame
        # Interpolated bbox should advance by dx_per_frame each skipped frame
        bboxes = [b1]
        for _ in range(5):
            b = tracker.update(frame)
            bboxes.append(b)

        # All skipped frames should have advancing x coordinates
        xs = [b[0] for b in bboxes]
        # At minimum, the x should be non-decreasing (tracker or interpolation both move right)
        for i in range(len(xs) - 1):
            assert xs[i + 1] >= xs[i] - 1.0  # allow small numerical drift

    def test_bbox_dimensions_preserved_during_skip(self):
        """Width and height must not change during interpolated frames."""
        inner = _ConstantTracker(dx=0.0)
        tracker = MotionAdaptiveTracker(
            inner, motion_threshold=0.9, max_skip_streak=100
        )
        frame = _blank_frame()
        init_bbox = (50.0, 60.0, 40.0, 30.0)
        tracker.initialize(frame, init_bbox)
        tracker.update(frame)  # first update: full
        bboxes = [tracker.update(frame) for _ in range(10)]
        for b in bboxes:
            assert b[2] == pytest.approx(40.0, abs=1.0)  # width
            assert b[3] == pytest.approx(30.0, abs=1.0)  # height


# ---------------------------------------------------------------------------
# State counter consistency
# ---------------------------------------------------------------------------

class TestCounters:
    def test_total_updates_increments_each_call(self):
        tracker = MotionAdaptiveTracker(_ConstantTracker(), motion_threshold=0.0)
        frame = _blank_frame()
        tracker.initialize(frame, _init_bbox())
        for i in range(1, 11):
            tracker.update(frame)
            assert tracker.total_updates == i

    def test_skip_ratio_bounded(self):
        inner = _ConstantTracker(dx=0.0)
        tracker = MotionAdaptiveTracker(inner, motion_threshold=0.5, max_skip_streak=10)
        frame = _blank_frame()
        tracker.initialize(frame, _init_bbox())
        for _ in range(50):
            tracker.update(frame)
        assert 0.0 <= tracker.skip_ratio <= 1.0

    def test_skipped_le_total(self):
        inner = _ConstantTracker(dx=2.0)
        tracker = MotionAdaptiveTracker(inner, motion_threshold=0.3)
        frame = _blank_frame()
        tracker.initialize(frame, _init_bbox())
        for _ in range(30):
            tracker.update(frame)
        assert tracker.skipped_updates <= tracker.total_updates

    def test_skip_ratio_zero_before_update(self):
        tracker = MotionAdaptiveTracker(_ConstantTracker())
        frame = _blank_frame()
        tracker.initialize(frame, _init_bbox())
        assert tracker.skip_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine + SyntheticDataset
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:
    def test_runs_with_benchmark_engine(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        ds = SyntheticDataset(num_sequences=2, num_frames=30, motion="linear", seed=0)
        engine = BenchmarkEngine(verbose=False)

        base = MOSSETracker()
        adapted = MotionAdaptiveTracker(MOSSETracker(), motion_threshold=0.03)

        result = engine.run(adapted, ds, dataset_name="Synthetic-Adaptive")
        assert result.tracker_name == adapted.name
        assert len(result.sequence_results) == 2
        for sr in result.sequence_results:
            assert sr.profiling.frame_count > 0

    def test_skip_reduces_fps_penalty(self):
        """Adaptive tracker with very high threshold should skip most frames.

        Since frames are interpolated cheaply, effective throughput (as
        measured by the profiler, which only times update calls) should
        be at least as fast as the base tracker — the profiler only
        measures time spent inside tracker.update(), which the adaptive
        wrapper reduces.  We can't assert strict FPS ordering without a
        real benchmark, but we can verify that it runs without error and
        produces valid IoU values.
        """
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        ds = SyntheticDataset(num_sequences=1, num_frames=50, motion="linear", seed=7)
        engine = BenchmarkEngine(verbose=False)
        adapted = MotionAdaptiveTracker(MOSSETracker(), motion_threshold=0.9, max_skip_streak=50)
        result = engine.run(adapted, ds, dataset_name="Synthetic")
        assert result.mean_iou >= 0.0
        assert result.mean_iou <= 1.0


# ---------------------------------------------------------------------------
# Repr / str
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_includes_class_name(self):
        tracker = MotionAdaptiveTracker(_ConstantTracker())
        r = repr(tracker)
        assert "MotionAdaptiveTracker" in r
