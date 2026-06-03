"""Tests for eovot.trackers.adaptive_skip."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.adaptive_skip import AdaptiveFrameSkipController, SkipStats
from eovot.trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Minimal stub tracker for testing
# ---------------------------------------------------------------------------

class _ConstantTracker(BaseTracker):
    """Returns a fixed bbox on every update call; counts invocations."""

    def __init__(self, bbox: BBox = (10.0, 10.0, 40.0, 30.0)) -> None:
        super().__init__(name="ConstantTracker")
        self._bbox = bbox
        self.call_count: int = 0

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self.call_count = 0

    def update(self, frame: np.ndarray) -> BBox:
        self.call_count += 1
        return self._bbox


def _make_frame(h: int = 64, w: int = 64, value: int = 0) -> np.ndarray:
    """Create a uniform BGR frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _make_frames(n: int, h: int = 64, w: int = 64, static: bool = True) -> list:
    """Return a list of BGR frames — either static or with random content."""
    if static:
        return [_make_frame(h, w, 42)] * n
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAdaptiveFrameSkipConstruction:
    def test_name_embeds_wrapped_tracker_name(self):
        inner = _ConstantTracker()
        ctrl = AdaptiveFrameSkipController(inner)
        assert "ConstantTracker" in ctrl.name

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="skip_threshold"):
            AdaptiveFrameSkipController(_ConstantTracker(), skip_threshold=-1.0)

    def test_zero_interval_raises(self):
        with pytest.raises(ValueError, match="min_active_interval"):
            AdaptiveFrameSkipController(_ConstantTracker(), min_active_interval=0)

    def test_wrapped_tracker_property(self):
        inner = _ConstantTracker()
        ctrl = AdaptiveFrameSkipController(inner)
        assert ctrl.wrapped_tracker is inner


# ---------------------------------------------------------------------------
# Basic runtime behaviour
# ---------------------------------------------------------------------------

class TestAdaptiveFrameSkipRuntime:
    def _run(self, frames, skip_threshold=6.0, min_active_interval=5, use_velocity=False):
        inner = _ConstantTracker()
        ctrl = AdaptiveFrameSkipController(
            inner,
            skip_threshold=skip_threshold,
            min_active_interval=min_active_interval,
            use_velocity=use_velocity,
        )
        init_bbox: BBox = (10.0, 10.0, 40.0, 30.0)
        ctrl.initialize(frames[0], init_bbox)
        results = []
        for frame in frames[1:]:
            results.append(ctrl.update(frame))
        return ctrl, inner, results

    def test_update_before_initialize_raises(self):
        ctrl = AdaptiveFrameSkipController(_ConstantTracker())
        with pytest.raises(RuntimeError):
            ctrl.update(_make_frame())

    def test_returns_bbox_for_every_frame(self):
        frames = _make_frames(20, static=True)
        _, _, results = self._run(frames)
        assert len(results) == 19

    def test_all_bboxes_are_4tuples(self):
        frames = _make_frames(10, static=True)
        _, _, results = self._run(frames)
        for bbox in results:
            assert len(bbox) == 4

    def test_static_scene_skips_most_frames(self):
        frames = _make_frames(30, static=True)
        ctrl, inner, _ = self._run(frames, skip_threshold=6.0, min_active_interval=5)
        stats = ctrl.skip_stats()
        # Static frames have zero motion; skipping should be high
        assert stats.skip_ratio > 0.5
        assert inner.call_count < 29

    def test_high_motion_scene_skips_nothing(self):
        frames = _make_frames(30, static=False)
        ctrl, inner, _ = self._run(frames, skip_threshold=0.0, min_active_interval=1)
        stats = ctrl.skip_stats()
        assert stats.skipped_frames == 0
        assert inner.call_count == 29

    def test_min_active_interval_forces_periodic_update(self):
        frames = _make_frames(20, static=True)
        interval = 3
        ctrl, inner, _ = self._run(frames, skip_threshold=255.0, min_active_interval=interval)
        # With threshold=255 (always below) and interval=3, updates must happen every 3 frames
        expected_active = 19 // interval  # floor
        assert inner.call_count >= expected_active


# ---------------------------------------------------------------------------
# Skip statistics
# ---------------------------------------------------------------------------

class TestSkipStats:
    def _run_and_stats(self, n_frames=20, static=True, skip_threshold=6.0, min_interval=5):
        inner = _ConstantTracker()
        ctrl = AdaptiveFrameSkipController(
            inner, skip_threshold=skip_threshold, min_active_interval=min_interval
        )
        frames = _make_frames(n_frames, static=static)
        ctrl.initialize(frames[0], (0.0, 0.0, 10.0, 10.0))
        for f in frames[1:]:
            ctrl.update(f)
        return ctrl.skip_stats()

    def test_total_frames_equals_updates(self):
        stats = self._run_and_stats(20)
        assert stats.total_frames == 19

    def test_active_plus_skipped_equals_total(self):
        stats = self._run_and_stats(20)
        assert stats.active_frames + stats.skipped_frames == stats.total_frames

    def test_skip_ratio_in_unit_interval(self):
        stats = self._run_and_stats(20)
        assert 0.0 <= stats.skip_ratio <= 1.0

    def test_zero_skip_when_threshold_is_zero(self):
        stats = self._run_and_stats(20, skip_threshold=0.0, min_interval=1)
        assert stats.skipped_frames == 0
        assert stats.skip_ratio == pytest.approx(0.0)

    def test_mean_motion_score_non_negative(self):
        stats = self._run_and_stats(20)
        assert stats.mean_motion_score >= 0.0

    def test_to_dict_contains_expected_keys(self):
        stats = self._run_and_stats(20)
        d = stats.to_dict()
        for key in ("total_frames", "skipped_frames", "active_frames",
                    "skip_ratio", "inference_fps", "effective_fps", "mean_motion_score"):
            assert key in d

    def test_str_representation(self):
        stats = SkipStats(
            total_frames=100, skipped_frames=60, active_frames=40,
            skip_ratio=0.60, inference_fps=200.0, effective_fps=500.0,
            mean_motion_score=3.5,
        )
        s = str(stats)
        assert "60" in s
        assert "200.0" in s


# ---------------------------------------------------------------------------
# Velocity prediction
# ---------------------------------------------------------------------------

class TestVelocityPrediction:
    def test_velocity_mode_returns_tuple(self):
        frames = _make_frames(10, static=True)
        inner = _ConstantTracker(bbox=(10.0, 10.0, 20.0, 20.0))
        ctrl = AdaptiveFrameSkipController(
            inner, skip_threshold=0.0, min_active_interval=2, use_velocity=True
        )
        ctrl.initialize(frames[0], (10.0, 10.0, 20.0, 20.0))
        for f in frames[1:]:
            bbox = ctrl.update(f)
            assert len(bbox) == 4

    def test_velocity_extrapolates_position(self):
        """Skipped frame should extrapolate bbox with constant-velocity model.

        Timeline (min_active_interval=3, threshold=255 → always skip-eligible):
          Frame 0: initialize → init_bbox=(0, 0, 20, 20)
          Frame 1: frames_since_active=1 < 3 → SKIP, last_bbox held
          Frame 2: frames_since_active=2 < 3 → SKIP, last_bbox held
          Frame 3: frames_since_active=3 (not < 3) → ACTIVE, inner returns (10, 0, 20, 20)
          Frame 4: frames_since_active=1 < 3 → SKIP, velocity predict
                   prev_prev_active=(0,0,20,20), prev_active=(10,0,20,20)
                   → predicted x = 10 + (10-0) = 20
        """
        frames = _make_frames(5, static=True)

        class _MovingTracker(BaseTracker):
            def __init__(self):
                super().__init__(name="Moving")
                self._step = 0

            def initialize(self, frame, bbox):
                self._step = 0

            def update(self, frame):
                self._step += 1
                return (float(self._step * 10), 0.0, 20.0, 20.0)

        inner = _MovingTracker()
        ctrl = AdaptiveFrameSkipController(
            inner, skip_threshold=255.0, min_active_interval=3, use_velocity=True
        )
        ctrl.initialize(frames[0], (0.0, 0.0, 20.0, 20.0))

        ctrl.update(frames[1])  # skipped (frames_since_active=1)
        ctrl.update(frames[2])  # skipped (frames_since_active=2)
        ctrl.update(frames[3])  # ACTIVE  (frames_since_active=3, inner → (10,0,20,20))
        b4 = ctrl.update(frames[4])  # skipped → velocity predict → (20, 0, 20, 20)

        assert len(b4) == 4
        assert b4[0] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Integration: works with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestBenchmarkEngineIntegration:
    def test_runs_through_benchmark_engine(self):
        """Smoke-test: AdaptiveFrameSkipController passes through BenchmarkEngine."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=2, num_frames=30, seed=0)
        base = MOSSETracker()
        tracker = AdaptiveFrameSkipController(
            base, skip_threshold=4.0, min_active_interval=3
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic-Skip")
        assert result.mean_fps > 0
        assert result.mean_iou >= 0.0
        assert len(result.sequence_results) == 2
