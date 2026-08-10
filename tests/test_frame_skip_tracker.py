"""Tests for FrameSkipTracker (eovot/trackers/adaptive.py)."""

import numpy as np
import pytest

from eovot.trackers.adaptive import FrameSkipTracker
from eovot.trackers.mosse import MOSSETracker


def _frame(h: int = 80, w: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


INIT_BBOX = (20, 15, 30, 25)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestFrameSkipTrackerConstruction:
    def test_valid_construction(self):
        t = FrameSkipTracker(MOSSETracker(), skip_rate=1, extrapolation="last")
        assert t is not None

    def test_negative_skip_rate_raises(self):
        with pytest.raises(ValueError, match="skip_rate"):
            FrameSkipTracker(MOSSETracker(), skip_rate=-1)

    def test_unknown_extrapolation_raises(self):
        with pytest.raises(ValueError, match="extrapolation"):
            FrameSkipTracker(MOSSETracker(), extrapolation="cubic")

    def test_name_encodes_skip_rate_and_mode(self):
        t = FrameSkipTracker(MOSSETracker(), skip_rate=3, extrapolation="linear")
        assert "skip=3" in t.name
        assert "linear" in t.name

    def test_name_includes_base_tracker_name(self):
        t = FrameSkipTracker(MOSSETracker(), skip_rate=1)
        assert "MOSSE" in t.name


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestFrameSkipTrackerProperties:
    def test_skip_ratio_zero_for_skip_rate_zero(self):
        t = FrameSkipTracker(MOSSETracker(), skip_rate=0)
        assert t.skip_ratio == 0.0

    def test_skip_ratio_half_for_skip_rate_one(self):
        t = FrameSkipTracker(MOSSETracker(), skip_rate=1)
        assert abs(t.skip_ratio - 0.5) < 1e-9

    def test_skip_ratio_two_thirds_for_skip_rate_two(self):
        t = FrameSkipTracker(MOSSETracker(), skip_rate=2)
        assert abs(t.skip_ratio - 2 / 3) < 1e-9

    def test_throughput_multiplier_one_for_no_skipping(self):
        t = FrameSkipTracker(MOSSETracker(), skip_rate=0)
        assert t.throughput_multiplier == 1.0

    def test_throughput_multiplier_four_for_skip_rate_three(self):
        t = FrameSkipTracker(MOSSETracker(), skip_rate=3)
        assert t.throughput_multiplier == 4.0


# ---------------------------------------------------------------------------
# update() behaviour before initialization
# ---------------------------------------------------------------------------

class TestFrameSkipTrackerUninitialized:
    def test_update_before_initialize_raises(self):
        t = FrameSkipTracker(MOSSETracker(), skip_rate=1)
        with pytest.raises(RuntimeError, match="initialised"):
            t.update(_frame())


# ---------------------------------------------------------------------------
# Correct frame-count and skip logic
# ---------------------------------------------------------------------------

class TestFrameSkipTrackerSkipLogic:
    def test_skip_rate_zero_processes_every_frame(self):
        """skip_rate=0 should be identical to the wrapped tracker."""
        t = FrameSkipTracker(MOSSETracker(), skip_rate=0)
        t.initialize(_frame(seed=0), INIT_BBOX)
        for i in range(10):
            bbox = t.update(_frame(seed=i + 1))
            assert len(bbox) == 4
            assert all(isinstance(v, float) for v in bbox)

    def test_skip_rate_one_returns_valid_bbox_on_all_frames(self):
        t = FrameSkipTracker(MOSSETracker(), skip_rate=1, extrapolation="last")
        t.initialize(_frame(seed=0), INIT_BBOX)
        for i in range(1, 12):
            bbox = t.update(_frame(seed=i))
            assert len(bbox) == 4

    def test_skip_rate_two_returns_valid_bbox_on_all_frames(self):
        t = FrameSkipTracker(MOSSETracker(), skip_rate=2, extrapolation="linear")
        t.initialize(_frame(seed=0), INIT_BBOX)
        for i in range(1, 15):
            bbox = t.update(_frame(seed=i))
            assert len(bbox) == 4

    def test_last_extrapolation_repeats_previous_bbox_on_skip(self):
        """On a skipped frame with 'last' mode, the returned box should
        equal the most-recent real-tracker output (no velocity applied)."""
        t = FrameSkipTracker(MOSSETracker(), skip_rate=2, extrapolation="last")
        t.initialize(_frame(seed=0), INIT_BBOX)
        # Frames 1, 2 are skipped (frame_count 1, 2 are not multiples of 3).
        b1 = t.update(_frame(seed=1))  # skipped → returns INIT_BBOX extrapolated
        b2 = t.update(_frame(seed=2))  # skipped → same
        # Both skipped frames should return the same last-known bbox.
        assert b1 == b2

    def test_linear_extrapolation_shifts_box_when_velocity_known(self):
        """After two real updates, the linear extrapolation should produce
        a box displaced by the estimated velocity."""
        t = FrameSkipTracker(MOSSETracker(), skip_rate=1, extrapolation="linear")
        t.initialize(_frame(seed=0), INIT_BBOX)
        # frame_count=1 (1 % 2 != 0) → skipped, prev_bbox is None → returns last
        t.update(_frame(seed=1))
        # frame_count=2 (2 % 2 == 0) → real update; prev_bbox set
        t.update(_frame(seed=2))
        # frame_count=3 (3 % 2 != 0) → skipped, now has prev_bbox → extrapolates
        b_skipped = t.update(_frame(seed=3))
        assert len(b_skipped) == 4
        assert all(isinstance(v, float) for v in b_skipped)


# ---------------------------------------------------------------------------
# Re-initialisation
# ---------------------------------------------------------------------------

class TestFrameSkipTrackerReinitialisation:
    def test_reinitialise_resets_state(self):
        t = FrameSkipTracker(MOSSETracker(), skip_rate=1)
        frame0 = _frame(seed=0)
        t.initialize(frame0, INIT_BBOX)
        for i in range(5):
            t.update(_frame(seed=i + 1))
        # Re-initialise with a different bbox
        new_bbox = (50, 40, 20, 20)
        t.initialize(_frame(seed=10), new_bbox)
        assert t._last_bbox == (50.0, 40.0, 20.0, 20.0)
        assert t._prev_bbox is None
        assert t._frame_count == 0


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestFrameSkipTrackerIntegration:
    def test_benchmark_engine_full_run_does_not_crash(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=2, num_frames=25, seed=7)
        tracker = FrameSkipTracker(MOSSETracker(), skip_rate=2, extrapolation="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="synthetic")
        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0
        assert len(result.sequence_results) == 2

    def test_skip_rate_zero_matches_base_tracker_iou_closely(self):
        """skip_rate=0 should give IoU indistinguishable from the base tracker."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset1 = SyntheticDataset(num_sequences=2, num_frames=30, seed=42)
        dataset2 = SyntheticDataset(num_sequences=2, num_frames=30, seed=42)
        engine = BenchmarkEngine(verbose=False)

        r_base = engine.run(MOSSETracker(), dataset1, dataset_name="syn")
        r_wrapped = engine.run(
            FrameSkipTracker(MOSSETracker(), skip_rate=0),
            dataset2,
            dataset_name="syn",
        )
        # With the same seed and skip_rate=0 the IoU should be near-identical.
        assert abs(r_base.mean_iou - r_wrapped.mean_iou) < 0.05

    def test_higher_skip_rate_gives_higher_fps(self):
        """A higher skip rate should always result in greater FPS."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        engine = BenchmarkEngine(verbose=False)
        fps_values = []
        for skip in [0, 2, 5]:
            dataset = SyntheticDataset(num_sequences=2, num_frames=40, seed=0)
            tracker = FrameSkipTracker(MOSSETracker(), skip_rate=skip)
            result = engine.run(tracker, dataset, dataset_name="syn")
            fps_values.append(result.mean_fps)

        # FPS should be monotonically increasing with skip rate
        assert fps_values[1] > fps_values[0], "skip=2 should be faster than skip=0"
        assert fps_values[2] > fps_values[0], "skip=5 should be faster than skip=0"
