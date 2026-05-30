"""Tests for FrameSkipTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.frame_skip import FrameSkipTracker


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


class _LinearTracker(BaseTracker):
    """Deterministic tracker: moves the box 1 px right per update() call."""

    def __init__(self) -> None:
        super().__init__(name="LinearTracker")
        self.update_count = 0
        self._bbox: BBox = (0.0, 0.0, 40.0, 40.0)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox
        self.update_count = 0

    def update(self, frame: np.ndarray) -> BBox:
        self.update_count += 1
        x, y, w, h = self._bbox
        self._bbox = (x + 1.0, y, w, h)
        return self._bbox


def _blank_frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _run(tracker: BaseTracker, n_updates: int = 10) -> list[BBox]:
    frame = _blank_frame()
    tracker.initialize(frame, (50.0, 50.0, 40.0, 40.0))
    return [tracker.update(frame) for _ in range(n_updates)]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_name_includes_skip_rate(self) -> None:
        t = FrameSkipTracker(_LinearTracker(), skip_rate=4)
        assert "skip=4" in t.name

    def test_name_includes_base_tracker_name(self) -> None:
        t = FrameSkipTracker(_LinearTracker(), skip_rate=2)
        assert "LinearTracker" in t.name

    def test_invalid_skip_rate_zero(self) -> None:
        with pytest.raises(ValueError, match="skip_rate must be >= 1"):
            FrameSkipTracker(_LinearTracker(), skip_rate=0)

    def test_invalid_skip_rate_negative(self) -> None:
        with pytest.raises(ValueError, match="skip_rate must be >= 1"):
            FrameSkipTracker(_LinearTracker(), skip_rate=-3)

    def test_repr_contains_skip_rate(self) -> None:
        t = FrameSkipTracker(_LinearTracker(), skip_rate=3)
        assert "skip_rate=3" in repr(t)


# ---------------------------------------------------------------------------
# Underlying tracker invocation counts
# ---------------------------------------------------------------------------


class TestInvocationCounts:
    """Verify that update() is called on the base tracker exactly the right
    number of times for various (skip_rate, n_updates) combinations."""

    def _count_calls(self, skip_rate: int, n_updates: int) -> int:
        base = _LinearTracker()
        wrapped = FrameSkipTracker(base, skip_rate=skip_rate)
        _run(wrapped, n_updates)
        return base.update_count

    def test_skip1_calls_every_frame(self) -> None:
        assert self._count_calls(skip_rate=1, n_updates=9) == 9

    def test_skip2_calls_half_frames(self) -> None:
        # 10 updates → frame_index 1..10, process at 2,4,6,8,10 → 5 calls
        assert self._count_calls(skip_rate=2, n_updates=10) == 5

    def test_skip3_correct_count(self) -> None:
        # 9 updates → frame_index 1..9, process at 3,6,9 → 3 calls
        assert self._count_calls(skip_rate=3, n_updates=9) == 3

    def test_skip5_correct_count(self) -> None:
        # 15 updates → process at 5,10,15 → 3 calls
        assert self._count_calls(skip_rate=5, n_updates=15) == 3

    def test_no_frames_after_init(self) -> None:
        base = _LinearTracker()
        wrapped = FrameSkipTracker(base, skip_rate=2)
        wrapped.initialize(_blank_frame(), (50.0, 50.0, 40.0, 40.0))
        assert base.update_count == 0


# ---------------------------------------------------------------------------
# Output shape and positivity
# ---------------------------------------------------------------------------


class TestOutputProperties:
    def test_returns_four_element_tuple(self) -> None:
        results = _run(FrameSkipTracker(_LinearTracker(), skip_rate=2), n_updates=9)
        for bbox in results:
            assert len(bbox) == 4

    def test_width_and_height_positive(self) -> None:
        results = _run(FrameSkipTracker(_LinearTracker(), skip_rate=5), n_updates=20)
        for x, y, w, h in results:
            assert w >= 1.0
            assert h >= 1.0

    def test_correct_number_of_predictions(self) -> None:
        n = 15
        results = _run(FrameSkipTracker(_LinearTracker(), skip_rate=3), n_updates=n)
        assert len(results) == n


# ---------------------------------------------------------------------------
# skip_rate=1 passes through to base tracker exactly
# ---------------------------------------------------------------------------


class TestSkipRate1Passthrough:
    def test_predictions_match_base(self) -> None:
        base_a = _LinearTracker()
        base_b = _LinearTracker()
        wrapped = FrameSkipTracker(base_a, skip_rate=1)

        frame = _blank_frame()
        init_bbox: BBox = (50.0, 50.0, 40.0, 40.0)
        base_b.initialize(frame, init_bbox)
        wrapped.initialize(frame, init_bbox)

        for _ in range(8):
            expected = base_b.update(frame)
            actual = wrapped.update(frame)
            assert actual == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Extrapolation correctness
# ---------------------------------------------------------------------------


class TestExtrapolation:
    def test_skipped_frame_advances_position(self) -> None:
        """Extrapolated frame should move in the same direction as the tracker."""
        base = _LinearTracker()
        wrapped = FrameSkipTracker(base, skip_rate=3)
        results = _run(wrapped, n_updates=6)

        # Frame indices 1..6; processed at 3 and 6.
        # Index 1 (skip): extrapolation from init (velocity=0) → same as init
        # Index 2 (skip): same
        # Index 3 (process): real tracker call
        # Index 4 (skip): should be ahead of index-3 prediction
        x_at_3 = results[2][0]  # result index 2 → frame_index 3 (processed)
        x_at_4 = results[3][0]  # result index 3 → frame_index 4 (skipped)
        # The tracker moves +1 px per call; we expect extrapolation to project
        # forward, so x_at_4 >= x_at_3 (velocity >= 0).
        assert x_at_4 >= x_at_3

    def test_process_frame_updates_velocity(self) -> None:
        """After the second process-frame the velocity estimate should be non-zero."""
        base = _LinearTracker()
        wrapped = FrameSkipTracker(base, skip_rate=2)
        frame = _blank_frame()
        wrapped.initialize(frame, (100.0, 100.0, 40.0, 40.0))
        # Run enough frames to get two process-frames.
        results = [wrapped.update(frame) for _ in range(6)]

        # Frame 2 (process): tracker moved to (101,100) → _last_bbox=(101,100,40,40)
        # Frame 4 (process): tracker moved to (102,100) → velocity=(1,0)
        # Frame 5 (skip): extrapolated → x should be ~103
        x5 = results[4][0]
        x4 = results[3][0]
        assert x5 > x4


# ---------------------------------------------------------------------------
# Re-initialization resets state
# ---------------------------------------------------------------------------


class TestReinitialization:
    def test_frame_index_resets_to_zero(self) -> None:
        base = _LinearTracker()
        wrapped = FrameSkipTracker(base, skip_rate=2)
        _run(wrapped, n_updates=8)
        assert wrapped._frame_index == 8

        # Re-initialize.
        wrapped.initialize(_blank_frame(), (10.0, 10.0, 20.0, 20.0))
        assert wrapped._frame_index == 0

    def test_predictions_reproducible_after_reinit(self) -> None:
        base = _LinearTracker()
        wrapped = FrameSkipTracker(base, skip_rate=2)
        frame = _blank_frame()
        init_bbox: BBox = (50.0, 50.0, 40.0, 40.0)

        wrapped.initialize(frame, init_bbox)
        run1 = [wrapped.update(frame) for _ in range(6)]

        wrapped.initialize(frame, init_bbox)
        run2 = [wrapped.update(frame) for _ in range(6)]

        for b1, b2 in zip(run1, run2):
            assert b1 == pytest.approx(b2, abs=1e-9)


# ---------------------------------------------------------------------------
# Integration: FrameSkipTracker + BenchmarkEngine
# ---------------------------------------------------------------------------


class TestBenchmarkEngineIntegration:
    def test_engine_runs_without_error(self) -> None:
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        base = MOSSETracker()
        tracker = FrameSkipTracker(base, skip_rate=2)
        dataset = SyntheticDataset(num_sequences=2, num_frames=30, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="synthetic")

        assert result.tracker_name == tracker.name
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0.0

    def test_skip1_and_no_wrap_have_similar_iou(self) -> None:
        """skip_rate=1 should produce the same mIoU as the unwrapped tracker."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=3, num_frames=30, motion="linear", seed=0)
        engine = BenchmarkEngine(verbose=False)

        base = MOSSETracker()
        skip1 = FrameSkipTracker(MOSSETracker(), skip_rate=1)

        r_base = engine.run(base, dataset, dataset_name="test")
        r_skip = engine.run(skip1, dataset, dataset_name="test")

        assert r_base.mean_iou == pytest.approx(r_skip.mean_iou, abs=1e-6)
