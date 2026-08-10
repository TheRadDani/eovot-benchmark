"""Tests for the FrameDropTracker adaptive frame-skip wrapper.

Covers:
- Correct skip detection based on stability threshold
- Warm-up window enforcement (no skips during first N frames)
- Max consecutive skip cap
- FrameDropStats accuracy
- Integration with BenchmarkEngine via SyntheticDataset
- Validation error handling
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.frame_drop import FrameDropTracker, FrameDropStats, _iou


# ---------------------------------------------------------------------------
# Minimal inner tracker fixtures
# ---------------------------------------------------------------------------

class ConstantTracker(BaseTracker):
    """Returns the same fixed box every update — maximally stable."""

    def __init__(self, box: BBox = (10.0, 10.0, 50.0, 50.0)) -> None:
        super().__init__(name="Constant")
        self._box = box

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        pass

    def update(self, frame: np.ndarray) -> BBox:
        return self._box


class MovingTracker(BaseTracker):
    """Shifts the box by `step` pixels each frame — maximally unstable."""

    def __init__(self, init_box: BBox = (0.0, 0.0, 50.0, 50.0), step: float = 60.0) -> None:
        super().__init__(name="Moving")
        self._box = list(init_box)
        self.step = step

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._box = list(bbox)

    def update(self, frame: np.ndarray) -> BBox:
        self._box[0] += self.step   # advance x each frame
        return tuple(self._box)  # type: ignore[return-value]


DUMMY_FRAME = np.zeros((240, 320, 3), dtype=np.uint8)
INIT_BBOX: BBox = (10.0, 10.0, 50.0, 50.0)


# ---------------------------------------------------------------------------
# _iou helper
# ---------------------------------------------------------------------------

class TestIouHelper:
    def test_identical_boxes(self):
        box = (10.0, 10.0, 50.0, 50.0)
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (20.0, 20.0, 10.0, 10.0)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_zero_area(self):
        a = (0.0, 0.0, 0.0, 10.0)
        b = (0.0, 0.0, 10.0, 10.0)
        assert _iou(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestFrameDropTrackerValidation:
    def test_invalid_threshold_zero(self):
        with pytest.raises(ValueError, match="stability_threshold"):
            FrameDropTracker(ConstantTracker(), stability_threshold=0.0)

    def test_invalid_threshold_negative(self):
        with pytest.raises(ValueError, match="stability_threshold"):
            FrameDropTracker(ConstantTracker(), stability_threshold=-0.1)

    def test_threshold_one_is_valid(self):
        tracker = FrameDropTracker(ConstantTracker(), stability_threshold=1.0)
        assert tracker.stability_threshold == pytest.approx(1.0)

    def test_invalid_max_consecutive_zero(self):
        with pytest.raises(ValueError, match="max_consecutive_skips"):
            FrameDropTracker(ConstantTracker(), max_consecutive_skips=0)

    def test_name_has_fd_suffix(self):
        tracker = FrameDropTracker(ConstantTracker())
        assert "(fd)" in tracker.name


# ---------------------------------------------------------------------------
# Warm-up window
# ---------------------------------------------------------------------------

class TestWarmUpWindow:
    def test_no_skips_during_warmup(self):
        """With a very low threshold, skipping would happen immediately — but warm-up prevents it."""
        tracker = FrameDropTracker(
            ConstantTracker(),
            stability_threshold=0.01,  # almost always triggers
            warm_up_frames=10,
        )
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        for _ in range(10):
            tracker.update(DUMMY_FRAME)
        stats = tracker.stats()
        assert stats.skipped_frames == 0

    def test_skips_start_after_warmup(self):
        """After warm-up, a constant tracker with low threshold should start skipping."""
        warm_up = 3
        tracker = FrameDropTracker(
            ConstantTracker(),
            stability_threshold=0.01,
            max_consecutive_skips=100,
            warm_up_frames=warm_up,
        )
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        total = 20
        for _ in range(total):
            tracker.update(DUMMY_FRAME)
        stats = tracker.stats()
        # The first `warm_up` frames can't skip; two frames after that are needed
        # to populate both _last_bbox and _prev_bbox; rest should skip
        assert stats.skipped_frames > 0


# ---------------------------------------------------------------------------
# Stable scene — constant tracker
# ---------------------------------------------------------------------------

class TestStableScene:
    def test_constant_tracker_skips_above_threshold(self):
        """When predictions never change (IoU=1.0), high-threshold should skip heavily."""
        tracker = FrameDropTracker(
            ConstantTracker(INIT_BBOX),
            stability_threshold=0.85,
            max_consecutive_skips=5,
            warm_up_frames=2,
        )
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        n = 30
        for _ in range(n):
            tracker.update(DUMMY_FRAME)

        stats = tracker.stats()
        assert stats.total_frames == n
        assert stats.skip_ratio > 0.5, f"Expected high skip ratio, got {stats.skip_ratio}"

    def test_returned_bbox_is_last_known(self):
        """Skipped frames should return the last non-skipped prediction."""
        fixed_box: BBox = (5.0, 5.0, 40.0, 40.0)
        tracker = FrameDropTracker(
            ConstantTracker(fixed_box),
            stability_threshold=0.5,
            max_consecutive_skips=10,
            warm_up_frames=2,
        )
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        results = [tracker.update(DUMMY_FRAME) for _ in range(15)]
        for result in results:
            assert result == fixed_box


# ---------------------------------------------------------------------------
# Unstable scene — moving tracker
# ---------------------------------------------------------------------------

class TestUnstableScene:
    def test_no_skips_when_target_moves_far(self):
        """When predicted boxes have IoU=0 between frames, nothing should be skipped."""
        tracker = FrameDropTracker(
            MovingTracker(step=200.0),   # jumps 200 px every frame → IoU=0
            stability_threshold=0.85,
            max_consecutive_skips=3,
            warm_up_frames=2,
        )
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        for _ in range(20):
            tracker.update(DUMMY_FRAME)
        stats = tracker.stats()
        assert stats.skipped_frames == 0, (
            f"Expected 0 skips on a fast-moving target, got {stats.skipped_frames}"
        )


# ---------------------------------------------------------------------------
# Max consecutive skip cap
# ---------------------------------------------------------------------------

class TestMaxConsecutiveSkips:
    def test_cap_is_respected(self):
        """Consecutive skips must never exceed max_consecutive_skips."""
        max_skip = 2
        tracker = FrameDropTracker(
            ConstantTracker(),
            stability_threshold=0.01,
            max_consecutive_skips=max_skip,
            warm_up_frames=1,
        )
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        for _ in range(30):
            tracker.update(DUMMY_FRAME)
        stats = tracker.stats()
        assert stats.max_consecutive_run <= max_skip


# ---------------------------------------------------------------------------
# FrameDropStats
# ---------------------------------------------------------------------------

class TestFrameDropStats:
    def test_stats_zero_after_init(self):
        tracker = FrameDropTracker(ConstantTracker())
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        stats = tracker.stats()
        assert stats.total_frames == 0
        assert stats.skipped_frames == 0
        assert stats.skip_ratio == pytest.approx(0.0)

    def test_skip_ratio_range(self):
        tracker = FrameDropTracker(
            ConstantTracker(), stability_threshold=0.5, warm_up_frames=1
        )
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        for _ in range(50):
            tracker.update(DUMMY_FRAME)
        stats = tracker.stats()
        assert 0.0 <= stats.skip_ratio <= 1.0

    def test_stats_total_frames_matches_updates(self):
        tracker = FrameDropTracker(ConstantTracker(), warm_up_frames=0)
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        n = 25
        for _ in range(n):
            tracker.update(DUMMY_FRAME)
        assert tracker.stats().total_frames == n

    def test_stats_to_dict_keys(self):
        tracker = FrameDropTracker(ConstantTracker())
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        tracker.update(DUMMY_FRAME)
        d = tracker.stats().to_dict()
        for key in ("tracker_name", "total_frames", "skipped_frames", "skip_ratio",
                    "max_consecutive_run", "stability_threshold", "max_consecutive_skips"):
            assert key in d

    def test_stats_str_contains_name(self):
        tracker = FrameDropTracker(ConstantTracker())
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        s = str(tracker.stats())
        assert "Constant(fd)" in s


# ---------------------------------------------------------------------------
# Reset on re-initialize
# ---------------------------------------------------------------------------

class TestReinitialization:
    def test_stats_reset_on_initialize(self):
        tracker = FrameDropTracker(
            ConstantTracker(), stability_threshold=0.5, warm_up_frames=1
        )
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        for _ in range(20):
            tracker.update(DUMMY_FRAME)
        first_stats = tracker.stats()
        assert first_stats.total_frames == 20

        # Re-initialize should reset counters
        tracker.initialize(DUMMY_FRAME, INIT_BBOX)
        fresh_stats = tracker.stats()
        assert fresh_stats.total_frames == 0
        assert fresh_stats.skipped_frames == 0


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:
    def test_frame_drop_tracker_runs_through_engine(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=2, num_frames=30, motion="linear", seed=0)
        base = MOSSETracker()
        tracker = FrameDropTracker(base, stability_threshold=0.80, max_consecutive_skips=3)

        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic", max_sequences=2)

        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_frame_drop_preserves_result_shape(self):
        """FrameDropTracker must return the same number of predictions as frames."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.kcf import KCFTracker

        dataset = SyntheticDataset(num_sequences=1, num_frames=25)
        tracker = FrameDropTracker(KCFTracker(), warm_up_frames=3)

        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic", max_sequences=1)

        sr = result.sequence_results[0]
        assert len(sr.ious) == 25

    def test_frame_drop_import_from_package(self):
        """Verify public API accessibility via eovot.trackers namespace."""
        from eovot.trackers import FrameDropTracker as FDT, FrameDropStats as FDS
        assert FDT is FrameDropTracker
        assert FDS is FrameDropStats
