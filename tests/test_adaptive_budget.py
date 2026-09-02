"""Tests for AdaptiveBudgetTracker."""

from __future__ import annotations

import math
import pytest
import numpy as np

from eovot.trackers.adaptive_budget import AdaptiveBudgetTracker
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker(easy=0.05, hard=0.15, smoothing=0.3):
    return AdaptiveBudgetTracker(
        MOSSETracker(),
        easy_threshold=easy,
        hard_threshold=hard,
        smoothing=smoothing,
    )


def _run_sequence(tracker, n_frames=60, motion="linear"):
    ds = SyntheticDataset(num_sequences=1, num_frames=n_frames, motion=motion)
    seq = ds[0]
    frames = list(seq)
    gt = seq.ground_truth

    tracker.initialize(frames[0], tuple(gt[0]))
    preds = [tuple(gt[0])]
    for frame in frames[1:]:
        preds.append(tracker.update(frame))
    return preds


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_name(self):
        t = AdaptiveBudgetTracker(MOSSETracker())
        assert "MOSSE" in t.name
        assert "Adaptive" in t.name

    def test_custom_name(self):
        t = AdaptiveBudgetTracker(MOSSETracker(), name="MyAdaptive")
        assert t.name == "MyAdaptive"

    def test_thresholds_must_be_ordered(self):
        with pytest.raises(ValueError, match="easy_threshold"):
            AdaptiveBudgetTracker(MOSSETracker(), easy_threshold=0.2, hard_threshold=0.1)

    def test_equal_thresholds_allowed(self):
        # easy == hard is degenerate but should not raise
        t = AdaptiveBudgetTracker(MOSSETracker(), easy_threshold=0.1, hard_threshold=0.1)
        assert t.easy_threshold == t.hard_threshold

    def test_invalid_smoothing_raises(self):
        with pytest.raises(ValueError, match="smoothing"):
            AdaptiveBudgetTracker(MOSSETracker(), smoothing=0.0)
        with pytest.raises(ValueError, match="smoothing"):
            AdaptiveBudgetTracker(MOSSETracker(), smoothing=1.5)

    def test_smoothing_of_one_allowed(self):
        t = AdaptiveBudgetTracker(MOSSETracker(), smoothing=1.0)
        assert t.smoothing == 1.0

    def test_wrapped_tracker_stored(self):
        base = MOSSETracker()
        t = AdaptiveBudgetTracker(base)
        assert t.wrapped_tracker is base


# ---------------------------------------------------------------------------
# initialize resets state
# ---------------------------------------------------------------------------

class TestInitialize:
    def test_stats_reset_on_second_initialize(self):
        t = _make_tracker()
        ds = SyntheticDataset(num_sequences=1, num_frames=50)
        seq = ds[0]
        frames = list(seq)
        gt = seq.ground_truth

        t.initialize(frames[0], tuple(gt[0]))
        for f in frames[1:]:
            t.update(f)

        t.initialize(frames[0], tuple(gt[0]))
        assert t.n_updates == 0
        assert t.n_coasted == 0
        assert len(t.complexity_history) == 0

    def test_complexity_history_cleared(self):
        t = _make_tracker()
        _run_sequence(t, n_frames=30)
        assert len(t.complexity_history) > 0

        ds = SyntheticDataset(num_sequences=1, num_frames=5)
        seq = ds[0]
        frames = list(seq)
        t.initialize(frames[0], tuple(seq.ground_truth[0]))
        assert len(t.complexity_history) == 0


# ---------------------------------------------------------------------------
# update produces valid bounding boxes
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_bbox_shape_valid(self):
        t = _make_tracker()
        preds = _run_sequence(t, n_frames=50)
        for bbox in preds:
            assert len(bbox) == 4

    def test_bbox_width_height_positive(self):
        t = _make_tracker()
        preds = _run_sequence(t, n_frames=50)
        for bbox in preds[1:]:
            x, y, w, h = bbox
            assert w > 0, f"Width non-positive: {w}"
            assert h > 0, f"Height non-positive: {h}"

    def test_total_calls_sum_to_n_frames_minus_one(self):
        t = _make_tracker()
        n = 60
        _run_sequence(t, n_frames=n)
        assert t.n_updates + t.n_coasted == n - 1

    def test_complexity_history_length(self):
        t = _make_tracker()
        n = 40
        _run_sequence(t, n_frames=n)
        assert len(t.complexity_history) == n - 1


# ---------------------------------------------------------------------------
# skip_rate property
# ---------------------------------------------------------------------------

class TestSkipRate:
    def test_skip_rate_range(self):
        t = _make_tracker()
        _run_sequence(t, n_frames=80, motion="linear")
        assert 0.0 <= t.skip_rate <= 1.0

    def test_zero_skip_when_always_hard(self):
        # easy_threshold=0.0 means complexity < 0 is never true (complexity >= 0),
        # so the tracker never enters easy mode.
        t = AdaptiveBudgetTracker(
            MOSSETracker(), easy_threshold=0.0, hard_threshold=0.0
        )
        _run_sequence(t, n_frames=50, motion="random")
        assert t.n_coasted == 0
        assert t.skip_rate == 0.0

    def test_skip_rate_zero_before_any_updates(self):
        t = _make_tracker()
        assert t.skip_rate == 0.0


# ---------------------------------------------------------------------------
# Benchmark engine integration
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:
    def test_runs_through_benchmark_engine(self):
        t = AdaptiveBudgetTracker(MOSSETracker(), easy_threshold=0.03, hard_threshold=0.10)
        ds = SyntheticDataset(num_sequences=2, num_frames=50, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(t, ds, dataset_name="Syn")
        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0

    def test_iou_finite(self):
        t = AdaptiveBudgetTracker(KCFTracker(), easy_threshold=0.05, hard_threshold=0.2)
        ds = SyntheticDataset(num_sequences=2, num_frames=60, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(t, ds, dataset_name="Syn")
        assert math.isfinite(result.mean_iou)

    def test_different_wrapped_trackers(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=50, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        for cls in [MOSSETracker, KCFTracker]:
            t = AdaptiveBudgetTracker(cls())
            result = engine.run(t, ds, dataset_name="Syn")
            assert result.mean_iou >= 0.0


# ---------------------------------------------------------------------------
# Motion model: coast prediction
# ---------------------------------------------------------------------------

class TestCoastPredict:
    def test_coast_produces_bbox(self):
        t = _make_tracker()
        t.initialize(np.zeros((100, 100, 3), dtype=np.uint8), (10.0, 10.0, 20.0, 20.0))
        # Manually trigger coast by entering easy mode
        t._in_easy_mode = True
        t._prev_bbox = (10.0, 10.0, 20.0, 20.0)
        t._velocity = (1.0, 0.5)
        coast = t._coast_predict()
        assert len(coast) == 4
        x, y, w, h = coast
        assert w > 0 and h > 0

    def test_coast_applies_velocity(self):
        t = _make_tracker()
        t.initialize(np.zeros((200, 200, 3), dtype=np.uint8), (50.0, 50.0, 30.0, 30.0))
        t._prev_bbox = (50.0, 50.0, 30.0, 30.0)
        t._velocity = (5.0, 3.0)
        x, y, w, h = t._coast_predict()
        # x should shift by vx=5, y by vy=3
        assert abs(x - 55.0) < 1e-6
        assert abs(y - 53.0) < 1e-6


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_tracker_name(self):
        t = AdaptiveBudgetTracker(MOSSETracker())
        r = repr(t)
        assert "MOSSE" in r

    def test_repr_shows_thresholds(self):
        t = AdaptiveBudgetTracker(MOSSETracker(), easy_threshold=0.04, hard_threshold=0.12)
        r = repr(t)
        assert "0.04" in r
        assert "0.12" in r
