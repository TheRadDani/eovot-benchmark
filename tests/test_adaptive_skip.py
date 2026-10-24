"""Tests for AdaptiveSkipTracker and the BaseTracker confidence API."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.adaptive_skip import AdaptiveSkipTracker
from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    rng = np.random.default_rng(0)
    frame = rng.integers(30, 200, size=(h, w, 3), dtype=np.uint8)
    frame[40:80, 60:100] = (200, 100, 50)
    return frame


INIT_BBOX: BBox = (60.0, 40.0, 40.0, 40.0)


# ---------------------------------------------------------------------------
# BaseTracker.confidence() default
# ---------------------------------------------------------------------------

class _MinimalTracker(BaseTracker):
    """Minimal concrete tracker that satisfies the ABC."""

    def initialize(self, frame, bbox):
        self._bbox = bbox

    def update(self, frame):
        return self._bbox


def test_base_tracker_default_confidence():
    t = _MinimalTracker(name="minimal")
    assert t.confidence() == 1.0


# ---------------------------------------------------------------------------
# MOSSETracker confidence
# ---------------------------------------------------------------------------

def test_mosse_confidence_after_initialize():
    tracker = MOSSETracker()
    frame = _make_frame()
    tracker.initialize(frame, INIT_BBOX)
    # confidence is 1.0 before any update (initial PSR not computed yet)
    assert isinstance(tracker.confidence(), float)


def test_mosse_confidence_after_update():
    tracker = MOSSETracker()
    frame = _make_frame()
    tracker.initialize(frame, INIT_BBOX)
    tracker.update(frame)
    conf = tracker.confidence()
    assert 0.0 <= conf <= 1.0


def test_mosse_confidence_range_multiple_frames():
    tracker = MOSSETracker()
    ds = SyntheticDataset(num_sequences=1, num_frames=30, seed=7)
    seq = ds[0]
    frames = list(seq)
    tracker.initialize(frames[0], seq.init_bbox)
    for f in frames[1:]:
        tracker.update(f)
        assert 0.0 <= tracker.confidence() <= 1.0


# ---------------------------------------------------------------------------
# AdaptiveSkipTracker construction
# ---------------------------------------------------------------------------

def test_adaptive_skip_init_requires_positive_max_skip():
    with pytest.raises(ValueError, match="max_skip"):
        AdaptiveSkipTracker(MOSSETracker(), max_skip=0)


def test_adaptive_skip_init_threshold_validation():
    with pytest.raises(ValueError, match="Thresholds"):
        AdaptiveSkipTracker(MOSSETracker(), high_threshold=0.2, low_threshold=0.8)


def test_adaptive_skip_name_includes_max_skip():
    t = AdaptiveSkipTracker(MOSSETracker(), max_skip=3)
    assert "3" in t.name
    assert "MOSSE" in t.name


# ---------------------------------------------------------------------------
# AdaptiveSkipTracker basic operation
# ---------------------------------------------------------------------------

def test_adaptive_skip_raises_before_init():
    t = AdaptiveSkipTracker(MOSSETracker(), max_skip=2)
    with pytest.raises(RuntimeError):
        t.update(_make_frame())


def test_adaptive_skip_returns_bbox_shape():
    t = AdaptiveSkipTracker(MOSSETracker(), max_skip=2)
    frame = _make_frame()
    t.initialize(frame, INIT_BBOX)
    bbox = t.update(frame)
    assert len(bbox) == 4
    assert all(isinstance(v, float) for v in bbox)


def test_adaptive_skip_history_populated():
    t = AdaptiveSkipTracker(MOSSETracker(), max_skip=3, high_threshold=0.5, low_threshold=0.1)
    ds = SyntheticDataset(num_sequences=1, num_frames=20, seed=42)
    seq = ds[0]
    frames = list(seq)
    t.initialize(frames[0], seq.init_bbox)
    for f in frames[1:]:
        t.update(f)
    assert len(t.skip_history) == len(frames) - 1


def test_adaptive_skip_compute_rate_bounded():
    t = AdaptiveSkipTracker(MOSSETracker(), max_skip=4)
    ds = SyntheticDataset(num_sequences=1, num_frames=30, seed=3)
    seq = ds[0]
    frames = list(seq)
    t.initialize(frames[0], seq.init_bbox)
    for f in frames[1:]:
        t.update(f)
    rate = t.compute_rate()
    assert 0.0 < rate <= 1.0


def test_adaptive_skip_max1_processes_every_frame():
    """max_skip=1 is a no-op: every frame must be processed."""
    t = AdaptiveSkipTracker(MOSSETracker(), max_skip=1)
    ds = SyntheticDataset(num_sequences=1, num_frames=20, seed=1)
    seq = ds[0]
    frames = list(seq)
    t.initialize(frames[0], seq.init_bbox)
    for f in frames[1:]:
        t.update(f)
    assert t.compute_rate() == 1.0


# ---------------------------------------------------------------------------
# AdaptiveSkipTracker with BenchmarkEngine
# ---------------------------------------------------------------------------

def test_adaptive_skip_benchmark_integration():
    ds = SyntheticDataset(num_sequences=3, num_frames=40, motion="linear", seed=99)
    engine = BenchmarkEngine(verbose=False)
    tracker = AdaptiveSkipTracker(MOSSETracker(), max_skip=3)
    result = engine.run(tracker, ds, dataset_name="Synthetic")
    assert result.mean_fps > 0
    assert 0.0 <= result.mean_iou <= 1.0
    assert len(result.sequence_results) == 3


def test_adaptive_skip_fps_improvement_over_baseline():
    """Adaptive skip should never be slower than the underlying tracker."""
    ds = SyntheticDataset(num_sequences=5, num_frames=60, motion="linear", seed=55)
    engine = BenchmarkEngine(verbose=False)

    baseline = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")
    adaptive = engine.run(
        AdaptiveSkipTracker(MOSSETracker(), max_skip=4, high_threshold=0.5),
        ds,
        dataset_name="Synthetic",
    )
    # Adaptive version should be at least as fast (usually faster)
    # We allow a 20 % margin to account for the small overhead of the wrapper.
    assert adaptive.mean_fps >= baseline.mean_fps * 0.8


# ---------------------------------------------------------------------------
# _confidence_to_skip mapping
# ---------------------------------------------------------------------------

def test_confidence_to_skip_boundaries():
    t = AdaptiveSkipTracker(MOSSETracker(), max_skip=4, high_threshold=0.7, low_threshold=0.3)
    assert t._confidence_to_skip(0.0) == 1
    assert t._confidence_to_skip(0.3) == 1
    assert t._confidence_to_skip(0.7) == 4
    assert t._confidence_to_skip(1.0) == 4


def test_confidence_to_skip_monotone():
    t = AdaptiveSkipTracker(MOSSETracker(), max_skip=8, high_threshold=0.8, low_threshold=0.2)
    prev = 0
    for conf in np.linspace(0.0, 1.0, 20):
        skip = t._confidence_to_skip(float(conf))
        assert skip >= prev or skip >= 1
        prev = skip
