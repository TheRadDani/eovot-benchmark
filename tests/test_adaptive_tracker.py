"""Tests for AdaptiveTracker — adaptive frame-skipping wrapper.

All tests use in-memory synthetic frames and bounding boxes; no real
dataset files are needed.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

import numpy as np
import pytest

from eovot.trackers.adaptive import AdaptiveTracker
from eovot.trackers.base import BaseTracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.registry import build_tracker

# ---------------------------------------------------------------------------
# Minimal fake tracker for unit tests (avoids real tracker initialization)
# ---------------------------------------------------------------------------

class FakeTracker(BaseTracker):
    """Returns a fixed bounding box; counts how many times update() is called."""

    def __init__(self, box=(10.0, 10.0, 50.0, 50.0)):
        super().__init__(name="FakeTracker")
        self.box = box
        self.update_calls = 0

    def initialize(self, frame: np.ndarray, bbox) -> None:
        pass

    def update(self, frame: np.ndarray):
        self.update_calls += 1
        return self.box


def _blank_frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

def test_construction_defaults():
    tracker = AdaptiveTracker(FakeTracker())
    assert tracker.confidence_threshold == 0.80
    assert tracker.max_consecutive_skips == 3
    assert tracker.window_size == 8


def test_construction_custom_params():
    tracker = AdaptiveTracker(
        FakeTracker(), confidence_threshold=0.6, max_consecutive_skips=5, window_size=4
    )
    assert tracker.confidence_threshold == 0.6
    assert tracker.max_consecutive_skips == 5
    assert tracker.window_size == 4


def test_invalid_confidence_threshold():
    with pytest.raises(ValueError):
        AdaptiveTracker(FakeTracker(), confidence_threshold=1.5)


def test_invalid_max_consecutive_skips():
    with pytest.raises(ValueError):
        AdaptiveTracker(FakeTracker(), max_consecutive_skips=-1)


def test_invalid_window_size():
    with pytest.raises(ValueError):
        AdaptiveTracker(FakeTracker(), window_size=1)


def test_invalid_tracker_type():
    with pytest.raises(TypeError):
        AdaptiveTracker("not_a_tracker")  # type: ignore[arg-type]


def test_name_includes_inner_tracker_name():
    inner = FakeTracker()
    at = AdaptiveTracker(inner)
    assert "FakeTracker" in at.name


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

def test_update_before_initialize_raises():
    at = AdaptiveTracker(FakeTracker())
    with pytest.raises(RuntimeError):
        at.update(_blank_frame())


def test_initialize_resets_metrics():
    at = AdaptiveTracker(FakeTracker())
    frame = _blank_frame()
    at.initialize(frame, (10.0, 10.0, 50.0, 50.0))
    at.update(frame)
    at.initialize(frame, (10.0, 10.0, 50.0, 50.0))  # re-initialize
    assert at.skip_count == 0
    assert at.total_updates == 0


def test_returns_valid_bbox():
    at = AdaptiveTracker(FakeTracker())
    frame = _blank_frame()
    at.initialize(frame, (10.0, 10.0, 50.0, 50.0))
    for _ in range(15):
        bbox = at.update(frame)
    assert len(bbox) == 4
    x, y, w, h = bbox
    assert w > 0 and h > 0


# ---------------------------------------------------------------------------
# Skipping behaviour tests
# ---------------------------------------------------------------------------

def test_no_skip_before_window_full():
    """Skipping must not occur until the history window is filled."""
    inner = FakeTracker()
    at = AdaptiveTracker(inner, confidence_threshold=0.0, window_size=8)
    frame = _blank_frame()
    at.initialize(frame, (10.0, 10.0, 50.0, 50.0))
    # window_size = 8; initialize adds 1 entry, so we need 7 updates to fill it
    for _ in range(7):
        at.update(frame)
    assert at.skip_count == 0, "should not skip until window is full"


def test_forced_update_after_max_consecutive_skips():
    """After max_consecutive_skips, a full update must always be forced."""
    inner = FakeTracker()
    max_skip = 2
    # confidence_threshold=0.0 means skip whenever allowed
    at = AdaptiveTracker(inner, confidence_threshold=0.0, max_consecutive_skips=max_skip, window_size=2)
    frame = _blank_frame()
    at.initialize(frame, (10.0, 10.0, 50.0, 50.0))

    # Fill history (window_size=2; init adds 1, so 1 update fills it)
    at.update(frame)

    # Now run max_skip + 1 more updates.  At most max_skip should be skipped.
    consecutive_skips_seen = 0
    forced_update_after_max = False
    for _ in range(max_skip + 2):
        prev_skip = at.skip_count
        at.update(frame)
        if at.skip_count > prev_skip:
            consecutive_skips_seen += 1
        else:
            if consecutive_skips_seen >= max_skip:
                forced_update_after_max = True
            consecutive_skips_seen = 0

    assert forced_update_after_max, "full update was not forced after max_consecutive_skips"


def test_skip_rate_never_exceeds_one():
    inner = FakeTracker()
    at = AdaptiveTracker(inner, confidence_threshold=0.0, max_consecutive_skips=100, window_size=2)
    frame = _blank_frame()
    at.initialize(frame, (10.0, 10.0, 50.0, 50.0))
    for _ in range(50):
        at.update(frame)
    assert 0.0 <= at.skip_rate <= 1.0


def test_no_skip_when_max_consecutive_skips_is_zero():
    """max_consecutive_skips=0 must disable all skipping, even at confidence=0."""
    inner = FakeTracker()
    at = AdaptiveTracker(inner, confidence_threshold=0.0, max_consecutive_skips=0, window_size=4)
    frame = _blank_frame()
    at.initialize(frame, (10.0, 10.0, 50.0, 50.0))
    for _ in range(20):
        at.update(frame)
    assert at.skip_count == 0


def test_skip_count_plus_real_updates_equals_total():
    """skip_count + (total_updates - skip_count) must always equal total_updates."""
    inner = FakeTracker()
    at = AdaptiveTracker(inner, confidence_threshold=0.5, window_size=4)
    frame = _blank_frame()
    at.initialize(frame, (10.0, 10.0, 50.0, 50.0))
    for _ in range(30):
        at.update(frame)
    assert at.total_updates == 30
    # inner.update_calls counts real (non-skipped) updates
    assert inner.update_calls + at.skip_count == at.total_updates


# ---------------------------------------------------------------------------
# Constant-velocity prediction test
# ---------------------------------------------------------------------------

def test_constant_velocity_prediction_follows_motion():
    """On skipped frames, the predicted bbox extrapolates linear motion correctly.

    The history window needs to be full (window_size entries) before any skip
    can occur.  With window_size=4 and max_consecutive_skips=1:
      - init adds 1 entry
      - calls 1–3 are real (filling the window to 4 entries each time)
      - call 4 triggers the first skip (confidence≥0, consec=0 < 1)
      - call 5 forces a real update (consec reached max)
    """

    class LinearTracker(BaseTracker):
        """Simulates a tracker that follows a linearly moving target exactly."""
        def __init__(self):
            super().__init__(name="LinearTracker")
            self._step = 0

        def initialize(self, frame, bbox):
            self._step = 0

        def update(self, frame):
            self._step += 1
            return (10.0 + self._step * 5.0, 10.0, 50.0, 50.0)

    inner = LinearTracker()
    window = 4
    # confidence_threshold=0.0 → always skip when allowed
    at = AdaptiveTracker(inner, confidence_threshold=0.0, max_consecutive_skips=1, window_size=window)
    frame = _blank_frame()
    at.initialize(frame, (10.0, 10.0, 50.0, 50.0))

    # window_size=4; init inserts 1 entry; we need 3 more real updates to fill.
    for _ in range(window - 1):
        at.update(frame)
    # After 3 real updates: steps 1,2,3 → x=15,20,25
    # history is now full with 4 entries; next call can skip.

    pre_skip_count = at.skip_count
    skip_bbox = at.update(frame)

    assert at.skip_count > pre_skip_count, "expected at least one skip after window is full"
    # The skip uses constant-velocity from the last 4 boxes.
    # The inner tracker moved +5 px/frame, so the extrapolated x should be ≈30.
    assert abs(skip_bbox[0] - 30.0) < 5.0, (
        f"Constant-velocity prediction should land near x=30, got {skip_bbox[0]:.2f}"
    )


# ---------------------------------------------------------------------------
# Registry integration test
# ---------------------------------------------------------------------------

def test_build_tracker_adaptive_default_inner():
    tracker = build_tracker("Adaptive")
    assert isinstance(tracker, AdaptiveTracker)
    assert isinstance(tracker.underlying_tracker, MOSSETracker)


def test_build_tracker_adaptive_custom_inner():
    tracker = build_tracker("Adaptive", inner="KCF", confidence_threshold=0.70)
    assert isinstance(tracker, AdaptiveTracker)
    assert isinstance(tracker.underlying_tracker, KCFTracker)
    assert tracker.confidence_threshold == 0.70


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

def test_adaptive_tracker_runs_through_benchmark_engine():
    """AdaptiveTracker must be accepted by BenchmarkEngine without error."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.base import BaseDataset, Sequence

    N = 25
    gt = np.tile(np.array([10.0, 10.0, 50.0, 50.0]), (N, 1))

    class Seq(Sequence):
        def __init__(self):
            super().__init__("s0", [f"f{i}.jpg" for i in range(N)], gt)
        def __iter__(self) -> Iterator[np.ndarray]:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            for _ in range(N):
                yield frame

    class DS(BaseDataset):
        def __len__(self): return 1
        def __getitem__(self, idx): return Seq()

    tracker = AdaptiveTracker(MOSSETracker(), confidence_threshold=0.75)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(tracker, DS(), dataset_name="Synthetic")

    assert result.mean_iou >= 0.0
    assert result.mean_fps > 0.0
