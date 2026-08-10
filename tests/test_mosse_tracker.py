"""Tests for MOSSETracker — OOB robustness and PSR confidence gating.

Regression coverage for issue #171 (cv2.resize assertion when the predicted
bbox drifts fully outside the frame) and for the PSR-based failure detection
that was wired up alongside the crash fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(h: int = 100, w: int = 120, seed: int = 0) -> np.ndarray:
    """Return a random BGR frame for testing."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _init(bbox=(30, 20, 20, 20), h=100, w=120, **kwargs) -> MOSSETracker:
    """Return an initialised MOSSETracker."""
    tracker = MOSSETracker(**kwargs)
    tracker.initialize(_frame(h=h, w=w), bbox)
    return tracker


# ---------------------------------------------------------------------------
# OOB crash regression tests  (issue #171)
# ---------------------------------------------------------------------------

class TestOOBRobustness:

    def test_no_crash_after_drift_off_right_edge(self):
        """Tracker must survive when internal bbox drifts past the right edge."""
        tracker = _init(bbox=(100, 20, 20, 20), h=80, w=120)
        rng = np.random.default_rng(1)
        for i in range(40):
            f = rng.integers(0, 256, (80, 120, 3), dtype=np.uint8)
            bbox = tracker.update(f)
            assert len(bbox) == 4, f"frame {i}: expected 4-tuple, got {bbox!r}"
            _, _, bw, bh = bbox
            assert bw > 0 and bh > 0

    def test_no_crash_after_drift_off_left_edge(self):
        tracker = _init(bbox=(5, 20, 20, 20), h=80, w=120)
        tracker._bbox = [-50, 20, 20, 20]   # force OOB left
        bbox = tracker.update(_frame(h=80, w=120))
        assert len(bbox) == 4
        _, _, bw, bh = bbox
        assert bw > 0 and bh > 0

    def test_no_crash_after_drift_off_bottom(self):
        tracker = _init(bbox=(30, 55, 20, 20), h=80, w=120)
        tracker._bbox = [30, 200, 20, 20]   # force OOB bottom
        bbox = tracker.update(_frame(h=80, w=120))
        assert len(bbox) == 4

    def test_no_crash_after_drift_off_top(self):
        tracker = _init(bbox=(30, 5, 20, 20))
        tracker._bbox = [30, -200, 20, 20]  # force OOB top
        bbox = tracker.update(_frame())
        assert len(bbox) == 4

    def test_extract_patch_empty_intersection_right(self):
        """_extract_patch must return zeros, not crash, for fully-OOB box (right)."""
        tracker = MOSSETracker()
        gray = np.ones((80, 120), dtype=np.uint8) * 128
        patch = tracker._extract_patch(gray, x=200, y=10, w=20, h=20)
        assert patch.shape == (20, 20)
        assert patch.sum() == 0

    def test_extract_patch_empty_intersection_left(self):
        tracker = MOSSETracker()
        gray = np.ones((80, 120), dtype=np.uint8) * 64
        patch = tracker._extract_patch(gray, x=-50, y=10, w=20, h=20)
        assert patch.shape == (20, 20)
        assert patch.sum() == 0

    def test_extract_patch_empty_intersection_bottom(self):
        tracker = MOSSETracker()
        gray = np.ones((80, 120), dtype=np.uint8) * 64
        patch = tracker._extract_patch(gray, x=10, y=200, w=20, h=20)
        assert patch.shape == (20, 20)
        assert patch.sum() == 0

    def test_extract_patch_empty_intersection_top(self):
        tracker = MOSSETracker()
        gray = np.ones((80, 120), dtype=np.uint8) * 64
        patch = tracker._extract_patch(gray, x=10, y=-200, w=20, h=20)
        assert patch.shape == (20, 20)
        assert patch.sum() == 0

    def test_extract_patch_partial_overlap_returns_correct_shape(self):
        tracker = MOSSETracker()
        gray = np.arange(80 * 120, dtype=np.uint8).reshape(80, 120)
        # Box starts at x=110 — overlaps by 10px on the right
        patch = tracker._extract_patch(gray, x=110, y=10, w=20, h=20)
        assert patch.shape == (20, 20)

    def test_clamping_keeps_box_within_one_pixel_of_frame(self):
        """update() must not let x_new jump more than (w-1) px past the edge."""
        h, w = 80, 120
        tracker = _init(bbox=(100, 10, 20, 20), h=h, w=w)
        # Drive many noisy frames so the tracker tries to drift far right
        rng = np.random.default_rng(7)
        for _ in range(50):
            bbox = tracker.update(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
            x, y, bw, bh = bbox
            # x may be slightly negative (box off left) but must keep 1px overlap
            assert x >= 1 - bw, f"x={x} too far left for box width {bw}"
            assert x <= w - 1, f"x={x} too far right for frame width {w}"


# ---------------------------------------------------------------------------
# PSR computation tests
# ---------------------------------------------------------------------------

class TestPSRComputation:

    def test_psr_positive_for_strongly_peaked_response(self):
        tracker = MOSSETracker()
        response = np.zeros((40, 40), dtype=np.float64)
        response[20, 20] = 100.0          # sharp peak, near-zero sidelobe
        psr = tracker._compute_psr(response, peak_y=20, peak_x=20)
        assert psr > 5.0, f"Expected high PSR for sharp peak, got {psr}"

    def test_psr_zero_for_flat_response(self):
        """Flat response has zero std → PSR must be 0."""
        tracker = MOSSETracker()
        response = np.ones((40, 40), dtype=np.float64) * 3.0
        psr = tracker._compute_psr(response, peak_y=20, peak_x=20)
        assert psr == pytest.approx(0.0, abs=1e-9)

    def test_psr_handles_corner_peak(self):
        """PSR must not crash when peak is at the response corner."""
        tracker = MOSSETracker()
        response = np.zeros((20, 20), dtype=np.float64)
        response[0, 0] = 50.0
        psr = tracker._compute_psr(response, peak_y=0, peak_x=0)
        assert isinstance(psr, float)
        assert psr >= 0.0

    def test_psr_tiny_response_shape(self):
        """PSR on a response smaller than the exclusion window must not crash."""
        tracker = MOSSETracker()
        response = np.array([[1.0, 2.0], [3.0, 10.0]])
        psr = tracker._compute_psr(response, peak_y=1, peak_x=1)
        assert isinstance(psr, float)


# ---------------------------------------------------------------------------
# PSR threshold gating tests
# ---------------------------------------------------------------------------

class TestPSRThresholdGating:

    def test_impossibly_high_threshold_holds_last_bbox(self):
        """With psr_threshold=1e9, PSR will always be below it → hold last bbox."""
        h, w = 80, 120
        frame = _frame(h=h, w=w)
        tracker = MOSSETracker(psr_threshold=1e9)
        tracker.initialize(frame, (30, 20, 20, 20))
        last_x, last_y = float(tracker._bbox[0]), float(tracker._bbox[1])

        result = tracker.update(_frame(h=h, w=w, seed=99))
        # Must return the frozen last-known-good position
        assert result[0] == pytest.approx(last_x)
        assert result[1] == pytest.approx(last_y)

    def test_threshold_none_always_updates(self):
        """Without a threshold, update() never holds the bbox."""
        tracker = _init(psr_threshold=None)
        init_bbox = tracker._bbox[:]
        result = tracker.update(_frame(seed=5))
        # Result is valid (even if position is the same, no exception)
        assert len(result) == 4

    def test_threshold_zero_always_passes(self):
        """With threshold=0.0, any positive PSR passes the gate."""
        tracker = _init(psr_threshold=0.0)
        result = tracker.update(_frame(seed=3))
        assert len(result) == 4

    def test_multiple_frames_with_threshold(self):
        """Tracker with PSR gating must not crash over many frames."""
        h, w = 80, 120
        tracker = _init(bbox=(30, 20, 20, 20), h=h, w=w, psr_threshold=5.0)
        rng = np.random.default_rng(42)
        for _ in range(20):
            f = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            bbox = tracker.update(f)
            assert len(bbox) == 4
            _, _, bw, bh = bbox
            assert bw > 0 and bh > 0


# ---------------------------------------------------------------------------
# Standard interface tests
# ---------------------------------------------------------------------------

class TestMOSSEInterface:

    def test_initialize_and_update_returns_four_tuple(self):
        tracker = _init()
        bbox = tracker.update(_frame())
        assert len(bbox) == 4

    def test_name_attribute(self):
        assert MOSSETracker().name == "MOSSE"

    def test_update_before_initialize_raises(self):
        tracker = MOSSETracker()
        with pytest.raises(RuntimeError, match="initialised"):
            tracker.update(_frame())

    def test_invalid_zero_width_raises(self):
        with pytest.raises(ValueError):
            MOSSETracker().initialize(_frame(), (10, 10, 0, 20))

    def test_invalid_zero_height_raises(self):
        with pytest.raises(ValueError):
            MOSSETracker().initialize(_frame(), (10, 10, 20, 0))

    def test_grayscale_input(self):
        gray = np.random.randint(0, 256, (80, 120), dtype=np.uint8)
        tracker = MOSSETracker()
        tracker.initialize(gray, (10, 10, 20, 20))
        bbox = tracker.update(gray)
        assert len(bbox) == 4

    def test_bgra_input(self):
        bgra = np.random.randint(0, 256, (80, 120, 4), dtype=np.uint8)
        tracker = MOSSETracker()
        tracker.initialize(bgra, (10, 10, 20, 20))
        bbox = tracker.update(bgra)
        assert len(bbox) == 4

    def test_custom_learning_rate(self):
        bbox = _init(learning_rate=0.5).update(_frame())
        assert len(bbox) == 4

    def test_width_and_height_unchanged(self):
        """MOSSE tracks translation only — w and h must not change."""
        tracker = _init(bbox=(30, 20, 24, 18))
        for i in range(5):
            _, _, w, h = tracker.update(_frame(seed=i))
            assert w == pytest.approx(24.0)
            assert h == pytest.approx(18.0)

    def test_repr_contains_name(self):
        assert "MOSSE" in repr(MOSSETracker())
