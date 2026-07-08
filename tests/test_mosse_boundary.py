"""Tests for the MOSSE out-of-frame boundary crash fix (issue #171).

Verifies that MOSSETracker._extract_patch and update() never raise an
exception when the predicted target position drifts fully outside the
frame — matching the failure-handling pattern used by CSRT, MIL, and
MedianFlow.
"""

import numpy as np
import pytest

from eovot.trackers.mosse import MOSSETracker


def _gray(h: int = 100, w: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w), dtype=np.uint8)


def _bgr(h: int = 100, w: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# _extract_patch unit tests
# ---------------------------------------------------------------------------

class TestExtractPatchBoundary:
    """Unit tests for MOSSETracker._extract_patch."""

    def setup_method(self):
        self.tracker = MOSSETracker()

    def test_normal_centre_patch(self):
        gray = _gray(100, 120)
        patch = self.tracker._extract_patch(gray, 30, 30, 30, 30)
        assert patch.shape == (30, 30)
        assert patch.any()  # should contain image data, not zeros

    def test_partial_oob_right_edge(self):
        gray = _gray(100, 120)
        patch = self.tracker._extract_patch(gray, 105, 10, 30, 30)
        assert patch.shape == (30, 30)  # resized from partial overlap

    def test_partial_oob_bottom_edge(self):
        gray = _gray(100, 120)
        patch = self.tracker._extract_patch(gray, 10, 90, 30, 30)
        assert patch.shape == (30, 30)

    def test_partial_oob_top_left(self):
        gray = _gray(100, 120)
        patch = self.tracker._extract_patch(gray, -5, -5, 30, 30)
        assert patch.shape == (30, 30)

    # -- Fully out-of-frame cases that previously crashed with cv2.error --

    def test_fully_oob_right_returns_zeros(self):
        gray = _gray(100, 120)
        patch = self.tracker._extract_patch(gray, 200, 10, 30, 30)
        assert patch.shape == (30, 30)
        assert not patch.any(), "fully-OOB patch must be all zeros"

    def test_fully_oob_left_returns_zeros(self):
        gray = _gray(100, 120)
        patch = self.tracker._extract_patch(gray, -100, 10, 30, 30)
        assert patch.shape == (30, 30)
        assert not patch.any()

    def test_fully_oob_above_returns_zeros(self):
        gray = _gray(100, 120)
        patch = self.tracker._extract_patch(gray, 10, -200, 30, 30)
        assert patch.shape == (30, 30)
        assert not patch.any()

    def test_fully_oob_below_returns_zeros(self):
        gray = _gray(100, 120)
        patch = self.tracker._extract_patch(gray, 10, 500, 30, 30)
        assert patch.shape == (30, 30)
        assert not patch.any()

    def test_exactly_at_right_boundary_returns_zeros(self):
        """x == image_width means the box starts at the first pixel beyond."""
        gray = _gray(100, 120)
        patch = self.tracker._extract_patch(gray, 120, 10, 30, 30)
        assert patch.shape == (30, 30)
        assert not patch.any()

    def test_exactly_at_bottom_boundary_returns_zeros(self):
        gray = _gray(100, 120)
        patch = self.tracker._extract_patch(gray, 10, 100, 30, 30)
        assert patch.shape == (30, 30)
        assert not patch.any()


# ---------------------------------------------------------------------------
# update() integration tests
# ---------------------------------------------------------------------------

class TestMOSSEUpdateOOB:
    """MOSSETracker.update() must not raise when target drifts out of frame."""

    def test_update_does_not_crash_target_exits_right(self):
        tracker = MOSSETracker(learning_rate=0.125)
        tracker.initialize(_bgr(100, 120, 0), (80, 40, 30, 30))
        for i in range(30):
            bbox = tracker.update(_bgr(100, 120, i + 1))
            assert len(bbox) == 4
            assert all(isinstance(v, float) for v in bbox)

    def test_update_does_not_crash_target_exits_bottom(self):
        tracker = MOSSETracker()
        tracker.initialize(_bgr(100, 120, 5), (40, 85, 30, 20))
        for i in range(20):
            bbox = tracker.update(_bgr(100, 120, i + 10))
            assert len(bbox) == 4

    def test_update_does_not_crash_target_exits_left(self):
        tracker = MOSSETracker()
        tracker.initialize(_bgr(100, 120, 7), (5, 30, 30, 30))
        for i in range(20):
            bbox = tracker.update(_bgr(100, 120, i + 20))
            assert len(bbox) == 4

    def test_update_returns_four_floats_always(self):
        """A 50-frame run on pure-noise frames should never raise."""
        tracker = MOSSETracker()
        tracker.initialize(_bgr(80, 80, 99), (30, 30, 20, 20))
        for i in range(50):
            bbox = tracker.update(_bgr(80, 80, i))
            assert len(bbox) == 4
            assert all(isinstance(v, float) for v in bbox)

    def test_update_without_initialize_raises_runtime_error(self):
        tracker = MOSSETracker()
        with pytest.raises(RuntimeError, match="not initialised"):
            tracker.update(_bgr())

    def test_initialize_invalid_bbox_raises_value_error(self):
        tracker = MOSSETracker()
        with pytest.raises(ValueError, match="Invalid bounding box"):
            tracker.initialize(_bgr(), (10, 10, 0, 30))


# ---------------------------------------------------------------------------
# Regression: benchmark engine on synthetic data does not crash
# ---------------------------------------------------------------------------

class TestMOSSEBoundaryBenchmarkIntegration:
    """End-to-end regression: MOSSE + SyntheticDataset must not raise."""

    def test_mosse_on_synthetic_completes_without_error(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(
            num_sequences=3,
            num_frames=30,
            motion="random",
            seed=42,
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="synthetic")
        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0
        assert len(result.sequence_results) == 3
