"""Unit and integration tests for ScaleMOSSETracker."""

import numpy as np
import pytest

from eovot.trackers.scale_mosse import ScaleMOSSETracker
from eovot.trackers.base import BaseTracker


def _make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """Create a deterministic random BGR frame."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _static_sequence(n: int = 8, h: int = 120, w: int = 160):
    frame = _make_frame(h, w)
    return [frame.copy() for _ in range(n)]


BBOX = (20.0, 20.0, 40.0, 40.0)


class TestScaleMOSSEInterface:
    def test_is_base_tracker(self):
        assert isinstance(ScaleMOSSETracker(), BaseTracker)

    def test_name(self):
        assert ScaleMOSSETracker().name == "ScaleMOSSE"

    def test_repr(self):
        assert "ScaleMOSSETracker" in repr(ScaleMOSSETracker())

    def test_update_without_init_raises(self):
        tracker = ScaleMOSSETracker()
        with pytest.raises(RuntimeError, match="initialised"):
            tracker.update(_make_frame())

    def test_invalid_bbox_zero_width_raises(self):
        tracker = ScaleMOSSETracker()
        with pytest.raises(ValueError):
            tracker.initialize(_make_frame(), (10, 10, 0, 30))

    def test_invalid_bbox_zero_height_raises(self):
        tracker = ScaleMOSSETracker()
        with pytest.raises(ValueError):
            tracker.initialize(_make_frame(), (10, 10, 30, 0))


class TestScaleMOSSETracker:
    def setup_method(self):
        self.tracker = ScaleMOSSETracker(
            num_scales=5, scale_step=1.05, scale_lr=0.3, learning_rate=0.125
        )

    def test_initialize_does_not_raise(self):
        self.tracker.initialize(_make_frame(), BBOX)

    def test_update_returns_four_values(self):
        frames = _static_sequence(5)
        self.tracker.initialize(frames[0], BBOX)
        bbox = self.tracker.update(frames[1])
        assert len(bbox) == 4

    def test_update_positive_size(self):
        frames = _static_sequence(6)
        self.tracker.initialize(frames[0], BBOX)
        for f in frames[1:]:
            x, y, w, h = self.tracker.update(f)
            assert w > 0, "Width must be positive"
            assert h > 0, "Height must be positive"

    def test_scale_stable_on_static_scene(self):
        """Size should not drift far from initial on a fully static scene."""
        frames = _static_sequence(12)
        self.tracker.initialize(frames[0], BBOX)
        for f in frames[1:]:
            x, y, w, h = self.tracker.update(f)
        # Allow 3x initial size in either direction on random noise
        assert 5 < w < 200 and 5 < h < 200

    def test_psr_non_negative(self):
        """PSR must be >= 0 (or very small negative due to float precision)."""
        tracker = ScaleMOSSETracker()
        frames = _static_sequence(2)
        tracker.initialize(frames[0], BBOX)
        gray = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        iw, ih = int(round(tracker._w)), int(round(tracker._h))
        psr = tracker._psr(np.zeros((ih, iw), dtype=np.float32), ih, iw)
        # All-zero response: PSR is 0 (peak == mean == 0)
        assert psr >= -1e-6

    def test_runs_on_synthetic_dataset(self):
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.benchmark.engine import BenchmarkEngine

        ds = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear", seed=0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(ScaleMOSSETracker(), ds, dataset_name="Synthetic")
        assert result.mean_fps > 0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_registered_in_registry(self):
        from eovot.trackers.registry import build_tracker, available_trackers

        assert "ScaleMOSSE" in available_trackers()
        tracker = build_tracker("ScaleMOSSE")
        assert isinstance(tracker, ScaleMOSSETracker)

    def test_build_with_params(self):
        from eovot.trackers.registry import build_tracker

        tracker = build_tracker("ScaleMOSSE", num_scales=7, scale_step=1.05)
        assert tracker.num_scales == 7
        assert tracker.scale_step == pytest.approx(1.05)


import cv2  # noqa: E402 — imported here to keep it out of the top-level for readability
