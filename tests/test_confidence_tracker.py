"""Unit tests for ConfidenceAdaptiveTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.confidence import ConfidenceAdaptiveTracker, _ncc, _extract_patch
from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Stubs and helpers
# ---------------------------------------------------------------------------

class _FixedTracker(BaseTracker):
    """Tracker stub that always returns a fixed bbox.

    Used in unit tests that verify ConfidenceAdaptiveTracker's own logic
    (confidence math, history tracking, loss flagging) without depending on
    a real tracker's accuracy or OpenCV internals.
    """

    def __init__(self, bbox: BBox = (100.0, 80.0, 40.0, 40.0)) -> None:
        super().__init__(name="Fixed")
        self._bbox = bbox

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        return self._bbox


def _make_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Create a synthetic BGR frame with a coloured square target."""
    rng = np.random.default_rng(0)
    frame = rng.integers(50, 150, (h, w, 3), dtype=np.uint8)
    frame[80:120, 100:140, :] = [0, 200, 0]
    return frame


def _make_static_sequence(n_frames: int = 20, bbox: BBox = (100.0, 80.0, 40.0, 40.0)) -> tuple:
    """Return (frame, [bbox]*n) for a stationary target.

    Using the same frame + bbox every update avoids MOSSE predicting
    out-of-bounds locations when the target drifts off-screen.
    """
    frame = _make_frame()
    return frame, [bbox] * n_frames


# ---------------------------------------------------------------------------
# _ncc unit tests
# ---------------------------------------------------------------------------

class TestNCC:
    def test_identical_patches_give_one(self):
        rng = np.random.default_rng(1)
        patch = rng.random((64, 64)).astype(np.float32)
        assert _ncc(patch, patch) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_patches_give_minus_one(self):
        rng = np.random.default_rng(2)
        patch = rng.random((64, 64)).astype(np.float32)
        patch -= patch.mean()
        neg = -patch
        assert _ncc(patch, neg) == pytest.approx(-1.0, abs=1e-5)

    def test_flat_patch_returns_zero(self):
        flat = np.ones((64, 64), dtype=np.float32)
        rng = np.random.default_rng(3)
        rand = rng.random((64, 64)).astype(np.float32)
        assert _ncc(flat, rand) == pytest.approx(0.0)

    def test_range_is_bounded(self):
        rng = np.random.default_rng(4)
        for _ in range(10):
            a = rng.random((32, 32)).astype(np.float32)
            b = rng.random((32, 32)).astype(np.float32)
            assert -1.0 <= _ncc(a, b) <= 1.0


# ---------------------------------------------------------------------------
# _extract_patch unit tests
# ---------------------------------------------------------------------------

class TestExtractPatch:
    def test_normal_bbox_returns_correct_shape(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        patch = _extract_patch(frame, (50.0, 60.0, 80.0, 80.0), (64, 64))
        assert patch is not None
        assert patch.shape == (64, 64)

    def test_out_of_bounds_bbox_returns_none(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        patch = _extract_patch(frame, (350.0, 260.0, 40.0, 40.0), (64, 64))
        assert patch is None

    def test_grayscale_frame_works(self):
        frame = np.zeros((240, 320), dtype=np.uint8)
        patch = _extract_patch(frame, (10.0, 10.0, 50.0, 50.0), (32, 32))
        assert patch is not None
        assert patch.shape == (32, 32)

    def test_partial_out_of_bounds_clamps(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        patch = _extract_patch(frame, (300.0, 220.0, 40.0, 40.0), (64, 64))
        assert patch is not None


# ---------------------------------------------------------------------------
# ConfidenceAdaptiveTracker unit tests (using _FixedTracker stub)
# ---------------------------------------------------------------------------

class TestConfidenceAdaptiveTracker:
    """Tests using _FixedTracker so results are deterministic and MOSSE
    internals cannot cause out-of-bounds cv2 errors."""

    def _make_cat(self, **kwargs) -> ConfidenceAdaptiveTracker:
        return ConfidenceAdaptiveTracker(_FixedTracker(), **kwargs)

    def test_init_sets_name_with_suffix(self):
        t = self._make_cat()
        assert t.name.endswith("_conf")

    def test_constructor_validates_thresholds(self):
        with pytest.raises(ValueError, match="loss_threshold"):
            ConfidenceAdaptiveTracker(_FixedTracker(), loss_threshold=0.7, update_threshold=0.5)

    def test_update_before_initialize_raises(self):
        t = self._make_cat()
        frame = _make_frame()
        with pytest.raises(RuntimeError, match="initialize"):
            t.update(frame)

    def test_initialize_confidence_is_one(self):
        t = self._make_cat()
        frame, bboxes = _make_static_sequence(n_frames=1)
        t.initialize(frame, bboxes[0])
        assert t.confidence == pytest.approx(1.0)
        assert not t.is_lost

    def test_single_update_returns_four_tuple(self):
        t = self._make_cat()
        frame, bboxes = _make_static_sequence(n_frames=2)
        t.initialize(frame, bboxes[0])
        bbox = t.update(frame)
        assert len(bbox) == 4
        assert len(t.confidence_history) == 1

    def test_confidence_history_length_matches_update_count(self):
        t = self._make_cat()
        frame, bboxes = _make_static_sequence(n_frames=20)
        t.initialize(frame, bboxes[0])
        for _ in range(14):
            t.update(frame)
        assert len(t.confidence_history) == 14

    def test_confidence_values_in_valid_range(self):
        """NCC output must always be in [-1, 1]."""
        t = self._make_cat()
        frame, bboxes = _make_static_sequence(n_frames=20)
        t.initialize(frame, bboxes[0])
        for _ in range(19):
            t.update(frame)
        for c in t.confidence_history:
            assert -1.0 <= c <= 1.0

    def test_loss_rate_zero_when_threshold_is_minus_one(self):
        """loss_threshold=-1.0 → nothing is ever flagged as lost."""
        t = self._make_cat(loss_threshold=-1.0, update_threshold=0.5)
        frame, bboxes = _make_static_sequence(n_frames=15)
        t.initialize(frame, bboxes[0])
        for _ in range(14):
            t.update(frame)
        assert t.total_lost_frames == 0
        assert t.loss_rate == pytest.approx(0.0)

    def test_loss_rate_one_when_threshold_is_one(self):
        """loss_threshold=1.0 → every real frame is flagged (NCC is never exactly 1)."""
        t = self._make_cat(loss_threshold=1.0, update_threshold=2.0)
        frame, bboxes = _make_static_sequence(n_frames=10)
        t.initialize(frame, bboxes[0])
        for _ in range(9):
            t.update(frame)
        assert t.total_lost_frames == 9
        assert t.loss_rate == pytest.approx(1.0)

    def test_mean_confidence_is_float_in_range(self):
        t = self._make_cat()
        frame, bboxes = _make_static_sequence(n_frames=10)
        t.initialize(frame, bboxes[0])
        for _ in range(9):
            t.update(frame)
        mc = t.mean_confidence
        assert isinstance(mc, float)
        assert -1.0 <= mc <= 1.0

    def test_lost_frame_indices_are_1_indexed(self):
        t = self._make_cat(loss_threshold=1.0, update_threshold=2.0)
        frame, bboxes = _make_static_sequence(n_frames=5)
        t.initialize(frame, bboxes[0])
        for _ in range(4):
            t.update(frame)
        assert t.lost_frame_indices == [1, 2, 3, 4]

    def test_reinitialize_resets_all_state(self):
        t = self._make_cat(loss_threshold=1.0, update_threshold=2.0)
        frame, bboxes = _make_static_sequence(n_frames=10)
        t.initialize(frame, bboxes[0])
        for _ in range(9):
            t.update(frame)
        assert t.total_lost_frames > 0

        t.initialize(frame, bboxes[0])
        assert t.total_lost_frames == 0
        assert t.confidence_history == []
        assert t.confidence == pytest.approx(1.0)
        assert t.loss_rate == pytest.approx(0.0)

    def test_underlying_tracker_accessible(self):
        inner = _FixedTracker()
        t = ConfidenceAdaptiveTracker(inner)
        assert t.underlying_tracker is inner

    def test_mean_confidence_before_updates_is_one(self):
        t = self._make_cat()
        assert t.mean_confidence == pytest.approx(1.0)

    def test_loss_rate_before_updates_is_zero(self):
        t = self._make_cat()
        assert t.loss_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration tests with real trackers
# ---------------------------------------------------------------------------

class TestConfidenceAdaptiveTrackerIntegration:
    """Integration tests using MOSSE + SyntheticDataset.

    SyntheticDataset keeps the target well inside the frame boundary,
    so MOSSE's internal _extract_patch will not encounter empty crops.
    """

    def test_compatible_with_benchmark_engine(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=2, num_frames=30, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        t = ConfidenceAdaptiveTracker(MOSSETracker(), loss_threshold=0.1)
        result = engine.run(t, dataset, dataset_name="Synthetic")

        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0
        assert isinstance(t.confidence_history, list)

    def test_mosse_static_target_gives_high_confidence(self):
        """MOSSE on a stationary target should produce non-trivial NCC values."""
        frame = _make_frame()
        bbox = (100.0, 80.0, 40.0, 40.0)
        t = ConfidenceAdaptiveTracker(MOSSETracker(), loss_threshold=0.05)
        t.initialize(frame, bbox)
        for _ in range(5):
            t.update(frame)
        assert len(t.confidence_history) == 5
        for c in t.confidence_history:
            assert -1.0 <= c <= 1.0
