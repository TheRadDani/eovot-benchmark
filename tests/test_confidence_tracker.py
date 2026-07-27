"""Unit tests for ConfidenceAdaptiveTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.confidence import ConfidenceAdaptiveTracker, _ncc, _extract_patch
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _make_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Create a synthetic BGR frame with a coloured square target."""
    frame = np.random.randint(50, 150, (h, w, 3), dtype=np.uint8)
    # Green target square at fixed position
    frame[80:120, 100:140, :] = [0, 200, 0]
    return frame


def _make_moving_sequence(n_frames: int = 30) -> tuple:
    """Return (frames, bbox_list) for a slowly moving target."""
    frames = []
    bboxes = []
    for i in range(n_frames):
        frame = np.random.randint(40, 120, (240, 320, 3), dtype=np.uint8)
        x_offset = i * 2  # 2-pixel drift per frame
        x1 = min(80 + x_offset, 280)
        frame[80:120, x1:x1 + 40, :] = [0, 180, 0]
        frames.append(frame)
        bboxes.append((float(x1), 80.0, 40.0, 40.0))
    return frames, bboxes


# ---------------------------------------------------------------------------
# _ncc unit tests
# ---------------------------------------------------------------------------

class TestNCC:
    def test_identical_patches_give_one(self):
        patch = np.random.rand(64, 64).astype(np.float32)
        assert _ncc(patch, patch) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_patches_give_minus_one(self):
        patch = np.random.rand(64, 64).astype(np.float32)
        neg = -patch
        # NCC is invariant to sign iff both have the same mean; shift to zero-mean
        patch -= patch.mean()
        neg = -patch
        assert _ncc(patch, neg) == pytest.approx(-1.0, abs=1e-5)

    def test_flat_patch_returns_zero(self):
        flat = np.ones((64, 64), dtype=np.float32)
        rand = np.random.rand(64, 64).astype(np.float32)
        assert _ncc(flat, rand) == pytest.approx(0.0)

    def test_range_is_bounded(self):
        for _ in range(10):
            a = np.random.rand(32, 32).astype(np.float32)
            b = np.random.rand(32, 32).astype(np.float32)
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
        # Partially outside, should still return a patch (the clamped crop)
        assert patch is not None


# ---------------------------------------------------------------------------
# ConfidenceAdaptiveTracker unit tests
# ---------------------------------------------------------------------------

class TestConfidenceAdaptiveTracker:
    def _make_tracker(self, **kwargs):
        inner = MOSSETracker()
        return ConfidenceAdaptiveTracker(inner, **kwargs)

    def test_init_sets_name_with_suffix(self):
        t = self._make_tracker()
        assert t.name.endswith("_conf")

    def test_constructor_validates_thresholds(self):
        with pytest.raises(ValueError, match="loss_threshold"):
            ConfidenceAdaptiveTracker(MOSSETracker(), loss_threshold=0.7, update_threshold=0.5)

    def test_update_before_initialize_raises(self):
        t = self._make_tracker()
        frame = _make_frame()
        with pytest.raises(RuntimeError, match="initialize"):
            t.update(frame)

    def test_initialize_and_single_update(self):
        t = self._make_tracker()
        frames, bboxes = _make_moving_sequence(n_frames=5)
        t.initialize(frames[0], bboxes[0])

        # Confidence starts at 1.0 (before any update)
        assert t.confidence == pytest.approx(1.0)
        assert not t.is_lost

        bbox = t.update(frames[1])
        assert len(bbox) == 4
        assert len(t.confidence_history) == 1

    def test_confidence_history_length_matches_update_count(self):
        t = self._make_tracker()
        frames, bboxes = _make_moving_sequence(n_frames=20)
        t.initialize(frames[0], bboxes[0])
        for i in range(1, 15):
            t.update(frames[i])
        assert len(t.confidence_history) == 14

    def test_confidence_values_in_valid_range(self):
        t = self._make_tracker()
        frames, bboxes = _make_moving_sequence(n_frames=20)
        t.initialize(frames[0], bboxes[0])
        for i in range(1, 20):
            t.update(frames[i])
        for c in t.confidence_history:
            assert -1.0 <= c <= 1.0

    def test_loss_rate_zero_when_no_losses(self):
        """On a clean, stable target the loss rate should be very low."""
        # Use a high loss_threshold so almost everything is 'lost' by default;
        # but here we set it very low so nothing is flagged.
        t = self._make_tracker(loss_threshold=-1.0, update_threshold=0.5)
        frames, bboxes = _make_moving_sequence(n_frames=15)
        t.initialize(frames[0], bboxes[0])
        for i in range(1, 15):
            t.update(frames[i])
        assert t.total_lost_frames == 0
        assert t.loss_rate == pytest.approx(0.0)

    def test_loss_rate_high_when_threshold_is_high(self):
        """With threshold=1.0 every frame is flagged as lost (NCC < 1)."""
        t = self._make_tracker(loss_threshold=1.0, update_threshold=2.0)
        frames, bboxes = _make_moving_sequence(n_frames=10)
        t.initialize(frames[0], bboxes[0])
        for i in range(1, 10):
            t.update(frames[i])
        assert t.total_lost_frames == 9
        assert t.loss_rate == pytest.approx(1.0)

    def test_mean_confidence_is_scalar(self):
        t = self._make_tracker()
        frames, bboxes = _make_moving_sequence(n_frames=10)
        t.initialize(frames[0], bboxes[0])
        for i in range(1, 10):
            t.update(frames[i])
        mc = t.mean_confidence
        assert isinstance(mc, float)
        assert -1.0 <= mc <= 1.0

    def test_lost_frame_indices_are_recorded(self):
        t = self._make_tracker(loss_threshold=1.0, update_threshold=2.0)
        frames, bboxes = _make_moving_sequence(n_frames=5)
        t.initialize(frames[0], bboxes[0])
        for i in range(1, 5):
            t.update(frames[i])
        # All 4 update frames should be marked lost
        assert t.lost_frame_indices == [1, 2, 3, 4]

    def test_reinitialize_resets_state(self):
        t = self._make_tracker(loss_threshold=1.0, update_threshold=2.0)
        frames, bboxes = _make_moving_sequence(n_frames=10)
        t.initialize(frames[0], bboxes[0])
        for i in range(1, 10):
            t.update(frames[i])
        assert t.total_lost_frames > 0

        # Re-initialize should reset everything
        t.initialize(frames[0], bboxes[0])
        assert t.total_lost_frames == 0
        assert t.confidence_history == []
        assert t.confidence == pytest.approx(1.0)

    def test_underlying_tracker_accessible(self):
        inner = MOSSETracker()
        t = ConfidenceAdaptiveTracker(inner)
        assert t.underlying_tracker is inner

    def test_compatible_with_benchmark_engine(self):
        """Full integration: ConfidenceAdaptiveTracker runs through BenchmarkEngine."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=2, num_frames=30, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        t = ConfidenceAdaptiveTracker(MOSSETracker(), loss_threshold=0.1)
        result = engine.run(t, dataset, dataset_name="Synthetic")

        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0
        # Confidence history should have been populated across both sequences
        # (each sequence re-initializes, so history resets; final sequence state)
        assert isinstance(t.confidence_history, list)
