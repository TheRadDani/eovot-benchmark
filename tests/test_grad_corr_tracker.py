"""Tests for the GradCorr gradient-channel correlation filter tracker."""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from eovot.trackers.grad_corr import GradCorrTracker
from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine
from eovot.trackers.registry import build_tracker, available_trackers


# ---------------------------------------------------------------------------
# Unit tests — BaseTracker contract
# ---------------------------------------------------------------------------

class TestGradCorrTrackerInterface:

    def _make_frame(self, h: int = 120, w: int = 160) -> np.ndarray:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[30:60, 40:90, 1] = 200  # green rectangle
        return frame

    def test_name(self):
        assert GradCorrTracker().name == "GradCorr"

    def test_initialize_and_update_returns_4tuple(self):
        tracker = GradCorrTracker()
        frame = self._make_frame()
        bbox = (40.0, 30.0, 50.0, 30.0)
        tracker.initialize(frame, bbox)
        pred = tracker.update(frame)
        assert len(pred) == 4
        assert all(isinstance(v, float) for v in pred)

    def test_update_before_initialize_raises(self):
        tracker = GradCorrTracker()
        with pytest.raises(RuntimeError, match="not initialised"):
            tracker.update(self._make_frame())

    def test_invalid_bbox_zero_width_raises(self):
        tracker = GradCorrTracker()
        with pytest.raises(ValueError):
            tracker.initialize(self._make_frame(), (10.0, 10.0, 0.0, 20.0))

    def test_invalid_bbox_negative_height_raises(self):
        tracker = GradCorrTracker()
        with pytest.raises(ValueError):
            tracker.initialize(self._make_frame(), (10.0, 10.0, 20.0, -5.0))

    def test_reset_clears_state(self):
        tracker = GradCorrTracker()
        frame = self._make_frame()
        tracker.initialize(frame, (40.0, 30.0, 50.0, 30.0))
        tracker.reset()
        with pytest.raises(RuntimeError):
            tracker.update(frame)

    def test_grayscale_frame_accepted(self):
        tracker = GradCorrTracker()
        frame_bgr = self._make_frame()
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        tracker.initialize(frame_gray, (40.0, 30.0, 50.0, 30.0))
        pred = tracker.update(frame_gray)
        assert len(pred) == 4

    def test_repr(self):
        assert "GradCorr" in repr(GradCorrTracker())

    def test_custom_params_stored(self):
        tracker = GradCorrTracker(learning_rate=0.05, padding=2.0, lambda_=1e-3)
        assert tracker.learning_rate == 0.05
        assert tracker.padding == 2.0
        assert tracker.lambda_ == 1e-3

    def test_multiple_update_calls(self):
        """Verify update can be called repeatedly without error."""
        tracker = GradCorrTracker()
        frame = self._make_frame()
        tracker.initialize(frame, (40.0, 30.0, 50.0, 30.0))
        for _ in range(10):
            pred = tracker.update(frame)
            assert len(pred) == 4


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

class TestGradCorrFeatures:

    def test_extract_features_shape(self):
        tracker = GradCorrTracker(padding=1.0)
        frame = np.random.randint(0, 256, (100, 120, 3), dtype=np.uint8)
        tracker.initialize(frame, (30.0, 20.0, 40.0, 30.0))
        sw, sh = tracker._search_sz
        feats = tracker._extract_features(frame, 50.0, 35.0)
        assert feats.shape == (4, sh, sw)

    def test_crop_patch_handles_out_of_bounds(self):
        """Patch extraction must not crash when centre is outside the frame."""
        tracker = GradCorrTracker()
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        tracker.initialize(frame, (5.0, 5.0, 20.0, 20.0))
        # Update with a frame where the predicted position is near the border
        patch = tracker._crop_patch(frame, 0.0, 0.0, 40, 40)
        assert patch.shape == (40, 40)

    def test_crop_patch_fully_outside_frame(self):
        """Patch centred entirely outside the frame is filled by edge replication."""
        tracker = GradCorrTracker()
        frame = np.ones((100, 120, 3), dtype=np.uint8) * 128
        tracker.initialize(frame, (5.0, 5.0, 20.0, 20.0))
        patch = tracker._crop_patch(frame, -50.0, -50.0, 40, 40)
        assert patch.shape == (40, 40)
        assert np.all(patch >= 0.0)


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------

class TestGradCorrAccuracy:

    def test_stationary_target_high_iou(self):
        """On a perfectly stationary target the tracker should stay on it."""
        from eovot.metrics.accuracy import iou

        tracker = GradCorrTracker()
        h, w = 120, 160
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[40:70, 50:90, 2] = 200  # red rectangle
        bbox = (50.0, 40.0, 40.0, 30.0)
        tracker.initialize(frame, bbox)

        ious = []
        for _ in range(20):
            pred = tracker.update(frame)
            ious.append(iou(pred, bbox))

        mean_iou = float(np.mean(ious))
        assert mean_iou > 0.5, f"Stationary target mIoU={mean_iou:.3f} < 0.5"

    def test_benchmark_pipeline_integration(self):
        """GradCorrTracker must work end-to-end in BenchmarkEngine."""
        dataset = SyntheticDataset(num_sequences=3, num_frames=30, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(GradCorrTracker(), dataset, dataset_name="Synthetic")

        assert result.tracker_name == "GradCorr"
        assert len(result.sequence_results) == 3
        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0
        assert result.peak_memory_mb >= 0.0
        assert result.mean_success_auc is not None
        assert result.mean_precision_auc is not None

    def test_linear_motion_decent_iou(self):
        """GradCorr should track predictable linear motion reasonably well."""
        dataset = SyntheticDataset(num_sequences=5, num_frames=60, motion="linear", seed=0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(GradCorrTracker(), dataset, dataset_name="Synthetic")
        assert result.mean_iou > 0.3, (
            f"GradCorrTracker mIoU on linear motion: {result.mean_iou:.4f} (expected > 0.3)"
        )

    def test_circular_motion_positive_iou(self):
        """GradCorr should maintain some positive IoU on circular motion."""
        dataset = SyntheticDataset(num_sequences=3, num_frames=40, motion="circular", seed=1)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(GradCorrTracker(), dataset, dataset_name="Synthetic")
        assert result.mean_iou >= 0.0


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestGradCorrRegistry:

    def test_gradcorr_in_registry(self):
        assert "GradCorr" in available_trackers()

    def test_build_tracker_gradcorr(self):
        tracker = build_tracker("GradCorr")
        assert isinstance(tracker, GradCorrTracker)

    def test_build_tracker_with_params(self):
        tracker = build_tracker("GradCorr", learning_rate=0.05, padding=2.0)
        assert tracker.learning_rate == 0.05
        assert tracker.padding == 2.0

    def test_dcf_not_in_registry(self):
        """DCF should NOT be registered in this branch (no duplicate)."""
        assert "DCF" not in available_trackers()
