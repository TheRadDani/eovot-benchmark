"""Tests for the DCF multi-channel correlation filter tracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.dcf import DCFTracker
from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine
from eovot.trackers.registry import build_tracker, available_trackers


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestDCFTrackerInterface:
    """Verify BaseTracker contract is fulfilled."""

    def _make_frame(self, h: int = 120, w: int = 160) -> np.ndarray:
        """Create a simple BGR test frame with a red rectangle."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[30:60, 40:90, 2] = 200  # red rectangle
        return frame

    def test_name(self):
        assert DCFTracker().name == "DCF"

    def test_initialize_and_update_returns_bbox(self):
        tracker = DCFTracker()
        frame = self._make_frame()
        bbox = (40.0, 30.0, 50.0, 30.0)
        tracker.initialize(frame, bbox)
        pred = tracker.update(frame)
        assert len(pred) == 4, "update() must return a 4-tuple (x, y, w, h)"
        assert all(isinstance(v, float) for v in pred)

    def test_update_before_initialize_raises(self):
        tracker = DCFTracker()
        frame = self._make_frame()
        with pytest.raises(RuntimeError, match="not initialised"):
            tracker.update(frame)

    def test_invalid_bbox_raises(self):
        tracker = DCFTracker()
        frame = self._make_frame()
        with pytest.raises(ValueError):
            tracker.initialize(frame, (10.0, 10.0, 0.0, 20.0))

    def test_reset_clears_state(self):
        tracker = DCFTracker()
        frame = self._make_frame()
        tracker.initialize(frame, (40.0, 30.0, 50.0, 30.0))
        tracker.reset()
        with pytest.raises(RuntimeError):
            tracker.update(frame)

    def test_grayscale_frame_accepted(self):
        tracker = DCFTracker()
        frame_bgr = self._make_frame()
        import cv2
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        tracker.initialize(frame_gray, (40.0, 30.0, 50.0, 30.0))
        pred = tracker.update(frame_gray)
        assert len(pred) == 4

    def test_repr(self):
        tracker = DCFTracker()
        assert "DCF" in repr(tracker)

    def test_custom_params(self):
        tracker = DCFTracker(learning_rate=0.05, padding=2.0, lambda_=1e-3)
        assert tracker.learning_rate == 0.05
        assert tracker.padding == 2.0
        assert tracker.lambda_ == 1e-3


class TestDCFTrackerAccuracy:
    """Verify that DCF achieves reasonable accuracy on synthetic sequences."""

    def test_stationary_target_high_iou(self):
        """A stationary target should yield near-perfect IoU."""
        tracker = DCFTracker()
        h, w = 120, 160
        bbox = (50.0, 40.0, 40.0, 30.0)

        # All frames identical — the tracker should stay on the target.
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[40:70, 50:90, 1] = 200  # green rectangle
        tracker.initialize(frame, bbox)

        ious = []
        from eovot.metrics.accuracy import iou
        for _ in range(20):
            pred = tracker.update(frame)
            ious.append(iou(pred, bbox))

        mean_iou = float(np.mean(ious))
        assert mean_iou > 0.5, (
            f"DCFTracker mean IoU on stationary target: {mean_iou:.3f} (expected > 0.5)"
        )

    def test_benchmark_pipeline_integration(self):
        """DCFTracker must work end-to-end in the BenchmarkEngine pipeline."""
        dataset = SyntheticDataset(num_sequences=3, num_frames=30, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(DCFTracker(), dataset, dataset_name="Synthetic")

        assert result.tracker_name == "DCF"
        assert len(result.sequence_results) == 3
        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0
        assert result.peak_memory_mb >= 0.0
        assert result.mean_success_auc is not None
        assert result.mean_precision_auc is not None

    def test_dcf_on_linear_motion_achieves_decent_iou(self):
        """DCF should achieve reasonable mean IoU on linear-motion sequences."""
        dataset = SyntheticDataset(num_sequences=5, num_frames=60, motion="linear", seed=0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(DCFTracker(), dataset, dataset_name="Synthetic")

        # On linear (predictable) motion the tracker should stay on target.
        assert result.mean_iou > 0.3, (
            f"DCFTracker mean IoU on linear motion: {result.mean_iou:.4f} (expected > 0.3)"
        )
        assert result.mean_success_auc is not None
        assert result.mean_success_auc > 0.0


class TestDCFRegistry:
    """Verify DCFTracker is accessible via the tracker registry."""

    def test_dcf_in_registry(self):
        assert "DCF" in available_trackers()

    def test_build_tracker_dcf(self):
        tracker = build_tracker("DCF")
        assert isinstance(tracker, DCFTracker)

    def test_build_tracker_dcf_with_params(self):
        tracker = build_tracker("DCF", learning_rate=0.05, padding=2.0)
        assert tracker.learning_rate == 0.05
        assert tracker.padding == 2.0
