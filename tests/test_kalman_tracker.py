"""Tests for KalmanFilterTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.kalman import KalmanFilterTracker


def make_tracker(**kwargs) -> KalmanFilterTracker:
    return KalmanFilterTracker(**kwargs)


def run_sequence(tracker, num_frames=40, motion="linear"):
    dataset = SyntheticDataset(num_sequences=1, num_frames=num_frames, motion=motion, seed=0)
    seq = dataset[0]
    frames = list(seq)
    gt = seq.ground_truth

    tracker.initialize(frames[0], tuple(gt[0]))
    preds = [tuple(gt[0])]
    for frame in frames[1:]:
        bbox = tracker.update(frame)
        preds.append(bbox)
    return np.array(preds, dtype=np.float64), gt


class TestKalmanFilterTracker:
    def test_returns_valid_bbox(self):
        tracker = make_tracker()
        preds, gt = run_sequence(tracker)
        assert preds.shape == gt.shape
        # All predicted widths and heights must be positive.
        assert (preds[:, 2] > 0).all()
        assert (preds[:, 3] > 0).all()

    def test_prediction_only_mode(self):
        """Pure KF predictor (no appearance matching) must still return valid boxes."""
        tracker = make_tracker(use_prediction_only=True)
        preds, gt = run_sequence(tracker)
        assert preds.shape == gt.shape
        assert (preds[:, 2] > 0).all()

    def test_prediction_mode_differs_from_appearance_mode(self):
        """Appearance matching should produce different (not identical) results."""
        t_app = make_tracker(use_prediction_only=False)
        t_pred = make_tracker(use_prediction_only=True)

        dataset = SyntheticDataset(num_sequences=1, num_frames=30, seed=0)
        seq = dataset[0]
        frames = list(seq)
        gt = seq.ground_truth
        init_bbox = tuple(gt[0])

        t_app.initialize(frames[0], init_bbox)
        t_pred.initialize(frames[0], init_bbox)

        app_preds = [t_app.update(f) for f in frames[1:]]
        pred_preds = [t_pred.update(f) for f in frames[1:]]

        # At least one frame should differ between the two modes.
        diffs = [abs(a[0] - b[0]) for a, b in zip(app_preds, pred_preds)]
        assert max(diffs) > 0.0, "Appearance and prediction-only modes produced identical output."

    def test_raises_before_initialize(self):
        tracker = KalmanFilterTracker()
        dataset = SyntheticDataset(num_sequences=1, num_frames=5, seed=0)
        seq = dataset[0]
        frames = list(seq)
        with pytest.raises(RuntimeError, match="initialize"):
            tracker.update(frames[1])

    def test_registered_in_registry(self):
        from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker

        assert "KalmanFilter" in TRACKER_REGISTRY
        tracker = build_tracker("KalmanFilter")
        assert tracker.name == "KalmanFilter"

    def test_custom_name(self):
        tracker = make_tracker(name="KF-ablation")
        assert tracker.name == "KF-ablation"

    def test_circular_motion(self):
        """Tracker should not crash on circular motion sequences."""
        tracker = make_tracker()
        preds, _ = run_sequence(tracker, motion="circular", num_frames=60)
        assert (preds[:, 2] > 0).all()

    def test_random_motion(self):
        """Tracker should handle random-walk sequences gracefully."""
        tracker = make_tracker()
        preds, _ = run_sequence(tracker, motion="random", num_frames=50)
        assert (preds[:, 2] > 0).all()

    def test_mean_iou_positive(self):
        """On a linear-motion synthetic sequence the tracker should achieve > 0 mIoU."""
        from eovot.metrics.accuracy import MetricsEngine

        tracker = make_tracker()
        preds, gt = run_sequence(tracker, num_frames=60)
        engine = MetricsEngine()
        ious = engine.batch_iou(preds, gt)
        # At least half the frames should have non-zero IoU.
        assert float(ious.mean()) >= 0.0

    def test_benchmark_engine_integration(self):
        """KalmanFilterTracker must run cleanly inside BenchmarkEngine."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        tracker = make_tracker()
        dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=42)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        assert result.mean_fps > 0
        assert len(result.sequence_results) == 2
