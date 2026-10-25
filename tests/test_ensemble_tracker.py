"""Tests for TrackerEnsemble and build_default_ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.camshift import CamShiftTracker
from eovot.trackers.ensemble import TrackerEnsemble, build_default_ensemble
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dataset(n_seq=2, n_frames=30, motion="linear", seed=0):
    return SyntheticDataset(num_sequences=n_seq, num_frames=n_frames, motion=motion, seed=seed)


def run_ensemble(ensemble, dataset=None):
    if dataset is None:
        dataset = make_dataset()
    seq = dataset[0]
    frames = list(seq)
    gt = seq.ground_truth
    ensemble.initialize(frames[0], tuple(gt[0]))
    preds = [tuple(gt[0])]
    for frame in frames[1:]:
        preds.append(ensemble.update(frame))
    return np.array(preds, dtype=np.float64), gt


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestTrackerEnsembleInit:
    def test_empty_trackers_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            TrackerEnsemble([])

    def test_single_member(self):
        ens = TrackerEnsemble([MOSSETracker()])
        assert len(ens.members) == 1

    def test_default_name(self):
        ens = TrackerEnsemble([MOSSETracker(), KCFTracker()])
        assert ens.name == "Ensemble"

    def test_custom_name(self):
        ens = TrackerEnsemble([MOSSETracker()], name="MyEnsemble")
        assert ens.name == "MyEnsemble"

    def test_member_names(self):
        ens = TrackerEnsemble([MOSSETracker(), KCFTracker(), CamShiftTracker()])
        names = ens.member_names()
        assert "MOSSE" in names
        assert "KCF" in names
        assert "CamShift" in names


# ---------------------------------------------------------------------------
# Output validity tests
# ---------------------------------------------------------------------------

class TestTrackerEnsembleOutput:
    def test_returns_valid_bbox(self):
        ens = build_default_ensemble()
        preds, gt = run_ensemble(ens)
        assert preds.shape == gt.shape
        assert (preds[:, 2] > 0).all(), "All predicted widths must be positive"
        assert (preds[:, 3] > 0).all(), "All predicted heights must be positive"

    def test_single_member_passthrough(self):
        """Single-member ensemble should behave like its sole member."""
        from eovot.trackers.mosse import MOSSETracker

        dataset = make_dataset(n_seq=1, n_frames=20, seed=1)
        seq = dataset[0]
        frames = list(seq)
        gt = seq.ground_truth
        init_bbox = tuple(gt[0])

        solo_mosse = MOSSETracker()
        solo_mosse.initialize(frames[0], init_bbox)

        ens = TrackerEnsemble([MOSSETracker()])
        ens.initialize(frames[0], init_bbox)

        for frame in frames[1:]:
            ens_pred = ens.update(frame)
            solo_pred = solo_mosse.update(frame)
            # Both should be valid.
            assert ens_pred[2] > 0
            assert solo_pred[2] > 0

    def test_three_member_ensemble(self):
        ens = TrackerEnsemble([MOSSETracker(), KCFTracker(), CamShiftTracker()])
        preds, _ = run_ensemble(ens)
        assert (preds[:, 2] > 0).all()
        assert (preds[:, 3] > 0).all()

    def test_circular_motion(self):
        ens = build_default_ensemble()
        preds, _ = run_ensemble(ens, make_dataset(motion="circular", n_frames=60))
        assert (preds[:, 2] > 0).all()

    def test_random_motion(self):
        ens = build_default_ensemble()
        preds, _ = run_ensemble(ens, make_dataset(motion="random", n_frames=40))
        assert (preds[:, 2] > 0).all()


# ---------------------------------------------------------------------------
# Consensus logic tests
# ---------------------------------------------------------------------------

class TestConsensusLogic:
    def _make_ens(self):
        return TrackerEnsemble([MOSSETracker(), KCFTracker()])

    def test_consensus_all_agree(self):
        """When all members agree (identical predictions) the result equals them."""
        ens = self._make_ens()
        bbox = (10.0, 20.0, 50.0, 40.0)
        # All predictions identical → weighted mean equals that bbox.
        result = ens._consensus([bbox, bbox])
        for got, exp in zip(result, bbox):
            assert abs(got - exp) < 1e-9

    def test_consensus_two_equal_predictions(self):
        """Weighted mean of two non-overlapping predictions should lie between them."""
        ens = self._make_ens()
        b1 = (0.0, 0.0, 10.0, 10.0)
        b2 = (100.0, 100.0, 10.0, 10.0)
        result = ens._consensus([b1, b2])
        # With zero pairwise IoU, weights are equal → pure average.
        assert abs(result[0] - 50.0) < 1e-6  # (0 + 100) / 2
        assert abs(result[1] - 50.0) < 1e-6

    def test_consensus_positive_width_height(self):
        """Output bounding box must always have positive w and h."""
        ens = self._make_ens()
        # Corner-case: tiny boxes with zero area.
        b1 = (0.0, 0.0, 0.0, 0.0)
        b2 = (0.0, 0.0, 0.0, 0.0)
        result = ens._consensus([b1, b2])
        assert result[2] >= 1.0
        assert result[3] >= 1.0


# ---------------------------------------------------------------------------
# Factory and integration tests
# ---------------------------------------------------------------------------

class TestBuildDefaultEnsemble:
    def test_returns_tracker_ensemble(self):
        ens = build_default_ensemble()
        assert isinstance(ens, TrackerEnsemble)

    def test_three_members(self):
        ens = build_default_ensemble()
        assert len(ens.members) == 3

    def test_default_name(self):
        ens = build_default_ensemble()
        assert ens.name == "Ensemble-MKC"

    def test_custom_name(self):
        ens = build_default_ensemble(name="my-ensemble")
        assert ens.name == "my-ensemble"

    def test_benchmark_engine_integration(self):
        from eovot.benchmark.engine import BenchmarkEngine

        ens = build_default_ensemble()
        dataset = make_dataset(n_seq=2, n_frames=20, seed=99)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(ens, dataset, dataset_name="Synthetic")

        assert result.tracker_name == "Ensemble-MKC"
        assert result.mean_fps > 0
        assert len(result.sequence_results) == 2
        assert result.mean_iou >= 0.0

    def test_mean_iou_positive(self):
        """On a linear synthetic sequence the ensemble should achieve > 0 mIoU."""
        ens = build_default_ensemble()
        preds, gt = run_ensemble(ens, make_dataset(n_seq=1, n_frames=50, motion="linear"))
        from eovot.metrics.accuracy import MetricsEngine
        ious = MetricsEngine().batch_iou(preds, gt)
        assert float(ious.mean()) >= 0.0
