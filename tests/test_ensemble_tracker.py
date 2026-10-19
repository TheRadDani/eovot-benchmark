"""Unit and integration tests for EnsembleTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker
from eovot.trackers.ensemble import EnsembleTracker, _box_iou, _weighted_box_average
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ConstantTracker(BaseTracker):
    """Test double: always returns a fixed bounding box."""

    def __init__(self, bbox, name="Constant"):
        super().__init__(name=name)
        self._bbox = tuple(bbox)

    def initialize(self, frame, bbox):
        pass

    def update(self, frame):
        return self._bbox


def _blank(h=120, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


INIT_BBOX = (20.0, 15.0, 50.0, 40.0)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestBoxIou:
    def test_identical_boxes(self):
        b = (10, 10, 20, 20)
        assert abs(_box_iou(b, b) - 1.0) < 1e-9

    def test_non_overlapping(self):
        a = (0, 0, 10, 10)
        b = (20, 20, 10, 10)
        assert _box_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = (0, 0, 20, 20)
        b = (10, 0, 20, 20)  # half overlap
        iou = _box_iou(a, b)
        # Intersection = 10×20 = 200; union = 400+400-200 = 600; IoU ≈ 0.333
        assert abs(iou - 200 / 600) < 1e-6

    def test_zero_area(self):
        assert _box_iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


class TestWeightedBoxAverage:
    def test_equal_weights_two_identical_boxes(self):
        b = (10.0, 10.0, 20.0, 20.0)
        result = _weighted_box_average([b, b], [1.0, 1.0])
        assert result == b

    def test_equal_weights_midpoint(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (10.0, 10.0, 10.0, 10.0)
        result = _weighted_box_average([a, b], [1.0, 1.0])
        assert abs(result[0] - 5.0) < 1e-9
        assert abs(result[1] - 5.0) < 1e-9

    def test_biased_weight(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (100.0, 100.0, 10.0, 10.0)
        result = _weighted_box_average([a, b], [1.0, 0.0])
        # Weight 0 means b is ignored — result should equal a
        assert abs(result[0] - 0.0) < 1e-9

    def test_degenerate_zero_total_weight(self):
        # Zero-weight degenerate case falls back to first box
        b = (5.0, 5.0, 5.0, 5.0)
        result = _weighted_box_average([b], [0.0])
        assert result == b


# ---------------------------------------------------------------------------
# EnsembleTracker construction
# ---------------------------------------------------------------------------

class TestEnsembleTrackerConstruction:
    def test_requires_at_least_two_trackers(self):
        with pytest.raises(ValueError, match="at least 2"):
            EnsembleTracker(trackers=[MOSSETracker()])

    def test_weights_length_mismatch(self):
        with pytest.raises(ValueError, match="len\\(weights\\)"):
            EnsembleTracker(
                trackers=[MOSSETracker(), KCFTracker()],
                weights=[1.0, 2.0, 3.0],
            )

    def test_zero_weight_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            EnsembleTracker(
                trackers=[MOSSETracker(), KCFTracker()],
                weights=[1.0, 0.0],
            )

    def test_name_contains_member_names(self):
        e = EnsembleTracker([_ConstantTracker(INIT_BBOX, "A"), _ConstantTracker(INIT_BBOX, "B")])
        assert "A" in e.name
        assert "B" in e.name

    def test_default_uniform_weights(self):
        e = EnsembleTracker([MOSSETracker(), KCFTracker()])
        assert all(w == 1.0 for w in e._weights)

    def test_is_base_tracker(self):
        e = EnsembleTracker([MOSSETracker(), KCFTracker()])
        assert isinstance(e, BaseTracker)


# ---------------------------------------------------------------------------
# Fusion correctness with deterministic test doubles
# ---------------------------------------------------------------------------

class TestEnsembleFusion:
    def test_uniform_weights_averages_boxes(self):
        box_a = (0.0, 0.0, 20.0, 20.0)
        box_b = (20.0, 20.0, 20.0, 20.0)
        ensemble = EnsembleTracker(
            trackers=[_ConstantTracker(box_a, "A"), _ConstantTracker(box_b, "B")],
            outlier_threshold=0.0,  # no outlier rejection
        )
        ensemble.initialize(_blank(), INIT_BBOX)
        fused = ensemble.update(_blank())
        # Average x/y should be 10, 10
        assert abs(fused[0] - 10.0) < 1e-9
        assert abs(fused[1] - 10.0) < 1e-9
        assert abs(fused[2] - 20.0) < 1e-9

    def test_biased_weights_pull_toward_higher_weight(self):
        box_a = (0.0, 0.0, 10.0, 10.0)
        box_b = (100.0, 0.0, 10.0, 10.0)
        ensemble = EnsembleTracker(
            trackers=[_ConstantTracker(box_a, "A"), _ConstantTracker(box_b, "B")],
            weights=[3.0, 1.0],
            outlier_threshold=0.0,
        )
        ensemble.initialize(_blank(), INIT_BBOX)
        fused = ensemble.update(_blank())
        # Weighted x = (0*3 + 100*1) / 4 = 25
        assert abs(fused[0] - 25.0) < 1e-9

    def test_three_member_consensus(self):
        """Two trackers agree, one disagrees far away — outlier should be filtered."""
        consensus = (50.0, 50.0, 20.0, 20.0)
        outlier = (200.0, 200.0, 20.0, 20.0)  # far from consensus
        ensemble = EnsembleTracker(
            trackers=[
                _ConstantTracker(consensus, "T1"),
                _ConstantTracker(consensus, "T2"),
                _ConstantTracker(outlier, "Outlier"),
            ],
            outlier_threshold=0.3,  # outlier won't overlap reference
        )
        ensemble.initialize(_blank(), INIT_BBOX)
        fused = ensemble.update(_blank())
        # Fused should be close to consensus, not pulled to outlier
        cx_fused = fused[0] + fused[2] / 2.0
        cx_consensus = consensus[0] + consensus[2] / 2.0
        assert abs(cx_fused - cx_consensus) < 15.0, f"Fused {cx_fused} pulled too far from {cx_consensus}"

    def test_all_trackers_outlier_fallback(self):
        """When all predictions are outliers, fall back to unfiltered average."""
        box_a = (0.0, 0.0, 1.0, 1.0)   # tiny box — IoU with any normal box ≈ 0
        box_b = (200.0, 200.0, 1.0, 1.0)
        ensemble = EnsembleTracker(
            trackers=[_ConstantTracker(box_a, "A"), _ConstantTracker(box_b, "B")],
            outlier_threshold=0.999,  # extremely strict — both will be outliers
        )
        ensemble.initialize(_blank(), INIT_BBOX)
        # Should not raise, should return fallback average
        fused = ensemble.update(_blank())
        assert len(fused) == 4


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestEnsembleDiagnostics:
    def test_individual_predictions_stored_after_update(self):
        box_a = (0.0, 0.0, 10.0, 10.0)
        box_b = (20.0, 20.0, 10.0, 10.0)
        ensemble = EnsembleTracker(
            [_ConstantTracker(box_a, "A"), _ConstantTracker(box_b, "B")],
            outlier_threshold=0.0,
        )
        ensemble.initialize(_blank(), INIT_BBOX)
        ensemble.update(_blank())
        indiv = ensemble.last_individual_predictions
        assert len(indiv) == 2
        assert indiv[0] == box_a
        assert indiv[1] == box_b

    def test_variance_zero_for_identical_predictions(self):
        box = (30.0, 30.0, 10.0, 10.0)
        ensemble = EnsembleTracker(
            [_ConstantTracker(box, "A"), _ConstantTracker(box, "B")],
            outlier_threshold=0.0,
        )
        ensemble.initialize(_blank(), INIT_BBOX)
        ensemble.update(_blank())
        assert ensemble.prediction_variance() < 1e-9

    def test_variance_positive_for_different_predictions(self):
        ensemble = EnsembleTracker(
            [_ConstantTracker((0.0, 0.0, 10.0, 10.0), "A"),
             _ConstantTracker((100.0, 100.0, 10.0, 10.0), "B")],
            outlier_threshold=0.0,
        )
        ensemble.initialize(_blank(), INIT_BBOX)
        ensemble.update(_blank())
        assert ensemble.prediction_variance() > 0.0

    def test_variance_zero_before_update(self):
        ensemble = EnsembleTracker([MOSSETracker(), KCFTracker()])
        assert ensemble.prediction_variance() == 0.0

    def test_trackers_property_returns_list(self):
        e = EnsembleTracker([MOSSETracker(), KCFTracker()])
        assert len(e.trackers) == 2


# ---------------------------------------------------------------------------
# Benchmark pipeline integration
# ---------------------------------------------------------------------------

class TestEnsembleBenchmarkIntegration:
    def test_benchmark_engine_runs_ensemble(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear", seed=5)
        engine = BenchmarkEngine(verbose=False)
        ensemble = EnsembleTracker(
            trackers=[MOSSETracker(), KCFTracker()],
            weights=[0.6, 0.4],
        )
        result = engine.run(ensemble, dataset, dataset_name="Synthetic")

        assert "Ensemble" in result.tracker_name
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_ensemble_iou_at_least_as_good_as_worst_member(self):
        """On a well-behaved synthetic sequence, the ensemble should not
        dramatically underperform the weakest individual member."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=3, num_frames=50, motion="linear", seed=42)
        engine = BenchmarkEngine(verbose=False)

        mosse_result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")
        kcf_result = engine.run(KCFTracker(), dataset, dataset_name="Synthetic")
        ensemble_result = engine.run(
            EnsembleTracker([MOSSETracker(), KCFTracker()]),
            dataset,
            dataset_name="Synthetic",
        )

        worst_iou = min(mosse_result.mean_iou, kcf_result.mean_iou)
        # The ensemble should not be catastrophically worse than the worst member
        # (allow 15% degradation as fusing two poor predictions can still be bad)
        assert ensemble_result.mean_iou >= worst_iou * 0.85, (
            f"Ensemble IoU {ensemble_result.mean_iou:.3f} much worse than "
            f"worst member IoU {worst_iou:.3f}"
        )
