"""Tests for eovot.trackers.ensemble.EnsembleTracker.

All tests use in-memory fake frames so no external dataset download is needed.
"""
from __future__ import annotations

from typing import List, Tuple
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.ensemble import EnsembleTracker, _iou


# ---------------------------------------------------------------------------
# Minimal stub tracker (records calls, returns a fixed bbox)
# ---------------------------------------------------------------------------

class _StubTracker(BaseTracker):
    """Deterministic stub: always returns the bbox given at construction."""

    def __init__(self, name: str, fixed_bbox: BBox) -> None:
        super().__init__(name=name)
        self.fixed_bbox = fixed_bbox
        self.init_calls: List[Tuple] = []
        self.update_calls: List[np.ndarray] = []

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self.init_calls.append((frame, bbox))

    def update(self, frame: np.ndarray) -> BBox:
        self.update_calls.append(frame)
        return self.fixed_bbox


def _frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _make_ensemble(
    bboxes: List[BBox],
    fusion: str = "mean",
    weights=None,
    name=None,
) -> EnsembleTracker:
    trackers = [_StubTracker(f"T{i}", b) for i, b in enumerate(bboxes)]
    return EnsembleTracker(trackers, fusion=fusion, weights=weights, name=name)


# ---------------------------------------------------------------------------
# _iou helper
# ---------------------------------------------------------------------------

class TestIouHelper:
    def test_identical_boxes_give_one(self):
        b = (10.0, 20.0, 50.0, 40.0)
        assert _iou(b, b) == pytest.approx(1.0)

    def test_non_overlapping_gives_zero(self):
        assert _iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 10.0, 10.0)) == 0.0

    def test_zero_width_gives_zero(self):
        assert _iou((0.0, 0.0, 0.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 0.0

    def test_zero_height_gives_zero(self):
        assert _iou((0.0, 0.0, 10.0, 0.0), (0.0, 0.0, 10.0, 10.0)) == 0.0

    def test_partial_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 0.0, 10.0, 10.0)
        iou = _iou(a, b)
        # Intersection = 5×10=50; union = 100+100-50=150
        assert iou == pytest.approx(50.0 / 150.0, abs=1e-9)

    def test_symmetry(self):
        a = (0.0, 0.0, 30.0, 20.0)
        b = (10.0, 5.0, 30.0, 20.0)
        assert _iou(a, b) == pytest.approx(_iou(b, a))


# ---------------------------------------------------------------------------
# EnsembleTracker construction
# ---------------------------------------------------------------------------

class TestEnsembleConstruction:
    def test_empty_trackers_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            EnsembleTracker([])

    def test_invalid_fusion_raises(self):
        t = _StubTracker("T", (0, 0, 10, 10))
        with pytest.raises(ValueError, match="Unknown fusion"):
            EnsembleTracker([t], fusion="bad_strategy")

    def test_weight_length_mismatch_raises(self):
        trackers = [_StubTracker("A", (0, 0, 10, 10)), _StubTracker("B", (0, 0, 10, 10))]
        with pytest.raises(ValueError, match="len\\(weights\\)"):
            EnsembleTracker(trackers, weights=[1.0])

    def test_zero_weight_raises(self):
        t = _StubTracker("T", (0, 0, 10, 10))
        with pytest.raises(ValueError, match="strictly positive"):
            EnsembleTracker([t], weights=[0.0])

    def test_negative_weight_raises(self):
        trackers = [_StubTracker("A", (0, 0, 10, 10)), _StubTracker("B", (0, 0, 10, 10))]
        with pytest.raises(ValueError, match="strictly positive"):
            EnsembleTracker(trackers, weights=[1.0, -1.0])

    def test_default_fusion_is_mean(self):
        t = _StubTracker("T", (0, 0, 10, 10))
        e = EnsembleTracker([t])
        assert e.fusion == "mean"

    def test_base_trackers_stored(self):
        trackers = [_StubTracker("A", (0, 0, 10, 10)), _StubTracker("B", (5, 5, 10, 10))]
        e = EnsembleTracker(trackers)
        assert e.base_trackers is not trackers  # stored as a copy
        assert len(e.base_trackers) == 2

    def test_is_base_tracker_subclass(self):
        t = _StubTracker("T", (0, 0, 10, 10))
        e = EnsembleTracker([t])
        assert isinstance(e, BaseTracker)

    def test_default_name_generated(self):
        trackers = [_StubTracker("MOSSE", (0, 0, 10, 10)), _StubTracker("KCF", (5, 5, 10, 10))]
        e = EnsembleTracker(trackers, fusion="vote")
        assert "MOSSE" in e.name
        assert "KCF" in e.name
        assert "vote" in e.name

    def test_custom_name_respected(self):
        t = _StubTracker("T", (0, 0, 10, 10))
        e = EnsembleTracker([t], name="MyEnsemble")
        assert e.name == "MyEnsemble"

    def test_weights_normalised(self):
        trackers = [_StubTracker("A", (0, 0, 10, 10)), _StubTracker("B", (5, 5, 10, 10))]
        e = EnsembleTracker(trackers, weights=[2.0, 2.0])
        assert e._weights.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# initialize() and update() delegation
# ---------------------------------------------------------------------------

class TestDelegation:
    def test_initialize_called_on_all_trackers(self):
        bboxes = [(0, 0, 10, 10), (5, 5, 10, 10)]
        e = _make_ensemble(bboxes)
        frame = _frame()
        init_bbox = (10.0, 10.0, 50.0, 50.0)
        e.initialize(frame, init_bbox)
        for tracker in e.base_trackers:
            assert len(tracker.init_calls) == 1
            assert tracker.init_calls[0][1] == init_bbox

    def test_update_called_on_all_trackers(self):
        bboxes = [(0, 0, 10, 10), (5, 5, 10, 10)]
        e = _make_ensemble(bboxes)
        frame = _frame()
        e.initialize(frame, (0, 0, 10, 10))
        e.update(frame)
        for tracker in e.base_trackers:
            assert len(tracker.update_calls) == 1

    def test_single_tracker_passthrough(self):
        fixed = (7.0, 8.0, 30.0, 25.0)
        e = _make_ensemble([fixed], fusion="mean")
        e.initialize(_frame(), (0.0, 0.0, 10.0, 10.0))
        result = e.update(_frame())
        assert result == fixed

    def test_single_tracker_vote_passthrough(self):
        fixed = (7.0, 8.0, 30.0, 25.0)
        e = _make_ensemble([fixed], fusion="vote")
        e.initialize(_frame(), (0.0, 0.0, 10.0, 10.0))
        result = e.update(_frame())
        assert result == fixed


# ---------------------------------------------------------------------------
# Mean fusion
# ---------------------------------------------------------------------------

class TestMeanFusion:
    def test_equal_weights_average(self):
        b1 = (0.0, 0.0, 10.0, 10.0)
        b2 = (10.0, 10.0, 10.0, 10.0)
        e = _make_ensemble([b1, b2], fusion="mean")
        e.initialize(_frame(), b1)
        result = e.update(_frame())
        expected = (5.0, 5.0, 10.0, 10.0)
        for r, ex in zip(result, expected):
            assert r == pytest.approx(ex, abs=1e-6)

    def test_equal_weights_three_trackers(self):
        b1 = (0.0,  0.0,  10.0, 10.0)
        b2 = (6.0,  6.0,  10.0, 10.0)
        b3 = (3.0,  3.0,  10.0, 10.0)
        e = _make_ensemble([b1, b2, b3], fusion="mean")
        e.initialize(_frame(), b1)
        result = e.update(_frame())
        for r, ex in zip(result, (3.0, 3.0, 10.0, 10.0)):
            assert r == pytest.approx(ex, abs=1e-6)

    def test_custom_weights_pull_toward_heavier(self):
        b1 = (0.0, 0.0, 10.0, 10.0)
        b2 = (20.0, 20.0, 10.0, 10.0)
        # Weight b2 3× heavier → fused x should be 3/4 * 20 = 15
        e = _make_ensemble([b1, b2], fusion="mean", weights=[1.0, 3.0])
        e.initialize(_frame(), b1)
        result = e.update(_frame())
        assert result[0] == pytest.approx(15.0, abs=1e-6)
        assert result[1] == pytest.approx(15.0, abs=1e-6)

    def test_mean_fusion_returns_four_tuple(self):
        e = _make_ensemble([(0, 0, 10, 10), (5, 5, 10, 10)], fusion="mean")
        e.initialize(_frame(), (0, 0, 10, 10))
        result = e.update(_frame())
        assert len(result) == 4
        for v in result:
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Vote fusion
# ---------------------------------------------------------------------------

class TestVoteFusion:
    def test_vote_selects_consensus_of_two_agreeing(self):
        """Two trackers agree on one box, one outlier: consensus box wins."""
        agree1 = (10.0, 10.0, 20.0, 20.0)
        agree2 = (12.0, 12.0, 20.0, 20.0)  # slightly off but close
        outlier = (200.0, 200.0, 20.0, 20.0)
        e = _make_ensemble([agree1, agree2, outlier], fusion="vote")
        e.initialize(_frame(), agree1)
        result = e.update(_frame())
        # Result must be one of the three candidate predictions
        assert result in [agree1, agree2, outlier]
        # The outlier has very low pairwise IoU with agree1/agree2,
        # so the winner should be agree1 or agree2
        assert result != outlier

    def test_vote_with_identical_predictions(self):
        """All trackers agree → any of them should win."""
        b = (5.0, 5.0, 30.0, 30.0)
        e = _make_ensemble([b, b, b], fusion="vote")
        e.initialize(_frame(), b)
        result = e.update(_frame())
        assert result == b

    def test_vote_with_two_trackers(self):
        """Two trackers — the one with non-zero IoU wins when it overlaps itself."""
        b1 = (0.0, 0.0, 10.0, 10.0)
        b2 = (50.0, 50.0, 10.0, 10.0)
        e = _make_ensemble([b1, b2], fusion="vote")
        e.initialize(_frame(), b1)
        # With two non-overlapping trackers, mean pairwise IoU = 0 for both.
        # argmax of [0, 0] → index 0 (NumPy picks lowest index for ties).
        result = e.update(_frame())
        assert result == b1

    def test_vote_does_not_blend_boxes(self):
        """Vote never interpolates — returned bbox must be one of the inputs."""
        b1 = (0.0, 0.0, 10.0, 10.0)
        b2 = (100.0, 100.0, 10.0, 10.0)
        e = _make_ensemble([b1, b2], fusion="vote")
        e.initialize(_frame(), b1)
        result = e.update(_frame())
        assert result == b1 or result == b2

    def test_vote_outlier_rejected_four_trackers(self):
        """With four agreeing trackers and one outlier, outlier is rejected."""
        base = (20.0, 20.0, 40.0, 40.0)
        cluster = [
            (20.0, 20.0, 40.0, 40.0),
            (22.0, 22.0, 38.0, 38.0),
            (19.0, 21.0, 41.0, 39.0),
            (21.0, 20.0, 40.0, 40.0),
        ]
        outlier = (300.0, 300.0, 20.0, 20.0)
        all_bboxes = cluster + [outlier]
        e = _make_ensemble(all_bboxes, fusion="vote")
        e.initialize(_frame(), base)
        result = e.update(_frame())
        assert result != outlier


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_name(self):
        t = _StubTracker("MOSSE", (0, 0, 10, 10))
        e = EnsembleTracker([t], name="TestEnsemble")
        assert "TestEnsemble" in repr(e)

    def test_repr_contains_fusion(self):
        t = _StubTracker("MOSSE", (0, 0, 10, 10))
        e = EnsembleTracker([t], fusion="vote")
        assert "vote" in repr(e)


# ---------------------------------------------------------------------------
# Integration with real MOSSE and KCF trackers (synthetic frames)
# ---------------------------------------------------------------------------

class TestIntegrationWithRealTrackers:
    """Smoke tests using actual MOSSE + KCF trackers on synthetic data."""

    def _make_synthetic_frame(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        # Draw a bright white rectangle as the "target"
        frame[80:120, 130:170] = 255
        return frame

    def test_mosse_kcf_ensemble_initialize_update(self):
        from eovot.trackers.registry import build_tracker
        mosse = build_tracker("MOSSE")
        kcf = build_tracker("KCF")
        ensemble = EnsembleTracker([mosse, kcf], fusion="mean")

        frame = self._make_synthetic_frame(42)
        bbox = (130.0, 80.0, 40.0, 40.0)
        ensemble.initialize(frame, bbox)

        result = ensemble.update(self._make_synthetic_frame(43))
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)

    def test_vote_ensemble_with_three_real_trackers(self):
        from eovot.trackers.registry import build_tracker
        mosse = build_tracker("MOSSE")
        kcf = build_tracker("KCF")
        try:
            csrt = build_tracker("CSRT")
        except ImportError:
            pytest.skip("opencv-contrib-python not installed; skipping CSRT test")
        ensemble = EnsembleTracker([mosse, kcf, csrt], fusion="vote")

        frame = self._make_synthetic_frame(0)
        bbox = (130.0, 80.0, 40.0, 40.0)
        ensemble.initialize(frame, bbox)

        result = ensemble.update(self._make_synthetic_frame(1))
        assert len(result) == 4

    def test_ensemble_in_benchmark_engine(self):
        """Full benchmark loop with EnsembleTracker: must not raise."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.registry import build_tracker

        mosse = build_tracker("MOSSE")
        kcf = build_tracker("KCF")
        ensemble = EnsembleTracker([mosse, kcf], fusion="mean", name="MOSSE+KCF")

        dataset = SyntheticDataset(
            num_sequences=2, num_frames=20, frame_size=(160, 120), bbox_size=(20, 20), seed=1
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(ensemble, dataset, dataset_name="Synthetic")

        assert result.tracker_name == "MOSSE+KCF"
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0
        assert 0.0 <= result.mean_iou <= 1.0
