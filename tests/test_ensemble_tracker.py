"""Tests for ConsensusTracker and fusion helpers (ensemble.py)."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.ensemble import (
    ConsensusTracker,
    _bbox_iou,
    _iou_weighted_fusion,
    _mean_fusion,
    _pairwise_mean_iou,
    _smallest_box_fusion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BBOX: BBox = (10.0, 10.0, 40.0, 40.0)
_FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


class _FixedTracker(BaseTracker):
    """Tracker that always returns a predetermined bounding box."""

    def __init__(self, bbox: BBox, name: str = "Fixed"):
        super().__init__(name)
        self._bbox = bbox
        self.update_calls = 0

    def initialize(self, frame, bbox):
        self.update_calls = 0

    def update(self, frame) -> BBox:
        self.update_calls += 1
        return self._bbox


def _make_fixed(bbox: BBox, name: str = "Fixed") -> _FixedTracker:
    return _FixedTracker(bbox, name)


# ---------------------------------------------------------------------------
# _bbox_iou
# ---------------------------------------------------------------------------

class TestBboxIou:
    def test_identical_boxes_iou_one(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert _bbox_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap_iou_zero(self):
        a = (0.0, 0.0, 5.0, 5.0)
        b = (10.0, 10.0, 5.0, 5.0)
        assert _bbox_iou(a, b) == pytest.approx(0.0)

    def test_half_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 0.0, 10.0, 10.0)
        # intersection: 5×10=50, union: 100+100-50=150, IoU=1/3
        assert _bbox_iou(a, b) == pytest.approx(1 / 3, rel=1e-6)

    def test_zero_area_box_returns_zero(self):
        a = (0.0, 0.0, 0.0, 10.0)   # zero width
        b = (0.0, 0.0, 10.0, 10.0)
        assert _bbox_iou(a, b) == 0.0

    def test_touching_at_edge_no_overlap(self):
        a = (0.0, 0.0, 5.0, 5.0)
        b = (5.0, 0.0, 5.0, 5.0)
        assert _bbox_iou(a, b) == pytest.approx(0.0)

    def test_iou_symmetric(self):
        a = (1.0, 2.0, 8.0, 6.0)
        b = (3.0, 0.0, 10.0, 7.0)
        assert _bbox_iou(a, b) == pytest.approx(_bbox_iou(b, a))


# ---------------------------------------------------------------------------
# Fusion functions
# ---------------------------------------------------------------------------

class TestMeanFusion:
    def test_single_box_returns_itself(self):
        box = (5.0, 6.0, 20.0, 15.0)
        result = _mean_fusion([box])
        assert result == pytest.approx(box)

    def test_two_identical_boxes(self):
        box = (10.0, 10.0, 30.0, 30.0)
        result = _mean_fusion([box, box])
        assert result == pytest.approx(box)

    def test_average_of_two_boxes(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (10.0, 10.0, 20.0, 20.0)
        result = _mean_fusion([a, b])
        assert result == pytest.approx((5.0, 5.0, 15.0, 15.0))

    def test_three_boxes_correct_mean(self):
        boxes = [(0.0, 0.0, 6.0, 6.0), (6.0, 6.0, 6.0, 6.0), (12.0, 12.0, 6.0, 6.0)]
        result = _mean_fusion(boxes)
        assert result == pytest.approx((6.0, 6.0, 6.0, 6.0))


class TestIouWeightedFusion:
    def test_single_box_returns_itself(self):
        box = (5.0, 5.0, 20.0, 20.0)
        assert _iou_weighted_fusion([box]) == pytest.approx(box)

    def test_identical_boxes_returns_same(self):
        box = (5.0, 5.0, 20.0, 20.0)
        result = _iou_weighted_fusion([box, box, box])
        assert result == pytest.approx(box)

    def test_outlier_gets_low_weight(self):
        """Two close boxes + one distant outlier; result should be near the close pair."""
        close_a = (10.0, 10.0, 30.0, 30.0)
        close_b = (11.0, 11.0, 30.0, 30.0)
        outlier = (100.0, 100.0, 30.0, 30.0)
        result = _iou_weighted_fusion([close_a, close_b, outlier])
        rx, ry = result[0], result[1]
        # Should be much closer to (10-11, 10-11) than to (100, 100)
        assert rx < 50.0, "Fused x should be near the consensus, not the outlier"
        assert ry < 50.0, "Fused y should be near the consensus, not the outlier"

    def test_zero_iou_falls_back_to_mean(self):
        """When all pairwise IoUs are 0, fall back to mean fusion."""
        a = (0.0, 0.0, 5.0, 5.0)
        b = (100.0, 100.0, 5.0, 5.0)
        result = _iou_weighted_fusion([a, b])
        mean_result = _mean_fusion([a, b])
        assert result == pytest.approx(mean_result)


class TestSmallestBoxFusion:
    def test_single_box_returns_itself(self):
        box = (5.0, 5.0, 10.0, 10.0)
        assert _smallest_box_fusion([box]) == box

    def test_returns_smallest_area_box(self):
        a = (0.0, 0.0, 20.0, 20.0)   # area 400
        b = (0.0, 0.0, 10.0, 10.0)   # area 100 ← smallest
        c = (0.0, 0.0, 30.0, 30.0)   # area 900
        assert _smallest_box_fusion([a, b, c]) == b


class TestPairwiseMeanIou:
    def test_single_element_returns_one(self):
        assert _pairwise_mean_iou([_BBOX]) == 1.0

    def test_identical_boxes_returns_one(self):
        assert _pairwise_mean_iou([_BBOX, _BBOX]) == pytest.approx(1.0)

    def test_non_overlapping_returns_zero(self):
        a = (0.0, 0.0, 5.0, 5.0)
        b = (100.0, 100.0, 5.0, 5.0)
        assert _pairwise_mean_iou([a, b]) == pytest.approx(0.0)

    def test_symmetric_result(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 10.0, 10.0)
        assert _pairwise_mean_iou([a, b]) == pytest.approx(_pairwise_mean_iou([b, a]))

    def test_three_boxes_in_0_1(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 10.0, 10.0)
        c = (8.0, 8.0, 10.0, 10.0)
        score = _pairwise_mean_iou([a, b, c])
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# ConsensusTracker — construction
# ---------------------------------------------------------------------------

class TestConsensusTrackerConstruction:
    def test_empty_trackers_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ConsensusTracker([])

    def test_unknown_fusion_raises(self):
        with pytest.raises(ValueError, match="fusion"):
            ConsensusTracker([_make_fixed(_BBOX)], fusion="nonexistent")  # type: ignore[arg-type]

    def test_name_encodes_members_and_fusion(self):
        t = ConsensusTracker(
            [_make_fixed(_BBOX, "A"), _make_fixed(_BBOX, "B")],
            fusion="mean",
        )
        assert "A" in t.name and "B" in t.name
        assert "mean" in t.name

    def test_single_member_valid(self):
        t = ConsensusTracker([_make_fixed(_BBOX)], fusion="mean")
        assert len(t.members) == 1

    def test_members_property_returns_copy(self):
        inner = _make_fixed(_BBOX)
        t = ConsensusTracker([inner])
        members = t.members
        members.clear()
        assert len(t.members) == 1  # internal list unchanged

    def test_fusion_mode_property(self):
        t = ConsensusTracker([_make_fixed(_BBOX)], fusion="smallest_box")
        assert t.fusion_mode == "smallest_box"

    def test_all_fusion_modes_accepted(self):
        for mode in ("mean", "iou_weighted", "smallest_box"):
            t = ConsensusTracker([_make_fixed(_BBOX)], fusion=mode)  # type: ignore[arg-type]
            assert t.fusion_mode == mode


# ---------------------------------------------------------------------------
# ConsensusTracker — initialize / update
# ---------------------------------------------------------------------------

class TestConsensusTrackerBehavior:
    def test_initialize_calls_all_members(self):
        members = [_make_fixed(_BBOX, f"T{i}") for i in range(3)]
        ens = ConsensusTracker(members)
        ens.initialize(_FRAME, _BBOX)
        # update_calls reset on initialize — call once to verify driving
        for _ in range(2):
            ens.update(_FRAME)
        assert all(m.update_calls == 2 for m in members)

    def test_update_returns_4tuple(self):
        ens = ConsensusTracker([_make_fixed(_BBOX)], fusion="mean")
        ens.initialize(_FRAME, _BBOX)
        bbox = ens.update(_FRAME)
        assert len(bbox) == 4

    def test_single_member_mean_returns_member_prediction(self):
        pred = (5.0, 7.0, 20.0, 18.0)
        ens = ConsensusTracker([_make_fixed(pred)], fusion="mean")
        ens.initialize(_FRAME, _BBOX)
        assert ens.update(_FRAME) == pytest.approx(pred)

    def test_identical_members_all_modes_agree(self):
        """All fusion modes should return the same result when members agree."""
        pred = (12.0, 8.0, 35.0, 25.0)
        for mode in ("mean", "iou_weighted", "smallest_box"):
            ens = ConsensusTracker(
                [_make_fixed(pred), _make_fixed(pred)],
                fusion=mode,  # type: ignore[arg-type]
            )
            ens.initialize(_FRAME, _BBOX)
            result = ens.update(_FRAME)
            assert result == pytest.approx(pred, abs=1e-9), f"mode={mode}"

    def test_iou_weighted_outlier_suppressed(self):
        """IoU-weighted fusion should suppress the outlier prediction."""
        consensus = (10.0, 10.0, 30.0, 30.0)
        outlier = (200.0, 200.0, 30.0, 30.0)
        ens = ConsensusTracker(
            [_make_fixed(consensus), _make_fixed(consensus), _make_fixed(outlier)],
            fusion="iou_weighted",
        )
        ens.initialize(_FRAME, _BBOX)
        result = ens.update(_FRAME)
        # Fused x/y should be much closer to consensus than to outlier
        assert result[0] < 50.0

    def test_last_predictions_populated_after_update(self):
        ens = ConsensusTracker(
            [_make_fixed(_BBOX, "A"), _make_fixed(_BBOX, "B")],
            fusion="mean",
        )
        ens.initialize(_FRAME, _BBOX)
        ens.update(_FRAME)
        assert len(ens.last_predictions) == 2

    def test_last_predictions_empty_before_update(self):
        ens = ConsensusTracker([_make_fixed(_BBOX)], fusion="mean")
        ens.initialize(_FRAME, _BBOX)
        assert ens.last_predictions == []

    def test_last_predictions_is_copy(self):
        ens = ConsensusTracker([_make_fixed(_BBOX)], fusion="mean")
        ens.initialize(_FRAME, _BBOX)
        ens.update(_FRAME)
        preds = ens.last_predictions
        preds.clear()
        assert len(ens.last_predictions) == 1  # internal list unchanged

    def test_reinitialize_clears_last_predictions(self):
        ens = ConsensusTracker([_make_fixed(_BBOX)], fusion="mean")
        ens.initialize(_FRAME, _BBOX)
        ens.update(_FRAME)
        ens.initialize(_FRAME, _BBOX)
        assert ens.last_predictions == []


# ---------------------------------------------------------------------------
# Agreement score
# ---------------------------------------------------------------------------

class TestAgreementScore:
    def test_agreement_one_before_update(self):
        ens = ConsensusTracker([_make_fixed(_BBOX)], fusion="mean")
        ens.initialize(_FRAME, _BBOX)
        assert ens.agreement == pytest.approx(1.0)

    def test_agreement_one_when_all_identical(self):
        pred = (10.0, 10.0, 30.0, 30.0)
        ens = ConsensusTracker(
            [_make_fixed(pred), _make_fixed(pred)], fusion="mean"
        )
        ens.initialize(_FRAME, _BBOX)
        ens.update(_FRAME)
        assert ens.agreement == pytest.approx(1.0)

    def test_agreement_low_when_members_disagree(self):
        a = (0.0, 0.0, 5.0, 5.0)
        b = (200.0, 200.0, 5.0, 5.0)  # no overlap with a
        ens = ConsensusTracker([_make_fixed(a), _make_fixed(b)], fusion="mean")
        ens.initialize(_FRAME, _BBOX)
        ens.update(_FRAME)
        assert ens.agreement == pytest.approx(0.0)

    def test_agreement_bounded_in_0_1(self):
        a = (10.0, 10.0, 40.0, 40.0)
        b = (20.0, 20.0, 40.0, 40.0)
        ens = ConsensusTracker([_make_fixed(a), _make_fixed(b)], fusion="mean")
        ens.initialize(_FRAME, _BBOX)
        ens.update(_FRAME)
        assert 0.0 <= ens.agreement <= 1.0

    def test_agreement_updates_each_frame(self):
        """Agreement should reflect the CURRENT frame's predictions."""
        a = (10.0, 10.0, 40.0, 40.0)
        b = (10.0, 10.0, 40.0, 40.0)
        ens = ConsensusTracker([_make_fixed(a), _make_fixed(b)], fusion="mean")
        ens.initialize(_FRAME, _BBOX)
        ens.update(_FRAME)
        first_agreement = ens.agreement
        ens.update(_FRAME)
        # Same fixed trackers → same agreement each frame
        assert ens.agreement == pytest.approx(first_agreement)


# ---------------------------------------------------------------------------
# Integration: ConsensusTracker with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestConsensusWithEngine:
    def test_runs_through_benchmark_engine(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.base import BaseDataset, Sequence

        class _SimpleSeq(Sequence):
            def __init__(self):
                gt = np.tile(_BBOX, (10, 1))
                super().__init__("s0", ["x"] * 10, gt)

            def __len__(self):
                return 10

            def __iter__(self) -> Iterator[np.ndarray]:
                for _ in range(10):
                    yield np.zeros((120, 160, 3), dtype=np.uint8)

        class _DS(BaseDataset):
            def __len__(self): return 1
            def __getitem__(self, i): return _SimpleSeq()

        engine = BenchmarkEngine(verbose=False)
        ens = ConsensusTracker(
            [_make_fixed(_BBOX, "A"), _make_fixed(_BBOX, "B")],
            fusion="mean",
        )
        result = engine.run(ens, _DS(), dataset_name="Tiny")
        assert result.mean_iou == pytest.approx(1.0, abs=0.01)

    def test_tracker_name_appears_in_result(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.base import BaseDataset, Sequence

        class _SimpleSeq(Sequence):
            def __init__(self):
                gt = np.tile(_BBOX, (5, 1))
                super().__init__("s0", ["x"] * 5, gt)

            def __len__(self): return 5

            def __iter__(self) -> Iterator[np.ndarray]:
                for _ in range(5):
                    yield np.zeros((120, 160, 3), dtype=np.uint8)

        class _DS(BaseDataset):
            def __len__(self): return 1
            def __getitem__(self, i): return _SimpleSeq()

        ens = ConsensusTracker([_make_fixed(_BBOX, "MOSSE"), _make_fixed(_BBOX, "KCF")])
        result = BenchmarkEngine(verbose=False).run(ens, _DS(), dataset_name="T")
        assert "Consensus" in result.tracker_name
