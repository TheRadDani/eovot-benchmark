"""Tests for EnsembleTracker and registry Ensemble support."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.trackers.ensemble import EnsembleTracker, _iou
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Minimal stub tracker for testing without OpenCV
# ---------------------------------------------------------------------------

class _StubTracker(BaseTracker):
    """Returns a fixed bounding box on every update."""

    def __init__(self, bbox, name="Stub"):
        super().__init__(name=name)
        self._bbox = tuple(bbox)

    def initialize(self, frame, bbox):
        pass

    def update(self, frame):
        return self._bbox


# ---------------------------------------------------------------------------
# Unit tests for _iou helper
# ---------------------------------------------------------------------------

class TestIouHelper:
    def test_perfect_overlap(self):
        box = (10.0, 10.0, 50.0, 50.0)
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (20.0, 20.0, 10.0, 10.0)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = (0.0, 0.0, 20.0, 20.0)
        b = (10.0, 0.0, 20.0, 20.0)
        # intersection: 10×20 = 200, union: 400+400-200 = 600
        assert _iou(a, b) == pytest.approx(200.0 / 600.0)

    def test_zero_area_box(self):
        a = (0.0, 0.0, 0.0, 10.0)
        b = (0.0, 0.0, 10.0, 10.0)
        assert _iou(a, b) == 0.0


# ---------------------------------------------------------------------------
# EnsembleTracker unit tests
# ---------------------------------------------------------------------------

class TestEnsembleTracker:

    def _dummy_frame(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_single_member_returns_its_prediction(self):
        box = (10.0, 10.0, 40.0, 40.0)
        stub = _StubTracker(box, name="S1")
        tracker = EnsembleTracker(members=[stub])
        tracker.initialize(self._dummy_frame(), box)
        pred = tracker.update(self._dummy_frame())
        assert pred == box

    def test_consensus_selects_majority_box(self):
        """Two members agree on one box; the third gives a different box.
        The consensus should pick the agreed-upon box."""
        good_box = (10.0, 10.0, 40.0, 40.0)
        bad_box  = (200.0, 200.0, 10.0, 10.0)  # far away, no overlap with others

        members = [
            _StubTracker(good_box, "A"),
            _StubTracker(good_box, "B"),
            _StubTracker(bad_box,  "C"),
        ]
        tracker = EnsembleTracker(members=members)
        tracker.initialize(self._dummy_frame(), good_box)
        pred = tracker.update(self._dummy_frame())
        assert pred == good_box

    def test_three_identical_members(self):
        box = (5.0, 5.0, 30.0, 30.0)
        members = [_StubTracker(box, f"M{i}") for i in range(3)]
        tracker = EnsembleTracker(members=members)
        tracker.initialize(self._dummy_frame(), box)
        assert tracker.update(self._dummy_frame()) == box

    def test_empty_members_raises(self):
        with pytest.raises(ValueError, match="at least one member"):
            EnsembleTracker(members=[])

    def test_default_name_includes_member_names(self):
        members = [
            _StubTracker((0, 0, 10, 10), "MOSSE"),
            _StubTracker((0, 0, 10, 10), "KCF"),
        ]
        tracker = EnsembleTracker(members=members)
        assert "MOSSE" in tracker.name
        assert "KCF" in tracker.name

    def test_custom_name(self):
        stub = _StubTracker((0, 0, 10, 10), "S")
        tracker = EnsembleTracker(members=[stub], name="MyEnsemble")
        assert tracker.name == "MyEnsemble"

    def test_initialize_calls_all_members(self):
        frame = self._dummy_frame()
        bbox = (5.0, 5.0, 20.0, 20.0)
        mocks = [MagicMock(spec=BaseTracker) for _ in range(3)]
        for m in mocks:
            m.name = "Mock"
            m.update.return_value = bbox
        tracker = EnsembleTracker(members=mocks)
        tracker.initialize(frame, bbox)
        for m in mocks:
            m.initialize.assert_called_once_with(frame, bbox)

    def test_update_calls_all_members(self):
        frame = self._dummy_frame()
        bbox = (5.0, 5.0, 20.0, 20.0)
        mocks = [MagicMock(spec=BaseTracker) for _ in range(3)]
        for m in mocks:
            m.name = "Mock"
            m.update.return_value = bbox
        tracker = EnsembleTracker(members=mocks)
        tracker.initialize(frame, bbox)
        tracker.update(frame)
        for m in mocks:
            m.update.assert_called_once_with(frame)

    def test_consensus_scores_shape(self):
        predictions = [
            (0.0, 0.0, 20.0, 20.0),
            (5.0, 5.0, 20.0, 20.0),
            (200.0, 200.0, 10.0, 10.0),
        ]
        scores = EnsembleTracker._consensus_scores(predictions)
        assert scores.shape == (3,)
        assert scores[0] >= 0.0
        assert scores[2] < scores[0]  # the far-away box scores lowest

    def test_consensus_scores_two_identical_and_one_outlier(self):
        a = (0.0, 0.0, 20.0, 20.0)
        b = (0.0, 0.0, 20.0, 20.0)
        c = (100.0, 100.0, 5.0, 5.0)
        scores = EnsembleTracker._consensus_scores([a, b, c])
        assert scores[0] > scores[2]
        assert scores[1] > scores[2]


# ---------------------------------------------------------------------------
# Registry integration tests
# ---------------------------------------------------------------------------

class TestRegistryEnsemble:

    def test_build_tracker_ensemble_is_available(self):
        from eovot.trackers.registry import available_trackers
        assert "Ensemble" in available_trackers()

    def test_build_tracker_ensemble_default(self):
        from eovot.trackers.registry import build_tracker
        # Default members are MOSSE + KCF + LKOpticalFlow (contrib-free)
        tracker = build_tracker("Ensemble")
        assert isinstance(tracker, EnsembleTracker)
        assert len(tracker.members) == 3

    def test_build_tracker_ensemble_custom_members(self):
        from eovot.trackers.registry import build_tracker
        tracker = build_tracker("Ensemble", member_names=["MOSSE", "KCF"])
        assert isinstance(tracker, EnsembleTracker)
        assert len(tracker.members) == 2

    def test_build_tracker_ensemble_unknown_member_raises(self):
        from eovot.trackers.registry import build_tracker
        with pytest.raises(ValueError, match="Unknown tracker"):
            build_tracker("Ensemble", member_names=["NonExistentTracker"])

    def test_from_names_factory(self):
        tracker = EnsembleTracker.from_names(["MOSSE", "KCF"])
        assert isinstance(tracker, EnsembleTracker)
        assert len(tracker.members) == 2
