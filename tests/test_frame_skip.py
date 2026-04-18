"""Unit tests for FrameSkipWrapper and AET metrics."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.frame_skip import FrameSkipWrapper
from eovot.metrics.efficiency import AETCurve, AETPoint, build_aet_curve


# ---------------------------------------------------------------------------
# Minimal stub tracker for testing
# ---------------------------------------------------------------------------

class _StubTracker(BaseTracker):
    """Simple tracker that returns a fixed or incrementing bbox."""

    def __init__(self, return_box: BBox = (1.0, 1.0, 10.0, 10.0)) -> None:
        super().__init__(name="Stub")
        self._box = return_box
        self.update_calls: int = 0

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self.update_calls = 0

    def update(self, frame: np.ndarray) -> BBox:
        self.update_calls += 1
        return self._box


# ---------------------------------------------------------------------------
# FrameSkipWrapper tests
# ---------------------------------------------------------------------------

class TestFrameSkipWrapper:
    def _make_frame(self) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def test_name_includes_skip_rate(self):
        stub = _StubTracker()
        wrapper = FrameSkipWrapper(stub, skip_rate=3)
        assert "skip3" in wrapper.name
        assert "Stub" in wrapper.name

    def test_invalid_skip_rate_raises(self):
        stub = _StubTracker()
        with pytest.raises(ValueError):
            FrameSkipWrapper(stub, skip_rate=0)
        with pytest.raises(ValueError):
            FrameSkipWrapper(stub, skip_rate=-2)

    def test_skip_rate_one_calls_every_frame(self):
        stub = _StubTracker()
        wrapper = FrameSkipWrapper(stub, skip_rate=1)
        frame = self._make_frame()
        wrapper.initialize(frame, (0.0, 0.0, 5.0, 5.0))
        for _ in range(10):
            wrapper.update(frame)
        assert stub.update_calls == 10

    def test_skip_rate_two_calls_every_other_frame(self):
        stub = _StubTracker()
        wrapper = FrameSkipWrapper(stub, skip_rate=2)
        frame = self._make_frame()
        wrapper.initialize(frame, (0.0, 0.0, 5.0, 5.0))
        for _ in range(10):
            wrapper.update(frame)
        # frames 2,4,6,8,10 → 5 calls
        assert stub.update_calls == 5

    def test_skip_rate_four(self):
        stub = _StubTracker()
        wrapper = FrameSkipWrapper(stub, skip_rate=4)
        frame = self._make_frame()
        wrapper.initialize(frame, (0.0, 0.0, 5.0, 5.0))
        for _ in range(12):
            wrapper.update(frame)
        # frames 4,8,12 → 3 calls
        assert stub.update_calls == 3

    def test_returns_init_box_on_first_skipped_frames(self):
        init_box = (5.0, 5.0, 20.0, 20.0)
        update_box = (10.0, 10.0, 20.0, 20.0)
        stub = _StubTracker(return_box=update_box)
        wrapper = FrameSkipWrapper(stub, skip_rate=3)
        frame = self._make_frame()
        wrapper.initialize(frame, init_box)

        # frame 1 and 2 are skipped; should return init_box
        box1 = wrapper.update(frame)
        assert box1 == init_box
        box2 = wrapper.update(frame)
        assert box2 == init_box
        # frame 3: tracker is called → returns update_box
        box3 = wrapper.update(frame)
        assert box3 == update_box

    def test_reinitialize_resets_frame_count(self):
        stub = _StubTracker()
        wrapper = FrameSkipWrapper(stub, skip_rate=2)
        frame = self._make_frame()
        wrapper.initialize(frame, (0.0, 0.0, 5.0, 5.0))
        for _ in range(4):
            wrapper.update(frame)
        assert stub.update_calls == 2  # frames 2, 4

        # Re-initialise for a new sequence
        wrapper.initialize(frame, (0.0, 0.0, 5.0, 5.0))
        assert stub.update_calls == 0  # reset by initialize
        for _ in range(4):
            wrapper.update(frame)
        assert stub.update_calls == 2

    def test_base_tracker_property(self):
        stub = _StubTracker()
        wrapper = FrameSkipWrapper(stub, skip_rate=2)
        assert wrapper.base_tracker is stub

    def test_last_box_propagated_on_skip(self):
        """Skipped frames must return the most recently predicted box."""
        boxes: List[BBox] = [
            (1.0, 1.0, 10.0, 10.0),
            (2.0, 2.0, 10.0, 10.0),
            (3.0, 3.0, 10.0, 10.0),
        ]
        call_idx = [0]

        class _SeqStub(BaseTracker):
            def __init__(self):
                super().__init__(name="Seq")
                self.update_calls = 0

            def initialize(self, frame, bbox):
                self.update_calls = 0

            def update(self, frame):
                box = boxes[call_idx[0] % len(boxes)]
                call_idx[0] += 1
                self.update_calls += 1
                return box

        stub = _SeqStub()
        wrapper = FrameSkipWrapper(stub, skip_rate=3)
        frame = self._make_frame()
        init_box = (0.0, 0.0, 5.0, 5.0)
        wrapper.initialize(frame, init_box)

        # Frame 1, 2: skipped → init_box
        assert wrapper.update(frame) == init_box
        assert wrapper.update(frame) == init_box
        # Frame 3: tracker called → boxes[0]
        b3 = wrapper.update(frame)
        assert b3 == boxes[0]
        # Frame 4, 5: skipped → still boxes[0]
        assert wrapper.update(frame) == boxes[0]
        assert wrapper.update(frame) == boxes[0]
        # Frame 6: tracker called → boxes[1]
        b6 = wrapper.update(frame)
        assert b6 == boxes[1]


# ---------------------------------------------------------------------------
# AETPoint tests
# ---------------------------------------------------------------------------

class TestAETPoint:
    def test_to_dict_basic(self):
        p = AETPoint(skip_rate=2, mean_iou=0.5, fps=300.0)
        d = p.to_dict()
        assert d["skip_rate"] == 2
        assert d["mean_iou"] == pytest.approx(0.5)
        assert d["fps"] == pytest.approx(300.0)
        assert "memory_mb" not in d

    def test_to_dict_with_memory(self):
        p = AETPoint(skip_rate=1, mean_iou=0.6, fps=150.0, memory_mb=45.3)
        d = p.to_dict()
        assert "memory_mb" in d
        assert d["memory_mb"] == pytest.approx(45.3)


# ---------------------------------------------------------------------------
# AETCurve tests
# ---------------------------------------------------------------------------

class TestAETCurve:
    def _simple_curve(self) -> AETCurve:
        return AETCurve(
            tracker_name="TestTracker",
            points=[
                AETPoint(skip_rate=1, mean_iou=0.60, fps=100.0),
                AETPoint(skip_rate=2, mean_iou=0.55, fps=180.0),
                AETPoint(skip_rate=4, mean_iou=0.45, fps=320.0),
            ],
        )

    def test_points_sorted_by_skip_rate(self):
        curve = AETCurve(
            tracker_name="X",
            points=[
                AETPoint(skip_rate=4, mean_iou=0.4, fps=400.0),
                AETPoint(skip_rate=1, mean_iou=0.6, fps=100.0),
                AETPoint(skip_rate=2, mean_iou=0.5, fps=200.0),
            ],
        )
        skip_rates = [p.skip_rate for p in curve.points]
        assert skip_rates == sorted(skip_rates)

    def test_auc_in_range(self):
        curve = self._simple_curve()
        assert 0.0 <= curve.auc <= 1.0

    def test_auc_single_point_returns_iou(self):
        curve = AETCurve(
            tracker_name="X",
            points=[AETPoint(skip_rate=1, mean_iou=0.7, fps=200.0)],
        )
        assert curve.auc == pytest.approx(0.7)

    def test_auc_empty_returns_zero(self):
        curve = AETCurve(tracker_name="X", points=[])
        assert curve.auc == pytest.approx(0.0)

    def test_auc_flat_curve(self):
        # Constant IoU = 0.5 across all skip rates → AUC = 0.5
        curve = AETCurve(
            tracker_name="Flat",
            points=[
                AETPoint(skip_rate=1, mean_iou=0.5, fps=100.0),
                AETPoint(skip_rate=2, mean_iou=0.5, fps=200.0),
                AETPoint(skip_rate=4, mean_iou=0.5, fps=400.0),
            ],
        )
        assert curve.auc == pytest.approx(0.5, abs=1e-4)

    def test_baseline_returns_skip1_point(self):
        curve = self._simple_curve()
        base = curve.baseline
        assert base is not None
        assert base.skip_rate == 1

    def test_baseline_none_when_missing(self):
        curve = AETCurve(
            tracker_name="X",
            points=[AETPoint(skip_rate=2, mean_iou=0.5, fps=200.0)],
        )
        assert curve.baseline is None

    def test_fps_gain(self):
        curve = self._simple_curve()
        gain = curve.fps_gain(2)
        assert gain == pytest.approx(180.0 / 100.0, rel=1e-4)

    def test_fps_gain_baseline_is_one(self):
        curve = self._simple_curve()
        gain = curve.fps_gain(1)
        assert gain == pytest.approx(1.0)

    def test_fps_gain_missing_skip_rate(self):
        curve = self._simple_curve()
        assert curve.fps_gain(99) is None

    def test_iou_drop(self):
        curve = self._simple_curve()
        drop = curve.iou_drop(4)
        assert drop == pytest.approx(0.60 - 0.45, rel=1e-4)

    def test_iou_drop_baseline_is_zero(self):
        curve = self._simple_curve()
        assert curve.iou_drop(1) == pytest.approx(0.0)

    def test_to_dict_structure(self):
        curve = self._simple_curve()
        d = curve.to_dict()
        assert "tracker_name" in d
        assert "auc" in d
        assert "points" in d
        assert len(d["points"]) == 3

    def test_summary_table_is_string(self):
        curve = self._simple_curve()
        table = curve.summary_table()
        assert isinstance(table, str)
        assert "AET" in table
        assert "TestTracker" in table


# ---------------------------------------------------------------------------
# build_aet_curve helper tests
# ---------------------------------------------------------------------------

class TestBuildAETCurve:
    def test_basic(self):
        curve = build_aet_curve(
            tracker_name="MOSSE",
            skip_rates=[1, 2, 4],
            mean_ious=[0.52, 0.49, 0.42],
            fps_values=[480.0, 860.0, 1500.0],
        )
        assert curve.tracker_name == "MOSSE"
        assert len(curve.points) == 3

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_aet_curve(
                tracker_name="X",
                skip_rates=[1, 2],
                mean_ious=[0.5],
                fps_values=[100.0, 200.0],
            )

    def test_with_memory(self):
        curve = build_aet_curve(
            tracker_name="KCF",
            skip_rates=[1, 2],
            mean_ious=[0.55, 0.50],
            fps_values=[200.0, 380.0],
            memory_mb=[30.0, 30.0],
        )
        for p in curve.points:
            assert p.memory_mb is not None
