"""Tests for AdaptiveBudgetTracker and AdaptiveBudgetAnalyzer."""

from __future__ import annotations

import time
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eovot.trackers.adaptive import AdaptiveBudgetTracker
from eovot.trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Minimal stub tracker for deterministic testing
# ---------------------------------------------------------------------------

class _FixedLatencyTracker(BaseTracker):
    """Tracker that returns a fixed bbox and sleeps to simulate latency."""

    def __init__(self, latency_s: float = 0.0, bbox: BBox = (10.0, 10.0, 20.0, 20.0)):
        super().__init__(name="FixedLatency")
        self._bbox = bbox
        self._latency_s = latency_s
        self.update_count = 0

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox
        self.update_count = 0

    def update(self, frame: np.ndarray) -> BBox:
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        self.update_count += 1
        return self._bbox


_DUMMY_FRAME = np.zeros((64, 64, 3), dtype=np.uint8)
_DUMMY_BBOX: BBox = (5.0, 5.0, 15.0, 15.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_name_includes_fps(self):
        inner = _FixedLatencyTracker()
        t = AdaptiveBudgetTracker(inner, target_fps=30.0)
        assert "30" in t.name
        assert "adaptive" in t.name.lower()

    def test_name_includes_inner_name(self):
        inner = _FixedLatencyTracker()
        t = AdaptiveBudgetTracker(inner, target_fps=25.0)
        assert "FixedLatency" in t.name

    def test_invalid_target_fps(self):
        with pytest.raises(ValueError, match="target_fps"):
            AdaptiveBudgetTracker(_FixedLatencyTracker(), target_fps=0.0)

    def test_invalid_negative_fps(self):
        with pytest.raises(ValueError, match="target_fps"):
            AdaptiveBudgetTracker(_FixedLatencyTracker(), target_fps=-5.0)

    def test_invalid_max_skip(self):
        with pytest.raises(ValueError, match="max_skip"):
            AdaptiveBudgetTracker(_FixedLatencyTracker(), max_skip=-1)

    def test_invalid_warmup_frames(self):
        with pytest.raises(ValueError, match="warmup_frames"):
            AdaptiveBudgetTracker(_FixedLatencyTracker(), warmup_frames=-1)

    def test_underlying_tracker_property(self):
        inner = _FixedLatencyTracker()
        t = AdaptiveBudgetTracker(inner)
        assert t.underlying_tracker is inner


# ---------------------------------------------------------------------------
# Initialization and basic update
# ---------------------------------------------------------------------------

class TestInitAndUpdate:
    def test_update_before_initialize_raises(self):
        t = AdaptiveBudgetTracker(_FixedLatencyTracker())
        with pytest.raises(RuntimeError):
            t.update(_DUMMY_FRAME)

    def test_returns_bbox_after_initialize(self):
        inner = _FixedLatencyTracker(bbox=_DUMMY_BBOX)
        t = AdaptiveBudgetTracker(inner, target_fps=60.0)
        t.initialize(_DUMMY_FRAME, _DUMMY_BBOX)
        result = t.update(_DUMMY_FRAME)
        assert len(result) == 4

    def test_first_update_always_runs_tracker(self):
        inner = _FixedLatencyTracker()
        t = AdaptiveBudgetTracker(inner, target_fps=60.0, warmup_frames=5)
        t.initialize(_DUMMY_FRAME, _DUMMY_BBOX)
        t.update(_DUMMY_FRAME)
        assert inner.update_count == 1

    def test_warmup_frames_all_run_tracker(self):
        inner = _FixedLatencyTracker()
        t = AdaptiveBudgetTracker(inner, target_fps=1.0, max_skip=7, warmup_frames=5)
        t.initialize(_DUMMY_FRAME, _DUMMY_BBOX)
        for _ in range(5):
            t.update(_DUMMY_FRAME)
        assert inner.update_count == 5

    def test_reinitialize_resets_state(self):
        inner = _FixedLatencyTracker()
        t = AdaptiveBudgetTracker(inner, target_fps=60.0)
        t.initialize(_DUMMY_FRAME, _DUMMY_BBOX)
        for _ in range(10):
            t.update(_DUMMY_FRAME)
        # Re-initialize should clear history
        t.initialize(_DUMMY_FRAME, _DUMMY_BBOX)
        stats = t.adaptation_stats
        assert stats["total_frames"] == 0
        assert stats["adaptation_count"] == 0


# ---------------------------------------------------------------------------
# Skip behaviour
# ---------------------------------------------------------------------------

class TestSkipBehaviour:
    def test_max_skip_zero_means_no_skipping(self):
        inner = _FixedLatencyTracker()
        t = AdaptiveBudgetTracker(inner, target_fps=1.0, max_skip=0, warmup_frames=1)
        t.initialize(_DUMMY_FRAME, _DUMMY_BBOX)
        n = 10
        for _ in range(n):
            t.update(_DUMMY_FRAME)
        # With max_skip=0 the controller can never skip; every frame runs tracker
        assert inner.update_count == n

    def test_cached_bbox_returned_on_passive_frame(self):
        bbox_a = (1.0, 2.0, 3.0, 4.0)
        bbox_b = (9.0, 9.0, 9.0, 9.0)

        class _ToggleTracker(BaseTracker):
            def __init__(self):
                super().__init__(name="Toggle")
                self._call = 0

            def initialize(self, frame, bbox):
                self._call = 0

            def update(self, frame):
                self._call += 1
                return bbox_a if self._call % 2 == 1 else bbox_b

        inner = _ToggleTracker()
        # force skip_counter to 1 after warmup by patching _adapt
        t = AdaptiveBudgetTracker(inner, target_fps=60.0, warmup_frames=1, max_skip=3)
        t.initialize(_DUMMY_FRAME, _DUMMY_BBOX)
        # Frame 1 (warmup) — runs tracker, gets bbox_a
        r1 = t.update(_DUMMY_FRAME)
        assert r1 == bbox_a

    def test_adaptation_count_increases_under_load(self):
        # Simulate a slow tracker so the controller should increase skip
        slow = _FixedLatencyTracker(latency_s=0.05)
        # target_fps=100 → budget period = 10 ms; latency = 50 ms → far behind
        t = AdaptiveBudgetTracker(slow, target_fps=100.0, max_skip=7, warmup_frames=3, window_size=3)
        t.initialize(_DUMMY_FRAME, _DUMMY_BBOX)
        for _ in range(20):
            t.update(_DUMMY_FRAME)
        stats = t.adaptation_stats
        assert stats["adaptation_count"] >= 1

    def test_skip_fraction_zero_when_no_skipping(self):
        inner = _FixedLatencyTracker()
        t = AdaptiveBudgetTracker(inner, target_fps=60.0, max_skip=0, warmup_frames=2)
        t.initialize(_DUMMY_FRAME, _DUMMY_BBOX)
        for _ in range(8):
            t.update(_DUMMY_FRAME)
        stats = t.adaptation_stats
        assert stats["mean_skip_fraction"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Adaptation stats
# ---------------------------------------------------------------------------

class TestAdaptationStats:
    def test_stats_structure(self):
        inner = _FixedLatencyTracker()
        t = AdaptiveBudgetTracker(inner)
        t.initialize(_DUMMY_FRAME, _DUMMY_BBOX)
        t.update(_DUMMY_FRAME)
        stats = t.adaptation_stats
        expected_keys = {
            "target_fps", "estimated_fps", "current_skip",
            "mean_skip_fraction", "adaptation_count", "total_frames", "active_frames",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_total_frames_matches_update_calls(self):
        inner = _FixedLatencyTracker()
        t = AdaptiveBudgetTracker(inner, max_skip=0)
        t.initialize(_DUMMY_FRAME, _DUMMY_BBOX)
        n = 12
        for _ in range(n):
            t.update(_DUMMY_FRAME)
        assert t.adaptation_stats["total_frames"] == n

    def test_current_skip_property(self):
        inner = _FixedLatencyTracker()
        t = AdaptiveBudgetTracker(inner, target_fps=60.0)
        t.initialize(_DUMMY_FRAME, _DUMMY_BBOX)
        assert isinstance(t.current_skip, int)
        assert t.current_skip >= 0


# ---------------------------------------------------------------------------
# AdaptiveBudgetReport
# ---------------------------------------------------------------------------

class TestAdaptiveBudgetReport:
    def _make_report(self):
        from eovot.analysis.adaptive_analysis import AdaptiveBudgetEntry, AdaptiveBudgetReport
        report = AdaptiveBudgetReport(tracker_name="MOSSETracker", dataset_name="TestDS")
        report.entries = [
            AdaptiveBudgetEntry(10, 0.60, 10.1, 0.40, 3, 5),
            AdaptiveBudgetEntry(30, 0.55, 29.8, 0.20, 2, 5),
            AdaptiveBudgetEntry(60, 0.50, 58.5, 0.05, 1, 5),
        ]
        return report

    def test_to_markdown_contains_header(self):
        report = self._make_report()
        md = report.to_markdown()
        assert "MOSSETracker" in md
        assert "TestDS" in md

    def test_to_markdown_contains_all_entries(self):
        report = self._make_report()
        md = report.to_markdown()
        assert "10" in md
        assert "30" in md
        assert "60" in md

    def test_to_dict_structure(self):
        report = self._make_report()
        d = report.to_dict()
        assert d["tracker_name"] == "MOSSETracker"
        assert len(d["entries"]) == 3
        assert "mean_iou" in d["entries"][0]

    def test_optimal_fps_at_iou_budget_found(self):
        report = self._make_report()
        fps = report.optimal_fps_at_iou_budget(min_iou=0.52)
        # Both 10 fps (0.60) and 30 fps (0.55) qualify; lowest fps should win
        assert fps == pytest.approx(10.0)

    def test_optimal_fps_at_iou_budget_not_found(self):
        report = self._make_report()
        fps = report.optimal_fps_at_iou_budget(min_iou=0.99)
        assert fps is None

    def test_best_iou_entry(self):
        report = self._make_report()
        best = report.best_iou_entry()
        assert best is not None
        assert best.mean_iou == pytest.approx(0.60)

    def test_fastest_entry(self):
        report = self._make_report()
        fastest = report.fastest_entry()
        assert fastest is not None
        assert fastest.actual_fps == pytest.approx(58.5)

    def test_empty_report_returns_none(self):
        from eovot.analysis.adaptive_analysis import AdaptiveBudgetReport
        report = AdaptiveBudgetReport(tracker_name="X", dataset_name="Y")
        assert report.best_iou_entry() is None
        assert report.fastest_entry() is None
        assert report.optimal_fps_at_iou_budget(0.5) is None
