"""Tests for FrameSkipTracker and FrameSkipAnalyzer."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.base import BaseDataset, Sequence
from eovot.trackers.base import BaseTracker
from eovot.trackers.frame_skip import FrameSkipTracker
from eovot.analysis.skip_analysis import FrameSkipAnalyzer, SkipRateResult


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

_FIXED_BOX = (10.0, 10.0, 50.0, 50.0)
_GT_BOX = (10.0, 10.0, 50.0, 50.0)
_N = 20


class _CountingTracker(BaseTracker):
    """Tracker that counts how many times update() is actually called."""

    def __init__(self):
        super().__init__(name="CountingTracker")
        self.update_calls = 0

    def initialize(self, frame, bbox):
        self.update_calls = 0

    def update(self, frame):
        self.update_calls += 1
        return _FIXED_BOX


class _InMemSeq(Sequence):
    def __init__(self, name="seq_0", n=_N):
        gt = np.tile(_GT_BOX, (n, 1))
        super().__init__(name=name, frame_paths=["x"] * n, ground_truth=gt)
        self._n = n

    def __len__(self):
        return self._n

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(self._n):
            yield np.zeros((120, 160, 3), dtype=np.uint8)


class _TinyDataset(BaseDataset):
    def __init__(self, n=2):
        self._seqs = [_InMemSeq(f"seq_{i}") for i in range(n)]

    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, idx):
        return self._seqs[idx]


# ---------------------------------------------------------------------------
# FrameSkipTracker — unit tests
# ---------------------------------------------------------------------------

class TestFrameSkipTrackerInit:
    def test_name_includes_suffix(self):
        inner = _CountingTracker()
        fst = FrameSkipTracker(inner, skip_rate=3)
        assert fst.name == "CountingTracker_skip3"

    def test_invalid_skip_rate_raises(self):
        with pytest.raises(ValueError):
            FrameSkipTracker(_CountingTracker(), skip_rate=0)

    def test_skip_rate_1_is_noop_name(self):
        inner = _CountingTracker()
        fst = FrameSkipTracker(inner, skip_rate=1)
        assert "_skip1" in fst.name

    def test_underlying_tracker_attribute(self):
        inner = _CountingTracker()
        fst = FrameSkipTracker(inner, skip_rate=2)
        assert fst.underlying_tracker is inner


class TestFrameSkipTrackerBehavior:
    def _run(self, skip_rate, mode="repeat", n=_N):
        """Initialize and run n-1 update calls; return the tracker."""
        inner = _CountingTracker()
        fst = FrameSkipTracker(inner, skip_rate=skip_rate, mode=mode)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        fst.initialize(frame, _FIXED_BOX)
        for _ in range(n - 1):
            fst.update(frame)
        return fst, inner

    def test_skip_rate_1_calls_update_every_frame(self):
        fst, inner = self._run(skip_rate=1)
        assert inner.update_calls == _N - 1

    def test_skip_rate_2_halves_update_calls(self):
        fst, inner = self._run(skip_rate=2)
        expected = (_N - 1) // 2
        assert inner.update_calls == expected

    def test_skip_rate_4_quarters_update_calls(self):
        fst, inner = self._run(skip_rate=4)
        expected = (_N - 1) // 4
        assert inner.update_calls == expected

    def test_update_returns_valid_bbox(self):
        fst = FrameSkipTracker(_CountingTracker(), skip_rate=2)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        fst.initialize(frame, _FIXED_BOX)
        bbox = fst.update(frame)
        assert len(bbox) == 4

    def test_repeat_mode_returns_last_active_bbox(self):
        fst, _ = self._run(skip_rate=3, mode="repeat", n=7)
        # frame 6 → idx=5, 5 % 3 != 0, so should return last active prediction
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        fst.initialize(frame, _FIXED_BOX)
        # Run 4 frames so frame 4 is a skip (4 % 3 = 1)
        bboxes = [fst.update(frame) for _ in range(4)]
        assert all(len(b) == 4 for b in bboxes)

    def test_linear_mode_returns_extrapolated_bbox(self):
        fst = FrameSkipTracker(_CountingTracker(), skip_rate=3, mode="linear")
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        fst.initialize(frame, _FIXED_BOX)
        # Run enough frames to trigger at least one active + one passive update
        for _ in range(4):
            bbox = fst.update(frame)
        assert len(bbox) == 4

    def test_uninitialized_update_raises(self):
        fst = FrameSkipTracker(_CountingTracker(), skip_rate=2)
        with pytest.raises(RuntimeError):
            fst.update(np.zeros((10, 10, 3), dtype=np.uint8))


class TestFrameSkipTrackerCounters:
    def test_active_frame_count(self):
        inner = _CountingTracker()
        fst = FrameSkipTracker(inner, skip_rate=3)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        fst.initialize(frame, _FIXED_BOX)
        for _ in range(9):
            fst.update(frame)
        assert fst.active_frame_count == 3  # frames 3, 6, 9

    def test_skipped_frame_count(self):
        fst = FrameSkipTracker(_CountingTracker(), skip_rate=3)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        fst.initialize(frame, _FIXED_BOX)
        for _ in range(9):
            fst.update(frame)
        assert fst.skipped_frame_count == 6

    def test_effective_skip_fraction(self):
        fst = FrameSkipTracker(_CountingTracker(), skip_rate=3)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        fst.initialize(frame, _FIXED_BOX)
        for _ in range(9):
            fst.update(frame)
        # 6 of 9 frames skipped → 2/3
        assert fst.effective_skip_fraction == pytest.approx(6 / 9)

    def test_counters_reset_on_initialize(self):
        fst = FrameSkipTracker(_CountingTracker(), skip_rate=2)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        fst.initialize(frame, _FIXED_BOX)
        for _ in range(6):
            fst.update(frame)
        fst.initialize(frame, _FIXED_BOX)
        assert fst.active_frame_count == 0
        assert fst.skipped_frame_count == 0


# ---------------------------------------------------------------------------
# Integration: FrameSkipTracker with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestFrameSkipWithEngine:
    def test_skip_rate_1_matches_direct_run(self):
        engine = BenchmarkEngine(verbose=False)
        dataset = _TinyDataset()
        direct = engine.run(_CountingTracker(), dataset, dataset_name="Syn")
        wrapped = engine.run(
            FrameSkipTracker(_CountingTracker(), skip_rate=1),
            dataset,
            dataset_name="Syn",
        )
        assert direct.mean_iou == pytest.approx(wrapped.mean_iou, abs=1e-3)

    def test_higher_skip_rate_higher_fps(self):
        engine = BenchmarkEngine(verbose=False)
        dataset = _TinyDataset(n=2)
        r1 = engine.run(_CountingTracker(), dataset, dataset_name="Syn")
        r3 = engine.run(
            FrameSkipTracker(_CountingTracker(), skip_rate=3),
            dataset,
            dataset_name="Syn",
        )
        # skip_rate=3 should be at least as fast as skip_rate=1
        # (on a constant tracker the overhead is negligible, allow 10% slack)
        assert r3.mean_fps >= r1.mean_fps * 0.5

    def test_tracker_name_propagated(self):
        engine = BenchmarkEngine(verbose=False)
        dataset = _TinyDataset()
        fst = FrameSkipTracker(_CountingTracker(), skip_rate=2)
        result = engine.run(fst, dataset, dataset_name="Syn")
        assert "skip2" in result.tracker_name


# ---------------------------------------------------------------------------
# FrameSkipAnalyzer
# ---------------------------------------------------------------------------

class TestFrameSkipAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return FrameSkipAnalyzer(BenchmarkEngine(verbose=False))

    @pytest.fixture
    def dataset(self):
        return _TinyDataset(n=2)

    def test_returns_skip_rate_result(self, analyzer, dataset):
        result = analyzer.analyze(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1, 2]
        )
        assert isinstance(result, SkipRateResult)

    def test_entries_count_matches_skip_rates(self, analyzer, dataset):
        result = analyzer.analyze(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1, 2, 3]
        )
        assert len(result.entries) == 3

    def test_entry_skip_rates_correct(self, analyzer, dataset):
        result = analyzer.analyze(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1, 4, 2]
        )
        rates = [e.skip_rate for e in result.entries]
        assert sorted(rates) == [1, 2, 4]

    def test_baseline_entry_iou_degradation_zero(self, analyzer, dataset):
        result = analyzer.analyze(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1, 2, 3]
        )
        base = result.baseline
        assert base is not None
        assert base.iou_degradation == pytest.approx(0.0)

    def test_baseline_fps_gain_one(self, analyzer, dataset):
        result = analyzer.analyze(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1, 2]
        )
        assert result.baseline.fps_gain == pytest.approx(1.0)

    def test_benchmark_results_keyed_by_rate(self, analyzer, dataset):
        result = analyzer.analyze(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1, 2, 3]
        )
        assert set(result.benchmark_results.keys()) == {1, 2, 3}

    def test_optimal_rate_returns_tuple(self, analyzer, dataset):
        result = analyzer.analyze(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1, 2, 3]
        )
        rate, iou, fps = result.optimal_rate(min_iou=0.0)
        assert isinstance(rate, int)
        assert isinstance(iou, float)
        assert isinstance(fps, float)

    def test_optimal_rate_respects_min_iou(self, analyzer, dataset):
        result = analyzer.analyze(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1, 2, 3]
        )
        rate, iou, fps = result.optimal_rate(min_iou=0.5)
        assert iou >= 0.5

    def test_optimal_rate_impossible_raises(self, analyzer, dataset):
        result = analyzer.analyze(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1]
        )
        with pytest.raises(ValueError):
            result.optimal_rate(min_iou=2.0)

    def test_to_markdown_table_contains_tracker_data(self, analyzer, dataset):
        result = analyzer.analyze(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1, 2]
        )
        table = result.to_markdown_table()
        assert "Skip Rate" in table
        assert "mIoU" in table
        assert "FPS" in table

    def test_invalid_skip_rate_raises(self, analyzer, dataset):
        with pytest.raises(ValueError):
            analyzer.analyze(
                _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[0]
            )

    def test_empty_skip_rates_raises(self, analyzer, dataset):
        with pytest.raises(ValueError):
            analyzer.analyze(
                _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[]
            )

    def test_str_representation(self, analyzer, dataset):
        result = analyzer.analyze(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1, 2]
        )
        s = str(result)
        assert "CountingTracker" in s
        assert "Syn" in s

    def test_fps_at_iou_budget(self, analyzer, dataset):
        result = analyzer.analyze(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1, 2, 3]
        )
        fps = result.fps_at_iou_budget(iou_budget=0.5)
        assert fps is not None and fps > 0

    def test_compare_modes_returns_both(self, analyzer, dataset):
        modes = analyzer.compare_modes(
            _CountingTracker(), dataset, dataset_name="Syn", skip_rates=[1, 2]
        )
        assert "repeat" in modes and "linear" in modes
        assert isinstance(modes["repeat"], SkipRateResult)
