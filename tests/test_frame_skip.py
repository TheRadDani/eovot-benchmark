"""Tests for eovot.analysis.frame_skip — temporal frame-skip analysis."""

from __future__ import annotations

import json
from typing import Tuple

import numpy as np
import pytest

from eovot.analysis.frame_skip import FrameSkipAnalysis, FrameSkipEvaluator, SkipRateResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.base import BaseTracker
from eovot.trackers.mosse import MOSSETracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BBox = Tuple[float, float, float, float]


class _ConstantTracker(BaseTracker):
    """Always returns the initialisation bounding box — never drifts."""

    def __init__(self) -> None:
        super().__init__(name="ConstantTracker")
        self._bbox: BBox = (0.0, 0.0, 1.0, 1.0)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        return self._bbox


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_dataset():
    # Linear motion with short sequences — target stays well within frame.
    return SyntheticDataset(num_sequences=3, num_frames=20, motion="linear")


@pytest.fixture()
def const_tracker():
    return _ConstantTracker()


@pytest.fixture()
def mosse():
    return MOSSETracker()


# ---------------------------------------------------------------------------
# FrameSkipEvaluator construction
# ---------------------------------------------------------------------------


class TestFrameSkipEvaluatorInit:
    def test_default_skip_rates(self):
        ev = FrameSkipEvaluator()
        assert ev.skip_rates == [1, 2, 4, 8]

    def test_skip_rate_1_auto_inserted(self):
        ev = FrameSkipEvaluator(skip_rates=[2, 4])
        assert 1 in ev.skip_rates

    def test_sorted_deduplication(self):
        ev = FrameSkipEvaluator(skip_rates=[8, 2, 4, 2, 1])
        assert ev.skip_rates == [1, 2, 4, 8]

    def test_invalid_skip_rate_raises(self):
        with pytest.raises(ValueError):
            FrameSkipEvaluator(skip_rates=[0, 2])

    def test_interpolation_modes(self):
        ev_hold = FrameSkipEvaluator(interpolation="hold_last")
        ev_lin = FrameSkipEvaluator(interpolation="linear")
        assert ev_hold.interpolation == "hold_last"
        assert ev_lin.interpolation == "linear"


# ---------------------------------------------------------------------------
# SkipRateResult
# ---------------------------------------------------------------------------


class TestSkipRateResult:
    def test_attributes_stored(self):
        r = SkipRateResult(
            skip_rate=4,
            mean_iou=0.72,
            success_auc=0.68,
            failure_rate=0.05,
            effective_fps_multiplier=4.0,
        )
        assert r.skip_rate == 4
        assert r.mean_iou == pytest.approx(0.72)
        assert r.effective_fps_multiplier == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Full evaluation run (integration)
# ---------------------------------------------------------------------------


class TestFrameSkipEvaluatorRun:
    def test_returns_analysis_object(self, const_tracker, small_dataset):
        ev = FrameSkipEvaluator(skip_rates=[1, 2], verbose=False)
        result = ev.evaluate(const_tracker, small_dataset, "Synthetic", max_sequences=2)
        assert isinstance(result, FrameSkipAnalysis)

    def test_all_skip_rates_present(self, const_tracker, small_dataset):
        skip_rates = [1, 2, 4]
        ev = FrameSkipEvaluator(skip_rates=skip_rates, verbose=False)
        result = ev.evaluate(const_tracker, small_dataset, "Synthetic", max_sequences=2)
        returned = [r.skip_rate for r in result.skip_results]
        assert returned == skip_rates

    def test_baseline_iou_highest(self, const_tracker, small_dataset):
        """skip_rate=1 should produce highest or equal accuracy."""
        ev = FrameSkipEvaluator(skip_rates=[1, 2, 4], verbose=False)
        result = ev.evaluate(const_tracker, small_dataset, "Synthetic", max_sequences=3)
        ious = {r.skip_rate: r.mean_iou for r in result.skip_results}
        assert ious[1] >= ious[2] - 1e-6
        assert ious[1] >= ious[4] - 1e-6

    def test_effective_fps_multiplier(self, const_tracker, small_dataset):
        ev = FrameSkipEvaluator(skip_rates=[1, 2, 4, 8], verbose=False)
        result = ev.evaluate(const_tracker, small_dataset, "Synthetic", max_sequences=2)
        for r in result.skip_results:
            assert r.effective_fps_multiplier == pytest.approx(float(r.skip_rate))

    def test_max_sequences_respected(self, const_tracker, small_dataset):
        ev = FrameSkipEvaluator(skip_rates=[1], verbose=False)
        result = ev.evaluate(const_tracker, small_dataset, "Synthetic", max_sequences=1)
        assert result.tracker_name == const_tracker.name

    def test_interpolation_linear_runs(self, const_tracker, small_dataset):
        ev = FrameSkipEvaluator(skip_rates=[1, 2], interpolation="linear", verbose=False)
        result = ev.evaluate(const_tracker, small_dataset, "Synthetic", max_sequences=2)
        assert result.interpolation == "linear"
        assert len(result.skip_results) == 2

    def test_baseline_fps_positive(self, const_tracker, small_dataset):
        ev = FrameSkipEvaluator(skip_rates=[1], verbose=False)
        result = ev.evaluate(const_tracker, small_dataset, "Synthetic", max_sequences=2)
        assert result.baseline_fps > 0.0

    def test_failure_rate_in_unit_interval(self, const_tracker, small_dataset):
        ev = FrameSkipEvaluator(skip_rates=[1, 4], verbose=False)
        result = ev.evaluate(const_tracker, small_dataset, "Synthetic", max_sequences=2)
        for r in result.skip_results:
            assert 0.0 <= r.failure_rate <= 1.0

    def test_success_auc_in_unit_interval(self, const_tracker, small_dataset):
        ev = FrameSkipEvaluator(skip_rates=[1, 2], verbose=False)
        result = ev.evaluate(const_tracker, small_dataset, "Synthetic", max_sequences=2)
        for r in result.skip_results:
            assert 0.0 <= r.success_auc <= 1.0 + 1e-9

    def test_mosse_basic_run(self, mosse, small_dataset):
        """Smoke test: MOSSE completes at skip_rate=1 without crashing."""
        ev = FrameSkipEvaluator(skip_rates=[1], verbose=False)
        result = ev.evaluate(mosse, small_dataset, "Synthetic", max_sequences=1)
        assert isinstance(result, FrameSkipAnalysis)
        assert result.baseline_fps > 0.0


# ---------------------------------------------------------------------------
# FrameSkipAnalysis derived properties
# ---------------------------------------------------------------------------


class TestFrameSkipAnalysis:
    def _make_analysis(self, iou_values):
        results = [
            SkipRateResult(
                skip_rate=k,
                mean_iou=v,
                success_auc=v * 0.9,
                failure_rate=1.0 - v,
                effective_fps_multiplier=float(k),
            )
            for k, v in zip([1, 2, 4, 8], iou_values)
        ]
        return FrameSkipAnalysis(
            tracker_name="test",
            dataset_name="synthetic",
            baseline_fps=100.0,
            interpolation="hold_last",
            skip_results=results,
        )

    def test_optimal_skip_rate_no_drop(self):
        # No degradation — highest skip rate should be optimal
        analysis = self._make_analysis([0.80, 0.80, 0.80, 0.80])
        assert analysis.optimal_skip_rate == 8

    def test_optimal_skip_rate_significant_drop(self):
        # Large drop at skip=2 — optimal should be 1
        analysis = self._make_analysis([0.80, 0.50, 0.30, 0.10])
        assert analysis.optimal_skip_rate == 1

    def test_optimal_skip_rate_small_drop(self):
        # 5% drop at skip=2, 15% at skip=4 — optimal skip=2 within 10% tolerance
        baseline = 0.80
        analysis = self._make_analysis([baseline, baseline * 0.95, baseline * 0.83, baseline * 0.60])
        assert analysis.optimal_skip_rate == 2

    def test_accuracy_at_existing_rate(self):
        analysis = self._make_analysis([0.80, 0.75, 0.65, 0.45])
        assert analysis.accuracy_at(2) == pytest.approx(0.75)

    def test_accuracy_at_missing_rate(self):
        analysis = self._make_analysis([0.80, 0.75, 0.65, 0.45])
        assert analysis.accuracy_at(16) is None

    def test_degradation_by_skip_rate(self):
        analysis = self._make_analysis([0.80, 0.80, 0.60, 0.40])
        deg = analysis.degradation_by_skip_rate()
        assert deg[1] == pytest.approx(0.0)
        assert deg[4] == pytest.approx(0.25)

    def test_summary_table_is_string(self):
        analysis = self._make_analysis([0.80, 0.75, 0.65, 0.45])
        table = analysis.summary_table()
        assert isinstance(table, str)
        assert "Skip Rate" in table

    def test_to_dict_structure(self):
        analysis = self._make_analysis([0.80, 0.75, 0.65, 0.45])
        d = analysis.to_dict()
        assert "tracker_name" in d
        assert "skip_results" in d
        assert "optimal_skip_rate" in d
        assert len(d["skip_results"]) == 4

    def test_to_dict_json_serialisable(self):
        analysis = self._make_analysis([0.80, 0.75, 0.65, 0.45])
        blob = json.dumps(analysis.to_dict())
        recovered = json.loads(blob)
        assert recovered["tracker_name"] == "test"

    def test_empty_skip_results(self):
        analysis = FrameSkipAnalysis(
            tracker_name="x",
            dataset_name="y",
            baseline_fps=0.0,
            interpolation="hold_last",
            skip_results=[],
        )
        assert analysis.optimal_skip_rate == 1
        assert analysis.accuracy_at(1) is None
        assert analysis.degradation_by_skip_rate() == {}
