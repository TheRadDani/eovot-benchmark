"""Tests for eovot.benchmark.vot_engine.

All tests use :class:`SyntheticDataset` — no real dataset download needed.
A minimal stub tracker is also defined here to avoid OpenCV dependency issues
in CI environments where the tracker contrib modules may be absent.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from eovot.benchmark.vot_engine import (
    VOTBenchmarkResult,
    VOTResetEngine,
    VOTSegment,
    VOTSequenceResult,
)
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Stub trackers
# ---------------------------------------------------------------------------

class PerfectTracker(BaseTracker):
    """Always returns the exact GT box.  Useful to verify EAO = 1.0."""

    def __init__(self) -> None:
        super().__init__("PerfectTracker")
        self._bbox: BBox = (0.0, 0.0, 10.0, 10.0)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        return self._bbox  # Return last known GT (from init or reinit).


class DriftingTracker(BaseTracker):
    """Drifts off-target a little every frame — causes some failures."""

    def __init__(self, drift_px: float = 5.0) -> None:
        super().__init__("DriftingTracker")
        self._bbox: BBox = (0.0, 0.0, 10.0, 10.0)
        self._drift = drift_px

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        x, y, w, h = self._bbox
        # Drift toward (0, 0) — will eventually leave the GT box.
        x = max(0.0, x - self._drift)
        y = max(0.0, y - self._drift)
        self._bbox = (x, y, w, h)
        return self._bbox


class AlwaysFailTracker(BaseTracker):
    """Returns a box in the corner that never overlaps the target."""

    def __init__(self) -> None:
        super().__init__("AlwaysFailTracker")

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        pass

    def update(self, frame: np.ndarray) -> BBox:
        return (0.0, 0.0, 1.0, 1.0)  # Far away from any realistic target.


# ---------------------------------------------------------------------------
# Dataset fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=60, motion="linear", seed=0)


@pytest.fixture
def single_sequence_dataset():
    return SyntheticDataset(num_sequences=1, num_frames=50, motion="circular", seed=1)


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

class TestVOTResetEngineInit:
    def test_defaults_accepted(self):
        engine = VOTResetEngine()
        assert engine.failure_threshold == 0.1
        assert engine.gap_frames == 5

    def test_custom_params(self):
        engine = VOTResetEngine(failure_threshold=0.3, gap_frames=10)
        assert engine.failure_threshold == 0.3
        assert engine.gap_frames == 10

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            VOTResetEngine(failure_threshold=1.5)

    def test_invalid_gap_raises(self):
        with pytest.raises(ValueError):
            VOTResetEngine(gap_frames=-1)


# ---------------------------------------------------------------------------
# Run returns correct types
# ---------------------------------------------------------------------------

class TestVOTResetEngineRun:
    def test_run_returns_vot_benchmark_result(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(PerfectTracker(), small_dataset, "synthetic")
        assert isinstance(result, VOTBenchmarkResult)

    def test_sequence_count_matches_dataset(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(PerfectTracker(), small_dataset, "synthetic")
        assert len(result.sequence_results) == len(small_dataset)

    def test_max_sequences_respected(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(PerfectTracker(), small_dataset, "synthetic", max_sequences=2)
        assert len(result.sequence_results) == 2

    def test_tracker_name_stored(self, single_sequence_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(PerfectTracker(), single_sequence_dataset, "syn")
        assert result.tracker_name == "PerfectTracker"

    def test_dataset_name_stored(self, single_sequence_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(PerfectTracker(), single_sequence_dataset, "MySyntheticDS")
        assert result.dataset_name == "MySyntheticDS"

    def test_failure_threshold_stored(self, single_sequence_dataset):
        engine = VOTResetEngine(failure_threshold=0.25, verbose=False)
        result = engine.run(PerfectTracker(), single_sequence_dataset)
        assert result.failure_threshold == 0.25

    def test_gap_frames_stored(self, single_sequence_dataset):
        engine = VOTResetEngine(gap_frames=3, verbose=False)
        result = engine.run(PerfectTracker(), single_sequence_dataset)
        assert result.gap_frames == 3


# ---------------------------------------------------------------------------
# Per-sequence result properties
# ---------------------------------------------------------------------------

class TestVOTSequenceResult:
    def test_frame_ious_length_matches_sequence(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(PerfectTracker(), small_dataset)
        for seq_result, seq in zip(result.sequence_results, small_dataset):
            assert len(seq_result.frame_ious) == len(seq)

    def test_frame_ious_in_unit_interval(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(DriftingTracker(drift_px=3.0), small_dataset)
        for seq_result in result.sequence_results:
            assert np.all(seq_result.frame_ious >= 0.0)
            assert np.all(seq_result.frame_ious <= 1.0 + 1e-9)

    def test_perfect_tracker_zero_failures(self, small_dataset):
        engine = VOTResetEngine(failure_threshold=0.0, verbose=False)
        result = engine.run(PerfectTracker(), small_dataset)
        for seq_result in result.sequence_results:
            assert seq_result.num_failures == 0

    def test_always_fail_tracker_has_failures(self, small_dataset):
        engine = VOTResetEngine(failure_threshold=0.5, verbose=False)
        result = engine.run(AlwaysFailTracker(), small_dataset)
        total = sum(r.num_failures for r in result.sequence_results)
        assert total > 0

    def test_segments_non_empty(self, single_sequence_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(DriftingTracker(), single_sequence_dataset)
        for seq_result in result.sequence_results:
            assert len(seq_result.segments) >= 1

    def test_segment_ious_in_unit_interval(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(DriftingTracker(), small_dataset)
        for seq_result in result.sequence_results:
            for seg in seq_result.segments:
                assert np.all(seg.ious >= 0.0)
                assert np.all(seg.ious <= 1.0 + 1e-9)

    def test_gap_frames_zeroed_after_failure(self, single_sequence_dataset):
        engine = VOTResetEngine(failure_threshold=0.5, gap_frames=5, verbose=False)
        result = engine.run(AlwaysFailTracker(), single_sequence_dataset)
        for seq_result in result.sequence_results:
            for f in seq_result.failure_frames:
                for g in range(f + 1, min(f + engine.gap_frames + 1, len(seq_result.frame_ious))):
                    assert seq_result.frame_ious[g] == 0.0, (
                        f"Expected gap zero at frame {g} after failure at {f}"
                    )

    def test_summary_has_required_keys(self, single_sequence_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(PerfectTracker(), single_sequence_dataset)
        seq_result = result.sequence_results[0]
        summary = seq_result.summary()
        for key in ("sequence_name", "eao", "num_failures", "num_segments", "fps"):
            assert key in summary, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Aggregate (benchmark-level) properties
# ---------------------------------------------------------------------------

class TestVOTBenchmarkResult:
    def test_eao_in_unit_interval(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(DriftingTracker(), small_dataset)
        assert 0.0 <= result.eao <= 1.0

    def test_eao_curve_in_unit_interval(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(DriftingTracker(), small_dataset)
        assert 0.0 <= result.eao_curve() <= 1.0

    def test_mean_fps_positive(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(DriftingTracker(), small_dataset)
        assert result.mean_fps > 0.0

    def test_peak_memory_positive(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(PerfectTracker(), small_dataset)
        assert result.peak_memory_mb > 0.0

    def test_summary_has_required_keys(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(PerfectTracker(), small_dataset)
        s = result.summary()
        for key in ("tracker", "dataset", "eao", "total_failures",
                    "mean_failures_per_sequence", "mean_fps"):
            assert key in s, f"Missing key: {key}"

    def test_to_dict_has_sequences(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(PerfectTracker(), small_dataset)
        d = result.to_dict()
        assert "summary" in d
        assert "sequences" in d
        assert len(d["sequences"]) == len(small_dataset)

    def test_str_repr_contains_tracker_name(self, small_dataset):
        engine = VOTResetEngine(verbose=False)
        result = engine.run(PerfectTracker(), small_dataset)
        assert "PerfectTracker" in str(result)

    def test_always_fail_has_more_failures_than_perfect(self, small_dataset):
        engine = VOTResetEngine(failure_threshold=0.5, verbose=False)
        perfect = engine.run(PerfectTracker(), small_dataset)
        fail = engine.run(AlwaysFailTracker(), small_dataset)
        assert fail.total_failures >= perfect.total_failures

    def test_empty_dataset_eao_zero(self):
        """Engine should handle a dataset with 0 sequences gracefully."""
        empty_ds = SyntheticDataset(num_sequences=0, num_frames=50)
        engine = VOTResetEngine(verbose=False)
        result = engine.run(PerfectTracker(), empty_ds)
        assert result.eao == 0.0
        assert result.total_failures == 0


# ---------------------------------------------------------------------------
# VOTSegment dataclass
# ---------------------------------------------------------------------------

class TestVOTSegment:
    def test_length_property(self):
        seg = VOTSegment(
            start_frame=5,
            end_frame=20,
            ious=np.ones(15),
            predictions=np.zeros((15, 4)),
        )
        assert seg.length == 15

    def test_mean_iou(self):
        seg = VOTSegment(
            start_frame=0,
            end_frame=4,
            ious=np.array([1.0, 0.8, 0.6, 0.4]),
            predictions=np.zeros((4, 4)),
        )
        assert abs(seg.mean_iou - 0.7) < 1e-9

    def test_empty_segment_mean_iou(self):
        seg = VOTSegment(
            start_frame=0,
            end_frame=0,
            ious=np.array([]),
            predictions=np.zeros((0, 4)),
        )
        assert seg.mean_iou == 0.0
