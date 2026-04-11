"""Integration tests for AUC metric fields in BenchmarkResult and SequenceResult.

These tests verify the fixes introduced in engine.py:

- SequenceResult carries ``energy`` and ``accuracy`` fields.
- BenchmarkResult exposes ``success_auc`` and ``precision_auc`` aggregate
  properties computed from per-sequence AccuracyMetrics.
- summary() and to_dict() include AUC fields when data is available and
  omit them gracefully when it is not.
- __str__ surfaces the success AUC alongside mIoU and FPS.
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.metrics.accuracy import AccuracyMetrics
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profiling(fps: float = 100.0, mem_mb: float = 50.0) -> ProfilingResult:
    """Return a minimal ProfilingResult for stub sequences."""
    return ProfilingResult(
        tracker_name="stub",
        frame_count=10,
        fps=fps,
        latency_mean_ms=1000.0 / fps,
        latency_std_ms=0.1,
        latency_p95_ms=1000.0 / fps + 1.0,
        peak_memory_mb=mem_mb,
    )


def _accuracy(iou: float = 0.70, s_auc: float = 0.65, p_auc: float = 0.72) -> AccuracyMetrics:
    """Return a stub AccuracyMetrics instance."""
    return AccuracyMetrics(mean_iou=iou, success_auc=s_auc, precision_auc=p_auc)


def _seq_result(
    name: str,
    iou: float,
    fps: float = 100.0,
    mem_mb: float = 50.0,
    with_accuracy: bool = True,
) -> SequenceResult:
    """Construct a minimal SequenceResult with optional AccuracyMetrics."""
    sr = SequenceResult(
        sequence_name=name,
        ious=np.array([iou, iou, iou]),
        profiling=_profiling(fps=fps, mem_mb=mem_mb),
    )
    if with_accuracy:
        sr.accuracy = _accuracy(iou=iou)
    return sr


def _benchmark(
    tracker: str = "TestTracker",
    dataset: str = "TestDS",
    n: int = 3,
    with_accuracy: bool = True,
) -> BenchmarkResult:
    """Build a BenchmarkResult with *n* synthetic SequenceResults."""
    result = BenchmarkResult(tracker_name=tracker, dataset_name=dataset)
    for i in range(n):
        result.sequence_results.append(
            _seq_result(f"seq{i}", iou=0.60 + i * 0.05, fps=120.0, mem_mb=80.0,
                        with_accuracy=with_accuracy)
        )
    return result


# ---------------------------------------------------------------------------
# SequenceResult field tests
# ---------------------------------------------------------------------------

class TestSequenceResultFields:
    """Verify that the new fields added to SequenceResult behave correctly."""

    def test_accuracy_field_present_when_set(self):
        sr = _seq_result("s", iou=0.7, with_accuracy=True)
        assert sr.accuracy is not None
        assert isinstance(sr.accuracy, AccuracyMetrics)

    def test_accuracy_field_none_by_default(self):
        sr = _seq_result("s", iou=0.7, with_accuracy=False)
        assert sr.accuracy is None

    def test_energy_field_defaults_to_none(self):
        sr = _seq_result("s", iou=0.5)
        assert sr.energy is None

    def test_predictions_and_ground_truths_default_none(self):
        sr = _seq_result("s", iou=0.5)
        assert sr.predictions is None
        assert sr.ground_truths is None

    def test_mean_iou_matches_ious_array(self):
        sr = _seq_result("s", iou=0.8)
        assert abs(sr.mean_iou - 0.8) < 1e-9

    def test_mean_center_distance_none_when_not_stored(self):
        sr = _seq_result("s", iou=0.6)
        assert sr.mean_center_distance is None

    def test_mean_center_distance_computed_when_stored(self):
        sr = _seq_result("s", iou=0.6)
        sr.center_distances = np.array([10.0, 20.0, 30.0])
        assert abs(sr.mean_center_distance - 20.0) < 1e-9


# ---------------------------------------------------------------------------
# BenchmarkResult AUC aggregate properties
# ---------------------------------------------------------------------------

class TestBenchmarkResultAUCProperties:
    """Verify success_auc and precision_auc aggregation logic."""

    def test_success_auc_returns_float(self):
        result = _benchmark()
        assert isinstance(result.success_auc, float)

    def test_precision_auc_returns_float(self):
        result = _benchmark()
        assert isinstance(result.precision_auc, float)

    def test_success_auc_in_unit_interval(self):
        result = _benchmark()
        assert 0.0 <= result.success_auc <= 1.0  # type: ignore[operator]

    def test_precision_auc_in_unit_interval(self):
        result = _benchmark()
        assert 0.0 <= result.precision_auc <= 1.0  # type: ignore[operator]

    def test_success_auc_none_when_no_accuracy(self):
        result = _benchmark(with_accuracy=False)
        assert result.success_auc is None

    def test_precision_auc_none_when_no_accuracy(self):
        result = _benchmark(with_accuracy=False)
        assert result.precision_auc is None

    def test_success_auc_is_mean_of_sequence_aucs(self):
        """success_auc must equal the mean of per-sequence AUCs."""
        result = BenchmarkResult(tracker_name="T", dataset_name="D")
        expected_aucs = [0.50, 0.60, 0.70]
        for i, a in enumerate(expected_aucs):
            sr = _seq_result(f"seq{i}", iou=a)
            sr.accuracy = _accuracy(iou=a, s_auc=a)
            result.sequence_results.append(sr)
        assert abs(result.success_auc - np.mean(expected_aucs)) < 1e-9  # type: ignore[operator]

    def test_precision_auc_is_mean_of_sequence_aucs(self):
        result = BenchmarkResult(tracker_name="T", dataset_name="D")
        expected_aucs = [0.55, 0.65, 0.75]
        for i, a in enumerate(expected_aucs):
            sr = _seq_result(f"seq{i}", iou=a)
            sr.accuracy = _accuracy(iou=a, p_auc=a)
            result.sequence_results.append(sr)
        assert abs(result.precision_auc - np.mean(expected_aucs)) < 1e-9  # type: ignore[operator]

    def test_partial_accuracy_data_uses_available_only(self):
        """If only some sequences have AccuracyMetrics, AUC is mean of those only."""
        result = BenchmarkResult(tracker_name="T", dataset_name="D")
        sr0 = _seq_result("seq0", iou=0.6, with_accuracy=True)
        sr0.accuracy = _accuracy(s_auc=0.55)
        sr1 = _seq_result("seq1", iou=0.7, with_accuracy=False)  # no accuracy
        result.sequence_results.extend([sr0, sr1])
        # Should use only sr0's AUC
        assert abs(result.success_auc - 0.55) < 1e-9  # type: ignore[operator]


# ---------------------------------------------------------------------------
# summary() dict contents
# ---------------------------------------------------------------------------

class TestBenchmarkResultSummary:
    """Verify that summary() returns the expected keys."""

    def test_core_fields_always_present(self):
        result = _benchmark(with_accuracy=False)
        s = result.summary()
        for key in ("tracker", "dataset", "num_sequences", "mean_iou", "mean_fps", "peak_memory_mb"):
            assert key in s, f"Missing key: {key}"

    def test_auc_fields_present_when_accuracy_available(self):
        result = _benchmark(with_accuracy=True)
        s = result.summary()
        assert "success_auc" in s
        assert "precision_auc" in s

    def test_auc_fields_absent_when_no_accuracy(self):
        result = _benchmark(with_accuracy=False)
        s = result.summary()
        assert "success_auc" not in s
        assert "precision_auc" not in s

    def test_auc_values_are_floats_in_summary(self):
        result = _benchmark(with_accuracy=True)
        s = result.summary()
        assert isinstance(s["success_auc"], float)
        assert isinstance(s["precision_auc"], float)

    def test_energy_absent_without_profiling(self):
        result = _benchmark()
        s = result.summary()
        assert "total_energy_j" not in s
        assert "mean_energy_per_frame_mj" not in s


# ---------------------------------------------------------------------------
# to_dict() output
# ---------------------------------------------------------------------------

class TestBenchmarkResultToDict:
    """Verify to_dict() structure and AUC presence in sequence entries."""

    def test_top_level_keys(self):
        result = _benchmark()
        d = result.to_dict()
        assert "summary" in d
        assert "sequences" in d

    def test_sequence_count_matches(self):
        result = _benchmark(n=4)
        d = result.to_dict()
        assert len(d["sequences"]) == 4

    def test_per_sequence_core_fields(self):
        result = _benchmark(n=1)
        entry = result.to_dict()["sequences"][0]
        for key in ("sequence_name", "mean_iou", "fps", "mean_latency_ms", "peak_memory_mb"):
            assert key in entry

    def test_per_sequence_auc_present_when_accuracy(self):
        result = _benchmark(with_accuracy=True)
        for entry in result.to_dict()["sequences"]:
            assert "success_auc" in entry
            assert "precision_auc" in entry

    def test_per_sequence_auc_absent_when_no_accuracy(self):
        result = _benchmark(with_accuracy=False)
        for entry in result.to_dict()["sequences"]:
            assert "success_auc" not in entry
            assert "precision_auc" not in entry

    def test_summary_in_to_dict_matches_summary_method(self):
        result = _benchmark()
        assert result.to_dict()["summary"] == result.summary()


# ---------------------------------------------------------------------------
# __str__ representation
# ---------------------------------------------------------------------------

class TestBenchmarkResultStr:
    def test_str_contains_tracker_name(self):
        result = _benchmark(tracker="MOSSE")
        assert "MOSSE" in str(result)

    def test_str_contains_auc_when_available(self):
        result = _benchmark(with_accuracy=True)
        assert "AUC" in str(result)

    def test_str_omits_auc_when_no_accuracy(self):
        result = _benchmark(with_accuracy=False)
        assert "AUC" not in str(result)

    def test_str_contains_fps_and_mem(self):
        result = _benchmark()
        s = str(result)
        assert "FPS" in s
        assert "MiB" in s
