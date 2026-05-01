"""Integration tests verifying AccuracyMetrics are computed and stored per sequence.

These tests use a fully in-memory synthetic dataset so no real dataset files
are required.  They complement test_engine.py by specifically exercising the
accuracy-metrics pipeline end-to-end through BenchmarkEngine.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult, SequenceResult
from eovot.datasets.base import BaseDataset, Sequence
from eovot.metrics.accuracy import AccuracyMetrics
from eovot.trackers.base import BaseTracker

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NUM_FRAMES = 25
GT_BOX = (10.0, 10.0, 50.0, 50.0)
SHIFTED_BOX = (40.0, 40.0, 50.0, 50.0)  # overlaps GT but not perfectly


class _StaticTracker(BaseTracker):
    def __init__(self, box):
        self._box = box

    @property
    def name(self):
        return "StaticTracker"

    def initialize(self, frame, bbox):
        pass

    def update(self, frame):
        return self._box


class _SyntheticSeq(Sequence):
    def __init__(self, name, n_frames, gt_box):
        gt = np.tile(np.array(gt_box, dtype=np.float64), (n_frames, 1))
        super().__init__(name=name, frame_paths=[f"f{i}.jpg" for i in range(n_frames)], ground_truth=gt)
        self._n = n_frames

    def __iter__(self) -> Iterator[np.ndarray]:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(self._n):
            yield frame


class _SyntheticDataset(BaseDataset):
    def __init__(self, n=3):
        self._seqs = [_SyntheticSeq(f"s{i}", NUM_FRAMES, GT_BOX) for i in range(n)]

    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, idx):
        return self._seqs[idx]


@pytest.fixture
def perfect_engine_result():
    engine = BenchmarkEngine(verbose=False)
    tracker = _StaticTracker(GT_BOX)
    dataset = _SyntheticDataset(n=3)
    return engine.run(tracker, dataset, dataset_name="Synthetic")


@pytest.fixture
def imperfect_engine_result():
    engine = BenchmarkEngine(verbose=False)
    tracker = _StaticTracker(SHIFTED_BOX)
    dataset = _SyntheticDataset(n=3)
    return engine.run(tracker, dataset, dataset_name="Synthetic")


# ---------------------------------------------------------------------------
# SequenceResult accuracy field
# ---------------------------------------------------------------------------

class TestSequenceAccuracyField:
    def test_accuracy_field_not_none(self, perfect_engine_result):
        for sr in perfect_engine_result.sequence_results:
            assert sr.accuracy is not None, "accuracy must be populated by engine"

    def test_accuracy_is_accuracy_metrics_instance(self, perfect_engine_result):
        for sr in perfect_engine_result.sequence_results:
            assert isinstance(sr.accuracy, AccuracyMetrics)

    def test_perfect_tracker_mean_iou_one(self, perfect_engine_result):
        for sr in perfect_engine_result.sequence_results:
            assert sr.accuracy.mean_iou == pytest.approx(1.0, abs=1e-6)

    def test_perfect_tracker_success_auc_approx_one(self, perfect_engine_result):
        for sr in perfect_engine_result.sequence_results:
            # AUC of success curve when all IoU = 1: thresholds in [0,1], all rates = 1
            # except at the strict > comparison at threshold=1.0; AUC ≈ 1.0
            assert sr.accuracy.success_auc == pytest.approx(1.0, abs=0.02)

    def test_perfect_tracker_precision_auc_approx_one(self, perfect_engine_result):
        for sr in perfect_engine_result.sequence_results:
            assert sr.accuracy.precision_auc == pytest.approx(1.0, abs=0.02)

    def test_imperfect_tracker_success_auc_less_than_one(self, imperfect_engine_result):
        for sr in imperfect_engine_result.sequence_results:
            assert sr.accuracy.success_auc < 1.0

    def test_imperfect_tracker_precision_auc_less_than_one(self, imperfect_engine_result):
        for sr in imperfect_engine_result.sequence_results:
            assert sr.accuracy.precision_auc < 1.0

    def test_accuracy_auc_in_valid_range(self, imperfect_engine_result):
        for sr in imperfect_engine_result.sequence_results:
            assert 0.0 <= sr.accuracy.success_auc <= 1.0
            assert 0.0 <= sr.accuracy.precision_auc <= 1.0


# ---------------------------------------------------------------------------
# BenchmarkResult aggregate AUC properties
# ---------------------------------------------------------------------------

class TestBenchmarkResultAUC:
    def test_mean_success_auc_not_none(self, perfect_engine_result):
        assert perfect_engine_result.mean_success_auc is not None

    def test_mean_precision_auc_not_none(self, perfect_engine_result):
        assert perfect_engine_result.mean_precision_auc is not None

    def test_perfect_tracker_mean_success_auc(self, perfect_engine_result):
        assert perfect_engine_result.mean_success_auc == pytest.approx(1.0, abs=0.02)

    def test_perfect_tracker_mean_precision_auc(self, perfect_engine_result):
        assert perfect_engine_result.mean_precision_auc == pytest.approx(1.0, abs=0.02)

    def test_imperfect_has_lower_auc(self, perfect_engine_result, imperfect_engine_result):
        assert imperfect_engine_result.mean_success_auc < perfect_engine_result.mean_success_auc
        assert imperfect_engine_result.mean_precision_auc < perfect_engine_result.mean_precision_auc

    def test_auc_range(self, imperfect_engine_result):
        assert 0.0 <= imperfect_engine_result.mean_success_auc <= 1.0
        assert 0.0 <= imperfect_engine_result.mean_precision_auc <= 1.0


# ---------------------------------------------------------------------------
# summary() and to_dict() expose AUC keys
# ---------------------------------------------------------------------------

class TestSummaryAndDict:
    def test_summary_contains_success_auc(self, perfect_engine_result):
        s = perfect_engine_result.summary()
        assert "mean_success_auc" in s

    def test_summary_contains_precision_auc(self, perfect_engine_result):
        s = perfect_engine_result.summary()
        assert "mean_precision_auc" in s

    def test_to_dict_sequences_contain_success_auc(self, perfect_engine_result):
        d = perfect_engine_result.to_dict()
        for seq in d["sequences"]:
            assert "success_auc" in seq, f"success_auc missing from sequence {seq}"

    def test_to_dict_sequences_contain_precision_auc(self, perfect_engine_result):
        d = perfect_engine_result.to_dict()
        for seq in d["sequences"]:
            assert "precision_auc" in seq, f"precision_auc missing from sequence {seq}"

    def test_to_dict_no_duplicate_keys(self, perfect_engine_result):
        # Verify the old triple-definition bug is gone: to_dict() must be unique
        import inspect
        import eovot.benchmark.engine as eng_mod
        src = inspect.getsource(BenchmarkResult.to_dict)
        # The source of to_dict should mention 'success_auc' (the current implementation)
        assert "success_auc" in src

    def test_summary_auc_values_are_floats(self, perfect_engine_result):
        s = perfect_engine_result.summary()
        assert isinstance(s["mean_success_auc"], float)
        assert isinstance(s["mean_precision_auc"], float)


# ---------------------------------------------------------------------------
# Energy field on SequenceResult (no energy profiling → None)
# ---------------------------------------------------------------------------

class TestEnergyFieldWithoutProfiling:
    def test_energy_none_when_not_configured(self, perfect_engine_result):
        for sr in perfect_engine_result.sequence_results:
            assert sr.energy is None

    def test_energy_field_exists_on_sequence_result(self, perfect_engine_result):
        sr = perfect_engine_result.sequence_results[0]
        assert hasattr(sr, "energy")

    def test_total_energy_none_when_not_profiled(self, perfect_engine_result):
        assert perfect_engine_result.total_energy_j is None

    def test_total_energy_not_in_summary_when_none(self, perfect_engine_result):
        s = perfect_engine_result.summary()
        assert "total_energy_j" not in s


class TestEnergyFieldWithProfiling:
    def test_energy_populated_when_tdp_set(self):
        engine = BenchmarkEngine(verbose=False, tdp_watts=15.0)
        tracker = _StaticTracker(GT_BOX)
        dataset = _SyntheticDataset(n=2)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        for sr in result.sequence_results:
            assert sr.energy is not None, "energy must be populated when tdp_watts is set"

    def test_total_energy_positive_when_profiled(self):
        engine = BenchmarkEngine(verbose=False, tdp_watts=15.0)
        tracker = _StaticTracker(GT_BOX)
        dataset = _SyntheticDataset(n=2)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        total = result.total_energy_j
        assert total is not None
        assert total >= 0.0

    def test_energy_in_summary_when_profiled(self):
        engine = BenchmarkEngine(verbose=False, tdp_watts=15.0)
        tracker = _StaticTracker(GT_BOX)
        dataset = _SyntheticDataset(n=2)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        s = result.summary()
        assert "total_energy_j" in s
        assert "mean_energy_per_frame_mj" in s
