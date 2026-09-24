"""Integration tests: RobustnessAnalyzer wired into BenchmarkEngine."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset


@pytest.fixture
def small_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=40, seed=0)


@pytest.fixture
def engine():
    return BenchmarkEngine(verbose=False)


class TestRobustnessIntegration:
    def test_sequence_result_has_robustness(self, engine, small_dataset):
        from eovot.trackers.mosse import MOSSETracker

        result = engine.run(MOSSETracker(), small_dataset, dataset_name="Synthetic")
        for sr in result.sequence_results:
            assert sr.robustness is not None, "SequenceResult.robustness must be populated"

    def test_robustness_fields_populated(self, engine, small_dataset):
        from eovot.trackers.mosse import MOSSETracker

        result = engine.run(MOSSETracker(), small_dataset, dataset_name="Synthetic")
        for sr in result.sequence_results:
            rob = sr.robustness
            assert 0.0 <= rob.eao <= 1.0
            assert 0.0 <= rob.survival_rate <= 1.0
            assert rob.num_failures >= 0
            assert rob.mean_recovery_lag >= 0.0

    def test_benchmark_result_aggregate_robustness(self, engine, small_dataset):
        from eovot.trackers.kcf import KCFTracker

        result = engine.run(KCFTracker(), small_dataset, dataset_name="Synthetic")
        assert result.mean_eao is not None
        assert 0.0 <= result.mean_eao <= 1.0
        assert result.total_failures is not None
        assert result.total_failures >= 0
        assert result.mean_survival_rate is not None
        assert 0.0 <= result.mean_survival_rate <= 1.0

    def test_summary_includes_robustness(self, engine, small_dataset):
        from eovot.trackers.mosse import MOSSETracker

        result = engine.run(MOSSETracker(), small_dataset, dataset_name="Synthetic")
        s = result.summary()
        assert "mean_eao" in s
        assert "total_failures" in s
        assert "mean_survival_rate" in s

    def test_to_dict_round_trip_preserves_robustness(self, engine, small_dataset):
        from eovot.trackers.mosse import MOSSETracker

        result = engine.run(MOSSETracker(), small_dataset, dataset_name="Synthetic")
        d = result.to_dict()

        # Check per-sequence serialization
        for seq in d["sequences"]:
            assert "eao" in seq
            assert "num_failures" in seq
            assert "survival_rate" in seq
            assert "mean_recovery_lag" in seq

        # Round-trip via from_dict
        restored = BenchmarkResult.from_dict(d)
        assert restored.mean_eao is not None
        assert abs(restored.mean_eao - result.mean_eao) < 1e-3

    def test_save_load_preserves_robustness(self, engine, small_dataset, tmp_path):
        from eovot.trackers.mosse import MOSSETracker

        result = engine.run(MOSSETracker(), small_dataset, dataset_name="Synthetic")
        path = result.save(tmp_path / "robustness_test")
        loaded = BenchmarkResult.load(path)

        assert loaded.mean_eao is not None
        assert loaded.total_failures is not None
        assert abs(loaded.mean_eao - result.mean_eao) < 1e-3
        assert loaded.total_failures == result.total_failures

    def test_eao_is_consistent_with_mean_iou(self, engine, small_dataset):
        """EAO should be close to mean_iou since both measure mean overlap."""
        from eovot.trackers.mosse import MOSSETracker

        result = engine.run(MOSSETracker(), small_dataset, dataset_name="Synthetic")
        # EAO skips burn-in frames; mean_iou uses all. They should be in the same ballpark.
        assert abs(result.mean_eao - result.mean_iou) < 0.15

    def test_robustness_tracker_name_and_sequence_name(self, engine, small_dataset):
        from eovot.trackers.mosse import MOSSETracker

        tracker = MOSSETracker()
        result = engine.run(tracker, small_dataset, dataset_name="Synthetic")
        for sr in result.sequence_results:
            assert sr.robustness.tracker_name == tracker.name
            assert sr.robustness.sequence_name == sr.sequence_name
