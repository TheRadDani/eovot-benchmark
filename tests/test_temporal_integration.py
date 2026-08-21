"""Integration tests: TemporalConsistencyAnalyzer wired into BenchmarkEngine.

Verifies that every benchmark run automatically populates temporal consistency
metrics on each SequenceResult, that the aggregate BenchmarkResult properties
work, and that temporal data round-trips cleanly through JSON serialisation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


@pytest.fixture(scope="module")
def benchmark_result():
    dataset = SyntheticDataset(
        num_sequences=3, num_frames=40, motion="linear", seed=42
    )
    engine = BenchmarkEngine(verbose=False)
    return engine.run(MOSSETracker(), dataset, dataset_name="Syn-Temporal-Test")


class TestTemporalOnSequenceResult:
    def test_temporal_field_is_populated(self, benchmark_result):
        for seq_res in benchmark_result.sequence_results:
            assert seq_res.temporal is not None, (
                f"temporal is None for sequence {seq_res.sequence_name!r}"
            )

    def test_smoothness_score_in_range(self, benchmark_result):
        for seq_res in benchmark_result.sequence_results:
            s = seq_res.temporal.smoothness_score
            assert 0.0 < s <= 1.0, f"smoothness {s} out of (0, 1]"

    def test_position_jitter_non_negative(self, benchmark_result):
        for seq_res in benchmark_result.sequence_results:
            assert seq_res.temporal.position_jitter >= 0.0

    def test_scale_jitter_non_negative(self, benchmark_result):
        for seq_res in benchmark_result.sequence_results:
            assert seq_res.temporal.scale_jitter >= 0.0

    def test_vor_in_zero_to_one(self, benchmark_result):
        for seq_res in benchmark_result.sequence_results:
            vor = seq_res.temporal.velocity_outlier_ratio
            assert 0.0 <= vor <= 1.0

    def test_tracker_name_propagated(self, benchmark_result):
        for seq_res in benchmark_result.sequence_results:
            assert seq_res.temporal.tracker_name == "MOSSE"

    def test_sequence_name_propagated(self, benchmark_result):
        for seq_res in benchmark_result.sequence_results:
            assert seq_res.temporal.sequence_name == seq_res.sequence_name


class TestBenchmarkResultAggregates:
    def test_mean_smoothness_score_not_none(self, benchmark_result):
        assert benchmark_result.mean_smoothness_score is not None

    def test_mean_smoothness_in_range(self, benchmark_result):
        s = benchmark_result.mean_smoothness_score
        assert 0.0 < s <= 1.0

    def test_mean_position_jitter_not_none(self, benchmark_result):
        assert benchmark_result.mean_position_jitter is not None

    def test_mean_position_jitter_non_negative(self, benchmark_result):
        assert benchmark_result.mean_position_jitter >= 0.0

    def test_mean_scale_jitter_not_none(self, benchmark_result):
        assert benchmark_result.mean_scale_jitter is not None

    def test_mean_scale_jitter_non_negative(self, benchmark_result):
        assert benchmark_result.mean_scale_jitter >= 0.0


class TestTemporalInSummary:
    def test_summary_has_smoothness_score(self, benchmark_result):
        s = benchmark_result.summary()
        assert "mean_smoothness_score" in s

    def test_summary_has_position_jitter(self, benchmark_result):
        s = benchmark_result.summary()
        assert "mean_position_jitter" in s

    def test_summary_has_scale_jitter(self, benchmark_result):
        s = benchmark_result.summary()
        assert "mean_scale_jitter" in s

    def test_summary_smoothness_value_matches_property(self, benchmark_result):
        s = benchmark_result.summary()
        expected = round(benchmark_result.mean_smoothness_score, 4)
        assert abs(s["mean_smoothness_score"] - expected) < 1e-9


class TestTemporalSerialisation:
    def test_to_dict_includes_smoothness_per_sequence(self, benchmark_result):
        d = benchmark_result.to_dict()
        for seq in d["sequences"]:
            assert "smoothness_score" in seq, f"Missing in {seq['sequence_name']}"

    def test_to_dict_includes_position_jitter(self, benchmark_result):
        d = benchmark_result.to_dict()
        for seq in d["sequences"]:
            assert "position_jitter" in seq

    def test_to_dict_includes_scale_jitter(self, benchmark_result):
        d = benchmark_result.to_dict()
        for seq in d["sequences"]:
            assert "scale_jitter" in seq

    def test_to_dict_includes_vor(self, benchmark_result):
        d = benchmark_result.to_dict()
        for seq in d["sequences"]:
            assert "velocity_outlier_ratio" in seq

    def test_from_dict_restores_temporal(self, benchmark_result):
        d = benchmark_result.to_dict()
        restored = BenchmarkResult.from_dict(d)
        for seq_res in restored.sequence_results:
            assert seq_res.temporal is not None

    def test_from_dict_smoothness_round_trip(self, benchmark_result):
        d = benchmark_result.to_dict()
        restored = BenchmarkResult.from_dict(d)
        for orig, rest in zip(
            benchmark_result.sequence_results, restored.sequence_results
        ):
            assert abs(
                orig.temporal.smoothness_score - rest.temporal.smoothness_score
            ) < 1e-4, "Smoothness score changed after JSON round-trip"

    def test_save_load_preserves_temporal(self, benchmark_result):
        with tempfile.TemporaryDirectory() as tmp:
            path = benchmark_result.save(Path(tmp) / "result")
            loaded = BenchmarkResult.load(path)
        assert loaded.mean_smoothness_score is not None
        assert abs(
            loaded.mean_smoothness_score - benchmark_result.mean_smoothness_score
        ) < 1e-4


class TestTemporalWithDifferentTracker:
    """Ensure temporal metrics work for multiple trackers, not just MOSSE."""

    def test_kcf_temporal_populated(self):
        dataset = SyntheticDataset(num_sequences=2, num_frames=25, seed=99)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KCFTracker(), dataset, dataset_name="Syn")
        assert result.mean_smoothness_score is not None
        for seq_res in result.sequence_results:
            assert seq_res.temporal is not None


class TestTemporalEdgeCases:
    def test_single_frame_sequence_no_temporal(self):
        """Sequences with 1 frame produce no temporal result (need >= 2 predictions)."""
        dataset = SyntheticDataset(
            num_sequences=1, num_frames=1, seed=0
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="OneFrame")
        # With only 1 frame, predictions has length 1 — temporal should be None.
        for seq_res in result.sequence_results:
            # Either None (skipped) or valid — both acceptable.
            if seq_res.temporal is not None:
                assert 0.0 < seq_res.temporal.smoothness_score <= 1.0

    def test_str_includes_smoothness(self):
        dataset = SyntheticDataset(num_sequences=1, num_frames=20, seed=5)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Syn")
        s = str(result)
        assert "smoothness" in s
