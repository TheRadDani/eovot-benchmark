"""Tests for the VOT re-initialization protocol evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.reinit_protocol import (
    ReinitEvaluator,
    ReinitSequenceResult,
    ReinitBenchmarkResult,
    compare_reinit_results,
)
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=40)


@pytest.fixture
def mosse_tracker():
    return MOSSETracker()


# ---------------------------------------------------------------------------
# ReinitEvaluator._build_gap_mask
# ---------------------------------------------------------------------------

class TestGapMask:
    def test_no_failures(self):
        mask = ReinitEvaluator._build_gap_mask(10, [], skip_frames=5)
        assert not mask.any()

    def test_single_failure_at_frame_2(self):
        mask = ReinitEvaluator._build_gap_mask(10, [2], skip_frames=3)
        # Gap frames: 3, 4, 5
        assert not mask[2]
        assert mask[3]
        assert mask[4]
        assert mask[5]
        assert not mask[6]

    def test_failure_near_end_clipped(self):
        mask = ReinitEvaluator._build_gap_mask(10, [8], skip_frames=5)
        # Gap would be 9, 10, 11, 12, 13 — but only 9 exists
        assert mask[9]
        assert mask.sum() == 1

    def test_multiple_failures(self):
        mask = ReinitEvaluator._build_gap_mask(20, [3, 10], skip_frames=2)
        assert mask[4]
        assert mask[5]
        assert mask[11]
        assert mask[12]
        assert not mask[3]
        assert not mask[10]


# ---------------------------------------------------------------------------
# ReinitEvaluator.evaluate_sequence
# ---------------------------------------------------------------------------

class TestEvaluateSequence:
    def test_returns_reinit_sequence_result(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        seq = synthetic_dataset[0]
        result = evaluator.evaluate_sequence(mosse_tracker, seq)
        assert isinstance(result, ReinitSequenceResult)

    def test_ious_shape(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        seq = synthetic_dataset[0]
        result = evaluator.evaluate_sequence(mosse_tracker, seq)
        assert result.ious.shape == (min(len(seq), len(seq.ground_truth)),)

    def test_ious_in_range(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        seq = synthetic_dataset[0]
        result = evaluator.evaluate_sequence(mosse_tracker, seq)
        assert result.ious.min() >= 0.0
        assert result.ious.max() <= 1.0 + 1e-6

    def test_eao_equals_mean_ious(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        seq = synthetic_dataset[0]
        result = evaluator.evaluate_sequence(mosse_tracker, seq)
        assert result.eao == pytest.approx(float(result.ious.mean()), abs=1e-8)

    def test_reliability_range(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        seq = synthetic_dataset[0]
        result = evaluator.evaluate_sequence(mosse_tracker, seq)
        assert 0.0 <= result.reliability <= 1.0

    def test_accuracy_active_frames_only(self, mosse_tracker, synthetic_dataset):
        """Accuracy should equal mean IoU over non-gap frames."""
        evaluator = ReinitEvaluator(failure_threshold=0.1, skip_frames=3)
        seq = synthetic_dataset[0]
        result = evaluator.evaluate_sequence(mosse_tracker, seq)

        # Recompute manually
        n = result.num_frames
        gap_mask = ReinitEvaluator._build_gap_mask(n, result.failure_frames, skip_frames=3)
        expected_accuracy = float(result.ious[~gap_mask].mean()) if (~gap_mask).any() else 0.0
        assert result.accuracy == pytest.approx(expected_accuracy, abs=1e-8)

    def test_num_failures_matches_failure_frames(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        seq = synthetic_dataset[0]
        result = evaluator.evaluate_sequence(mosse_tracker, seq)
        assert result.num_failures == len(result.failure_frames)

    def test_failure_frames_within_bounds(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        seq = synthetic_dataset[0]
        result = evaluator.evaluate_sequence(mosse_tracker, seq)
        for f in result.failure_frames:
            assert 0 <= f < result.num_frames

    def test_fps_positive(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        seq = synthetic_dataset[0]
        result = evaluator.evaluate_sequence(mosse_tracker, seq)
        assert result.mean_fps > 0.0

    def test_permissive_threshold_fewer_failures(self, mosse_tracker, synthetic_dataset):
        """Higher threshold ⟹ more or equal failures."""
        seq = synthetic_dataset[0]
        strict = ReinitEvaluator(failure_threshold=0.5).evaluate_sequence(mosse_tracker, seq)
        lenient = ReinitEvaluator(failure_threshold=0.01).evaluate_sequence(mosse_tracker, seq)
        assert strict.num_failures >= lenient.num_failures


# ---------------------------------------------------------------------------
# ReinitEvaluator.evaluate_dataset
# ---------------------------------------------------------------------------

class TestEvaluateDataset:
    def test_returns_benchmark_result(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        result = evaluator.evaluate_dataset(mosse_tracker, synthetic_dataset, "Synthetic")
        assert isinstance(result, ReinitBenchmarkResult)

    def test_sequence_count(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        result = evaluator.evaluate_dataset(mosse_tracker, synthetic_dataset, "Synthetic")
        assert len(result.sequence_results) == 3

    def test_max_sequences(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        result = evaluator.evaluate_dataset(
            mosse_tracker, synthetic_dataset, "Synthetic", max_sequences=2
        )
        assert len(result.sequence_results) == 2

    def test_mean_eao_in_range(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        result = evaluator.evaluate_dataset(mosse_tracker, synthetic_dataset, "Synthetic")
        assert 0.0 <= result.mean_eao <= 1.0

    def test_summary_keys(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        result = evaluator.evaluate_dataset(mosse_tracker, synthetic_dataset, "Synthetic")
        s = result.summary()
        assert "mean_eao" in s
        assert "mean_accuracy" in s
        assert "mean_reliability" in s
        assert "total_failures" in s
        assert "mean_fps" in s

    def test_total_failures_sum(self, mosse_tracker, synthetic_dataset):
        evaluator = ReinitEvaluator()
        result = evaluator.evaluate_dataset(mosse_tracker, synthetic_dataset, "Synthetic")
        expected = sum(r.num_failures for r in result.sequence_results)
        assert result.total_failures == expected


# ---------------------------------------------------------------------------
# compare_reinit_results
# ---------------------------------------------------------------------------

class TestCompareReinitResults:
    def _make_result(self, tracker_name: str, eao: float) -> ReinitBenchmarkResult:
        r = ReinitBenchmarkResult(tracker_name=tracker_name, dataset_name="Synthetic")
        r.sequence_results.append(
            ReinitSequenceResult(
                sequence_name="seq1",
                num_frames=100,
                num_failures=2,
                failure_frames=[10, 50],
                ious=np.full(100, eao),
                accuracy=eao + 0.05,
                reliability=0.02,
                eao=eao,
                mean_fps=100.0,
            )
        )
        return r

    def test_returns_markdown_table(self):
        results = {
            "MOSSE": self._make_result("MOSSE", 0.4),
            "KCF": self._make_result("KCF", 0.5),
        }
        table = compare_reinit_results(results)
        assert "| Tracker |" in table
        assert "KCF" in table
        assert "MOSSE" in table

    def test_sorted_by_eao_descending(self):
        results = {
            "A": self._make_result("A", 0.3),
            "B": self._make_result("B", 0.7),
            "C": self._make_result("C", 0.5),
        }
        table = compare_reinit_results(results)
        lines = [l for l in table.split("\n") if "|" in l and "Tracker" not in l and "---" not in l]
        names = [l.split("|")[1].strip() for l in lines]
        assert names == ["B", "C", "A"]

    def test_empty_results(self):
        table = compare_reinit_results({})
        assert "| Tracker |" in table
