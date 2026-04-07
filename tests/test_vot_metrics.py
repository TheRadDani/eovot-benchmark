"""Tests for the VOT re-initialisation protocol and EAO computation."""

from __future__ import annotations

import pytest
import numpy as np

from eovot.metrics.vot_metrics import (
    simulate_reinit_overlaps,
    compute_sequence_accuracy,
    compute_eao,
    VOTEvaluator,
    VOTMetrics,
    VOTSequenceResult,
    FAILURE_THRESHOLD,
    REINIT_DELAY,
)


# ---------------------------------------------------------------------------
# simulate_reinit_overlaps
# ---------------------------------------------------------------------------


class TestSimulateReinitOverlaps:
    def test_no_failures_unchanged(self):
        overlaps = [0.8, 0.7, 0.6, 0.5]
        result, n_failures = simulate_reinit_overlaps(overlaps)
        assert n_failures == 0
        assert result == overlaps

    def test_single_failure_zeroed(self):
        # Failure at index 2; REINIT_DELAY=5 means frames 2..6 become 0
        overlaps = [0.8, 0.7, 0.0, 0.6, 0.5, 0.4, 0.3, 0.2]
        result, n_failures = simulate_reinit_overlaps(overlaps, reinit_delay=2)
        assert n_failures == 1
        # Frame 2 is the failure, frames 2..4 (delay=2) are zeroed
        assert result[0] == 0.8
        assert result[1] == 0.7
        assert result[2] == 0.0
        assert result[3] == 0.0
        assert result[4] == 0.0
        # Frames after reinit window are untouched
        assert result[5] == 0.4
        assert result[6] == 0.3

    def test_multiple_failures_counted(self):
        # Two zero-IoU events
        overlaps = [0.5, 0.0, 0.6, 0.7, 0.0, 0.4]
        result, n_failures = simulate_reinit_overlaps(overlaps, reinit_delay=0)
        assert n_failures == 2

    def test_empty_sequence(self):
        result, n = simulate_reinit_overlaps([])
        assert result == []
        assert n == 0

    def test_failure_at_end_does_not_overflow(self):
        overlaps = [0.5, 0.3, 0.0]  # failure at last frame (index 2)
        result, n = simulate_reinit_overlaps(overlaps, reinit_delay=5)
        assert n == 1
        assert len(result) == 3
        # Only the failure frame (index 2) is zeroed; earlier frames are intact
        assert result[0] == 0.5
        assert result[1] == 0.3
        assert result[2] == 0.0

    def test_consecutive_failures(self):
        overlaps = [0.0, 0.0, 0.5]
        result, n = simulate_reinit_overlaps(overlaps, reinit_delay=0)
        # First failure zeroes frame 0; next iteration is frame 1 which is also 0
        assert n == 2

    def test_custom_threshold(self):
        overlaps = [0.8, 0.05, 0.6]
        result, n = simulate_reinit_overlaps(overlaps, failure_threshold=0.1, reinit_delay=0)
        assert n == 1
        assert result[1] == 0.0


# ---------------------------------------------------------------------------
# compute_sequence_accuracy
# ---------------------------------------------------------------------------


class TestComputeSequenceAccuracy:
    def test_all_tracked(self):
        overlaps = [0.8, 0.6, 0.7]
        acc = compute_sequence_accuracy(overlaps)
        assert abs(acc - np.mean([0.8, 0.6, 0.7])) < 1e-9

    def test_all_zero_returns_zero(self):
        assert compute_sequence_accuracy([0.0, 0.0, 0.0]) == 0.0

    def test_mixed_excludes_zeros(self):
        overlaps = [0.8, 0.0, 0.6]
        acc = compute_sequence_accuracy(overlaps)
        assert abs(acc - np.mean([0.8, 0.6])) < 1e-9

    def test_empty_returns_zero(self):
        assert compute_sequence_accuracy([]) == 0.0


# ---------------------------------------------------------------------------
# compute_eao
# ---------------------------------------------------------------------------


class TestComputeEAO:
    def test_empty_input_returns_zero(self):
        assert compute_eao([]) == 0.0

    def test_perfect_tracker_eao_near_one(self):
        # 200 frames of perfect overlap
        seqs = [[1.0] * 200 for _ in range(5)]
        eao = compute_eao(seqs, low=50, high=150)
        assert abs(eao - 1.0) < 1e-6

    def test_zero_tracker_eao_zero(self):
        seqs = [[0.0] * 200 for _ in range(5)]
        eao = compute_eao(seqs, low=50, high=150)
        assert eao == 0.0

    def test_eao_in_valid_range(self):
        rng = np.random.default_rng(42)
        seqs = [list(rng.uniform(0.3, 0.8, 300)) for _ in range(10)]
        eao = compute_eao(seqs, low=100, high=356)
        assert 0.0 <= eao <= 1.0

    def test_short_sequences_padded(self):
        # Sequences shorter than `high` should be padded, not crash
        seqs = [[0.5] * 50, [0.6] * 80]
        eao = compute_eao(seqs, low=10, high=100)
        assert 0.0 <= eao <= 1.0

    def test_single_sequence(self):
        seqs = [[0.7] * 200]
        eao = compute_eao(seqs, low=50, high=150)
        assert abs(eao - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# VOTEvaluator
# ---------------------------------------------------------------------------


class TestVOTEvaluator:
    def test_evaluate_sequence_perfect_tracker(self):
        evaluator = VOTEvaluator()
        overlaps = [0.9] * 100
        result = evaluator.evaluate_sequence("seq1", overlaps)
        assert result.failures == 0
        assert abs(result.accuracy - 0.9) < 1e-9
        assert result.sequence_name == "seq1"

    def test_evaluate_sequence_with_failure(self):
        evaluator = VOTEvaluator(reinit_delay=3)
        # Failure at frame 10
        overlaps = [0.8] * 10 + [0.0] + [0.7] * 89
        result = evaluator.evaluate_sequence("seq", overlaps)
        assert result.failures == 1
        # Frames 10..13 are zeroed; rest tracked at 0.8 or 0.7
        assert result.accuracy > 0.0
        assert result.accuracy < 0.9

    def test_evaluate_empty_sequence_raises(self):
        evaluator = VOTEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate([], [])

    def test_evaluate_mismatched_lengths_raises(self):
        evaluator = VOTEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate(["a", "b"], [[0.5]])

    def test_evaluate_returns_vot_metrics(self):
        evaluator = VOTEvaluator()
        names = ["seq1", "seq2"]
        overlaps = [[0.8] * 150, [0.7] * 200]
        metrics = evaluator.evaluate(names, overlaps)
        assert isinstance(metrics, VOTMetrics)
        assert len(metrics.sequence_results) == 2

    def test_evaluate_metrics_range(self):
        evaluator = VOTEvaluator()
        rng = np.random.default_rng(7)
        names = [f"seq{i}" for i in range(10)]
        overlaps = [list(rng.uniform(0.2, 0.9, 200)) for _ in range(10)]
        metrics = evaluator.evaluate(names, overlaps)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.robustness >= 0.0
        assert 0.0 <= metrics.eao <= 1.0

    def test_summary_dict_keys(self):
        evaluator = VOTEvaluator()
        metrics = evaluator.evaluate(["s"], [[0.5] * 200])
        s = metrics.summary()
        assert "accuracy" in s
        assert "robustness" in s
        assert "eao" in s
        assert "n_sequences" in s

    def test_str_representation(self):
        evaluator = VOTEvaluator()
        metrics = evaluator.evaluate(["s"], [[0.6] * 150])
        text = str(metrics)
        assert "EAO" in text
        assert "Accuracy" in text
        assert "Robustness" in text

    def test_robustness_zero_for_perfect_tracker(self):
        evaluator = VOTEvaluator()
        names = [f"s{i}" for i in range(5)]
        overlaps = [[0.9] * 100 for _ in range(5)]
        metrics = evaluator.evaluate(names, overlaps)
        assert metrics.robustness == 0.0

    def test_evaluate_single_frame_sequences(self):
        """Single-frame sequences should not crash."""
        evaluator = VOTEvaluator()
        metrics = evaluator.evaluate(["a", "b"], [[0.5], [0.8]])
        assert isinstance(metrics, VOTMetrics)


# ---------------------------------------------------------------------------
# Integration: BenchmarkEngine with vot_protocol
# ---------------------------------------------------------------------------


class TestBenchmarkEngineVOTIntegration:
    """Smoke-tests that BenchmarkEngine correctly populates vot_metrics."""

    def _make_engine_and_result(self):
        import numpy as np
        from unittest.mock import MagicMock
        from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult, SequenceResult
        from eovot.profiling.profiler import ProfilingResult

        prof = ProfilingResult(
            tracker_name="mock",
            frame_count=50,
            fps=100.0,
            latency_mean_ms=10.0,
            latency_std_ms=1.0,
            latency_p95_ms=12.0,
            peak_memory_mb=50.0,
        )
        sr = SequenceResult(
            sequence_name="mock_seq",
            ious=np.array([0.7] * 50),
            profiling=prof,
        )
        result = BenchmarkResult(
            tracker_name="mock",
            dataset_name="mock_dataset",
            sequence_results=[sr],
        )
        return result

    def test_vot_metrics_none_by_default(self):
        result = self._make_engine_and_result()
        assert result.vot_metrics is None

    def test_summary_includes_vot_when_set(self):
        from eovot.metrics.vot_metrics import VOTEvaluator
        result = self._make_engine_and_result()
        names = [r.sequence_name for r in result.sequence_results]
        overlaps = [r.ious.tolist() for r in result.sequence_results]
        result.vot_metrics = VOTEvaluator().evaluate(names, overlaps)
        s = result.summary()
        assert "vot_eao" in s
        assert "vot_accuracy" in s
        assert "vot_robustness" in s

    def test_to_dict_includes_vot_section(self):
        from eovot.metrics.vot_metrics import VOTEvaluator
        result = self._make_engine_and_result()
        names = [r.sequence_name for r in result.sequence_results]
        overlaps = [r.ious.tolist() for r in result.sequence_results]
        result.vot_metrics = VOTEvaluator().evaluate(names, overlaps)
        d = result.to_dict()
        assert "vot" in d
        assert "eao" in d["vot"]
