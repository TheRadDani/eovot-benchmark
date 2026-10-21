"""Tests for VotEaoCalculator and VotEaoResult."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.vot_eao import VotEaoCalculator, VotEaoResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def perfect_ious(n: int = 200) -> np.ndarray:
    return np.ones(n, dtype=np.float64)


def zero_ious(n: int = 200) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def failing_ious(n: int = 200, fail_start: int = 50, fail_end: int = 60) -> np.ndarray:
    arr = np.ones(n, dtype=np.float64)
    arr[fail_start:fail_end] = 0.05
    return arr


# ---------------------------------------------------------------------------
# apply_penalty
# ---------------------------------------------------------------------------

class TestApplyPenalty:
    def test_no_failures_unchanged(self):
        calc = VotEaoCalculator()
        ious = perfect_ious(100)
        result = calc.apply_penalty(ious)
        np.testing.assert_array_almost_equal(result, ious)

    def test_penalty_window_zeroed(self):
        calc = VotEaoCalculator(failure_threshold=0.1, penalty_frames=3)
        ious = np.array([1.0, 1.0, 0.05, 1.0, 1.0, 1.0, 1.0])
        result = calc.apply_penalty(ious)
        # Failure at index 2; penalty zeroes indices 3, 4, 5
        assert result[2] == pytest.approx(0.05)
        assert result[3] == pytest.approx(0.0)
        assert result[4] == pytest.approx(0.0)
        assert result[5] == pytest.approx(0.0)
        assert result[6] == pytest.approx(1.0)

    def test_multiple_failures(self):
        calc = VotEaoCalculator(failure_threshold=0.1, penalty_frames=2)
        ious = np.array([1.0, 0.05, 1.0, 1.0, 0.05, 1.0, 1.0, 1.0])
        result = calc.apply_penalty(ious)
        # First failure at 1 → zeros 2, 3; second failure at 4 → zeros 5, 6
        assert result[1] == pytest.approx(0.05)
        assert result[2] == pytest.approx(0.0)
        assert result[3] == pytest.approx(0.0)
        assert result[4] == pytest.approx(0.05)
        assert result[5] == pytest.approx(0.0)
        assert result[6] == pytest.approx(0.0)
        assert result[7] == pytest.approx(1.0)

    def test_penalty_clipped_at_end(self):
        calc = VotEaoCalculator(failure_threshold=0.1, penalty_frames=10)
        ious = np.array([1.0, 0.05, 1.0, 1.0])
        result = calc.apply_penalty(ious)
        assert len(result) == 4
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.05)
        assert result[2] == pytest.approx(0.0)
        assert result[3] == pytest.approx(0.0)

    def test_empty_array(self):
        calc = VotEaoCalculator()
        result = calc.apply_penalty(np.array([]))
        assert len(result) == 0

    def test_returns_copy(self):
        calc = VotEaoCalculator()
        ious = np.array([1.0, 0.05, 1.0])
        result = calc.apply_penalty(ious)
        result[0] = 999.0
        assert ious[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# overlap_curve_for_sequences
# ---------------------------------------------------------------------------

class TestOverlapCurve:
    def test_perfect_sequences(self):
        calc = VotEaoCalculator(eao_low=1, eao_high=50)
        seqs = [perfect_ious(100), perfect_ious(80)]
        curve = calc.overlap_curve_for_sequences(seqs)
        assert curve.shape == (50,)
        np.testing.assert_array_almost_equal(curve, np.ones(50))

    def test_length_matches_eao_high(self):
        calc = VotEaoCalculator(eao_low=10, eao_high=100)
        seqs = [perfect_ious(200)]
        curve = calc.overlap_curve_for_sequences(seqs)
        assert len(curve) == 100

    def test_short_sequence_does_not_fill_curve(self):
        calc = VotEaoCalculator(eao_low=1, eao_high=20)
        short = np.ones(5, dtype=np.float64)
        curve = calc.overlap_curve_for_sequences([short])
        # Positions 0-4 have data; 5-19 are zero (no contributing sequence)
        assert curve[4] == pytest.approx(1.0)
        assert curve[5] == pytest.approx(0.0)

    def test_empty_sequence_list(self):
        calc = VotEaoCalculator(eao_low=1, eao_high=50)
        curve = calc.overlap_curve_for_sequences([])
        np.testing.assert_array_equal(curve, np.zeros(50))


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------

class TestCompute:
    def test_perfect_tracker_high_eao(self):
        calc = VotEaoCalculator(eao_low=1, eao_high=50)
        seq_ious = {f"seq{i}": perfect_ious(200) for i in range(5)}
        result = calc.compute(seq_ious, tracker_name="Perfect")
        assert isinstance(result, VotEaoResult)
        assert result.eao > 0.95
        assert result.total_failures == 0
        assert result.sequence_count == 5

    def test_zero_tracker_zero_eao(self):
        calc = VotEaoCalculator(eao_low=1, eao_high=50)
        seq_ious = {f"seq{i}": zero_ious(200) for i in range(3)}
        result = calc.compute(seq_ious, tracker_name="Worst")
        assert result.eao == pytest.approx(0.0)

    def test_failing_tracker_lower_eao_than_perfect(self):
        calc = VotEaoCalculator(eao_low=1, eao_high=100)
        perfect = calc.compute(
            {f"s{i}": perfect_ious(200) for i in range(4)}, tracker_name="Good"
        )
        failing = calc.compute(
            {f"s{i}": failing_ious(200, 30, 50) for i in range(4)},
            tracker_name="Bad",
        )
        assert perfect.eao > failing.eao

    def test_tracker_name_stored(self):
        calc = VotEaoCalculator()
        result = calc.compute({}, tracker_name="MyTracker")
        assert result.tracker_name == "MyTracker"

    def test_empty_sequences_graceful(self):
        calc = VotEaoCalculator()
        result = calc.compute({}, tracker_name="Empty")
        assert result.eao == pytest.approx(0.0)
        assert result.sequence_count == 0
        assert result.total_failures == 0

    def test_failure_count_correct(self):
        calc = VotEaoCalculator(failure_threshold=0.1, penalty_frames=5)
        # One failure per sequence, 3 sequences
        seq_ious = {f"s{i}": failing_ious(100, 40, 45) for i in range(3)}
        result = calc.compute(seq_ious, tracker_name="T")
        assert result.total_failures == 3

    def test_mean_failures_per_seq(self):
        calc = VotEaoCalculator(failure_threshold=0.1)
        seq_ious = {f"s{i}": failing_ious(100, 40, 45) for i in range(4)}
        result = calc.compute(seq_ious)
        assert result.mean_failures_per_seq == pytest.approx(1.0)

    def test_overlap_curve_shape(self):
        calc = VotEaoCalculator(eao_low=10, eao_high=200)
        seq_ious = {"s1": perfect_ious(300)}
        result = calc.compute(seq_ious)
        assert result.overlap_curve.shape == (200,)

    def test_eao_low_high_stored(self):
        calc = VotEaoCalculator(eao_low=50, eao_high=150)
        result = calc.compute({})
        assert result.eao_low == 50
        assert result.eao_high == 150

    def test_penalty_reduces_eao(self):
        """A tracker with failures should score lower EAO than the same IoU without failure."""
        calc = VotEaoCalculator(eao_low=1, eao_high=50, penalty_frames=10)
        # 30% of frames are failures
        low_iou = np.full(100, 0.05)
        result_fail = calc.compute({"s": low_iou}, tracker_name="bad")
        result_ok = calc.compute({"s": np.full(100, 0.5)}, tracker_name="ok")
        assert result_ok.eao > result_fail.eao


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_sorted_descending(self):
        calc = VotEaoCalculator(eao_low=1, eao_high=50)
        tracker_ious = {
            "Good": {f"s{i}": perfect_ious(200) for i in range(3)},
            "Bad":  {f"s{i}": np.full(200, 0.3) for i in range(3)},
        }
        results = calc.compare(tracker_ious)
        assert len(results) == 2
        assert results[0].tracker_name == "Good"
        assert results[1].tracker_name == "Bad"
        assert results[0].eao >= results[1].eao

    def test_single_tracker(self):
        calc = VotEaoCalculator(eao_low=1, eao_high=50)
        results = calc.compare({"Solo": {"s": perfect_ious(100)}})
        assert len(results) == 1
        assert results[0].tracker_name == "Solo"

    def test_empty_dict(self):
        calc = VotEaoCalculator()
        results = calc.compare({})
        assert results == []


# ---------------------------------------------------------------------------
# Integration: __str__ formatting
# ---------------------------------------------------------------------------

class TestStr:
    def test_str_contains_eao(self):
        calc = VotEaoCalculator(eao_low=1, eao_high=50)
        result = calc.compute({"s": perfect_ious(100)}, tracker_name="T")
        s = str(result)
        assert "EAO" in s
        assert "T" in s
