"""Unit tests for VOT-standard evaluation metrics (EAO, Robustness, AR).

Covers:
- Failure detection edge cases
- Subsequence extraction with burn-in
- Accuracy computation (excludes failure frames)
- Robustness (failure rate normalization)
- EAO computation (curve shape, scalar window, perfect/zero tracking)
- VOTMetricsEngine end-to-end
- AR score behaviour
- compare() multi-tracker ranking
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.vot_metrics import (
    VOTMetricsEngine,
    VOTResult,
    compute_accuracy,
    compute_eao,
    compute_robustness,
    detect_failures,
    extract_subsequences,
)


# ---------------------------------------------------------------------------
# detect_failures
# ---------------------------------------------------------------------------


class TestDetectFailures:
    def test_no_failures(self):
        iou = np.array([0.8, 0.7, 0.9, 0.6])
        assert detect_failures(iou) == []

    def test_single_failure_at_boundary(self):
        iou = np.array([0.8, 0.0, 0.7])
        assert detect_failures(iou) == [1]

    def test_first_frame_failure(self):
        iou = np.array([0.0, 0.8, 0.7])
        assert detect_failures(iou) == [0]

    def test_last_frame_failure(self):
        iou = np.array([0.8, 0.7, 0.0])
        assert detect_failures(iou) == [2]

    def test_all_failures(self):
        iou = np.zeros(5)
        assert detect_failures(iou) == [0, 1, 2, 3, 4]

    def test_custom_threshold(self):
        iou = np.array([0.8, 0.05, 0.7])
        assert detect_failures(iou, threshold=0.1) == [1]

    def test_empty_sequence(self):
        assert detect_failures(np.array([])) == []


# ---------------------------------------------------------------------------
# extract_subsequences
# ---------------------------------------------------------------------------


class TestExtractSubsequences:
    def test_no_failures_returns_full_sequence(self):
        iou = np.array([0.8, 0.7, 0.9])
        subs = extract_subsequences(iou)
        assert len(subs) == 1
        np.testing.assert_array_equal(subs[0], iou)

    def test_failure_at_end(self):
        iou = np.array([0.8, 0.7, 0.0])
        subs = extract_subsequences(iou, burn_in=0)
        assert len(subs) == 1
        np.testing.assert_array_almost_equal(subs[0], np.array([0.8, 0.7]))

    def test_failure_then_burn_in_then_recovery(self):
        # [good, failure, <5 burn-in frames>, good, good]
        iou = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.6])
        # burn_in=5: skip frame 1 (failure) + 5 frames → resume at index 7
        subs = extract_subsequences(iou, burn_in=5)
        assert len(subs) == 2
        np.testing.assert_array_almost_equal(subs[0], np.array([0.8]))
        np.testing.assert_array_almost_equal(subs[1], np.array([0.6]))

    def test_empty_sequence(self):
        subs = extract_subsequences(np.array([]))
        assert subs == []

    def test_all_failures(self):
        iou = np.zeros(5)
        subs = extract_subsequences(iou, burn_in=0)
        # Every frame is a failure; each failure ejects immediately
        # First failure at 0: subseq length 0 (not added), skip to 1
        # And so on — no subsequences with positive length
        assert all(len(s) == 0 for s in subs)


# ---------------------------------------------------------------------------
# compute_accuracy
# ---------------------------------------------------------------------------


class TestComputeAccuracy:
    def test_perfect_tracking(self):
        iou = np.ones(10)
        assert compute_accuracy(iou) == pytest.approx(1.0)

    def test_failures_excluded(self):
        iou = np.array([0.8, 0.0, 0.6, 0.0, 0.7])
        expected = np.mean([0.8, 0.6, 0.7])
        assert compute_accuracy(iou) == pytest.approx(expected)

    def test_all_failures_returns_zero(self):
        iou = np.zeros(5)
        assert compute_accuracy(iou) == 0.0

    def test_single_valid_frame(self):
        iou = np.array([0.0, 0.0, 0.65, 0.0])
        assert compute_accuracy(iou) == pytest.approx(0.65)

    def test_custom_threshold(self):
        iou = np.array([0.8, 0.05, 0.7])
        # With threshold=0.1, frame 1 (0.05) is excluded
        expected = np.mean([0.8, 0.7])
        assert compute_accuracy(iou, failure_threshold=0.1) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_robustness
# ---------------------------------------------------------------------------


class TestComputeRobustness:
    def test_no_failures(self):
        iou = np.ones(100)
        rob, n_fail = compute_robustness(iou)
        assert n_fail == 0
        assert rob == pytest.approx(0.0)

    def test_single_failure_in_100_frames(self):
        iou = np.ones(100)
        iou[50] = 0.0
        rob, n_fail = compute_robustness(iou, burn_in=5)
        assert n_fail == 1
        assert rob == pytest.approx(1.0)  # 1 failure / 100 frames * 100

    def test_multiple_failures(self):
        # Force two failures at frames 20 and 70 (after burn-in of 5)
        iou = np.ones(100)
        iou[20] = 0.0
        iou[70] = 0.0
        _, n_fail = compute_robustness(iou, burn_in=5)
        assert n_fail == 2

    def test_empty_sequence(self):
        rob, n_fail = compute_robustness(np.array([]))
        assert n_fail == 0
        assert rob == pytest.approx(0.0)

    def test_short_sequence_with_failure(self):
        iou = np.array([0.8, 0.0, 0.7])
        _, n_fail = compute_robustness(iou, burn_in=0)
        assert n_fail == 1


# ---------------------------------------------------------------------------
# compute_eao
# ---------------------------------------------------------------------------


class TestComputeEAO:
    def test_empty_sequences_returns_zeros(self):
        eao, curve = compute_eao([])
        assert eao == 0.0
        assert curve.shape == (356,)
        assert np.all(curve == 0.0)

    def test_curve_shape(self):
        seqs = [np.ones(200)] * 3
        _, curve = compute_eao(seqs, max_length=356)
        assert curve.shape == (356,)

    def test_perfect_tracking_eao_near_one(self):
        # All-ones IoU sequences, much longer than integration window
        seqs = [np.ones(500)] * 10
        eao, _ = compute_eao(seqs, min_length=100, max_length=356)
        assert eao == pytest.approx(1.0, abs=0.01)

    def test_zero_tracking_eao_is_zero(self):
        # Instant failure every time
        seqs = [np.zeros(500)] * 10
        eao, _ = compute_eao(seqs, min_length=100, max_length=356)
        assert eao == pytest.approx(0.0, abs=0.01)

    def test_short_sequences_below_window(self):
        # All subsequences shorter than min_length → EAO window is empty
        seqs = [np.ones(50)] * 10
        eao, _ = compute_eao(seqs, min_length=100, max_length=200)
        assert eao == pytest.approx(0.0, abs=0.01)

    def test_custom_max_length(self):
        seqs = [np.ones(200)] * 5
        eao, curve = compute_eao(seqs, min_length=10, max_length=50)
        assert curve.shape == (50,)
        assert eao > 0.5

    def test_eao_increases_with_tracking_quality(self):
        good_seqs = [np.full(300, 0.8)] * 5
        bad_seqs = [np.full(300, 0.2)] * 5
        eao_good, _ = compute_eao(good_seqs, min_length=100, max_length=200)
        eao_bad, _ = compute_eao(bad_seqs, min_length=100, max_length=200)
        assert eao_good > eao_bad


# ---------------------------------------------------------------------------
# VOTMetricsEngine
# ---------------------------------------------------------------------------


class TestVOTMetricsEngine:
    def setup_method(self):
        # Use a short window for test speed
        self.engine = VOTMetricsEngine(
            burn_in=5,
            eao_min_length=10,
            eao_max_length=50,
        )

    def test_evaluate_empty_returns_zero_result(self):
        result = self.engine.evaluate([])
        assert isinstance(result, VOTResult)
        assert result.eao == 0.0
        assert result.accuracy == 0.0
        assert result.robustness == 0.0
        assert result.failure_count == 0
        assert result.sequence_length == 0

    def test_evaluate_perfect_tracking(self):
        seqs = [np.ones(100)] * 5
        result = self.engine.evaluate(seqs)
        assert result.accuracy == pytest.approx(1.0, abs=0.01)
        assert result.failure_count == 0
        assert result.robustness == pytest.approx(0.0)
        assert result.eao > 0.8
        assert result.sequence_length == 500

    def test_evaluate_zero_tracking(self):
        seqs = [np.zeros(100)] * 5
        result = self.engine.evaluate(seqs)
        assert result.accuracy == pytest.approx(0.0)
        assert result.failure_count > 0

    def test_evaluate_mixed(self):
        seqs = [np.ones(100), np.zeros(100)]
        result = self.engine.evaluate(seqs)
        assert 0.0 < result.accuracy <= 1.0
        assert result.sequence_length == 200

    def test_eao_curve_is_ndarray(self):
        seqs = [np.ones(60)]
        result = self.engine.evaluate(seqs)
        assert isinstance(result.eao_curve, np.ndarray)
        assert result.eao_curve.shape == (50,)  # eao_max_length

    def test_str_representation(self):
        seqs = [np.ones(60)]
        result = self.engine.evaluate(seqs)
        s = str(result)
        assert "EAO" in s
        assert "accuracy" in s

    def test_ar_score_perfect(self):
        seqs = [np.ones(100)] * 5
        result = self.engine.evaluate(seqs)
        ar = self.engine.ar_score(result)
        assert ar == pytest.approx(1.0, abs=0.05)

    def test_ar_score_zero_accuracy_tracker(self):
        # All failures (IoU = 0 every frame): accuracy = 0 → AR = 0
        result = self.engine.evaluate([np.zeros(100)] * 5)
        ar = self.engine.ar_score(result)
        assert ar == pytest.approx(0.0, abs=0.01)

    def test_ar_score_in_range(self):
        seqs = [np.full(100, 0.5)] * 3
        result = self.engine.evaluate(seqs)
        ar = self.engine.ar_score(result)
        assert 0.0 <= ar <= 1.0

    def test_compare_sorts_by_eao(self):
        tracker_results = {
            "fast_but_bad": [np.full(100, 0.2)] * 3,
            "perfect": [np.ones(100)] * 3,
            "mediocre": [np.full(100, 0.5)] * 3,
        }
        ranking = self.engine.compare(tracker_results)
        names = list(ranking.keys())
        assert names[0] == "perfect"
        assert names[-1] == "fast_but_bad"

    def test_compare_returns_vot_results(self):
        tracker_results = {"t1": [np.ones(60)], "t2": [np.zeros(60)]}
        ranking = self.engine.compare(tracker_results)
        for name, result in ranking.items():
            assert isinstance(result, VOTResult)
