"""Tests for eovot.metrics.robustness."""

import numpy as np
import pytest

from eovot.metrics.robustness import RobustnessAnalyzer, RobustnessResult


@pytest.fixture
def analyzer():
    """Analyzer with burn_in=2 for short synthetic sequences."""
    return RobustnessAnalyzer(
        failure_threshold=0.1,
        recovery_threshold=0.1,
        burn_in_frames=2,
    )


# ---------------------------------------------------------------------------
# detect_failures
# ---------------------------------------------------------------------------

class TestDetectFailures:
    def test_no_failures_all_high(self, analyzer):
        ious = np.array([1.0, 1.0, 0.8, 0.7, 0.9])
        assert analyzer.detect_failures(ious) == []

    def test_single_failure(self, analyzer):
        # burn_in=2 → indices 0,1 ignored; failure at index 3
        ious = np.array([1.0, 1.0, 0.8, 0.05, 0.8])
        failures = analyzer.detect_failures(ious)
        assert failures == [3]

    def test_burn_in_respected(self, analyzer):
        # First 2 frames (burn-in) are sub-threshold but must not count
        ious = np.array([0.0, 0.0, 0.8, 0.9])
        assert analyzer.detect_failures(ious) == []

    def test_two_separate_failures(self, analyzer):
        # failure at 2 (recover at 4), failure again at 6
        ious = np.array([1.0, 1.0, 0.05, 0.05, 0.8, 0.8, 0.05])
        failures = analyzer.detect_failures(ious)
        assert failures == [2, 6]

    def test_failure_at_last_frame(self, analyzer):
        ious = np.array([1.0, 1.0, 0.8, 0.8, 0.05])
        failures = analyzer.detect_failures(ious)
        assert failures == [4]

    def test_no_failure_exactly_at_threshold(self, analyzer):
        # IoU == threshold is NOT a failure (strict less-than)
        ious = np.array([1.0, 1.0, 0.1, 0.1])
        assert analyzer.detect_failures(ious) == []

    def test_entire_sequence_below_threshold(self, analyzer):
        ious = np.zeros(8)
        failures = analyzer.detect_failures(ious)
        # Only one failure event (burn_in=2, first sub-threshold frame is index 2)
        assert len(failures) == 1
        assert failures[0] == 2


# ---------------------------------------------------------------------------
# compute_recovery_lags
# ---------------------------------------------------------------------------

class TestRecoveryLags:
    def test_recovers_quickly(self, analyzer):
        ious = np.array([1.0, 1.0, 0.8, 0.05, 0.05, 0.05, 0.8])
        failures = analyzer.detect_failures(ious)
        lags = analyzer.compute_recovery_lags(ious, failures)
        assert lags == [3]  # frames 4, 5, 6 → lag = 3

    def test_immediate_recovery(self, analyzer):
        ious = np.array([1.0, 1.0, 0.05, 0.8])
        failures = analyzer.detect_failures(ious)
        lags = analyzer.compute_recovery_lags(ious, failures)
        assert lags == [1]  # recovers on very next frame

    def test_never_recovers(self, analyzer):
        ious = np.array([1.0, 1.0, 0.8, 0.05, 0.05, 0.05])
        failures = analyzer.detect_failures(ious)
        lags = analyzer.compute_recovery_lags(ious, failures)
        # len(ious) - failure_frame = 6 - 3 = 3
        assert lags == [3]

    def test_no_failures_empty_lags(self, analyzer):
        ious = np.ones(10)
        failures = analyzer.detect_failures(ious)
        lags = analyzer.compute_recovery_lags(ious, failures)
        assert lags == []


# ---------------------------------------------------------------------------
# compute_eao
# ---------------------------------------------------------------------------

class TestComputeEAO:
    def test_perfect_tracking(self, analyzer):
        ious = np.ones(10)
        assert abs(analyzer.compute_eao(ious) - 1.0) < 1e-9

    def test_zero_tracking(self, analyzer):
        ious = np.zeros(10)
        assert abs(analyzer.compute_eao(ious) - 0.0) < 1e-9

    def test_known_mean(self, analyzer):
        # burn_in=2: mean over indices 2..9 of a ramp
        ious = np.array([0.0, 0.0, 0.2, 0.4, 0.6, 0.8])
        expected = np.mean([0.2, 0.4, 0.6, 0.8])
        assert abs(analyzer.compute_eao(ious) - expected) < 1e-9

    def test_too_short_returns_zero(self, analyzer):
        ious = np.array([1.0])  # shorter than burn_in=2
        assert analyzer.compute_eao(ious) == 0.0

    def test_exactly_burn_in_length_returns_zero(self, analyzer):
        ious = np.array([1.0, 1.0])  # len == burn_in
        assert analyzer.compute_eao(ious) == 0.0


# ---------------------------------------------------------------------------
# survival_curve
# ---------------------------------------------------------------------------

class TestSurvivalCurve:
    def test_shape(self, analyzer):
        ious_list = [np.ones(20), np.ones(20)]
        curve = analyzer.survival_curve(ious_list)
        assert curve.shape == (100,)

    def test_perfect_tracker_all_ones(self, analyzer):
        ious_list = [np.ones(50)]
        curve = analyzer.survival_curve(ious_list)
        np.testing.assert_allclose(curve, 1.0)

    def test_failed_tracker_all_zeros(self, analyzer):
        ious_list = [np.zeros(50)]
        curve = analyzer.survival_curve(ious_list)
        np.testing.assert_allclose(curve, 0.0)

    def test_empty_list(self, analyzer):
        curve = analyzer.survival_curve([])
        assert curve.shape == (100,)
        np.testing.assert_allclose(curve, 0.0)

    def test_custom_n_points(self, analyzer):
        ious_list = [np.ones(30)]
        curve = analyzer.survival_curve(ious_list, n_points=50)
        assert curve.shape == (50,)

    def test_values_in_range(self, analyzer):
        rng = np.random.default_rng(0)
        ious_list = [rng.uniform(0, 1, 40) for _ in range(5)]
        curve = analyzer.survival_curve(ious_list)
        assert np.all(curve >= 0.0)
        assert np.all(curve <= 1.0)


# ---------------------------------------------------------------------------
# analyze_sequence
# ---------------------------------------------------------------------------

class TestAnalyzeSequence:
    def test_returns_dataclass(self, analyzer):
        ious = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        result = analyzer.analyze_sequence(ious, tracker_name="T", sequence_name="S")
        assert isinstance(result, RobustnessResult)
        assert result.tracker_name == "T"
        assert result.sequence_name == "S"

    def test_perfect_tracker_no_failures(self, analyzer):
        ious = np.ones(20)
        result = analyzer.analyze_sequence(ious)
        assert result.num_failures == 0
        assert result.failure_frames == []
        assert result.recovery_lags == []
        assert result.mean_recovery_lag == 0.0
        assert abs(result.eao - 1.0) < 1e-9
        assert abs(result.survival_rate - 1.0) < 1e-9

    def test_known_failure_counted(self, analyzer):
        ious = np.array([1.0, 1.0, 0.8, 0.05, 0.8, 0.9, 0.8])
        result = analyzer.analyze_sequence(ious)
        assert result.num_failures == 1

    def test_survival_rate_partial(self, analyzer):
        # burn_in=2 → 4 active frames; 2 are above threshold
        ious = np.array([1.0, 1.0, 0.8, 0.05, 0.8, 0.05])
        result = analyzer.analyze_sequence(ious)
        assert abs(result.survival_rate - 0.5) < 1e-9

    def test_str_representation(self, analyzer):
        ious = np.ones(10)
        result = analyzer.analyze_sequence(ious, tracker_name="MOSSE", sequence_name="car1")
        s = str(result)
        assert "MOSSE" in s
        assert "car1" in s
        assert "EAO" in s


# ---------------------------------------------------------------------------
# analyze_benchmark
# ---------------------------------------------------------------------------

class TestAnalyzeBenchmark:
    def test_aggregate_keys_present(self, analyzer):
        seq_ious = {
            "seq1": np.ones(10),
            "seq2": np.array([1.0, 1.0, 0.8, 0.05, 0.8, 0.9]),
        }
        out = analyzer.analyze_benchmark(seq_ious, tracker_name="KCF")
        agg = out["aggregate"]
        assert "mean_eao" in agg
        assert "total_failures" in agg
        assert "mean_failures_per_sequence" in agg
        assert "mean_survival_rate" in agg
        assert "mean_recovery_lag_frames" in agg
        assert agg["num_sequences"] == 2
        assert agg["tracker_name"] == "KCF"

    def test_per_sequence_populated(self, analyzer):
        seq_ious = {"s1": np.ones(8), "s2": np.ones(8)}
        out = analyzer.analyze_benchmark(seq_ious)
        assert set(out["per_sequence"].keys()) == {"s1", "s2"}

    def test_empty_input(self, analyzer):
        out = analyzer.analyze_benchmark({})
        assert out["aggregate"]["num_sequences"] == 0
        assert out["aggregate"]["total_failures"] == 0

    def test_all_perfect_zero_failures(self, analyzer):
        seq_ious = {f"s{i}": np.ones(15) for i in range(5)}
        out = analyzer.analyze_benchmark(seq_ious)
        assert out["aggregate"]["total_failures"] == 0
        assert abs(out["aggregate"]["mean_eao"] - 1.0) < 1e-6
