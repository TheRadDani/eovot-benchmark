"""Unit tests for eovot.metrics.robustness."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.robustness import RobustnessAnalyzer, RobustnessResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return RobustnessAnalyzer(failure_threshold=0.1, burn_in_frames=3)


@pytest.fixture
def perfect_ious():
    """Sequence with IoU=1.0 on every frame — no failures."""
    return np.ones(30, dtype=np.float64)


@pytest.fixture
def failing_ious():
    """Sequence with one clear failure at frame 10 recovering at frame 15."""
    ious = np.ones(30, dtype=np.float64)
    ious[10:15] = 0.0   # failure window
    return ious


@pytest.fixture
def never_recovering_ious():
    """Sequence with a failure that never recovers."""
    ious = np.ones(20, dtype=np.float64)
    ious[10:] = 0.0
    return ious


# ---------------------------------------------------------------------------
# detect_failures
# ---------------------------------------------------------------------------

class TestDetectFailures:
    def test_no_failures_when_perfect(self, analyzer, perfect_ious):
        failures = analyzer.detect_failures(perfect_ious)
        assert failures == []

    def test_detects_single_failure(self, analyzer, failing_ious):
        failures = analyzer.detect_failures(failing_ious)
        assert len(failures) == 1
        assert failures[0] == 10

    def test_failure_not_detected_in_burn_in(self, analyzer):
        ious = np.ones(20, dtype=np.float64)
        ious[1] = 0.0   # inside burn-in window (burn_in=3)
        failures = analyzer.detect_failures(ious)
        assert failures == []

    def test_two_separate_failures(self, analyzer):
        ious = np.ones(40, dtype=np.float64)
        ious[5:8] = 0.0    # first failure
        ious[20:23] = 0.0  # second failure
        failures = analyzer.detect_failures(ious)
        assert len(failures) == 2
        assert failures[0] == 5
        assert failures[1] == 20

    def test_continuous_failure_counts_once(self, analyzer):
        ious = np.zeros(20, dtype=np.float64)  # all below threshold
        ious[:3] = 1.0  # burn-in only
        failures = analyzer.detect_failures(ious)
        assert len(failures) == 1  # one failure, not one per frame

    def test_empty_sequence(self, analyzer):
        failures = analyzer.detect_failures(np.array([]))
        assert failures == []

    def test_sequence_shorter_than_burn_in(self, analyzer):
        ious = np.zeros(2, dtype=np.float64)
        failures = analyzer.detect_failures(ious)
        assert failures == []


# ---------------------------------------------------------------------------
# compute_recovery_lags
# ---------------------------------------------------------------------------

class TestComputeRecoveryLags:
    def test_lag_for_single_failure(self, analyzer, failing_ious):
        failures = [10]
        lags = analyzer.compute_recovery_lags(failing_ious, failures)
        assert len(lags) == 1
        # Frames 10–14 are failures; IoU returns to 1.0 at frame 15
        assert lags[0] == 5

    def test_never_recovering_lag(self, analyzer, never_recovering_ious):
        failures = [10]
        lags = analyzer.compute_recovery_lags(never_recovering_ious, failures)
        # Should equal len(ious) - failure_frame
        assert lags[0] == len(never_recovering_ious) - 10

    def test_no_failures_returns_empty(self, analyzer, perfect_ious):
        lags = analyzer.compute_recovery_lags(perfect_ious, [])
        assert lags == []

    def test_immediate_recovery(self, analyzer):
        ious = np.ones(20, dtype=np.float64)
        ious[5] = 0.0   # single-frame failure
        lags = analyzer.compute_recovery_lags(ious, [5])
        assert lags[0] == 1  # recovers on the very next frame


# ---------------------------------------------------------------------------
# compute_eao
# ---------------------------------------------------------------------------

class TestComputeEao:
    def test_perfect_sequence_eao_is_one(self, analyzer, perfect_ious):
        eao = analyzer.compute_eao(perfect_ious)
        assert eao == pytest.approx(1.0)

    def test_zero_ious_eao_is_zero(self, analyzer):
        ious = np.zeros(30, dtype=np.float64)
        eao = analyzer.compute_eao(ious)
        # Burn-in frames are skipped; remaining are all 0
        assert eao == pytest.approx(0.0)

    def test_eao_ignores_burn_in(self, analyzer):
        ious = np.zeros(30, dtype=np.float64)
        ious[:3] = 0.0   # burn-in: these are skipped
        ious[3:] = 1.0   # post-burn-in: all 1.0
        eao = analyzer.compute_eao(ious)
        assert eao == pytest.approx(1.0)

    def test_eao_short_sequence(self, analyzer):
        ious = np.ones(2, dtype=np.float64)  # shorter than burn_in=3
        eao = analyzer.compute_eao(ious)
        assert eao == pytest.approx(0.0)

    def test_eao_in_range(self, analyzer):
        rng = np.random.default_rng(0)
        ious = rng.uniform(0.0, 1.0, 50)
        eao = analyzer.compute_eao(ious)
        assert 0.0 <= eao <= 1.0


# ---------------------------------------------------------------------------
# analyze_sequence
# ---------------------------------------------------------------------------

class TestAnalyzeSequence:
    def test_returns_robustness_result(self, analyzer, perfect_ious):
        result = analyzer.analyze_sequence(perfect_ious, "MOSSE", "seq01")
        assert isinstance(result, RobustnessResult)

    def test_perfect_tracker_no_failures(self, analyzer, perfect_ious):
        result = analyzer.analyze_sequence(perfect_ious)
        assert result.num_failures == 0
        assert result.failure_frames == []
        assert result.recovery_lags == []
        assert result.mean_recovery_lag == pytest.approx(0.0)

    def test_perfect_tracker_eao_one(self, analyzer, perfect_ious):
        result = analyzer.analyze_sequence(perfect_ious)
        assert result.eao == pytest.approx(1.0)

    def test_perfect_tracker_survival_rate_one(self, analyzer, perfect_ious):
        result = analyzer.analyze_sequence(perfect_ious)
        assert result.survival_rate == pytest.approx(1.0)

    def test_failing_tracker_detects_failure(self, analyzer, failing_ious):
        result = analyzer.analyze_sequence(failing_ious)
        assert result.num_failures == 1

    def test_names_propagated(self, analyzer, perfect_ious):
        result = analyzer.analyze_sequence(perfect_ious, "KCF", "ball_seq")
        assert result.tracker_name == "KCF"
        assert result.sequence_name == "ball_seq"

    def test_survival_rate_range(self, analyzer):
        rng = np.random.default_rng(1)
        ious = rng.uniform(0.0, 1.0, 50)
        result = analyzer.analyze_sequence(ious)
        assert 0.0 <= result.survival_rate <= 1.0

    def test_eao_matches_compute_eao(self, analyzer, failing_ious):
        result = analyzer.analyze_sequence(failing_ious)
        expected_eao = analyzer.compute_eao(failing_ious)
        assert result.eao == pytest.approx(expected_eao)

    def test_str_representation(self, analyzer, perfect_ious):
        result = analyzer.analyze_sequence(perfect_ious, "MOSSE", "car1")
        s = str(result)
        assert "MOSSE" in s
        assert "car1" in s

    def test_to_dict_keys(self, analyzer, perfect_ious):
        result = analyzer.analyze_sequence(perfect_ious)
        d = result.to_dict()
        for key in ("num_failures", "eao", "survival_rate", "mean_recovery_lag"):
            assert key in d


# ---------------------------------------------------------------------------
# analyze_benchmark
# ---------------------------------------------------------------------------

class TestAnalyzeBenchmark:
    def test_aggregate_keys(self, analyzer):
        ious_map = {
            "seq1": np.ones(30),
            "seq2": np.ones(30) * 0.8,
        }
        output = analyzer.analyze_benchmark(ious_map, tracker_name="MOSSE")
        agg = output["aggregate"]
        for key in ("tracker_name", "num_sequences", "total_failures", "mean_eao",
                    "mean_survival_rate", "mean_recovery_lag_frames"):
            assert key in agg

    def test_per_sequence_present(self, analyzer):
        ious_map = {"s1": np.ones(20), "s2": np.ones(20)}
        output = analyzer.analyze_benchmark(ious_map)
        assert "s1" in output["per_sequence"]
        assert "s2" in output["per_sequence"]

    def test_aggregate_num_sequences(self, analyzer):
        ious_map = {f"seq_{i}": np.ones(20) for i in range(5)}
        output = analyzer.analyze_benchmark(ious_map)
        assert output["aggregate"]["num_sequences"] == 5

    def test_perfect_tracker_zero_failures(self, analyzer):
        ious_map = {f"seq_{i}": np.ones(20) for i in range(4)}
        output = analyzer.analyze_benchmark(ious_map, tracker_name="MOSSE")
        assert output["aggregate"]["total_failures"] == 0

    def test_empty_input(self, analyzer):
        output = analyzer.analyze_benchmark({})
        assert output["aggregate"]["num_sequences"] == 0
        assert output["aggregate"]["total_failures"] == 0


# ---------------------------------------------------------------------------
# survival_curve
# ---------------------------------------------------------------------------

class TestSurvivalCurve:
    def test_perfect_curve_all_ones(self, analyzer):
        ious_list = [np.ones(30)] * 5
        curve = analyzer.survival_curve(ious_list)
        np.testing.assert_allclose(curve, 1.0)

    def test_curve_shape(self, analyzer):
        ious_list = [np.ones(20)] * 3
        curve = analyzer.survival_curve(ious_list, n_points=50)
        assert curve.shape == (50,)

    def test_curve_values_in_range(self, analyzer):
        rng = np.random.default_rng(42)
        ious_list = [rng.uniform(0.0, 1.0, 30) for _ in range(10)]
        curve = analyzer.survival_curve(ious_list)
        assert np.all(curve >= 0.0) and np.all(curve <= 1.0)

    def test_empty_list_returns_zeros(self, analyzer):
        curve = analyzer.survival_curve([])
        assert np.all(curve == 0.0)
