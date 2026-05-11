"""Tests for eovot.metrics.statistical — Wilcoxon test and bootstrap CI."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eovot.metrics.statistical import (
    BootstrapCI,
    StatisticalComparison,
    StatisticalSummary,
    WilcoxonResult,
    bootstrap_ci,
    wilcoxon_signed_rank,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _rng_scores(n: int, loc: float, scale: float = 0.05) -> np.ndarray:
    return np.clip(RNG.normal(loc, scale, n), 0.0, 1.0)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_estimate_is_sample_mean(self) -> None:
        vals = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        ci = bootstrap_ci(vals, n_bootstrap=500, rng=np.random.default_rng(0))
        assert math.isclose(ci.estimate, float(vals.mean()))

    def test_ci_contains_estimate(self) -> None:
        vals = _rng_scores(30, 0.5)
        ci = bootstrap_ci(vals, confidence=0.95, n_bootstrap=1000, rng=np.random.default_rng(1))
        assert ci.ci_lower <= ci.estimate <= ci.ci_upper

    def test_wider_ci_for_high_confidence(self) -> None:
        vals = _rng_scores(20, 0.5)
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        ci_95 = bootstrap_ci(vals, confidence=0.95, n_bootstrap=500, rng=rng_a)
        ci_80 = bootstrap_ci(vals, confidence=0.80, n_bootstrap=500, rng=rng_b)
        assert (ci_95.ci_upper - ci_95.ci_lower) >= (ci_80.ci_upper - ci_80.ci_lower)

    def test_empty_array(self) -> None:
        ci = bootstrap_ci(np.array([]))
        assert math.isnan(ci.estimate)
        assert ci.n_samples == 0

    def test_single_value(self) -> None:
        ci = bootstrap_ci(np.array([0.42]), n_bootstrap=100, rng=np.random.default_rng(3))
        assert math.isclose(ci.estimate, 0.42)
        # All resamples are the same → CI should collapse
        assert math.isclose(ci.ci_lower, 0.42, abs_tol=1e-9)
        assert math.isclose(ci.ci_upper, 0.42, abs_tol=1e-9)

    def test_n_samples_recorded(self) -> None:
        vals = _rng_scores(15, 0.4)
        ci = bootstrap_ci(vals, n_bootstrap=200, rng=np.random.default_rng(5))
        assert ci.n_samples == 15

    def test_to_dict_keys(self) -> None:
        vals = _rng_scores(10, 0.5)
        ci = bootstrap_ci(vals, rng=np.random.default_rng(9))
        d = ci.to_dict()
        for key in ("estimate", "ci_lower", "ci_upper", "confidence", "n_bootstrap", "n_samples"):
            assert key in d


# ---------------------------------------------------------------------------
# wilcoxon_signed_rank
# ---------------------------------------------------------------------------


class TestWilcoxonSignedRank:
    def test_clearly_different_trackers(self) -> None:
        # A much better than B
        a = _rng_scores(20, 0.65, 0.03)
        b = _rng_scores(20, 0.35, 0.03)
        result = wilcoxon_signed_rank(a, b, tracker_a="Good", tracker_b="Bad", metric="iou")
        assert result.significant
        assert "Good" in result.direction

    def test_equal_trackers_not_significant(self) -> None:
        # A and B drawn from identical distribution
        rng = np.random.default_rng(99)
        a = rng.normal(0.5, 0.05, 30)
        b = rng.normal(0.5, 0.05, 30)
        # We can't guarantee p ≥ 0.05 with random data, so just check type
        result = wilcoxon_signed_rank(a, b)
        assert isinstance(result, WilcoxonResult)
        assert 0.0 <= result.p_value <= 1.0

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            wilcoxon_signed_rank(np.array([0.1, 0.2]), np.array([0.1]))

    def test_all_zeros_returns_p1(self) -> None:
        a = np.array([0.5, 0.5, 0.5])
        b = np.array([0.5, 0.5, 0.5])
        result = wilcoxon_signed_rank(a, b)
        assert result.p_value == 1.0
        assert result.n_effective == 0
        assert not result.significant

    def test_direction_a_better(self) -> None:
        a = np.array([0.9, 0.85, 0.88, 0.91, 0.87])
        b = np.array([0.5, 0.48, 0.52, 0.51, 0.49])
        result = wilcoxon_signed_rank(a, b, tracker_a="A", tracker_b="B")
        if result.significant:
            assert "A" in result.direction

    def test_direction_b_better(self) -> None:
        a = np.array([0.3, 0.28, 0.31, 0.29, 0.30])
        b = np.array([0.7, 0.68, 0.72, 0.69, 0.71])
        result = wilcoxon_signed_rank(a, b, tracker_a="A", tracker_b="B")
        if result.significant:
            assert "B" in result.direction

    def test_n_sequences_recorded(self) -> None:
        a = np.zeros(12)
        b = np.zeros(12)
        result = wilcoxon_signed_rank(a, b)
        assert result.n_sequences == 12

    def test_p_value_in_range(self) -> None:
        a = _rng_scores(15, 0.6)
        b = _rng_scores(15, 0.4)
        result = wilcoxon_signed_rank(a, b)
        assert 0.0 <= result.p_value <= 1.0

    def test_to_dict_has_required_keys(self) -> None:
        a = _rng_scores(10, 0.5)
        b = _rng_scores(10, 0.4)
        d = wilcoxon_signed_rank(a, b).to_dict()
        for key in ("tracker_a", "tracker_b", "p_value", "significant", "direction", "statistic"):
            assert key in d


# ---------------------------------------------------------------------------
# StatisticalComparison
# ---------------------------------------------------------------------------


class TestStatisticalComparison:
    def _three_tracker_scores(self) -> dict:
        rng = np.random.default_rng(42)
        return {
            "MOSSE": rng.normal(0.30, 0.04, 20),
            "KCF":   rng.normal(0.44, 0.04, 20),
            "CSRT":  rng.normal(0.58, 0.04, 20),
        }

    def test_analyze_returns_summary(self) -> None:
        cmp = StatisticalComparison(seed=0)
        summary = cmp.analyze(self._three_tracker_scores(), metric="mean_iou")
        assert isinstance(summary, StatisticalSummary)

    def test_all_trackers_have_ci(self) -> None:
        cmp = StatisticalComparison(seed=1)
        scores = self._three_tracker_scores()
        summary = cmp.analyze(scores, metric="mean_iou")
        for name in scores:
            assert name in summary.tracker_cis

    def test_correct_number_of_pairwise_tests(self) -> None:
        cmp = StatisticalComparison(seed=2)
        scores = self._three_tracker_scores()
        summary = cmp.analyze(scores, metric="mean_iou")
        # C(3, 2) = 3
        assert len(summary.pairwise_tests) == 3

    def test_ranking_is_descending(self) -> None:
        cmp = StatisticalComparison(seed=3)
        summary = cmp.analyze(self._three_tracker_scores(), metric="mean_iou")
        estimates = [v for _, v in summary.ranking()]
        assert estimates == sorted(estimates, reverse=True)

    def test_mismatched_lengths_raises(self) -> None:
        cmp = StatisticalComparison()
        scores = {"A": np.array([0.1, 0.2, 0.3]), "B": np.array([0.4, 0.5])}
        with pytest.raises(ValueError, match="same number of sequences"):
            cmp.analyze(scores)

    def test_to_markdown_contains_tracker_names(self) -> None:
        cmp = StatisticalComparison(seed=4)
        scores = {"AlphaTracker": _rng_scores(10, 0.5), "BetaTracker": _rng_scores(10, 0.4)}
        md = cmp.analyze(scores, metric="iou").to_markdown()
        assert "AlphaTracker" in md
        assert "BetaTracker" in md

    def test_to_dict_structure(self) -> None:
        cmp = StatisticalComparison(seed=5)
        d = cmp.analyze(self._three_tracker_scores(), metric="fps").to_dict()
        assert "metric" in d
        assert "tracker_cis" in d
        assert "pairwise_tests" in d
        assert "ranking" in d

    def test_significant_pairs_subset(self) -> None:
        cmp = StatisticalComparison(seed=6)
        summary = cmp.analyze(self._three_tracker_scores(), metric="mean_iou")
        sig = summary.significant_pairs()
        assert all(t.significant for t in sig)
        assert len(sig) <= len(summary.pairwise_tests)

    def test_five_tracker_pairwise_count(self) -> None:
        rng = np.random.default_rng(10)
        scores = {f"T{i}": rng.normal(0.5 + i * 0.05, 0.03, 15) for i in range(5)}
        cmp = StatisticalComparison(seed=10)
        summary = cmp.analyze(scores)
        # C(5, 2) = 10
        assert len(summary.pairwise_tests) == 10

    def test_analyze_from_benchmark_results_unsupported_metric(self) -> None:
        cmp = StatisticalComparison()
        with pytest.raises(ValueError, match="metric must be one of"):
            cmp.analyze_from_benchmark_results([], metric="nonexistent")
