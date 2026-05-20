"""Tests for the statistical significance testing module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eovot.metrics.statistical import (
    BootstrapCI,
    PairwiseSummary,
    StatisticalTestEngine,
    WilcoxonResult,
    _rank_with_ties,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_scores(mean: float, std: float, n: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n).clip(0, 1).tolist()


# ---------------------------------------------------------------------------
# _rank_with_ties
# ---------------------------------------------------------------------------

class TestRankWithTies:
    def test_no_ties(self):
        arr = np.array([3.0, 1.0, 4.0, 1.5])
        ranks = _rank_with_ties(arr)
        # Expected ranks: 1→3, 2→1, 3→4, 4→2
        assert ranks[1] == 1.0  # smallest
        assert ranks[3] == 2.0
        assert ranks[0] == 3.0
        assert ranks[2] == 4.0

    def test_all_tied(self):
        arr = np.array([5.0, 5.0, 5.0])
        ranks = _rank_with_ties(arr)
        assert np.allclose(ranks, [2.0, 2.0, 2.0])

    def test_two_tie_groups(self):
        arr = np.array([1.0, 1.0, 3.0, 3.0])
        ranks = _rank_with_ties(arr)
        # Group 1: positions 0,1 → avg rank (1+2)/2 = 1.5
        # Group 2: positions 2,3 → avg rank (3+4)/2 = 3.5
        assert np.allclose(ranks[[0, 1]], 1.5)
        assert np.allclose(ranks[[2, 3]], 3.5)

    def test_single_element(self):
        arr = np.array([7.0])
        ranks = _rank_with_ties(arr)
        assert ranks[0] == 1.0

    def test_sum_of_ranks_equals_n_n_plus1_over_2(self):
        arr = np.array([2.0, 1.0, 3.0, 4.0, 1.0])
        ranks = _rank_with_ties(arr)
        n = len(arr)
        assert abs(ranks.sum() - n * (n + 1) / 2) < 1e-9


# ---------------------------------------------------------------------------
# StatisticalTestEngine — constructor
# ---------------------------------------------------------------------------

class TestEngineConstructor:
    def test_default_params(self):
        engine = StatisticalTestEngine()
        assert engine.alpha == 0.05
        assert engine.n_bootstrap == 10_000
        assert engine.seed == 42

    def test_custom_params(self):
        engine = StatisticalTestEngine(alpha=0.01, n_bootstrap=1000, seed=7)
        assert engine.alpha == 0.01
        assert engine.n_bootstrap == 1000
        assert engine.seed == 7

    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha"):
            StatisticalTestEngine(alpha=0.0)

    def test_invalid_alpha_one(self):
        with pytest.raises(ValueError, match="alpha"):
            StatisticalTestEngine(alpha=1.0)

    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError, match="alpha"):
            StatisticalTestEngine(alpha=-0.05)

    def test_invalid_bootstrap_too_small(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            StatisticalTestEngine(n_bootstrap=50)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def setup_method(self):
        self.engine = StatisticalTestEngine(alpha=0.05, n_bootstrap=5_000, seed=42)

    def test_returns_bootstrap_ci(self):
        scores = _make_scores(0.5, 0.1, 30)
        ci = self.engine.bootstrap_ci(scores)
        assert isinstance(ci, BootstrapCI)

    def test_observed_is_sample_mean(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        ci = self.engine.bootstrap_ci(scores)
        assert abs(ci.observed - 0.3) < 1e-9

    def test_ci_contains_observed(self):
        scores = _make_scores(0.4, 0.1, 50)
        ci = self.engine.bootstrap_ci(scores)
        assert ci.lower <= ci.observed <= ci.upper

    def test_ci_width_decreases_with_more_samples(self):
        small = _make_scores(0.5, 0.1, 10)
        large = _make_scores(0.5, 0.1, 200)
        ci_small = self.engine.bootstrap_ci(small)
        ci_large = self.engine.bootstrap_ci(large)
        width_small = ci_small.upper - ci_small.lower
        width_large = ci_large.upper - ci_large.lower
        assert width_small > width_large

    def test_confidence_level_stored(self):
        scores = _make_scores(0.5, 0.1, 20)
        ci = self.engine.bootstrap_ci(scores)
        assert abs(ci.confidence_level - 0.95) < 1e-9

    def test_n_samples_stored(self):
        scores = _make_scores(0.5, 0.1, 30)
        ci = self.engine.bootstrap_ci(scores, "test_metric")
        assert ci.n_samples == 30
        assert ci.metric_name == "test_metric"

    def test_n_bootstrap_stored(self):
        scores = _make_scores(0.5, 0.1, 20)
        ci = self.engine.bootstrap_ci(scores)
        assert ci.n_bootstrap == 5_000

    def test_reproduciblity_with_same_seed(self):
        scores = _make_scores(0.5, 0.1, 30)
        ci1 = StatisticalTestEngine(seed=99).bootstrap_ci(scores)
        ci2 = StatisticalTestEngine(seed=99).bootstrap_ci(scores)
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper

    def test_different_seeds_produce_different_results(self):
        scores = _make_scores(0.5, 0.2, 20)
        ci1 = StatisticalTestEngine(seed=1).bootstrap_ci(scores)
        ci2 = StatisticalTestEngine(seed=2).bootstrap_ci(scores)
        # Very unlikely to be identical
        assert not (ci1.lower == ci2.lower and ci1.upper == ci2.upper)

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError):
            self.engine.bootstrap_ci([0.5])

    def test_str_contains_metric_name(self):
        scores = _make_scores(0.5, 0.1, 20)
        ci = self.engine.bootstrap_ci(scores, metric_name="success_AUC")
        assert "success_AUC" in str(ci)


# ---------------------------------------------------------------------------
# wilcoxon_test
# ---------------------------------------------------------------------------

class TestWilcoxonTest:
    def setup_method(self):
        self.engine = StatisticalTestEngine(alpha=0.05)

    def test_returns_wilcoxon_result(self):
        a = _make_scores(0.5, 0.1, 30, seed=0)
        b = _make_scores(0.5, 0.1, 30, seed=1)
        result = self.engine.wilcoxon_test(a, b)
        assert isinstance(result, WilcoxonResult)

    def test_p_value_in_0_1(self):
        a = _make_scores(0.5, 0.1, 30)
        b = _make_scores(0.5, 0.1, 30, seed=1)
        result = self.engine.wilcoxon_test(a, b)
        assert 0.0 <= result.p_value <= 1.0

    def test_identical_scores_high_p_value(self):
        scores = _make_scores(0.5, 0.1, 30)
        result = self.engine.wilcoxon_test(scores, scores)
        # All differences are zero → p = 1.0
        assert result.p_value == 1.0
        assert not result.significant
        assert result.n_pairs == 0

    def test_clearly_different_trackers_significant(self):
        # A is clearly better (shifted by 0.3 with low std)
        a = [0.8] * 40
        b = [0.4] * 40
        result = self.engine.wilcoxon_test(a, b, "A", "B")
        assert result.significant
        assert result.p_value < 0.001

    def test_direction_a_wins(self):
        a = [0.8] * 30
        b = [0.4] * 30
        result = self.engine.wilcoxon_test(a, b, "Fast", "Slow")
        assert "Fast" in result.direction
        assert result.effect_size > 0

    def test_direction_b_wins(self):
        a = [0.3] * 30
        b = [0.7] * 30
        result = self.engine.wilcoxon_test(a, b, "Weak", "Strong")
        assert "Strong" in result.direction
        assert result.effect_size < 0

    def test_effect_size_in_minus1_to_1(self):
        a = _make_scores(0.6, 0.1, 40)
        b = _make_scores(0.4, 0.1, 40)
        result = self.engine.wilcoxon_test(a, b)
        assert -1.0 <= result.effect_size <= 1.0

    def test_w_plus_w_minus_sum_is_n_n1_over2(self):
        a = _make_scores(0.6, 0.1, 30)
        b = _make_scores(0.4, 0.1, 30)
        result = self.engine.wilcoxon_test(a, b)
        n = result.n_pairs
        expected_total = n * (n + 1) / 2
        assert abs(result.w_plus + result.w_minus - expected_total) < 1e-6

    def test_unequal_length_uses_shorter(self):
        a = _make_scores(0.5, 0.1, 30)
        b = _make_scores(0.5, 0.1, 20)
        result = self.engine.wilcoxon_test(a, b)
        # Should not raise; uses min(30, 20)=20 pairs
        assert result.n_pairs <= 20

    def test_tracker_names_stored(self):
        a = _make_scores(0.5, 0.1, 20)
        b = _make_scores(0.5, 0.1, 20)
        result = self.engine.wilcoxon_test(a, b, "TrackerAlpha", "TrackerBeta")
        assert result.tracker_a == "TrackerAlpha"
        assert result.tracker_b == "TrackerBeta"

    def test_too_few_pairs_raises(self):
        with pytest.raises(ValueError):
            self.engine.wilcoxon_test([0.5], [0.6])

    def test_small_n_produces_warning(self):
        a = [0.5, 0.6, 0.7]
        b = [0.4, 0.5, 0.6]
        result = self.engine.wilcoxon_test(a, b)
        assert result.warning is not None

    def test_large_n_no_warning(self):
        a = _make_scores(0.5, 0.1, 50, seed=0)
        b = _make_scores(0.5, 0.1, 50, seed=7)
        result = self.engine.wilcoxon_test(a, b)
        assert result.warning is None

    def test_alpha_stored_in_result(self):
        engine = StatisticalTestEngine(alpha=0.01)
        a = _make_scores(0.5, 0.1, 20)
        b = _make_scores(0.5, 0.1, 20)
        result = engine.wilcoxon_test(a, b)
        assert result.alpha == 0.01

    def test_significant_flag_consistent_with_alpha(self):
        a = [0.8] * 50
        b = [0.2] * 50
        engine_strict = StatisticalTestEngine(alpha=0.001)
        result = engine_strict.wilcoxon_test(a, b)
        assert result.significant == (result.p_value < 0.001)

    def test_str_representation(self):
        a = _make_scores(0.5, 0.1, 30)
        b = _make_scores(0.4, 0.1, 30)
        result = self.engine.wilcoxon_test(a, b, "MOSSE", "KCF")
        s = str(result)
        assert "MOSSE" in s
        assert "KCF" in s
        assert "p=" in s

    def test_effect_label_large(self):
        a = [1.0] * 30
        b = [0.0] * 30
        result = self.engine.wilcoxon_test(a, b)
        assert result.effect_label == "large"

    def test_effect_label_negligible(self):
        a = _make_scores(0.5, 0.001, 30, seed=1)
        b = _make_scores(0.5, 0.001, 30, seed=2)
        result = self.engine.wilcoxon_test(a, b)
        assert result.effect_label in ("negligible", "small")


# ---------------------------------------------------------------------------
# pairwise_report
# ---------------------------------------------------------------------------

class TestPairwiseReport:
    def setup_method(self):
        self.engine = StatisticalTestEngine(alpha=0.05)
        self.scores = {
            "MOSSE": _make_scores(0.35, 0.1, 40, seed=0),
            "KCF": _make_scores(0.45, 0.1, 40, seed=1),
            "CSRT": _make_scores(0.65, 0.1, 40, seed=2),
        }

    def test_returns_pairwise_summary(self):
        summary = self.engine.pairwise_report(self.scores)
        assert isinstance(summary, PairwiseSummary)

    def test_correct_number_of_comparisons(self):
        summary = self.engine.pairwise_report(self.scores)
        # C(3,2) = 3
        assert summary.n_comparisons == 3
        assert len(summary.comparisons) == 3

    def test_four_trackers_six_pairs(self):
        scores_4 = dict(self.scores)
        scores_4["MIL"] = _make_scores(0.40, 0.1, 40, seed=3)
        summary = self.engine.pairwise_report(scores_4)
        assert summary.n_comparisons == 6  # C(4,2)

    def test_bonferroni_correction_applied(self):
        summary = self.engine.pairwise_report(self.scores)
        expected = 0.05 / 3
        assert abs(summary.alpha_bonferroni - expected) < 1e-12

    def test_trackers_stored(self):
        summary = self.engine.pairwise_report(self.scores)
        assert set(summary.trackers) == {"MOSSE", "KCF", "CSRT"}

    def test_one_tracker_raises(self):
        with pytest.raises(ValueError):
            self.engine.pairwise_report({"MOSSE": _make_scores(0.5, 0.1, 20)})

    def test_clearly_separated_trackers_produce_significant_pairs(self):
        summary = self.engine.pairwise_report(self.scores)
        sig = summary.significant_pairs()
        # CSRT vs MOSSE should be significant (0.65 vs 0.35 mean)
        assert len(sig) > 0

    def test_identical_trackers_no_significant_pairs(self):
        scores_same = {
            "A": [0.5] * 30,
            "B": [0.5] * 30,
        }
        summary = self.engine.pairwise_report(scores_same)
        assert len(summary.significant_pairs()) == 0


# ---------------------------------------------------------------------------
# PairwiseSummary.to_markdown
# ---------------------------------------------------------------------------

class TestPairwiseSummaryMarkdown:
    def setup_method(self):
        engine = StatisticalTestEngine(alpha=0.05)
        scores = {
            "MOSSE": _make_scores(0.35, 0.1, 40),
            "KCF": _make_scores(0.45, 0.1, 40, seed=1),
        }
        self.summary = engine.pairwise_report(scores)

    def test_markdown_has_header(self):
        md = self.summary.to_markdown()
        assert "Tracker A" in md
        assert "p-value" in md

    def test_markdown_has_bonferroni_note(self):
        md = self.summary.to_markdown()
        assert "Bonferroni" in md

    def test_markdown_has_data_rows(self):
        md = self.summary.to_markdown()
        rows = [l for l in md.splitlines() if l.startswith("|") and "---" not in l and "Tracker A" not in l]
        assert len(rows) >= 1

    def test_markdown_contains_tracker_names(self):
        md = self.summary.to_markdown()
        assert "MOSSE" in md
        assert "KCF" in md


# ---------------------------------------------------------------------------
# ci_table
# ---------------------------------------------------------------------------

class TestCITable:
    def setup_method(self):
        self.engine = StatisticalTestEngine(alpha=0.05, n_bootstrap=2_000, seed=0)

    def test_markdown_has_correct_columns(self):
        scores = {
            "A": _make_scores(0.4, 0.1, 30),
            "B": _make_scores(0.6, 0.1, 30, seed=1),
        }
        table = self.engine.ci_table(scores, metric_name="mIoU")
        assert "Tracker" in table
        assert "mIoU" in table
        assert "95%" in table

    def test_all_trackers_in_table(self):
        scores = {"X": [0.3] * 20, "Y": [0.6] * 20, "Z": [0.5] * 20}
        table = self.engine.ci_table(scores)
        assert "X" in table
        assert "Y" in table
        assert "Z" in table

    def test_ci_lower_lte_upper(self):
        scores = {"M": _make_scores(0.5, 0.1, 40)}
        ci = self.engine.bootstrap_ci(scores["M"])
        assert ci.lower <= ci.upper
