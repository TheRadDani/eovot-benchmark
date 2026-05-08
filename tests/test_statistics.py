"""Tests for eovot.analysis.statistics."""

import numpy as np
import pytest

from eovot.analysis.statistics import ComparisonResult, TrackerStatistics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(tracker: str, ious: list, fps_list: list) -> dict:
    """Build a minimal BenchmarkResult.to_dict()-compatible dict."""
    sequences = [
        {
            "sequence_name": f"seq{i}",
            "mean_iou": ious[i],
            "fps": fps_list[i],
            "mean_latency_ms": 1_000.0 / max(fps_list[i], 1),
            "peak_memory_mb": 50.0,
        }
        for i in range(len(ious))
    ]
    return {
        "summary": {
            "tracker": tracker,
            "mean_iou": float(np.mean(ious)),
            "mean_fps": float(np.mean(fps_list)),
        },
        "sequences": sequences,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stats():
    return TrackerStatistics(alpha=0.05, n_bootstrap=500, seed=42)


@pytest.fixture()
def high_iou_result():
    rng = np.random.default_rng(0)
    ious = rng.uniform(0.70, 0.90, 50).tolist()
    return _make_result("CSRT", ious, [50.0] * 50)


@pytest.fixture()
def low_iou_result():
    rng = np.random.default_rng(1)
    ious = rng.uniform(0.10, 0.30, 50).tolist()
    return _make_result("MOSSE", ious, [500.0] * 50)


@pytest.fixture()
def identical_result():
    """Two results built from the same scores."""
    rng = np.random.default_rng(2)
    ious = rng.uniform(0.50, 0.60, 30).tolist()
    fps = [100.0] * 30
    return (
        _make_result("TrackerA", ious, fps),
        _make_result("TrackerB", ious, fps),
    )


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------


class TestCompare:
    def test_returns_comparison_result(self, stats, high_iou_result, low_iou_result):
        result = stats.compare(high_iou_result, low_iou_result)
        assert isinstance(result, ComparisonResult)

    def test_significant_when_distributions_differ(
        self, stats, high_iou_result, low_iou_result
    ):
        result = stats.compare(high_iou_result, low_iou_result)
        assert result.significant
        assert result.p_value < 0.05

    def test_not_significant_identical_distributions(self, stats, identical_result):
        ra, rb = identical_result
        result = stats.compare(ra, rb)
        assert not result.significant

    def test_tracker_names_propagated(self, stats, high_iou_result, low_iou_result):
        result = stats.compare(high_iou_result, low_iou_result)
        assert result.tracker_a == "CSRT"
        assert result.tracker_b == "MOSSE"

    def test_effect_size_sign(self, stats, high_iou_result, low_iou_result):
        result = stats.compare(high_iou_result, low_iou_result)
        # CSRT has higher IoU → positive Cohen's d
        assert result.effect_size > 0

    def test_ci_contains_mean_a(self, stats, high_iou_result, low_iou_result):
        result = stats.compare(high_iou_result, low_iou_result)
        assert result.ci_lower <= result.mean_a <= result.ci_upper

    def test_ci_is_ordered(self, stats, high_iou_result, low_iou_result):
        result = stats.compare(high_iou_result, low_iou_result)
        assert result.ci_lower <= result.ci_upper

    def test_fps_metric(self, stats, high_iou_result, low_iou_result):
        result = stats.compare(high_iou_result, low_iou_result, metric="fps")
        assert result.metric == "fps"
        assert 0.0 <= result.p_value <= 1.0

    def test_winner_when_significant(self, stats, high_iou_result, low_iou_result):
        result = stats.compare(high_iou_result, low_iou_result)
        assert result.winner == "CSRT"

    def test_winner_none_when_not_significant(self, stats, identical_result):
        ra, rb = identical_result
        result = stats.compare(ra, rb)
        assert result.winner is None

    def test_p_value_in_unit_interval(self, stats, high_iou_result, low_iou_result):
        result = stats.compare(high_iou_result, low_iou_result)
        assert 0.0 <= result.p_value <= 1.0

    def test_metric_stored_on_result(self, stats, high_iou_result, low_iou_result):
        result = stats.compare(high_iou_result, low_iou_result, metric="iou")
        assert result.metric == "iou"

    def test_missing_metric_raises(self, stats, high_iou_result, low_iou_result):
        with pytest.raises(ValueError, match="No per-sequence data"):
            stats.compare(high_iou_result, low_iou_result, metric="energy_j")


# ---------------------------------------------------------------------------
# pairwise_comparison()
# ---------------------------------------------------------------------------


class TestPairwiseComparison:
    def test_three_trackers_produces_three_pairs(
        self, stats, high_iou_result, low_iou_result
    ):
        rng = np.random.default_rng(3)
        mid = _make_result("KCF", rng.uniform(0.4, 0.6, 50).tolist(), [150.0] * 50)
        results = stats.pairwise_comparison(
            [high_iou_result, low_iou_result, mid]
        )
        # C(3, 2) = 3 pairs
        assert len(results) == 3

    def test_two_trackers_produces_one_pair(
        self, stats, high_iou_result, low_iou_result
    ):
        results = stats.pairwise_comparison([high_iou_result, low_iou_result])
        assert len(results) == 1

    def test_empty_list_returns_empty(self, stats):
        assert stats.pairwise_comparison([]) == []

    def test_single_tracker_returns_empty(self, stats, high_iou_result):
        assert stats.pairwise_comparison([high_iou_result]) == []


# ---------------------------------------------------------------------------
# significance_table()
# ---------------------------------------------------------------------------


class TestSignificanceTable:
    def test_markdown_header_present(self, stats, high_iou_result, low_iou_result):
        comps = stats.pairwise_comparison([high_iou_result, low_iou_result])
        table = stats.significance_table(comps)
        assert "| Tracker A |" in table
        assert "| Tracker B |" in table

    def test_tracker_names_in_table(self, stats, high_iou_result, low_iou_result):
        comps = stats.pairwise_comparison([high_iou_result, low_iou_result])
        table = stats.significance_table(comps)
        assert "CSRT" in table
        assert "MOSSE" in table

    def test_significance_column_present(self, stats, high_iou_result, low_iou_result):
        comps = stats.pairwise_comparison([high_iou_result, low_iou_result])
        table = stats.significance_table(comps)
        assert "Yes" in table or "No" in table


class TestLatexTable:
    def test_booktabs_structure(self, stats, high_iou_result, low_iou_result):
        comps = stats.pairwise_comparison([high_iou_result, low_iou_result])
        latex = stats.latex_table(comps)
        assert r"\begin{table}" in latex
        assert r"\toprule" in latex
        assert r"\midrule" in latex
        assert r"\bottomrule" in latex
        assert r"\end{table}" in latex

    def test_checkmark_when_significant(self, stats, high_iou_result, low_iou_result):
        comps = stats.pairwise_comparison([high_iou_result, low_iou_result])
        latex = stats.latex_table(comps)
        assert r"\checkmark" in latex


# ---------------------------------------------------------------------------
# auc_confidence_interval()
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_lower_le_upper(self, stats, high_iou_result):
        lo, hi = stats.auc_confidence_interval(high_iou_result)
        assert lo <= hi

    def test_ci_contains_mean(self, stats, high_iou_result):
        mean_iou = high_iou_result["summary"]["mean_iou"]
        lo, hi = stats.auc_confidence_interval(high_iou_result, ci=0.95)
        assert lo <= mean_iou <= hi

    def test_tighter_ci_at_higher_confidence(self, stats, high_iou_result):
        lo_90, hi_90 = stats.auc_confidence_interval(high_iou_result, ci=0.90)
        lo_99, hi_99 = stats.auc_confidence_interval(high_iou_result, ci=0.99)
        width_90 = hi_90 - lo_90
        width_99 = hi_99 - lo_99
        assert width_99 >= width_90
