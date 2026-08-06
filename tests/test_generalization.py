"""Tests for cross-dataset generalization analysis (GeneralizationAnalyzer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.metrics.generalization import (
    GeneralizationAnalyzer,
    GeneralizationProfile,
    GeneralizationReport,
    RankCorrelationTable,
)


# ---------------------------------------------------------------------------
# Minimal BenchmarkResult stub
# ---------------------------------------------------------------------------

class _FakeResult:
    """Minimal stand-in for BenchmarkResult — only needs summary()."""

    def __init__(self, tracker_name: str, **metrics) -> None:
        self.tracker_name = tracker_name
        self._metrics = {"tracker": tracker_name, **metrics}

    def summary(self) -> Dict:
        return dict(self._metrics)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results(tracker_scores: Dict[str, Dict[str, float]]) -> Dict[str, List]:
    """Build a {dataset: [FakeResult]} dict from {tracker: {dataset: metric}} scores."""
    datasets = sorted({ds for scores in tracker_scores.values() for ds in scores})
    by_dataset: Dict[str, List[_FakeResult]] = {ds: [] for ds in datasets}
    for tracker, scores in tracker_scores.items():
        for ds, val in scores.items():
            by_dataset[ds].append(_FakeResult(tracker, mean_iou=val))
    return by_dataset


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

class TestGeneralizationAnalyzerInit:
    def test_default_metric(self):
        a = GeneralizationAnalyzer()
        assert a.metric == "mean_iou"

    def test_metric_alias_iou(self):
        a = GeneralizationAnalyzer(metric="iou")
        assert a.metric == "mean_iou"

    def test_metric_alias_auc(self):
        a = GeneralizationAnalyzer(metric="auc")
        assert a.metric == "success_auc"

    def test_metric_alias_fps(self):
        a = GeneralizationAnalyzer(metric="fps")
        assert a.metric == "mean_fps"

    def test_custom_metric_passthrough(self):
        a = GeneralizationAnalyzer(metric="peak_memory_mb")
        assert a.metric == "peak_memory_mb"

    def test_fewer_than_two_datasets_raises(self):
        a = GeneralizationAnalyzer()
        with pytest.raises(ValueError, match="at least 2"):
            a.analyze({"OTB100": [_FakeResult("MOSSE", mean_iou=0.5)]})


# ---------------------------------------------------------------------------
# Basic report properties
# ---------------------------------------------------------------------------

class TestGeneralizationReport:
    def setup_method(self):
        # Three trackers, two datasets — MOSSE dominates OTB but underperforms on GOT
        self.scores = {
            "MOSSE": {"OTB100": 0.70, "GOT10k": 0.40},
            "KCF":   {"OTB100": 0.60, "GOT10k": 0.55},
            "CSRT":  {"OTB100": 0.50, "GOT10k": 0.65},
        }
        results = _make_results(self.scores)
        self.report = GeneralizationAnalyzer().analyze(results)

    def test_report_has_correct_datasets(self):
        assert set(self.report.datasets) == {"GOT10k", "OTB100"}

    def test_report_has_correct_number_of_profiles(self):
        assert len(self.report.profiles) == 3

    def test_profile_tracker_names_match(self):
        names = {p.tracker_name for p in self.report.profiles}
        assert names == {"MOSSE", "KCF", "CSRT"}

    def test_rank_std_sorted_ascending(self):
        stds = [p.rank_std for p in self.report.profiles]
        assert stds == sorted(stds)

    def test_mean_metric_correct_for_kcf(self):
        kcf = next(p for p in self.report.profiles if p.tracker_name == "KCF")
        expected = (0.60 + 0.55) / 2
        assert kcf.mean_metric == pytest.approx(expected, abs=1e-6)

    def test_correlation_matrix_shape(self):
        mat = self.report.correlation_table.correlation_matrix
        assert mat.shape == (2, 2)

    def test_correlation_diagonal_is_one(self):
        mat = self.report.correlation_table.correlation_matrix
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-9)

    def test_correlation_matrix_is_symmetric(self):
        mat = self.report.correlation_table.correlation_matrix
        np.testing.assert_allclose(mat, mat.T, atol=1e-9)

    def test_mean_inter_dataset_correlation_range(self):
        rho = self.report.mean_inter_dataset_correlation
        assert -1.0 <= rho <= 1.0


# ---------------------------------------------------------------------------
# Rank assignment
# ---------------------------------------------------------------------------

class TestRankAssignment:
    def test_best_tracker_gets_rank_1(self):
        scores = {
            "A": {"DS1": 0.9, "DS2": 0.8},
            "B": {"DS1": 0.5, "DS2": 0.4},
        }
        report = GeneralizationAnalyzer().analyze(_make_results(scores))
        a_profile = next(p for p in report.profiles if p.tracker_name == "A")
        assert a_profile.ranks["DS1"] == 1
        assert a_profile.ranks["DS2"] == 1

    def test_worst_tracker_gets_highest_rank(self):
        scores = {
            "A": {"DS1": 0.9, "DS2": 0.9},
            "B": {"DS1": 0.1, "DS2": 0.1},
        }
        report = GeneralizationAnalyzer().analyze(_make_results(scores))
        b_profile = next(p for p in report.profiles if p.tracker_name == "B")
        assert b_profile.ranks["DS1"] == 2
        assert b_profile.ranks["DS2"] == 2

    def test_lower_is_better_flag(self):
        # For memory, lower is better
        scores = {
            "Tiny":   {"DS1": 50.0,  "DS2": 60.0},   # best (smallest)
            "Bloat":  {"DS1": 400.0, "DS2": 500.0},  # worst (largest)
        }
        report = GeneralizationAnalyzer(
            metric="mean_iou", higher_is_better=False
        ).analyze(_make_results(scores))
        tiny = next(p for p in report.profiles if p.tracker_name == "Tiny")
        assert tiny.ranks["DS1"] == 1  # lower value → rank 1


# ---------------------------------------------------------------------------
# Spearman rho
# ---------------------------------------------------------------------------

class TestSpearmanRho:
    def test_identical_rankings_give_rho_1(self):
        r = np.array([1.0, 2.0, 3.0])
        rho = GeneralizationAnalyzer._spearman_rho(r, r)
        assert rho == pytest.approx(1.0, abs=1e-9)

    def test_reversed_rankings_give_rho_minus1(self):
        r1 = np.array([1.0, 2.0, 3.0])
        r2 = np.array([3.0, 2.0, 1.0])
        rho = GeneralizationAnalyzer._spearman_rho(r1, r2)
        assert rho == pytest.approx(-1.0, abs=1e-9)

    def test_single_element_returns_zero(self):
        rho = GeneralizationAnalyzer._spearman_rho(np.array([1.0]), np.array([1.0]))
        assert rho == 0.0

    def test_rho_bounded(self):
        rng = np.random.default_rng(0)
        r1 = rng.permutation(10).astype(float) + 1
        r2 = rng.permutation(10).astype(float) + 1
        rho = GeneralizationAnalyzer._spearman_rho(r1, r2)
        assert -1.0 <= rho <= 1.0

    def test_correlated_rankings_positive_rho(self):
        # Trackers mostly agree on ranking across two datasets
        scores = {
            "A": {"D1": 0.9, "D2": 0.85},
            "B": {"D1": 0.7, "D2": 0.65},
            "C": {"D1": 0.5, "D2": 0.45},
        }
        report = GeneralizationAnalyzer().analyze(_make_results(scores))
        # Perfect rank agreement → rho = 1.0
        assert report.mean_inter_dataset_correlation == pytest.approx(1.0, abs=1e-6)

    def test_anti_correlated_rankings_negative_rho(self):
        # Rankings are completely reversed across datasets
        scores = {
            "A": {"D1": 0.9, "D2": 0.1},
            "B": {"D1": 0.5, "D2": 0.5},
            "C": {"D1": 0.1, "D2": 0.9},
        }
        report = GeneralizationAnalyzer().analyze(_make_results(scores))
        assert report.mean_inter_dataset_correlation == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Profiles: best/worst dataset
# ---------------------------------------------------------------------------

class TestProfileBestWorst:
    def test_best_and_worst_dataset_correct(self):
        # MOSSE ranks 1st on OTB (0.80 > 0.50) but 2nd on GOT (0.30 < 0.70)
        results = {
            "OTB": [
                _FakeResult("MOSSE", mean_iou=0.80),
                _FakeResult("KCF", mean_iou=0.50),
            ],
            "GOT": [
                _FakeResult("MOSSE", mean_iou=0.30),
                _FakeResult("KCF", mean_iou=0.70),
            ],
        }
        report = GeneralizationAnalyzer().analyze(results)
        mosse = next(p for p in report.profiles if p.tracker_name == "MOSSE")
        assert mosse.best_dataset == "OTB"   # rank 1 on OTB
        assert mosse.worst_dataset == "GOT"  # rank 2 on GOT


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

class TestMarkdownOutput:
    def test_markdown_contains_tracker_names(self):
        scores = {
            "MOSSE": {"DS1": 0.6, "DS2": 0.5},
            "KCF":   {"DS1": 0.7, "DS2": 0.6},
        }
        report = GeneralizationAnalyzer().analyze(_make_results(scores))
        md = report.to_markdown()
        assert "MOSSE" in md
        assert "KCF" in md

    def test_markdown_contains_dataset_names(self):
        scores = {
            "A": {"Alpha": 0.6, "Beta": 0.5},
            "B": {"Alpha": 0.7, "Beta": 0.6},
        }
        report = GeneralizationAnalyzer().analyze(_make_results(scores))
        md = report.to_markdown()
        assert "Alpha" in md
        assert "Beta" in md

    def test_correlation_table_markdown_shape(self):
        scores = {
            "A": {"D1": 0.5, "D2": 0.4, "D3": 0.3},
            "B": {"D1": 0.4, "D2": 0.3, "D3": 0.2},
        }
        report = GeneralizationAnalyzer().analyze(_make_results(scores))
        md = report.correlation_table.to_markdown()
        lines = [l for l in md.strip().split("\n") if l.startswith("|")]
        # header + separator + 3 datasets = 5 rows
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# Warning on missing metric
# ---------------------------------------------------------------------------

class TestMissingMetricWarning:
    def test_warns_on_missing_metric(self):
        # FakeResult only has mean_iou; requesting success_auc triggers a warning
        scores = {
            "A": {"D1": 0.5, "D2": 0.4},
            "B": {"D1": 0.4, "D2": 0.3},
        }
        results = _make_results(scores)
        analyzer = GeneralizationAnalyzer(metric="success_auc")
        with pytest.warns(UserWarning, match="not found"):
            report = analyzer.analyze(results)
        # Should still produce a report (defaulting missing values to 0.0)
        assert len(report.profiles) == 2
