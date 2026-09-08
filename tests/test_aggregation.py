"""Tests for MultiDatasetAggregator — cross-dataset rank averaging."""

from __future__ import annotations

from unittest.mock import MagicMock
from typing import List

import numpy as np
import pytest

from eovot.metrics.aggregation import (
    AGGREGATE_METRICS,
    AggregationReport,
    MultiDatasetAggregator,
    PerDatasetRow,
    _average_ranks,
    _latex_escape,
)


# ---------------------------------------------------------------------------
# Helpers: mock BenchmarkResult
# ---------------------------------------------------------------------------

def _make_result(tracker: str, dataset: str, iou: float, fps: float,
                 mem_mb: float = 100.0, success_auc: float = 0.0,
                 precision_auc: float = 0.0) -> MagicMock:
    """Create a minimal BenchmarkResult mock with a summary() method."""
    r = MagicMock()
    r.tracker_name = tracker
    r.dataset_name = dataset
    summary = {
        "tracker": tracker,
        "dataset": dataset,
        "mean_iou": iou,
        "mean_fps": fps,
        "peak_memory_mb": mem_mb,
        "success_auc": success_auc,
        "precision_auc": precision_auc,
    }
    r.summary.return_value = summary
    return r


def _three_tracker_two_dataset_results() -> List[MagicMock]:
    """Standard fixture: 3 trackers × 2 datasets."""
    return [
        # Dataset A: CSRT best, KCF middle, MOSSE worst
        _make_result("CSRT",  "A", iou=0.70, fps=10, success_auc=0.65),
        _make_result("KCF",   "A", iou=0.55, fps=30, success_auc=0.50),
        _make_result("MOSSE", "A", iou=0.40, fps=80, success_auc=0.35),
        # Dataset B: MOSSE best on IoU, CSRT middle, KCF worst
        _make_result("CSRT",  "B", iou=0.60, fps=10, success_auc=0.55),
        _make_result("KCF",   "B", iou=0.45, fps=30, success_auc=0.40),
        _make_result("MOSSE", "B", iou=0.65, fps=80, success_auc=0.60),
    ]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_primary_metric():
    agg = MultiDatasetAggregator()
    assert agg.primary_metric == "success_auc"


def test_custom_primary_metric():
    agg = MultiDatasetAggregator(primary_metric="mean_iou")
    assert agg.primary_metric == "mean_iou"


def test_custom_metrics_list():
    custom = [("mean_iou", "mIoU", True)]
    agg = MultiDatasetAggregator(metrics=custom)
    assert agg.metrics == custom


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_aggregate_requires_two_trackers():
    agg = MultiDatasetAggregator()
    results = [_make_result("MOSSE", "A", 0.5, 30)]
    with pytest.raises(ValueError, match="at least 2 trackers"):
        agg.aggregate(results)


# ---------------------------------------------------------------------------
# Core aggregation correctness
# ---------------------------------------------------------------------------

def test_aggregate_trackers_and_datasets():
    agg = MultiDatasetAggregator()
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)

    assert set(report.trackers) == {"CSRT", "KCF", "MOSSE"}
    assert set(report.datasets) == {"A", "B"}


def test_aggregate_per_dataset_length():
    agg = MultiDatasetAggregator()
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    # 3 trackers × 2 datasets = 6 PerDatasetRow entries
    assert len(report.per_dataset) == 6


def test_ranks_assigned_on_dataset_a():
    """CSRT has best success_auc on A → rank 1."""
    agg = MultiDatasetAggregator(primary_metric="success_auc")
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)

    ds_a_rows = {r.tracker_name: r for r in report.per_dataset if r.dataset_name == "A"}
    assert ds_a_rows["CSRT"].ranks["success_auc"] == pytest.approx(1.0)
    assert ds_a_rows["KCF"].ranks["success_auc"] == pytest.approx(2.0)
    assert ds_a_rows["MOSSE"].ranks["success_auc"] == pytest.approx(3.0)


def test_ranks_assigned_on_dataset_b():
    """MOSSE has best success_auc on B → rank 1."""
    agg = MultiDatasetAggregator(primary_metric="success_auc")
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)

    ds_b_rows = {r.tracker_name: r for r in report.per_dataset if r.dataset_name == "B"}
    assert ds_b_rows["MOSSE"].ranks["success_auc"] == pytest.approx(1.0)
    assert ds_b_rows["CSRT"].ranks["success_auc"] == pytest.approx(2.0)
    assert ds_b_rows["KCF"].ranks["success_auc"] == pytest.approx(3.0)


def test_avg_rank_csrt_success_auc():
    """CSRT: rank 1 on A, rank 2 on B → avg = 1.5."""
    agg = MultiDatasetAggregator(primary_metric="success_auc")
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    assert report.avg_ranks["CSRT"]["success_auc"] == pytest.approx(1.5)


def test_avg_rank_mosse_success_auc():
    """MOSSE: rank 3 on A, rank 1 on B → avg = 2.0."""
    agg = MultiDatasetAggregator(primary_metric="success_auc")
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    assert report.avg_ranks["MOSSE"]["success_auc"] == pytest.approx(2.0)


def test_final_ranking_order():
    """CSRT has lowest avg rank on success_auc → should be first."""
    agg = MultiDatasetAggregator(primary_metric="success_auc")
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    # CSRT avg=1.5, MOSSE avg=2.0, KCF avg=2.5
    assert report.final_ranking[0] == "CSRT"


def test_avg_metrics_populated():
    agg = MultiDatasetAggregator()
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    # MOSSE mean_iou: (0.40 + 0.65) / 2 = 0.525
    assert report.avg_metrics["MOSSE"]["mean_iou"] == pytest.approx(0.525, abs=1e-4)


def test_summary_dict_structure():
    agg = MultiDatasetAggregator()
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    d = report.summary_dict()
    assert set(d.keys()) == {"CSRT", "KCF", "MOSSE"}
    for tracker, entry in d.items():
        assert "avg_rank" in entry
        assert "avg_metrics" in entry


# ---------------------------------------------------------------------------
# FPS ranking (higher is better)
# ---------------------------------------------------------------------------

def test_fps_rank_order():
    """MOSSE is fastest → rank 1 on FPS."""
    agg = MultiDatasetAggregator(primary_metric="mean_fps")
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    # Both datasets: MOSSE fps=80 > KCF fps=30 > CSRT fps=10
    assert report.avg_ranks["MOSSE"]["mean_fps"] == pytest.approx(1.0)
    assert report.avg_ranks["KCF"]["mean_fps"] == pytest.approx(2.0)
    assert report.avg_ranks["CSRT"]["mean_fps"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Tie handling
# ---------------------------------------------------------------------------

def test_tied_ranks_averaged():
    """Two trackers tied should each receive the average of their tied ranks."""
    # Use two datasets so aggregate() doesn't raise the "≥2 trackers" error
    results2 = [
        _make_result("A_tracker", "DS1", iou=0.5, fps=10, success_auc=0.5),
        _make_result("B_tracker", "DS1", iou=0.5, fps=10, success_auc=0.5),
        _make_result("C_tracker", "DS1", iou=0.7, fps=10, success_auc=0.7),
    ] + [
        _make_result("A_tracker", "DS2", iou=0.5, fps=10, success_auc=0.5),
        _make_result("B_tracker", "DS2", iou=0.5, fps=10, success_auc=0.5),
        _make_result("C_tracker", "DS2", iou=0.7, fps=10, success_auc=0.7),
    ]
    agg = MultiDatasetAggregator(primary_metric="success_auc")
    report = agg.aggregate(results2)

    # A and B tied at rank 2.5 (= (2+3)/2), C at rank 1
    assert report.avg_ranks["C_tracker"]["success_auc"] == pytest.approx(1.0)
    assert report.avg_ranks["A_tracker"]["success_auc"] == pytest.approx(2.5)
    assert report.avg_ranks["B_tracker"]["success_auc"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def test_to_markdown_contains_tracker_names():
    agg = MultiDatasetAggregator()
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    md = agg.to_markdown(report)
    assert "CSRT" in md
    assert "KCF" in md
    assert "MOSSE" in md


def test_to_markdown_contains_dataset_names():
    agg = MultiDatasetAggregator()
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    md = agg.to_markdown(report)
    assert "A" in md
    assert "B" in md


def test_to_markdown_has_table_structure():
    agg = MultiDatasetAggregator()
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    md = agg.to_markdown(report)
    assert "|" in md
    assert "---" in md


def test_to_markdown_empty_metrics():
    """When no metrics match, return graceful fallback string."""
    agg = MultiDatasetAggregator(metrics=[("nonexistent_key", "X", True)])
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    md = agg.to_markdown(report)
    assert "No metrics" in md


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------

def test_to_latex_contains_tabular():
    agg = MultiDatasetAggregator()
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    latex = agg.to_latex(report)
    assert r"\begin{tabular}" in latex
    assert r"\end{tabular}" in latex
    assert r"\toprule" in latex
    assert r"\bottomrule" in latex


def test_to_latex_contains_tracker_names():
    agg = MultiDatasetAggregator()
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    latex = agg.to_latex(report)
    assert "CSRT" in latex
    assert "KCF" in latex
    assert "MOSSE" in latex


def test_to_latex_bolds_best_rank():
    agg = MultiDatasetAggregator(primary_metric="success_auc")
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    latex = agg.to_latex(report)
    # The best avg_rank should be bolded
    assert r"\textbf{" in latex


# ---------------------------------------------------------------------------
# Per-dataset breakdown
# ---------------------------------------------------------------------------

def test_per_dataset_markdown_contains_all_datasets():
    agg = MultiDatasetAggregator()
    results = _three_tracker_two_dataset_results()
    report = agg.aggregate(results)
    breakdown = agg.to_per_dataset_markdown(report)
    assert "### A" in breakdown
    assert "### B" in breakdown


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

def test_aggregate_with_real_benchmark_results():
    """End-to-end: run trackers on two synthetic datasets and aggregate."""
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker

    engine = BenchmarkEngine(verbose=False)
    ds1 = SyntheticDataset(num_sequences=2, num_frames=20, seed=10)
    ds2 = SyntheticDataset(num_sequences=2, num_frames=20, seed=20)

    results = [
        engine.run(MOSSETracker(), ds1, dataset_name="Syn-A"),
        engine.run(KCFTracker(),   ds1, dataset_name="Syn-A"),
        engine.run(MOSSETracker(), ds2, dataset_name="Syn-B"),
        engine.run(KCFTracker(),   ds2, dataset_name="Syn-B"),
    ]

    agg = MultiDatasetAggregator()
    report = agg.aggregate(results)

    assert set(report.trackers) == {"MOSSE", "KCF"}
    assert set(report.datasets) == {"Syn-A", "Syn-B"}
    assert len(report.final_ranking) == 2
    assert report.final_ranking[0] in {"MOSSE", "KCF"}

    md = agg.to_markdown(report)
    assert "MOSSE" in md
    assert "KCF" in md

    latex = agg.to_latex(report)
    assert r"\begin{tabular}" in latex


# ---------------------------------------------------------------------------
# _average_ranks utility
# ---------------------------------------------------------------------------

def test_average_ranks_no_ties():
    sorted_t = [("A", 0.9), ("B", 0.7), ("C", 0.5)]
    ranks = _average_ranks(sorted_t)
    assert ranks == {"A": 1.0, "B": 2.0, "C": 3.0}


def test_average_ranks_all_tied():
    sorted_t = [("A", 0.5), ("B", 0.5), ("C", 0.5)]
    ranks = _average_ranks(sorted_t)
    expected = (1 + 2 + 3) / 3  # = 2.0
    assert ranks == {"A": expected, "B": expected, "C": expected}


def test_average_ranks_partial_tie():
    sorted_t = [("A", 0.9), ("B", 0.5), ("C", 0.5)]
    ranks = _average_ranks(sorted_t)
    assert ranks["A"] == 1.0
    assert ranks["B"] == pytest.approx(2.5)
    assert ranks["C"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# _latex_escape utility
# ---------------------------------------------------------------------------

def test_latex_escape_underscore():
    assert r"\_" in _latex_escape("success_auc")


def test_latex_escape_ampersand():
    assert r"\&" in _latex_escape("a&b")


def test_latex_escape_no_op():
    s = "plaintext"
    assert _latex_escape(s) == s
