"""Tests for eovot.benchmark.multi_tracker — MultiTrackerBenchmark API."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eovot.benchmark.multi_tracker import MultiTrackerBenchmark, MultiTrackerReport
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.camshift import CamShiftTracker
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    return SyntheticDataset(num_sequences=2, num_frames=15, seed=42)


@pytest.fixture
def two_trackers():
    return [MOSSETracker(), KCFTracker()]


@pytest.fixture
def three_trackers():
    return [MOSSETracker(), KCFTracker(), CamShiftTracker()]


# ---------------------------------------------------------------------------
# MultiTrackerBenchmark.run — basic structure
# ---------------------------------------------------------------------------

def test_run_returns_report(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset, dataset_name="Syn")

    assert isinstance(report, MultiTrackerReport)
    assert report.dataset_name == "Syn"
    assert len(report.results) == 2


def test_run_efficiency_ranking_length(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    assert len(report.efficiency_ranking) == 2


def test_three_trackers(small_dataset, three_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(three_trackers, small_dataset)

    assert len(report.results) == 3
    assert len(report.efficiency_ranking) == 3


def test_single_tracker(small_dataset):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run([MOSSETracker()], small_dataset)

    assert len(report.results) == 1
    assert report.efficiency_ranking[0].on_pareto_front


def test_max_sequences_limits_eval(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset, max_sequences=1)

    for r in report.results:
        assert len(r.sequence_results) == 1


def test_tracker_names_preserved(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset, dataset_name="Syn")

    tracker_names = {r.tracker_name for r in report.results}
    assert "MOSSE" in tracker_names
    assert "KCF" in tracker_names


# ---------------------------------------------------------------------------
# MultiTrackerReport.best_tracker
# ---------------------------------------------------------------------------

def test_best_tracker_ees_is_valid(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    best = report.best_tracker("ees")
    assert best in ["MOSSE", "KCF"]


def test_best_tracker_mean_iou(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    best = report.best_tracker("mean_iou")
    assert best in ["MOSSE", "KCF"]


def test_best_tracker_fps(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    best = report.best_tracker("fps")
    assert best in ["MOSSE", "KCF"]


def test_best_tracker_memory(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    # peak_memory_mb: lower is better
    best = report.best_tracker("peak_memory_mb")
    assert best in ["MOSSE", "KCF"]


def test_best_tracker_invalid_metric(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    with pytest.raises(ValueError, match="Unknown metric"):
        report.best_tracker("nonexistent_metric")


def test_best_tracker_empty_report():
    report = MultiTrackerReport(dataset_name="empty")
    with pytest.raises(ValueError, match="no results"):
        report.best_tracker("ees")


def test_best_tracker_ees_empty_ranking():
    report = MultiTrackerReport(dataset_name="empty")
    from eovot.benchmark.engine import BenchmarkResult
    report.results = [BenchmarkResult(tracker_name="T", dataset_name="D")]
    # efficiency_ranking is still empty
    with pytest.raises(ValueError, match="Efficiency ranking"):
        report.best_tracker("ees")


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def test_at_least_one_pareto_tracker(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    pareto_count = sum(1 for e in report.efficiency_ranking if e.on_pareto_front)
    assert pareto_count >= 1


def test_single_tracker_always_pareto(small_dataset):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run([MOSSETracker()], small_dataset)

    assert report.efficiency_ranking[0].on_pareto_front


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def test_markdown_table_contains_headers(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    md = report.to_markdown_table()
    assert "Tracker" in md
    assert "EES" in md
    assert "mIoU" in md
    assert "Pareto" in md


def test_markdown_table_contains_tracker_names(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    md = report.to_markdown_table()
    assert "MOSSE" in md or "KCF" in md


def test_markdown_table_row_count(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    lines = [l for l in report.to_markdown_table().splitlines() if l.startswith("|")]
    # header + separator + 2 data rows
    assert len(lines) == 4


def test_markdown_table_empty_report():
    report = MultiTrackerReport(dataset_name="empty")
    md = report.to_markdown_table()
    assert "No results" in md


# ---------------------------------------------------------------------------
# summary_table
# ---------------------------------------------------------------------------

def test_summary_table_length(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    table = report.summary_table()
    assert len(table) == 2


def test_summary_table_keys(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    for row in report.summary_table():
        assert "mean_iou" in row
        assert "mean_fps" in row
        assert "tracker" in row


# ---------------------------------------------------------------------------
# Serialisation: to_dict / save / load
# ---------------------------------------------------------------------------

def test_to_dict_structure(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset, dataset_name="Synthetic")

    d = report.to_dict()
    assert d["dataset_name"] == "Synthetic"
    assert len(d["trackers"]) == 2
    assert len(d["efficiency_ranking"]) == 2


def test_to_dict_efficiency_ranking_fields(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    d = report.to_dict()
    for entry in d["efficiency_ranking"]:
        assert "tracker_name" in entry
        assert "ees" in entry
        assert "on_pareto_front" in entry
        assert "rank" in entry


def test_save_creates_json_file(tmp_path, small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    out = report.save(tmp_path / "report")
    assert out.exists()
    assert out.suffix == ".json"


def test_save_json_is_valid(tmp_path, small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset, dataset_name="Syn")

    out = report.save(tmp_path / "report.json")
    with open(out) as f:
        data = json.load(f)
    assert data["dataset_name"] == "Syn"
    assert len(data["efficiency_ranking"]) == 2


def test_save_appends_json_extension(tmp_path, small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    out = report.save(tmp_path / "no_ext")
    assert out.suffix == ".json"


def test_save_creates_parent_dirs(tmp_path, small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False)
    report = bench.run(two_trackers, small_dataset)

    nested = tmp_path / "deep" / "nested" / "report.json"
    out = report.save(nested)
    assert out.exists()


# ---------------------------------------------------------------------------
# Energy profiling integration
# ---------------------------------------------------------------------------

def test_energy_profiling_enabled(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False, tdp_watts=15.0)
    report = bench.run(two_trackers, small_dataset)

    # With energy enabled, total_energy_j should be populated
    for r in report.results:
        assert r.total_energy_j is not None
        assert r.total_energy_j >= 0.0


# ---------------------------------------------------------------------------
# Memory budget
# ---------------------------------------------------------------------------

def test_custom_memory_budget(small_dataset, two_trackers):
    bench = MultiTrackerBenchmark(verbose=False, memory_budget_mb=128.0)
    report = bench.run(two_trackers, small_dataset)

    # EES values depend on memory_budget_mb; just ensure they're non-negative
    for e in report.efficiency_ranking:
        assert e.ees >= 0.0
