"""Tests for FullBenchmarkEvaluator and FullEvaluationReport."""
from __future__ import annotations

import json

import pytest

from eovot.benchmark.full_evaluator import FullBenchmarkEvaluator, FullEvaluationReport
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


@pytest.fixture()
def small_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=30, seed=0)


@pytest.fixture()
def two_trackers():
    return [MOSSETracker(), KCFTracker()]


def test_returns_full_evaluation_report(small_dataset, two_trackers):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run(two_trackers, small_dataset, dataset_name="SynTest")
    assert isinstance(report, FullEvaluationReport)
    assert len(report.tracker_reports) == 2


def test_all_accuracy_metrics_present(small_dataset, two_trackers):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run(two_trackers, small_dataset, dataset_name="SynTest")
    for r in report.tracker_reports:
        br = r.benchmark_result
        assert 0.0 <= br.mean_iou <= 1.0
        assert br.mean_fps > 0.0
        assert br.peak_memory_mb >= 0.0


def test_robustness_metrics_present(small_dataset, two_trackers):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run(two_trackers, small_dataset, dataset_name="SynTest")
    for r in report.tracker_reports:
        assert r.total_failures >= 0
        assert 0.0 <= r.mean_eao <= 1.0
        assert 0.0 <= r.mean_survival_rate <= 1.0
        assert r.mean_recovery_lag_frames >= 0.0


def test_temporal_consistency_metrics_present(small_dataset, two_trackers):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run(two_trackers, small_dataset, dataset_name="SynTest")
    for r in report.tracker_reports:
        assert 0.0 <= r.mean_smoothness_score <= 1.0
        assert r.mean_position_jitter >= 0.0
        assert r.mean_scale_jitter >= 0.0
        assert r.mean_velocity_outlier_ratio >= 0.0


def test_efficiency_ranking_length(small_dataset, two_trackers):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run(two_trackers, small_dataset, dataset_name="SynTest")
    assert len(report.efficiency_ranking) == 2


def test_efficiency_ranking_sorted_descending(small_dataset, two_trackers):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run(two_trackers, small_dataset, dataset_name="SynTest")
    ees = [e.ees for e in report.efficiency_ranking]
    assert ees == sorted(ees, reverse=True)


def test_efficiency_entry_wired_into_tracker_reports(small_dataset, two_trackers):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run(two_trackers, small_dataset, dataset_name="SynTest")
    for r in report.tracker_reports:
        assert r.efficiency_entry is not None
        assert r.efficiency_entry.tracker_name == r.tracker_name


def test_summary_markdown_contains_all_sections(small_dataset, two_trackers):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run(two_trackers, small_dataset, dataset_name="SynTest")
    md = report.summary_markdown()
    assert "Accuracy" in md
    assert "Robustness" in md
    assert "Temporal Consistency" in md
    assert "Edge Efficiency" in md


def test_save_and_reload(tmp_path, small_dataset, two_trackers):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run(two_trackers, small_dataset, dataset_name="SynTest")
    out = report.save(tmp_path / "full_report")
    assert out.exists()
    with open(out) as f:
        data = json.load(f)
    assert "trackers" in data
    assert len(data["trackers"]) == 2
    assert "efficiency_ranking" in data
    assert len(data["efficiency_ranking"]) == 2


def test_single_tracker(small_dataset):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run([MOSSETracker()], small_dataset, dataset_name="SynTest")
    assert len(report.tracker_reports) == 1
    assert len(report.efficiency_ranking) == 1


def test_max_sequences_respected(small_dataset, two_trackers):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run(two_trackers, small_dataset, dataset_name="SynTest", max_sequences=2)
    for r in report.tracker_reports:
        assert len(r.benchmark_result.sequence_results) == 2


def test_wall_clock_positive(small_dataset, two_trackers):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run(two_trackers, small_dataset, dataset_name="SynTest")
    assert report.wall_clock_seconds > 0.0


def test_to_dict_structure(small_dataset, two_trackers):
    evaluator = FullBenchmarkEvaluator(verbose=False)
    report = evaluator.run(two_trackers, small_dataset, dataset_name="SynTest")
    for tr in report.tracker_reports:
        d = tr.to_dict()
        assert "accuracy" in d
        assert "robustness" in d
        assert "temporal_consistency" in d
        assert "efficiency" in d
