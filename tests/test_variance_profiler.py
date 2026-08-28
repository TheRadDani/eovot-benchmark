"""Tests for eovot.analysis.variance_profiler."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.analysis.variance_profiler import MetricStats, VarianceProfiler, VarianceReport
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# MetricStats
# ---------------------------------------------------------------------------

def test_metric_stats_cv_zero_mean():
    """CV must be 0.0 when mean is zero to avoid division by zero."""
    profiler = VarianceProfiler(n_runs=2)
    stats = profiler._metric_stats("x", [0.0, 0.0])
    assert stats.cv == 0.0


def test_metric_stats_str_contains_name():
    profiler = VarianceProfiler(n_runs=2)
    stats = profiler._metric_stats("fps", [30.0, 35.0])
    assert "fps" in str(stats)


# ---------------------------------------------------------------------------
# VarianceProfiler
# ---------------------------------------------------------------------------

def test_variance_profiler_requires_at_least_two_runs():
    with pytest.raises(ValueError, match="n_runs"):
        VarianceProfiler(n_runs=1)


def test_variance_profiler_returns_report():
    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=0)
    engine = BenchmarkEngine(verbose=False)
    profiler = VarianceProfiler(n_runs=3, seed=0)
    report = profiler.profile(MOSSETracker(), dataset, dataset_name="Synth", engine=engine)

    assert isinstance(report, VarianceReport)
    assert report.tracker_name == "MOSSE"
    assert report.dataset_name == "Synth"
    assert report.n_runs == 3
    assert "fps" in report.metrics
    assert "mean_iou" in report.metrics
    assert "peak_memory_mb" in report.metrics


def test_variance_profiler_metrics_are_positive():
    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=1)
    engine = BenchmarkEngine(verbose=False)
    profiler = VarianceProfiler(n_runs=2, seed=1)
    report = profiler.profile(MOSSETracker(), dataset, dataset_name="Synth", engine=engine)

    assert report.metrics["fps"].mean > 0
    assert report.metrics["mean_iou"].mean >= 0
    assert report.metrics["peak_memory_mb"].mean > 0


def test_variance_profiler_ci_lo_le_ci_hi():
    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=2)
    engine = BenchmarkEngine(verbose=False)
    profiler = VarianceProfiler(n_runs=2, seed=2)
    report = profiler.profile(MOSSETracker(), dataset, dataset_name="Synth", engine=engine)

    for stats in report.metrics.values():
        assert stats.ci_lo <= stats.ci_hi, f"CI inverted for {stats.name}"


def test_variance_profiler_reliable_flag_type():
    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=10)
    engine = BenchmarkEngine(verbose=False)
    profiler = VarianceProfiler(n_runs=2, seed=10)
    report = profiler.profile(MOSSETracker(), dataset, dataset_name="Synth", engine=engine)
    assert isinstance(report.reliable, bool)


# ---------------------------------------------------------------------------
# VarianceReport
# ---------------------------------------------------------------------------

def test_variance_report_str_contains_tracker_name():
    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=11)
    engine = BenchmarkEngine(verbose=False)
    profiler = VarianceProfiler(n_runs=2, seed=11)
    report = profiler.profile(MOSSETracker(), dataset, dataset_name="Synth", engine=engine)
    assert "MOSSE" in str(report)


def test_variance_report_to_markdown():
    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=5)
    engine = BenchmarkEngine(verbose=False)
    profiler = VarianceProfiler(n_runs=2, seed=5)
    report = profiler.profile(MOSSETracker(), dataset, dataset_name="Synth", engine=engine)
    md = report.to_markdown()
    assert "Metric" in md
    assert "fps" in md
    assert "CV" in md


def test_variance_report_max_sequences():
    dataset = SyntheticDataset(num_sequences=5, num_frames=20, seed=6)
    engine = BenchmarkEngine(verbose=False)
    profiler = VarianceProfiler(n_runs=2, seed=6)
    report = profiler.profile(
        MOSSETracker(), dataset, dataset_name="Synth", engine=engine, max_sequences=2
    )
    assert report.n_runs == 2
