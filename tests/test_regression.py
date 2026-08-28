"""Tests for eovot.analysis.regression."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.analysis.regression import BenchmarkDiff, DiffReport, RegressionError, SequenceDelta
from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(tracker_name: str, n_sequences: int = 3, n_frames: int = 30, seed: int = 0):
    """Run MOSSE on a small synthetic dataset and return BenchmarkResult."""
    dataset = SyntheticDataset(num_sequences=n_sequences, num_frames=n_frames, seed=seed)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")
    return result


# ---------------------------------------------------------------------------
# BenchmarkDiff constructor
# ---------------------------------------------------------------------------

def test_invalid_regression_threshold():
    with pytest.raises(ValueError, match="regression_threshold"):
        BenchmarkDiff(regression_threshold=-0.01)


def test_invalid_regression_rate_limit():
    with pytest.raises(ValueError, match="regression_rate_limit"):
        BenchmarkDiff(regression_rate_limit=1.5)


# ---------------------------------------------------------------------------
# compare: identical runs should yield zero delta and be accepted
# ---------------------------------------------------------------------------

def test_identical_runs_zero_delta():
    r = _run("MOSSE", seed=0)
    diff = BenchmarkDiff()
    report = diff.compare(r, r)

    assert report.iou_delta == pytest.approx(0.0)
    assert report.fps_delta == pytest.approx(0.0)
    assert report.num_regressed == 0
    assert report.accepted is True


def test_identical_runs_all_stable():
    r = _run("MOSSE", seed=0)
    diff = BenchmarkDiff(regression_threshold=0.01)
    report = diff.compare(r, r)

    assert all(d.status == "stable" for d in report.sequence_deltas)


# ---------------------------------------------------------------------------
# compare: two different seeds give different results
# ---------------------------------------------------------------------------

def test_different_seeds_produce_report():
    r1 = _run("MOSSE", seed=0)
    r2 = _run("MOSSE", seed=5)
    diff = BenchmarkDiff()
    report = diff.compare(r1, r2)

    assert isinstance(report, DiffReport)
    assert len(report.sequence_deltas) == 3   # 3 sequences in common


# ---------------------------------------------------------------------------
# DiffReport: acceptance logic
# ---------------------------------------------------------------------------

def test_accepted_when_regression_rate_below_limit():
    r = _run("MOSSE", seed=0)
    diff = BenchmarkDiff(regression_rate_limit=1.0)  # always accept
    report = diff.compare(r, r)
    assert report.accepted is True


def test_rejected_when_regression_rate_above_limit():
    r1 = _run("MOSSE", seed=0)
    r2 = _run("MOSSE", seed=7)
    diff = BenchmarkDiff(regression_rate_limit=0.0)  # never accept any regression
    report = diff.compare(r1, r2)
    # With limit=0 any single regressed sequence causes rejection
    if report.num_regressed > 0:
        assert report.accepted is False


def test_require_iou_improvement_rejects_degraded():
    r1 = _run("MOSSE", seed=0)
    r2 = _run("MOSSE", seed=7)
    diff = BenchmarkDiff(require_iou_improvement=True, regression_rate_limit=1.0)
    report = diff.compare(r1, r2)
    # If candidate mIoU < baseline mIoU, must be rejected
    if report.iou_delta < 0:
        assert report.accepted is False


# ---------------------------------------------------------------------------
# DiffReport.raise_if_regressed
# ---------------------------------------------------------------------------

def test_raise_if_regressed_does_not_raise_when_accepted():
    r = _run("MOSSE", seed=0)
    diff = BenchmarkDiff()
    report = diff.compare(r, r)
    report.raise_if_regressed()  # should not raise


def test_raise_if_regressed_raises_regression_error():
    # Force rejection by setting limit=0 and finding a pair with regressions
    r1 = _run("MOSSE", seed=0)
    r2 = _run("MOSSE", seed=9)
    diff = BenchmarkDiff(regression_rate_limit=0.0, regression_threshold=0.001)
    report = diff.compare(r1, r2)
    if not report.accepted:
        with pytest.raises(RegressionError):
            report.raise_if_regressed()


# ---------------------------------------------------------------------------
# DiffReport: string and Markdown output
# ---------------------------------------------------------------------------

def test_str_contains_tracker_and_dataset():
    r = _run("MOSSE", seed=0)
    diff = BenchmarkDiff()
    report = diff.compare(r, r)
    s = str(report)
    assert "MOSSE" in s
    assert "Synthetic" in s


def test_to_markdown_contains_table_header():
    r = _run("MOSSE", seed=0)
    diff = BenchmarkDiff()
    report = diff.compare(r, r)
    md = report.to_markdown()
    assert "Benchmark Regression Report" in md
    assert "| Sequence |" in md
    assert "ΔIoU" in md
