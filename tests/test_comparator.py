"""Tests for BenchmarkComparator.

Validates:
- Metric delta computation (absolute + relative)
- Regression / improvement classification
- Threshold sensitivity
- All-better, all-worse, and mixed scenarios
- Per-sequence delta computation and matching
- Output format helpers: Markdown, CSV, JSON/dict
- Edge cases: missing metrics, identical runs, single-sequence runs
"""

from __future__ import annotations

import csv
import io
import json
import math
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

from eovot.analysis.comparator import (
    BenchmarkComparator,
    ComparisonResult,
    MetricDelta,
    SequenceDelta,
)
from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profiling(fps: float = 30.0, mem_mb: float = 50.0) -> ProfilingResult:
    lat = 1000.0 / fps if fps > 0 else 0.0
    return ProfilingResult(
        tracker_name="test",
        frame_count=50,
        fps=fps,
        latency_mean_ms=lat,
        latency_std_ms=lat * 0.1,
        latency_p95_ms=lat * 1.2,
        latency_p99_ms=lat * 1.5,
        latency_cv=0.1,
        peak_memory_mb=mem_mb,
    )


def _make_seq_result(
    name: str,
    iou: float,
    fps: float = 30.0,
    mem_mb: float = 50.0,
) -> SequenceResult:
    n = 50
    ious = np.full(n, iou, dtype=np.float64)
    return SequenceResult(
        sequence_name=name,
        ious=ious,
        profiling=_make_profiling(fps=fps, mem_mb=mem_mb),
    )


def _make_result(
    tracker: str = "TestTracker",
    dataset: str = "Synthetic",
    seqs: List[SequenceResult] = None,
) -> BenchmarkResult:
    r = BenchmarkResult(tracker_name=tracker, dataset_name=dataset)
    if seqs:
        r.sequence_results = seqs
    return r


def _baseline() -> BenchmarkResult:
    return _make_result(
        tracker="KCF",
        seqs=[
            _make_seq_result("car1", iou=0.40, fps=120.0, mem_mb=80.0),
            _make_seq_result("person1", iou=0.50, fps=120.0, mem_mb=80.0),
            _make_seq_result("bird1", iou=0.30, fps=120.0, mem_mb=80.0),
        ],
    )


def _candidate_better() -> BenchmarkResult:
    """Candidate with higher IoU but also higher memory."""
    return _make_result(
        tracker="KCF-v2",
        seqs=[
            _make_seq_result("car1", iou=0.45, fps=130.0, mem_mb=90.0),
            _make_seq_result("person1", iou=0.56, fps=130.0, mem_mb=90.0),
            _make_seq_result("bird1", iou=0.36, fps=130.0, mem_mb=90.0),
        ],
    )


def _candidate_worse() -> BenchmarkResult:
    """Candidate with lower IoU and lower FPS."""
    return _make_result(
        tracker="KCF-bad",
        seqs=[
            _make_seq_result("car1", iou=0.30, fps=80.0, mem_mb=60.0),
            _make_seq_result("person1", iou=0.40, fps=80.0, mem_mb=60.0),
            _make_seq_result("bird1", iou=0.20, fps=80.0, mem_mb=60.0),
        ],
    )


def _candidate_identical() -> BenchmarkResult:
    """Candidate with exactly the same values as baseline."""
    return _make_result(
        tracker="KCF",
        seqs=[
            _make_seq_result("car1", iou=0.40, fps=120.0, mem_mb=80.0),
            _make_seq_result("person1", iou=0.50, fps=120.0, mem_mb=80.0),
            _make_seq_result("bird1", iou=0.30, fps=120.0, mem_mb=80.0),
        ],
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestComparatorConstruction:
    def test_default_threshold(self):
        cmp = BenchmarkComparator()
        assert cmp.regression_threshold == 0.01

    def test_custom_threshold(self):
        cmp = BenchmarkComparator(regression_threshold=0.05)
        assert cmp.regression_threshold == 0.05

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="regression_threshold"):
            BenchmarkComparator(regression_threshold=-0.01)

    def test_zero_threshold_allowed(self):
        # Threshold of 0 means any change is flagged.
        cmp = BenchmarkComparator(regression_threshold=0.0)
        assert cmp.regression_threshold == 0.0


# ---------------------------------------------------------------------------
# Metric delta computation
# ---------------------------------------------------------------------------

class TestMetricDeltaComputation:
    def test_positive_delta_for_better_iou(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        iou_delta = next(d for d in result.metric_deltas if d.metric == "mean_iou")
        assert iou_delta.delta > 0

    def test_negative_delta_for_worse_iou(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_worse())
        iou_delta = next(d for d in result.metric_deltas if d.metric == "mean_iou")
        assert iou_delta.delta < 0

    def test_zero_delta_for_identical_runs(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_identical())
        iou_delta = next(d for d in result.metric_deltas if d.metric == "mean_iou")
        assert math.isclose(iou_delta.delta, 0.0, abs_tol=1e-9)

    def test_pct_change_computed_correctly(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        iou_delta = next(d for d in result.metric_deltas if d.metric == "mean_iou")
        # baseline mean_iou ~ 0.40, candidate ~ 0.4567
        expected_pct = (iou_delta.delta / abs(iou_delta.baseline)) * 100
        assert math.isclose(iou_delta.pct_change, expected_pct, rel_tol=1e-5)

    def test_baseline_and_candidate_populated(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        for d in result.metric_deltas:
            assert d.baseline is not None
            assert d.candidate is not None

    def test_all_required_metric_keys_present(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        metric_names = {d.metric for d in result.metric_deltas}
        # At minimum these two must always be present.
        assert "mean_iou" in metric_names
        assert "mean_fps" in metric_names
        assert "peak_memory_mb" in metric_names


# ---------------------------------------------------------------------------
# Regression / improvement classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_improved_when_iou_rises(self):
        cmp = BenchmarkComparator(regression_threshold=0.01)
        result = cmp.compare(_baseline(), _candidate_better())
        iou_delta = next(d for d in result.metric_deltas if d.metric == "mean_iou")
        assert iou_delta.improved is True
        assert iou_delta.regressed is False

    def test_regressed_when_iou_falls(self):
        cmp = BenchmarkComparator(regression_threshold=0.01)
        result = cmp.compare(_baseline(), _candidate_worse())
        iou_delta = next(d for d in result.metric_deltas if d.metric == "mean_iou")
        assert iou_delta.regressed is True
        assert iou_delta.improved is False

    def test_no_flag_for_identical_runs(self):
        cmp = BenchmarkComparator(regression_threshold=0.01)
        result = cmp.compare(_baseline(), _candidate_identical())
        for d in result.metric_deltas:
            assert not d.improved
            assert not d.regressed

    def test_has_improvements_flag(self):
        cmp = BenchmarkComparator(regression_threshold=0.01)
        result = cmp.compare(_baseline(), _candidate_better())
        assert result.has_improvements is True

    def test_has_regressions_flag(self):
        cmp = BenchmarkComparator(regression_threshold=0.01)
        result = cmp.compare(_baseline(), _candidate_worse())
        assert result.has_regressions is True

    def test_no_regressions_for_identical(self):
        cmp = BenchmarkComparator(regression_threshold=0.01)
        result = cmp.compare(_baseline(), _candidate_identical())
        assert not result.has_regressions
        assert not result.has_improvements

    def test_memory_higher_is_bad(self):
        """Memory is lower-is-better: higher candidate memory = regression."""
        cmp = BenchmarkComparator(regression_threshold=0.01)
        result = cmp.compare(_baseline(), _candidate_better())
        mem_delta = next(d for d in result.metric_deltas if d.metric == "peak_memory_mb")
        # candidate has higher memory → should be flagged as regression
        assert mem_delta.regressed is True
        assert mem_delta.improved is False

    def test_threshold_sensitivity(self):
        """A tight threshold (0%) should flag any non-zero change."""
        cmp = BenchmarkComparator(regression_threshold=0.0)
        result = cmp.compare(_baseline(), _candidate_better())
        iou_delta = next(d for d in result.metric_deltas if d.metric == "mean_iou")
        assert iou_delta.improved is True

    def test_large_threshold_suppresses_small_change(self):
        """A very large threshold should suppress the ~14 % IoU improvement."""
        cmp = BenchmarkComparator(regression_threshold=0.99)
        result = cmp.compare(_baseline(), _candidate_better())
        assert not result.has_improvements
        assert not result.has_regressions


# ---------------------------------------------------------------------------
# Direction symbol
# ---------------------------------------------------------------------------

class TestDirectionSymbol:
    def test_improved_symbol(self):
        d = MetricDelta(
            metric="mean_iou", label="Mean IoU",
            baseline=0.4, candidate=0.5, delta=0.1, pct_change=25.0,
            higher_is_better=True, improved=True, regressed=False,
        )
        assert d.direction_symbol() == "↑"

    def test_regressed_symbol(self):
        d = MetricDelta(
            metric="mean_iou", label="Mean IoU",
            baseline=0.5, candidate=0.4, delta=-0.1, pct_change=-20.0,
            higher_is_better=True, improved=False, regressed=True,
        )
        assert d.direction_symbol() == "↓"

    def test_unchanged_symbol(self):
        d = MetricDelta(
            metric="mean_iou", label="Mean IoU",
            baseline=0.4, candidate=0.4, delta=0.0, pct_change=0.0,
            higher_is_better=True, improved=False, regressed=False,
        )
        assert d.direction_symbol() == "="


# ---------------------------------------------------------------------------
# Sequence-level deltas
# ---------------------------------------------------------------------------

class TestSequenceDeltas:
    def test_seq_deltas_computed_for_common_seqs(self):
        cmp = BenchmarkComparator(sequence_match_by_name=True)
        result = cmp.compare(_baseline(), _candidate_better())
        assert len(result.sequence_deltas) == 3

    def test_seq_deltas_skipped_when_disabled(self):
        cmp = BenchmarkComparator(sequence_match_by_name=False)
        result = cmp.compare(_baseline(), _candidate_better())
        assert len(result.sequence_deltas) == 0

    def test_seq_delta_values_correct(self):
        cmp = BenchmarkComparator(sequence_match_by_name=True)
        result = cmp.compare(_baseline(), _candidate_better())
        car1 = next(s for s in result.sequence_deltas if s.name == "car1")
        assert math.isclose(car1.baseline_iou, 0.40, abs_tol=1e-4)
        assert math.isclose(car1.candidate_iou, 0.45, abs_tol=1e-4)
        assert math.isclose(car1.delta_iou, 0.05, abs_tol=1e-4)

    def test_seq_delta_improved_property(self):
        cmp = BenchmarkComparator(sequence_match_by_name=True)
        result = cmp.compare(_baseline(), _candidate_better())
        for s in result.sequence_deltas:
            assert s.improved

    def test_seq_delta_regressed_property(self):
        cmp = BenchmarkComparator(sequence_match_by_name=True)
        result = cmp.compare(_baseline(), _candidate_worse())
        for s in result.sequence_deltas:
            assert s.regressed

    def test_unmatched_sequences_excluded(self):
        b = _make_result(
            seqs=[_make_seq_result("seq_a", 0.5), _make_seq_result("seq_b", 0.4)]
        )
        c = _make_result(
            seqs=[_make_seq_result("seq_b", 0.45), _make_seq_result("seq_c", 0.6)]
        )
        cmp = BenchmarkComparator(sequence_match_by_name=True)
        result = cmp.compare(b, c)
        # Only seq_b is common
        assert len(result.sequence_deltas) == 1
        assert result.sequence_deltas[0].name == "seq_b"


# ---------------------------------------------------------------------------
# summary() and regression_report()
# ---------------------------------------------------------------------------

class TestSummaryOutput:
    def test_summary_keys_match_metric_names(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        s = result.summary()
        for d in result.metric_deltas:
            assert d.metric in s

    def test_regression_report_empty_when_no_regressions(self):
        cmp = BenchmarkComparator(regression_threshold=0.01)
        result = cmp.compare(_baseline(), _candidate_identical())
        report = result.regression_report()
        assert "No regressions" in report

    def test_regression_report_contains_metric_names(self):
        cmp = BenchmarkComparator(regression_threshold=0.01)
        result = cmp.compare(_baseline(), _candidate_worse())
        report = result.regression_report()
        assert "Mean IoU" in report or "FPS" in report


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

class TestMarkdownOutput:
    def test_markdown_contains_tracker_names(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        md = cmp.to_markdown_table(result)
        assert "KCF" in md
        assert "KCF-v2" in md

    def test_markdown_contains_metric_labels(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        md = cmp.to_markdown_table(result)
        assert "Mean IoU" in md
        assert "FPS" in md

    def test_markdown_has_status_line(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_identical())
        md = cmp.to_markdown_table(result)
        assert "no significant change" in md

    def test_markdown_regression_warning(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_worse())
        md = cmp.to_markdown_table(result)
        assert "regressions detected" in md

    def test_markdown_sequence_tables_present(self):
        cmp = BenchmarkComparator(sequence_match_by_name=True)
        result = cmp.compare(_baseline(), _candidate_better())
        md = cmp.to_markdown_table(result)
        assert "IoU drops" in md or "IoU gains" in md


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

class TestCsvOutput:
    def test_csv_parseable(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        csv_str = cmp.to_csv(result)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == len(result.metric_deltas)

    def test_csv_header_present(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        csv_str = cmp.to_csv(result)
        first_line = csv_str.splitlines()[0]
        assert "metric" in first_line
        assert "delta" in first_line

    def test_csv_tracker_names_in_rows(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        csv_str = cmp.to_csv(result)
        assert "KCF" in csv_str
        assert "KCF-v2" in csv_str


# ---------------------------------------------------------------------------
# Dict / JSON output
# ---------------------------------------------------------------------------

class TestDictOutput:
    def test_dict_json_serialisable(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        d = cmp.to_dict(result)
        # Must not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_dict_has_regression_flag(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_worse())
        d = cmp.to_dict(result)
        assert d["has_regressions"] is True

    def test_dict_has_improvement_flag(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        d = cmp.to_dict(result)
        assert d["has_improvements"] is True

    def test_dict_sequence_deltas_present(self):
        cmp = BenchmarkComparator(sequence_match_by_name=True)
        result = cmp.compare(_baseline(), _candidate_better())
        d = cmp.to_dict(result)
        assert len(d["sequence_deltas"]) == 3


# ---------------------------------------------------------------------------
# Persistence round-trip (save → load → compare)
# ---------------------------------------------------------------------------

class TestPersistenceRoundTrip:
    def test_compare_from_saved_json(self):
        baseline = _baseline()
        candidate = _candidate_better()
        with tempfile.TemporaryDirectory() as tmp:
            b_path = Path(tmp) / "baseline.json"
            c_path = Path(tmp) / "candidate.json"
            baseline.save(b_path)
            candidate.save(c_path)

            b_loaded = BenchmarkResult.load(b_path)
            c_loaded = BenchmarkResult.load(c_path)

        cmp = BenchmarkComparator()
        result = cmp.compare(b_loaded, c_loaded)
        assert result.has_improvements
        iou_delta = next(d for d in result.metric_deltas if d.metric == "mean_iou")
        assert iou_delta.delta > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_sequence_list(self):
        b = _make_result(tracker="T1", seqs=[])
        c = _make_result(tracker="T2", seqs=[])
        cmp = BenchmarkComparator()
        result = cmp.compare(b, c)
        # mean_iou for empty result is 0.0; both are 0 so no change.
        assert not result.has_regressions
        assert not result.has_improvements

    def test_single_sequence(self):
        b = _make_result(seqs=[_make_seq_result("only", 0.4)])
        c = _make_result(seqs=[_make_seq_result("only", 0.5)])
        cmp = BenchmarkComparator()
        result = cmp.compare(b, c)
        assert result.has_improvements

    def test_metadata_preserved_in_result(self):
        cmp = BenchmarkComparator()
        result = cmp.compare(_baseline(), _candidate_better())
        assert result.baseline_tracker == "KCF"
        assert result.candidate_tracker == "KCF-v2"
        assert result.dataset == "Synthetic"
