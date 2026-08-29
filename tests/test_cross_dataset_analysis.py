"""Tests for CrossDatasetAnalyzer — generalization analysis across datasets."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.analysis.cross_dataset import (
    CrossDatasetAnalyzer,
    GeneralizationEntry,
    GeneralizationReport,
    MetricBundle,
)
from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_profiling(fps: float = 30.0, mem: float = 100.0) -> ProfilingResult:
    lat = 1000.0 / fps if fps > 0 else 0.0
    return ProfilingResult(
        tracker_name="test",
        frame_count=50,
        fps=fps,
        latency_mean_ms=lat,
        latency_std_ms=0.5,
        latency_p95_ms=lat * 1.1,
        latency_p99_ms=lat * 1.2,
        latency_cv=0.02,
        peak_memory_mb=mem,
    )


def _make_result(
    tracker: str,
    dataset: str,
    mean_iou: float,
    fps: float = 30.0,
    mem: float = 150.0,
    n_seq: int = 5,
    n_frames: int = 50,
) -> BenchmarkResult:
    """Build a minimal BenchmarkResult with uniform IoU across all sequences."""
    result = BenchmarkResult(tracker_name=tracker, dataset_name=dataset)
    for i in range(n_seq):
        ious = np.full(n_frames, mean_iou, dtype=np.float64)
        prof = _make_profiling(fps=fps, mem=mem)
        prof.tracker_name = tracker
        result.sequence_results.append(
            SequenceResult(
                sequence_name=f"seq_{i:02d}",
                ious=ious,
                profiling=prof,
            )
        )
    return result


# ---------------------------------------------------------------------------
# MetricBundle extraction
# ---------------------------------------------------------------------------

class TestMetricBundleExtraction:
    def test_extracts_mean_iou(self):
        res = _make_result("MOSSE", "OTB", 0.45)
        analyzer = CrossDatasetAnalyzer()
        bundle = analyzer._extract_bundle(res)
        assert abs(bundle.mean_iou - 0.45) < 1e-6

    def test_ees_positive_for_valid_result(self):
        res = _make_result("KCF", "GOT10k", 0.60, fps=60.0, mem=200.0)
        analyzer = CrossDatasetAnalyzer()
        bundle = analyzer._extract_bundle(res)
        assert bundle.ees > 0.0

    def test_ees_zero_for_zero_fps(self):
        res = _make_result("MOSSE", "OTB", 0.5, fps=0.0)
        analyzer = CrossDatasetAnalyzer()
        bundle = analyzer._extract_bundle(res)
        assert bundle.ees == 0.0


# ---------------------------------------------------------------------------
# CrossDatasetAnalyzer.analyze
# ---------------------------------------------------------------------------

class TestCrossDatasetAnalyzer:
    @pytest.fixture
    def two_dataset_results(self):
        """Two datasets, three trackers with different cross-dataset profiles."""
        return {
            "OTB": [
                _make_result("MOSSE", "OTB", 0.40),
                _make_result("KCF",   "OTB", 0.55),
                _make_result("CSRT",  "OTB", 0.70),
            ],
            "GOT10k": [
                _make_result("MOSSE",  "GOT10k", 0.35),
                _make_result("KCF",    "GOT10k", 0.50),
                _make_result("CSRT",   "GOT10k", 0.60),
            ],
        }

    def test_returns_report_with_correct_datasets(self, two_dataset_results):
        report = CrossDatasetAnalyzer().analyze(two_dataset_results)
        assert set(report.datasets) == {"OTB", "GOT10k"}

    def test_all_trackers_in_entries(self, two_dataset_results):
        report = CrossDatasetAnalyzer().analyze(two_dataset_results)
        names = {e.tracker_name for e in report.entries}
        assert names == {"MOSSE", "KCF", "CSRT"}

    def test_entries_sorted_by_gen_score_descending(self, two_dataset_results):
        report = CrossDatasetAnalyzer().analyze(two_dataset_results)
        scores = [e.generalization_score for e in report.entries]
        assert scores == sorted(scores, reverse=True)

    def test_gen_score_nonnegative(self, two_dataset_results):
        report = CrossDatasetAnalyzer().analyze(two_dataset_results)
        for e in report.entries:
            assert e.generalization_score >= 0.0

    def test_consistency_penalty_zero_for_uniform_tracker(self):
        """A tracker with the same relative rank on every dataset gets penalty ≈ 0."""
        # MOSSE ranks last on both, KCF middle on both, CSRT top on both.
        results = {
            "D1": [
                _make_result("MOSSE", "D1", 0.30),
                _make_result("KCF",   "D1", 0.50),
                _make_result("CSRT",  "D1", 0.70),
            ],
            "D2": [
                _make_result("MOSSE", "D2", 0.30),
                _make_result("KCF",   "D2", 0.50),
                _make_result("CSRT",  "D2", 0.70),
            ],
        }
        report = CrossDatasetAnalyzer().analyze(results)
        # Each tracker has identical normalized IoU on both datasets → penalty ≈ 0.
        for e in report.entries:
            assert e.consistency_penalty < 1e-9

    def test_single_dataset_analysis(self):
        results = {
            "OTB": [
                _make_result("MOSSE", "OTB", 0.40),
                _make_result("KCF",   "OTB", 0.60),
            ],
        }
        report = CrossDatasetAnalyzer().analyze(results)
        assert len(report.entries) == 2
        # Single dataset → no cross-dataset variation → penalty = 0
        for e in report.entries:
            assert e.consistency_penalty == 0.0

    def test_missing_tracker_on_one_dataset(self):
        """A tracker present on only one dataset should still get an entry."""
        results = {
            "OTB": [
                _make_result("MOSSE", "OTB", 0.40),
                _make_result("KCF",   "OTB", 0.55),
            ],
            "GOT10k": [
                _make_result("MOSSE", "GOT10k", 0.35),
                # KCF absent from GOT10k
            ],
        }
        report = CrossDatasetAnalyzer().analyze(results)
        names = {e.tracker_name for e in report.entries}
        assert "KCF" in names

    def test_metrics_dict_populated(self, two_dataset_results):
        report = CrossDatasetAnalyzer().analyze(two_dataset_results)
        for e in report.entries:
            for ds in e.datasets:
                if ds in e.metrics:
                    b = e.metrics[ds]
                    assert isinstance(b, MetricBundle)
                    assert 0.0 <= b.mean_iou <= 1.0


# ---------------------------------------------------------------------------
# GeneralizationReport formatting
# ---------------------------------------------------------------------------

class TestGeneralizationReportFormatting:
    @pytest.fixture
    def simple_report(self):
        results = {
            "OTB": [
                _make_result("MOSSE", "OTB", 0.40),
                _make_result("KCF",   "OTB", 0.60),
            ],
            "LaSOT": [
                _make_result("MOSSE", "LaSOT", 0.35),
                _make_result("KCF",   "LaSOT", 0.55),
            ],
        }
        return CrossDatasetAnalyzer().analyze(results)

    def test_markdown_table_contains_dataset_names(self, simple_report):
        md = simple_report.to_markdown_table()
        assert "OTB" in md
        assert "LaSOT" in md

    def test_markdown_table_contains_tracker_names(self, simple_report):
        md = simple_report.to_markdown_table()
        assert "MOSSE" in md
        assert "KCF" in md

    def test_generalization_ranking_contains_header(self, simple_report):
        ranking = simple_report.to_generalization_ranking()
        assert "Gen. Score" in ranking
        assert "MOSSE" in ranking

    def test_csv_has_correct_number_of_columns(self, simple_report):
        csv = simple_report.to_csv()
        lines = csv.strip().split("\n")
        n_cols = len(lines[0].split(","))
        # tracker + 2 datasets + gen_score + consistency_penalty = 5
        assert n_cols == 5

    def test_latex_table_begins_with_table_env(self, simple_report):
        latex = simple_report.to_latex_table()
        assert latex.startswith("\\begin{table}")
        assert latex.endswith("\\end{table}")

    def test_markdown_table_fps_metric(self, simple_report):
        md = simple_report.to_markdown_table(metric="fps")
        assert "fps" in md.lower() or "FPS" in md


# ---------------------------------------------------------------------------
# compare_pair helper
# ---------------------------------------------------------------------------

class TestComparePair:
    def test_advantage_detected_correctly(self):
        results_mosse = [
            _make_result("MOSSE", "OTB",    0.40),
            _make_result("MOSSE", "GOT10k", 0.45),
        ]
        results_kcf = [
            _make_result("KCF",   "OTB",    0.55),
            _make_result("KCF",   "GOT10k", 0.40),
        ]
        analyzer = CrossDatasetAnalyzer()
        comparison = analyzer.compare_pair(results_mosse, results_kcf, "MOSSE", "KCF")
        assert "OTB" in comparison["advantage_b"]   # KCF wins on OTB
        assert "GOT10k" in comparison["advantage_a"] # MOSSE wins on GOT10k
