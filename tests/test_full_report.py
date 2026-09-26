"""Tests for eovot.reporting.full_report.FullBenchmarkReport.

Exercises all metric dimensions (accuracy, robustness, temporal consistency,
edge efficiency), output formats (Markdown, CSV, JSON), and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.reporting.full_report import FullBenchmarkReport, TrackerRow, _fmt
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_tracker_results():
    """Run MOSSE and KCF on a small synthetic dataset."""
    ds = SyntheticDataset(num_sequences=4, num_frames=60, motion="linear")
    engine = BenchmarkEngine(verbose=False)
    mosse = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")
    kcf   = engine.run(KCFTracker(),   ds, dataset_name="Synthetic")
    return [mosse, kcf]


@pytest.fixture(scope="module")
def report(two_tracker_results):
    return FullBenchmarkReport(two_tracker_results)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            FullBenchmarkReport([])

    def test_rows_count(self, report):
        assert len(report.rows) == 2

    def test_rows_sorted_by_ees(self, report):
        eess = [r.ees for r in report.rows]
        assert eess == sorted(eess, reverse=True)


# ---------------------------------------------------------------------------
# TrackerRow fields
# ---------------------------------------------------------------------------

class TestTrackerRowFields:
    def test_tracker_names_present(self, report):
        names = {r.tracker_name for r in report.rows}
        assert "MOSSE" in names
        assert "KCF" in names

    def test_mean_iou_in_range(self, report):
        for row in report.rows:
            assert 0.0 <= row.mean_iou <= 1.0

    def test_mean_fps_positive(self, report):
        for row in report.rows:
            assert row.mean_fps > 0

    def test_ees_positive(self, report):
        for row in report.rows:
            assert row.ees >= 0

    def test_eao_in_range(self, report):
        for row in report.rows:
            assert 0.0 <= row.eao <= 1.0

    def test_total_failures_non_negative(self, report):
        for row in report.rows:
            assert row.total_failures >= 0

    def test_survival_rate_in_range(self, report):
        for row in report.rows:
            assert 0.0 <= row.mean_survival_rate <= 1.0

    def test_smoothness_score_populated(self, report):
        for row in report.rows:
            # The benchmark engine stores predictions, so smoothness should be present
            assert row.smoothness_score is not None
            assert 0.0 < row.smoothness_score <= 1.0

    def test_pareto_front_has_at_least_one(self, report):
        pareto = [r for r in report.rows if r.on_pareto_front]
        assert len(pareto) >= 1


# ---------------------------------------------------------------------------
# TrackerRow.to_dict
# ---------------------------------------------------------------------------

class TestTrackerRowToDict:
    def test_all_required_keys(self, report):
        required = {
            "tracker", "dataset", "mean_iou", "mean_fps", "peak_memory_mb",
            "ees", "eao", "total_failures", "mean_survival_rate",
            "smoothness_score", "position_jitter", "scale_jitter",
        }
        for row in report.rows:
            d = row.to_dict()
            assert required.issubset(d.keys())

    def test_pareto_front_is_bool(self, report):
        for row in report.rows:
            assert isinstance(row.to_dict()["on_pareto_front"], bool)


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

class TestToMarkdown:
    def test_contains_section_headers(self, report):
        md = report.to_markdown()
        assert "### Accuracy" in md
        assert "### Robustness" in md
        assert "### Edge Efficiency" in md

    def test_contains_tracker_names(self, report):
        md = report.to_markdown()
        assert "MOSSE" in md
        assert "KCF" in md

    def test_temporal_section_present_with_predictions(self, report):
        md = report.to_markdown()
        assert "### Temporal Consistency" in md

    def test_custom_title(self, report):
        md = report.to_markdown(title="My Title")
        assert "## My Title" in md

    def test_bold_best_value(self, report):
        md = report.to_markdown()
        assert "**" in md  # At least one value should be bolded

    def test_markdown_is_string(self, report):
        md = report.to_markdown()
        assert isinstance(md, str) and len(md) > 0


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

class TestToCSV:
    def test_file_created(self, report, tmp_path):
        path = report.to_csv(tmp_path / "out.csv")
        assert path.is_file()

    def test_csv_has_correct_row_count(self, report, tmp_path):
        import csv as csv_module
        path = report.to_csv(tmp_path / "out.csv")
        with open(path) as f:
            rows = list(csv_module.DictReader(f))
        assert len(rows) == len(report.rows)

    def test_csv_has_tracker_column(self, report, tmp_path):
        import csv as csv_module
        path = report.to_csv(tmp_path / "out.csv")
        with open(path) as f:
            reader = csv_module.DictReader(f)
            names = [row["tracker"] for row in reader]
        assert "MOSSE" in names
        assert "KCF" in names

    def test_suffix_appended_when_missing(self, report, tmp_path):
        path = report.to_csv(tmp_path / "out_no_ext")
        assert path.suffix == ".csv"

    def test_parent_dirs_created(self, report, tmp_path):
        path = report.to_csv(tmp_path / "nested/deep/out.csv")
        assert path.is_file()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

class TestToJSON:
    def test_file_created(self, report, tmp_path):
        path = report.to_json(tmp_path / "out.json")
        assert path.is_file()

    def test_valid_json(self, report, tmp_path):
        path = report.to_json(tmp_path / "out.json")
        with open(path) as f:
            data = json.load(f)
        assert "trackers" in data
        assert "dataset" in data

    def test_tracker_count_in_json(self, report, tmp_path):
        path = report.to_json(tmp_path / "out.json")
        with open(path) as f:
            data = json.load(f)
        assert len(data["trackers"]) == len(report.rows)

    def test_json_dataset_name(self, report, tmp_path):
        path = report.to_json(tmp_path / "out.json")
        with open(path) as f:
            data = json.load(f)
        assert data["dataset"] == "Synthetic"


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------

class TestFmt:
    def test_none_passthrough(self):
        assert _fmt(None) is None

    def test_rounding(self):
        assert _fmt(0.123456789, 4) == pytest.approx(0.1235)

    def test_integer_input(self):
        assert _fmt(1, 2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Single-tracker report (edge case)
# ---------------------------------------------------------------------------

class TestSingleTracker:
    def test_single_tracker_pareto(self):
        ds = SyntheticDataset(num_sequences=3, num_frames=40)
        engine = BenchmarkEngine(verbose=False)
        r = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")
        report = FullBenchmarkReport([r])
        assert len(report.rows) == 1
        assert report.rows[0].on_pareto_front is True

    def test_markdown_single_tracker(self):
        ds = SyntheticDataset(num_sequences=3, num_frames=40)
        engine = BenchmarkEngine(verbose=False)
        r = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")
        report = FullBenchmarkReport([r])
        md = report.to_markdown()
        assert "MOSSE" in md
