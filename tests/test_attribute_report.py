"""Tests for MultiTrackerAttributeReport.

Covers:
- Construction and validation
- Active attribute detection
- Markdown, LaTeX, CSV, JSON output formats
- Best-value bolding in Markdown and LaTeX
- Save helpers (files written correctly)
- Overall ranking
- Integration with BenchmarkEngine + SyntheticDataset
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.base import BaseDataset, Sequence
from eovot.datasets.synthetic import SyntheticDataset
from eovot.reporting.attribute_report import MultiTrackerAttributeReport
from eovot.trackers.base import BaseTracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeTracker(BaseTracker):
    """Non-crashing tracker that always returns the initialization bbox.

    Used in tests that care about report *structure* rather than tracker
    accuracy.  Returns the bbox seen at initialize() on every subsequent
    update() call, giving a stable (though inaccurate) prediction that
    never invokes OpenCV and never raises cv2.error.
    """

    def __init__(self, name: str = "FakeTracker"):
        super().__init__(name=name)
        self._bbox = (0.0, 0.0, 10.0, 10.0)

    def initialize(self, frame: np.ndarray, bbox) -> None:
        self._bbox = tuple(float(v) for v in bbox)

    def update(self, frame: np.ndarray):
        return self._bbox


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestMultiTrackerAttributeReportInit:
    def test_basic_construction(self):
        ds = SyntheticDataset(num_sequences=4, num_frames=60, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        results = [
            engine.run(_FakeTracker(name="MOSSE"), ds, dataset_name="Syn"),
            engine.run(KCFTracker(), ds, dataset_name="Syn"),
        ]
        report = MultiTrackerAttributeReport(results)
        assert report.metric == "mean_iou"
        assert "MOSSE" in report._tracker_names
        assert "KCF" in report._tracker_names

    def test_invalid_metric_raises(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=30)
        engine = BenchmarkEngine(verbose=False)
        results = [engine.run(_FakeTracker(), ds, dataset_name="Syn")]
        with pytest.raises(ValueError, match="metric must be one of"):
            MultiTrackerAttributeReport(results, metric="bogus")

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="results must not be empty"):
            MultiTrackerAttributeReport([])

    def test_single_tracker(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=30)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(_FakeTracker(name="MOSSE"), ds, dataset_name="Syn")
        report = MultiTrackerAttributeReport([result])
        assert len(report._tracker_names) == 1

    def test_repr(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=30)
        engine = BenchmarkEngine(verbose=False)
        results = [engine.run(_FakeTracker(name="MOSSE"), ds, dataset_name="Syn")]
        report = MultiTrackerAttributeReport(results)
        r = repr(report)
        assert "MultiTrackerAttributeReport" in r
        assert "MOSSE" in r

    def test_success_auc_metric(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=30)
        engine = BenchmarkEngine(verbose=False)
        results = [engine.run(_FakeTracker(), ds, dataset_name="Syn")]
        report = MultiTrackerAttributeReport(results, metric="success_auc")
        assert report.metric == "success_auc"

    def test_precision_auc_metric(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=30)
        engine = BenchmarkEngine(verbose=False)
        results = [engine.run(_FakeTracker(), ds, dataset_name="Syn")]
        report = MultiTrackerAttributeReport(results, metric="precision_auc")
        assert report.metric == "precision_auc"


# ---------------------------------------------------------------------------
# Active attributes
# ---------------------------------------------------------------------------

class TestActiveAttributes:
    def test_returns_list(self):
        # Circular motion triggers fast_motion attribute (arc ~14px > threshold).
        ds = SyntheticDataset(num_sequences=5, num_frames=80, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        results = [engine.run(_FakeTracker(), ds, dataset_name="Syn")]
        report = MultiTrackerAttributeReport(results)
        attrs = report.active_attributes()
        assert isinstance(attrs, list)

    def test_all_active_are_known(self):
        from eovot.metrics.attributes import ALL_ATTRIBUTES
        ds = SyntheticDataset(num_sequences=5, num_frames=80, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        results = [engine.run(_FakeTracker(), ds, dataset_name="Syn")]
        report = MultiTrackerAttributeReport(results)
        for attr in report.active_attributes():
            assert attr in ALL_ATTRIBUTES


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

class TestMarkdownOutput:
    @pytest.fixture(scope="class")
    def report(self):
        # Circular motion ensures fast_motion attribute is active so the
        # markdown table has rows rather than returning the no-data message.
        ds = SyntheticDataset(num_sequences=5, num_frames=80, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        results = [
            engine.run(_FakeTracker(name="MOSSE"), ds, dataset_name="Syn"),
            engine.run(KCFTracker(), ds, dataset_name="Syn"),
        ]
        return MultiTrackerAttributeReport(results)

    def test_markdown_is_string(self, report):
        md = report.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_markdown_contains_tracker_names(self, report):
        md = report.to_markdown()
        assert "MOSSE" in md
        assert "KCF" in md

    def test_markdown_contains_dataset_name(self, report):
        md = report.to_markdown()
        assert "Syn" in md

    def test_markdown_has_table_separator(self, report):
        md = report.to_markdown()
        assert "|" in md

    def test_markdown_best_value_bolded(self, report):
        md = report.to_markdown()
        # At least one cell should be bolded
        assert "**" in md

    def test_markdown_show_counts(self, report):
        md = report.to_markdown(show_counts=True)
        # With counts, cells should contain parentheses
        assert "(" in md

    def test_no_attribute_data_returns_message(self):
        """Report with no active attributes returns a no-data message."""
        ds = SyntheticDataset(num_sequences=1, num_frames=5)
        engine = BenchmarkEngine(verbose=False)
        results = [engine.run(_FakeTracker(), ds, dataset_name="Syn")]
        report = MultiTrackerAttributeReport(results, min_sequences=9999)
        md = report.to_markdown()
        assert "No attribute data" in md


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

class TestLatexOutput:
    @pytest.fixture(scope="class")
    def report(self):
        ds = SyntheticDataset(num_sequences=5, num_frames=80, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        results = [
            engine.run(_FakeTracker(name="MOSSE"), ds, dataset_name="Syn"),
            engine.run(KCFTracker(), ds, dataset_name="Syn"),
        ]
        return MultiTrackerAttributeReport(results)

    def test_latex_contains_tabular(self, report):
        tex = report.to_latex()
        assert r"\begin{tabular}" in tex
        assert r"\end{tabular}" in tex

    def test_latex_contains_tracker_names(self, report):
        tex = report.to_latex()
        assert "MOSSE" in tex
        assert "KCF" in tex

    def test_latex_bold_best(self, report):
        tex = report.to_latex(bold_best=True)
        assert r"\textbf{" in tex

    def test_latex_no_bold(self, report):
        tex = report.to_latex(bold_best=False)
        assert r"\textbf{" not in tex

    def test_latex_with_caption_wraps_table(self, report):
        tex = report.to_latex(caption="Test caption")
        assert r"\begin{table}" in tex
        assert r"\caption{Test caption}" in tex

    def test_latex_without_caption_no_table_env(self, report):
        tex = report.to_latex()
        assert r"\begin{table}" not in tex

    def test_latex_has_booktabs_rules(self, report):
        tex = report.to_latex()
        assert r"\toprule" in tex
        assert r"\midrule" in tex
        assert r"\bottomrule" in tex


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

class TestCSVOutput:
    @pytest.fixture(scope="class")
    def report(self):
        ds = SyntheticDataset(num_sequences=5, num_frames=80, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        results = [
            engine.run(_FakeTracker(name="MOSSE"), ds, dataset_name="Syn"),
            engine.run(KCFTracker(), ds, dataset_name="Syn"),
        ]
        return MultiTrackerAttributeReport(results)

    def test_csv_is_parseable(self, report):
        csv_str = report.to_csv()
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) >= 0  # may be empty if no attrs

    def test_csv_has_attribute_column(self, report):
        csv_str = report.to_csv()
        assert "attribute" in csv_str

    def test_csv_has_tracker_columns(self, report):
        csv_str = report.to_csv()
        assert "MOSSE" in csv_str
        assert "KCF" in csv_str


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

class TestJSONOutput:
    @pytest.fixture(scope="class")
    def report_dict(self):
        ds = SyntheticDataset(num_sequences=5, num_frames=80, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        results = [
            engine.run(_FakeTracker(name="MOSSE"), ds, dataset_name="Syn"),
            engine.run(KCFTracker(), ds, dataset_name="Syn"),
        ]
        return MultiTrackerAttributeReport(results).to_dict()

    def test_json_has_required_keys(self, report_dict):
        assert "dataset" in report_dict
        assert "metric" in report_dict
        assert "trackers" in report_dict
        assert "attributes" in report_dict

    def test_json_tracker_names(self, report_dict):
        assert "MOSSE" in report_dict["trackers"]
        assert "KCF" in report_dict["trackers"]

    def test_json_serializable(self, report_dict):
        # Should not raise
        s = json.dumps(report_dict)
        assert len(s) > 0

    def test_json_attribute_entries_have_n_sequences(self, report_dict):
        for attr, data in report_dict["attributes"].items():
            assert "n_sequences" in data
            assert data["n_sequences"] >= 1


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

class TestSaveHelpers:
    @pytest.fixture(scope="class")
    def report(self):
        # Circular motion ensures at least one attribute is active so
        # save_markdown produces a table (not the no-data fallback).
        ds = SyntheticDataset(num_sequences=4, num_frames=60, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        results = [
            engine.run(_FakeTracker(name="MOSSE"), ds, dataset_name="Syn"),
            engine.run(KCFTracker(), ds, dataset_name="Syn"),
        ]
        return MultiTrackerAttributeReport(results)

    def test_save_markdown(self, report, tmp_path):
        p = report.save_markdown(tmp_path / "report.md")
        assert p.exists()
        assert p.suffix == ".md"
        content = p.read_text()
        assert "MOSSE" in content

    def test_save_markdown_adds_extension(self, report, tmp_path):
        p = report.save_markdown(tmp_path / "report_no_ext")
        assert p.suffix == ".md"

    def test_save_latex(self, report, tmp_path):
        p = report.save_latex(tmp_path / "report.tex")
        assert p.exists()
        content = p.read_text()
        assert r"\begin{tabular}" in content

    def test_save_csv(self, report, tmp_path):
        p = report.save_csv(tmp_path / "report.csv")
        assert p.exists()
        content = p.read_text()
        assert "attribute" in content

    def test_save_json(self, report, tmp_path):
        p = report.save_json(tmp_path / "report.json")
        assert p.exists()
        data = json.loads(p.read_text())
        assert "trackers" in data


# ---------------------------------------------------------------------------
# Overall ranking
# ---------------------------------------------------------------------------

class TestOverallRanking:
    def test_ranking_returns_list_of_tuples(self):
        ds = SyntheticDataset(num_sequences=4, num_frames=60, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        results = [
            engine.run(_FakeTracker(name="MOSSE"), ds, dataset_name="Syn"),
            engine.run(KCFTracker(), ds, dataset_name="Syn"),
        ]
        report = MultiTrackerAttributeReport(results)
        ranking = report.overall_ranking()
        assert isinstance(ranking, list)
        assert len(ranking) == 2
        for name, score in ranking:
            assert isinstance(name, str)
            assert isinstance(score, float)

    def test_ranking_is_descending(self):
        ds = SyntheticDataset(num_sequences=4, num_frames=60, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        results = [
            engine.run(_FakeTracker(name="MOSSE"), ds, dataset_name="Syn"),
            engine.run(KCFTracker(), ds, dataset_name="Syn"),
        ]
        report = MultiTrackerAttributeReport(results)
        ranking = report.overall_ranking()
        scores = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True)
