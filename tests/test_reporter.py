"""Unit tests for eovot.reporting.reporter."""

import json
import csv
from pathlib import Path

import pytest

from eovot.reporting.reporter import BenchmarkReporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result_dict(tracker="MOSSE", dataset="OTB100", n_seqs=3):
    """Build a minimal result dict in the format produced by BenchmarkResult.to_dict()."""
    sequences = [
        {
            "sequence_name": f"seq_{i}",
            "mean_iou": 0.5 + i * 0.05,
            "fps": 200.0 + i * 10,
            "mean_latency_ms": 5.0 - i * 0.5,
            "peak_memory_mb": 50.0 + i,
        }
        for i in range(n_seqs)
    ]
    return {
        "summary": {
            "tracker": tracker,
            "tracker_name": tracker,   # reporter.to_markdown_row uses "tracker_name"
            "dataset": dataset,
            "dataset_name": dataset,   # reporter.to_markdown_row uses "dataset_name"
            "num_sequences": n_seqs,
            "mean_iou": 0.55,
            "mean_fps": 210.0,
            "peak_memory_mb": 51.0,
        },
        "sequences": sequences,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBenchmarkReporter:
    def setup_method(self, tmp_path_factory):
        # Use a temp directory per test class (pytest fixture not available in
        # setup_method, so we create one manually)
        import tempfile, os
        self._tmpdir = tempfile.mkdtemp()
        self.reporter = BenchmarkReporter(output_dir=self._tmpdir)

    def test_output_dir_created(self, tmp_path):
        subdir = tmp_path / "new_subdir" / "nested"
        BenchmarkReporter(output_dir=str(subdir))
        assert subdir.exists()

    def test_save_json_creates_file(self):
        result = _make_result_dict()
        path = self.reporter.save_json(result, name="test_run")
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_json_valid_content(self):
        result = _make_result_dict()
        path = self.reporter.save_json(result, name="test_run")
        with open(path) as f:
            data = json.load(f)
        assert data["summary"]["tracker"] == "MOSSE"
        assert data["summary"]["mean_iou"] == pytest.approx(0.55)

    def test_save_csv_creates_file(self):
        result = _make_result_dict()
        path = self.reporter.save_csv(result, name="test_run")
        assert path.exists()
        assert path.suffix == ".csv"

    def test_save_csv_row_count(self):
        result = _make_result_dict(n_seqs=4)
        path = self.reporter.save_csv(result, name="test_run")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4

    def test_save_csv_has_expected_columns(self):
        result = _make_result_dict()
        path = self.reporter.save_csv(result, name="test_run")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        assert "sequence_name" in cols
        assert "mean_iou" in cols
        assert "fps" in cols

    def test_save_all_returns_both_formats(self):
        result = _make_result_dict()
        saved = self.reporter.save_all(result, name="test_run")
        assert "json" in saved
        assert "csv" in saved
        assert saved["json"].exists()
        assert saved["csv"].exists()

    def test_print_summary_does_not_raise(self, capsys):
        result = _make_result_dict()
        self.reporter.print_summary(result)
        captured = capsys.readouterr()
        assert "BENCHMARK SUMMARY" in captured.out

    def test_comparison_table_contains_tracker_names(self):
        results = [_make_result_dict("MOSSE"), _make_result_dict("KCF")]
        table = self.reporter.comparison_table(results)
        assert "MOSSE" in table or "KCF" in table

    def test_save_comparison_creates_md(self):
        results = [_make_result_dict("MOSSE"), _make_result_dict("KCF")]
        path = self.reporter.save_comparison(results, name="cmp_test")
        assert path.exists()
        assert path.suffix == ".md"
        content = path.read_text()
        assert "EOVOT Tracker Comparison" in content

    def test_save_csv_empty_sequences(self):
        """save_csv returns early without writing when 'sequences' is empty."""
        result = {"summary": {}, "sequences": []}
        path = self.reporter.save_csv(result, name="empty")
        # The method returns the path but skips writing when sequences is empty
        assert isinstance(path, Path)
