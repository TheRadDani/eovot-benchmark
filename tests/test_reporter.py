"""Unit tests for eovot.reporting.reporter — CSV export and comparison tables."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from eovot.reporting.reporter import BenchmarkReporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    tracker: str = "MOSSE",
    dataset: str = "OTB100",
    n_seqs: int = 3,
    with_energy: bool = False,
    with_auc: bool = True,
) -> Dict[str, Any]:
    """Build a minimal result dict matching BenchmarkResult.to_dict() output."""
    sequences = []
    for i in range(n_seqs):
        seq: Dict[str, Any] = {
            "sequence_name": f"seq_{i:02d}",
            "mean_iou": round(0.8 - i * 0.05, 4),
            "fps": 200.0 - i * 10,
            "mean_latency_ms": 5.0 + i * 0.5,
            "peak_memory_mb": 250.0 + i * 5,
        }
        if with_auc:
            seq["success_auc"] = round(0.75 - i * 0.04, 4)
            seq["precision_auc"] = round(0.70 - i * 0.03, 4)
        if with_energy:
            seq["energy_j"] = round(0.012 + i * 0.001, 6)
            seq["energy_per_frame_mj"] = round(0.08 + i * 0.01, 4)
        sequences.append(seq)

    summary: Dict[str, Any] = {
        "tracker": tracker,
        "tracker_name": tracker,
        "dataset": dataset,
        "dataset_name": dataset,
        "num_sequences": n_seqs,
        "mean_iou": round(0.775, 4),
        "mean_fps": 195.0,
        "peak_memory_mb": 255.0,
    }
    if with_auc:
        summary["success_auc"] = 0.7100
        summary["precision_auc"] = 0.6700
    if with_energy:
        summary["total_energy_j"] = 0.039
        summary["mean_energy_per_frame_mj"] = 0.09

    return {"summary": summary, "sequences": sequences}


# ---------------------------------------------------------------------------
# Original reporter tests (preserved and extended)
# ---------------------------------------------------------------------------

class TestBenchmarkReporter:
    def setup_method(self, _method):
        self._tmpdir = tempfile.mkdtemp()
        self.reporter = BenchmarkReporter(output_dir=self._tmpdir)

    def test_output_dir_created(self, tmp_path):
        subdir = tmp_path / "new_subdir" / "nested"
        BenchmarkReporter(output_dir=str(subdir))
        assert subdir.exists()

    def test_save_json_creates_file(self):
        result = _make_result()
        path = self.reporter.save_json(result, name="test_run")
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_json_valid_content(self):
        result = _make_result()
        path = self.reporter.save_json(result, name="test_run")
        with open(path) as f:
            data = json.load(f)
        assert data["summary"]["tracker"] == "MOSSE"
        assert data["summary"]["mean_iou"] == pytest.approx(0.775, abs=1e-3)

    def test_save_csv_creates_file(self):
        result = _make_result()
        path = self.reporter.save_csv(result, name="test_run")
        assert path.exists()
        assert path.suffix == ".csv"

    def test_save_csv_row_count(self):
        result = _make_result(n_seqs=4)
        path = self.reporter.save_csv(result, name="test_run")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4

    def test_save_csv_has_core_columns(self):
        result = _make_result()
        path = self.reporter.save_csv(result, name="test_run")
        with open(path, newline="") as f:
            cols = csv.DictReader(f).fieldnames or []
        for col in ("sequence_name", "mean_iou", "fps", "mean_latency_ms", "peak_memory_mb"):
            assert col in cols

    def test_save_csv_no_stale_precision_score_column(self):
        """The old bogus 'precision_score' column must no longer appear."""
        result = _make_result()
        path = self.reporter.save_csv(result, name="test_run")
        with open(path, newline="") as f:
            cols = csv.DictReader(f).fieldnames or []
        assert "precision_score" not in cols

    def test_save_csv_includes_auc_when_present(self):
        result = _make_result(with_auc=True)
        path = self.reporter.save_csv(result, name="test_run")
        with open(path, newline="") as f:
            cols = csv.DictReader(f).fieldnames or []
        assert "success_auc" in cols
        assert "precision_auc" in cols

    def test_save_csv_excludes_auc_when_absent(self):
        result = _make_result(with_auc=False)
        path = self.reporter.save_csv(result, name="test_run")
        with open(path, newline="") as f:
            cols = csv.DictReader(f).fieldnames or []
        assert "success_auc" not in cols
        assert "precision_auc" not in cols

    def test_save_csv_includes_energy_when_present(self):
        result = _make_result(with_energy=True)
        path = self.reporter.save_csv(result, name="test_run")
        with open(path, newline="") as f:
            cols = csv.DictReader(f).fieldnames or []
        assert "energy_j" in cols
        assert "energy_per_frame_mj" in cols

    def test_save_csv_excludes_energy_when_absent(self):
        result = _make_result(with_energy=False)
        path = self.reporter.save_csv(result, name="test_run")
        with open(path, newline="") as f:
            cols = csv.DictReader(f).fieldnames or []
        assert "energy_j" not in cols

    def test_save_csv_empty_sequences_returns_path(self):
        result = {"summary": {}, "sequences": []}
        path = self.reporter.save_csv(result, name="empty")
        assert isinstance(path, Path)

    def test_save_all_returns_both_formats(self):
        result = _make_result()
        saved = self.reporter.save_all(result, name="test_run")
        assert "json" in saved
        assert "csv" in saved
        assert saved["json"].exists()
        assert saved["csv"].exists()

    def test_print_summary_does_not_raise(self, capsys):
        result = _make_result()
        self.reporter.print_summary(result)
        captured = capsys.readouterr()
        assert "BENCHMARK SUMMARY" in captured.out

    def test_comparison_table_contains_tracker_names(self):
        results = [_make_result("MOSSE"), _make_result("KCF")]
        table = BenchmarkReporter.comparison_table(results)
        assert "MOSSE" in table
        assert "KCF" in table

    def test_save_comparison_creates_md(self):
        results = [_make_result("MOSSE"), _make_result("KCF")]
        path = self.reporter.save_comparison(results, name="cmp_test")
        assert path.exists()
        assert path.suffix == ".md"
        assert "EOVOT Tracker Comparison" in path.read_text()


# ---------------------------------------------------------------------------
# save_comparison_csv tests
# ---------------------------------------------------------------------------

class TestSaveComparisonCSV:
    def _reporters(self, tmp_path):
        return BenchmarkReporter(output_dir=str(tmp_path))

    def test_one_row_per_tracker(self, tmp_path):
        reporter = self._reporters(tmp_path)
        results = [_make_result("MOSSE"), _make_result("KCF"), _make_result("CSRT")]
        path = reporter.save_comparison_csv(results, name="cmp")
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3

    def test_tracker_and_dataset_columns(self, tmp_path):
        reporter = self._reporters(tmp_path)
        path = reporter.save_comparison_csv([_make_result("MOSSE", "OTB100")], name="cmp")
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["tracker"] == "MOSSE"
        assert rows[0]["dataset"] == "OTB100"

    def test_sorted_by_success_auc(self, tmp_path):
        reporter = self._reporters(tmp_path)
        r1, r2 = _make_result("MOSSE"), _make_result("KCF")
        r1["summary"]["success_auc"] = 0.50
        r2["summary"]["success_auc"] = 0.90
        path = reporter.save_comparison_csv([r1, r2], sort_by="success_auc")
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["tracker"] == "KCF"

    def test_sorted_by_fps(self, tmp_path):
        reporter = self._reporters(tmp_path)
        r1, r2 = _make_result("MOSSE"), _make_result("KCF")
        r1["summary"]["mean_fps"] = 10.0
        r2["summary"]["mean_fps"] = 500.0
        path = reporter.save_comparison_csv([r1, r2], sort_by="mean_fps")
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["tracker"] == "KCF"

    def test_energy_columns_when_present(self, tmp_path):
        reporter = self._reporters(tmp_path)
        results = [_make_result(with_energy=True), _make_result(with_energy=True)]
        path = reporter.save_comparison_csv(results, name="cmp")
        with open(path, newline="") as fh:
            cols = csv.DictReader(fh).fieldnames or []
        assert "total_energy_j" in cols
        assert "mean_energy_per_frame_mj" in cols

    def test_energy_columns_absent_when_not_present(self, tmp_path):
        reporter = self._reporters(tmp_path)
        results = [_make_result(with_energy=False)]
        path = reporter.save_comparison_csv(results, name="cmp")
        with open(path, newline="") as fh:
            cols = csv.DictReader(fh).fieldnames or []
        assert "total_energy_j" not in cols

    def test_legacy_tracker_name_key(self, tmp_path):
        """Results that use tracker_name/dataset_name (old format) still work."""
        reporter = self._reporters(tmp_path)
        old_result = {
            "summary": {
                "tracker_name": "OldTracker",
                "dataset_name": "OTB",
                "num_sequences": 5,
                "mean_iou": 0.55,
                "mean_fps": 100.0,
                "peak_memory_mb": 200.0,
            },
            "sequences": [],
        }
        path = reporter.save_comparison_csv([old_result], name="legacy")
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["tracker"] == "OldTracker"
        assert rows[0]["dataset"] == "OTB"
