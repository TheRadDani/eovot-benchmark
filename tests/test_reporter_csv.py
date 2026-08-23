"""Tests for the fixed and extended CSV reporter.

Covers:
- save_csv() writes all available sequence-level fields (fixes the broken
  `precision_score` reference from the original implementation).
- save_csv() handles optional fields (energy, AUC metrics) correctly when
  present or absent.
- save_summary_csv() writes one row per tracker with the correct columns.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from eovot.reporting.reporter import BenchmarkReporter


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_result(
    tracker: str = "MOSSE",
    dataset: str = "Synthetic",
    with_energy: bool = False,
    with_accuracy_aucs: bool = True,
) -> dict:
    """Minimal result dict matching BenchmarkResult.to_dict() format."""
    seq: dict = {
        "sequence_name": "seq_001",
        "mean_iou": 0.623,
        "fps": 312.5,
        "mean_latency_ms": 3.2,
        "latency_std_ms": 0.4,
        "latency_p95_ms": 3.9,
        "latency_p99_ms": 4.5,
        "latency_cv": 0.125,
        "peak_memory_mb": 48.3,
    }
    if with_accuracy_aucs:
        seq["success_auc"] = 0.548
        seq["precision_auc"] = 0.721
        seq["normalized_precision_auc"] = 0.634
    if with_energy:
        seq["energy_j"] = 0.010240
        seq["energy_per_frame_mj"] = 0.0320
        seq["energy_tdp_watts"] = 6.0
        seq["energy_mean_power_w"] = 5.1

    summary: dict = {
        "tracker": tracker,
        "dataset": dataset,
        "num_sequences": 1,
        "mean_iou": 0.623,
        "mean_fps": 312.5,
        "peak_memory_mb": 48.3,
    }
    if with_accuracy_aucs:
        summary["success_auc"] = 0.548
        summary["precision_auc"] = 0.721
        summary["normalized_precision_auc"] = 0.634
    if with_energy:
        summary["total_energy_j"] = 0.010240
        summary["mean_energy_per_frame_mj"] = 0.0320

    return {"summary": summary, "sequences": [seq]}


# ---------------------------------------------------------------------------
# Tests: save_csv
# ---------------------------------------------------------------------------

class TestSaveCsv:
    def test_core_fields_written(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        result = _make_result(with_energy=False, with_accuracy_aucs=False)
        path = reporter.save_csv(result, name="run")

        assert path.exists()
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == 1
        row = rows[0]
        assert row["sequence_name"] == "seq_001"
        assert float(row["fps"]) == pytest.approx(312.5, rel=1e-3)
        assert float(row["mean_latency_ms"]) == pytest.approx(3.2, rel=1e-3)
        assert float(row["latency_p99_ms"]) == pytest.approx(4.5, rel=1e-3)
        assert float(row["latency_cv"]) == pytest.approx(0.125, rel=1e-3)
        assert float(row["peak_memory_mb"]) == pytest.approx(48.3, rel=1e-3)

    def test_no_precision_score_column(self, tmp_path):
        """The old broken column 'precision_score' must not appear."""
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        result = _make_result()
        path = reporter.save_csv(result, name="run")
        rows = list(csv.DictReader(open(path)))
        assert "precision_score" not in rows[0]

    def test_accuracy_auc_columns_present_when_available(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        result = _make_result(with_accuracy_aucs=True)
        path = reporter.save_csv(result, name="run")
        rows = list(csv.DictReader(open(path)))
        row = rows[0]
        assert "success_auc" in row
        assert "precision_auc" in row
        assert "normalized_precision_auc" in row
        assert float(row["success_auc"]) == pytest.approx(0.548, rel=1e-3)
        assert float(row["precision_auc"]) == pytest.approx(0.721, rel=1e-3)

    def test_accuracy_auc_columns_absent_when_not_available(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        result = _make_result(with_accuracy_aucs=False)
        path = reporter.save_csv(result, name="run")
        rows = list(csv.DictReader(open(path)))
        assert "success_auc" not in rows[0]
        assert "precision_auc" not in rows[0]

    def test_energy_columns_present_when_available(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        result = _make_result(with_energy=True)
        path = reporter.save_csv(result, name="run")
        rows = list(csv.DictReader(open(path)))
        row = rows[0]
        assert "energy_j" in row
        assert "energy_per_frame_mj" in row
        assert float(row["energy_j"]) == pytest.approx(0.010240, rel=1e-3)

    def test_energy_columns_absent_when_not_available(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        result = _make_result(with_energy=False)
        path = reporter.save_csv(result, name="run")
        rows = list(csv.DictReader(open(path)))
        assert "energy_j" not in rows[0]

    def test_empty_sequences_creates_file(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        result = {"summary": {}, "sequences": []}
        path = reporter.save_csv(result, name="empty")
        assert path.exists()

    def test_multiple_sequences(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        result = _make_result()
        # Duplicate the sequence to simulate multi-sequence results.
        seq2 = dict(result["sequences"][0])
        seq2["sequence_name"] = "seq_002"
        seq2["mean_iou"] = 0.71
        result["sequences"].append(seq2)

        path = reporter.save_csv(result, name="multi")
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == 2
        assert rows[1]["sequence_name"] == "seq_002"


# ---------------------------------------------------------------------------
# Tests: save_summary_csv
# ---------------------------------------------------------------------------

class TestSaveSummaryCsv:
    def test_one_row_per_tracker(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        results = [
            _make_result("MOSSE", "Synthetic"),
            _make_result("KCF", "Synthetic"),
        ]
        path = reporter.save_summary_csv(results, name="summary")

        assert path.exists()
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == 2
        trackers = [r["tracker"] for r in rows]
        assert "MOSSE" in trackers
        assert "KCF" in trackers

    def test_core_summary_columns(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        results = [_make_result()]
        path = reporter.save_summary_csv(results, name="summary")
        rows = list(csv.DictReader(open(path)))
        row = rows[0]
        assert "tracker" in row
        assert "dataset" in row
        assert "mean_iou" in row
        assert "mean_fps" in row
        assert "peak_memory_mb" in row

    def test_optional_auc_columns(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        results = [_make_result(with_accuracy_aucs=True)]
        path = reporter.save_summary_csv(results, name="summary")
        rows = list(csv.DictReader(open(path)))
        assert "success_auc" in rows[0]
        assert "precision_auc" in rows[0]

    def test_optional_auc_absent_when_not_in_any_result(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        results = [_make_result(with_accuracy_aucs=False)]
        path = reporter.save_summary_csv(results, name="summary")
        rows = list(csv.DictReader(open(path)))
        assert "success_auc" not in rows[0]

    def test_energy_columns_in_summary(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        results = [_make_result(with_energy=True)]
        path = reporter.save_summary_csv(results, name="summary")
        rows = list(csv.DictReader(open(path)))
        assert "total_energy_j" in rows[0]
        assert "mean_energy_per_frame_mj" in rows[0]

    def test_mixed_results_optional_columns_present_when_any_has_them(self, tmp_path):
        """Optional columns appear when at least one result has them."""
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        results = [
            _make_result("MOSSE", with_energy=True),
            _make_result("KCF", with_energy=False),
        ]
        path = reporter.save_summary_csv(results, name="summary")
        rows = list(csv.DictReader(open(path)))
        assert "total_energy_j" in rows[0]
        # KCF row should have the column but it may be empty
        kcf_row = next(r for r in rows if r["tracker"] == "KCF")
        assert "total_energy_j" in kcf_row

    def test_empty_results_creates_file(self, tmp_path):
        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        path = reporter.save_summary_csv([], name="empty")
        assert path.exists()
