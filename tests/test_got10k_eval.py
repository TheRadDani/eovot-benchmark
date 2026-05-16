"""Tests for eovot.metrics.got10k_eval and eovot.datasets.got10k (bug fix)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterator, List

import cv2
import numpy as np
import pytest

from eovot.metrics.got10k_eval import (
    GOT10kEvaluator,
    GOT10kReport,
    GOT10kSequenceResult,
    compute_ao,
    compute_sr,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic BenchmarkResult-like objects
# ---------------------------------------------------------------------------

class _FakeSequenceResult:
    def __init__(self, name: str, ious, predictions=None):
        self.sequence_name = name
        self.ious = np.asarray(ious, dtype=np.float64)
        self.predictions = (
            np.array(predictions, dtype=np.float64)
            if predictions is not None
            else None
        )


class _FakeBenchmarkResult:
    def __init__(self, tracker_name: str, seq_results):
        self.tracker_name = tracker_name
        self.sequence_results = seq_results


def _make_result(tracker_name, seq_ious_map, with_predictions=True):
    seq_results = []
    for name, ious in seq_ious_map.items():
        n = len(ious)
        preds = np.tile([10.0, 10.0, 50.0, 50.0], (n, 1)) if with_predictions else None
        seq_results.append(_FakeSequenceResult(name, ious, preds))
    return _FakeBenchmarkResult(tracker_name, seq_results)


# ---------------------------------------------------------------------------
# compute_ao
# ---------------------------------------------------------------------------

class TestComputeAO:
    def test_perfect_tracking_excludes_init(self):
        # All frames including init are 1.0 — AO should still be 1.0
        ious = np.ones(10)
        assert compute_ao(ious) == pytest.approx(1.0)

    def test_zero_tracking(self):
        ious = np.zeros(10)
        assert compute_ao(ious) == pytest.approx(0.0)

    def test_init_frame_excluded(self):
        # Init frame (index 0) = 1.0, rest = 0.0 → AO should be 0.0
        ious = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        assert compute_ao(ious) == pytest.approx(0.0)

    def test_known_mean(self):
        ious = np.array([1.0, 0.2, 0.4, 0.6, 0.8])
        expected = np.mean([0.2, 0.4, 0.6, 0.8])
        assert compute_ao(ious) == pytest.approx(expected)

    def test_single_frame_returns_zero(self):
        assert compute_ao(np.array([1.0])) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert compute_ao(np.array([])) == pytest.approx(0.0)

    def test_two_frames(self):
        # Only one non-init frame
        ious = np.array([1.0, 0.6])
        assert compute_ao(ious) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# compute_sr
# ---------------------------------------------------------------------------

class TestComputeSR:
    def test_sr50_all_above(self):
        ious = np.array([1.0, 0.6, 0.7, 0.8, 0.9])
        assert compute_sr(ious, 0.5) == pytest.approx(1.0)

    def test_sr50_none_above(self):
        ious = np.array([1.0, 0.1, 0.2, 0.3, 0.4])
        assert compute_sr(ious, 0.5) == pytest.approx(0.0)

    def test_sr50_half(self):
        ious = np.array([1.0, 0.6, 0.4, 0.6, 0.4])
        assert compute_sr(ious, 0.5) == pytest.approx(0.5)

    def test_sr75_stricter(self):
        ious = np.array([1.0, 0.8, 0.6, 0.8, 0.6])
        sr50 = compute_sr(ious, 0.5)
        sr75 = compute_sr(ious, 0.75)
        assert sr75 <= sr50

    def test_exact_threshold_included(self):
        # IoU == threshold qualifies (≥)
        ious = np.array([1.0, 0.5, 0.5])
        assert compute_sr(ious, 0.5) == pytest.approx(1.0)

    def test_single_frame_returns_zero(self):
        assert compute_sr(np.array([1.0]), 0.5) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert compute_sr(np.array([]), 0.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GOT10kEvaluator.evaluate
# ---------------------------------------------------------------------------

class TestGOT10kEvaluator:
    def setup_method(self):
        self.evaluator = GOT10kEvaluator(split="val")

    def test_returns_report_type(self):
        result = _make_result("MOSSE", {"seq1": np.ones(10)})
        report = self.evaluator.evaluate(result)
        assert isinstance(report, GOT10kReport)

    def test_tracker_name_preserved(self):
        result = _make_result("KCF", {"seq1": np.ones(5)})
        report = self.evaluator.evaluate(result)
        assert report.tracker_name == "KCF"

    def test_split_preserved(self):
        ev = GOT10kEvaluator(split="test")
        result = _make_result("MOSSE", {"seq1": np.ones(5)})
        report = ev.evaluate(result)
        assert report.dataset_split == "test"

    def test_num_sequences(self):
        result = _make_result("T", {"s1": np.ones(5), "s2": np.ones(5), "s3": np.ones(5)})
        report = self.evaluator.evaluate(result)
        assert report.num_sequences == 3

    def test_perfect_tracker_ao_one(self):
        result = _make_result("T", {"s1": np.ones(20), "s2": np.ones(15)})
        report = self.evaluator.evaluate(result)
        assert report.ao == pytest.approx(1.0)

    def test_perfect_tracker_sr50_one(self):
        result = _make_result("T", {"s1": np.ones(20)})
        report = self.evaluator.evaluate(result)
        assert report.sr50 == pytest.approx(1.0)

    def test_perfect_tracker_sr75_one(self):
        result = _make_result("T", {"s1": np.ones(20)})
        report = self.evaluator.evaluate(result)
        assert report.sr75 == pytest.approx(1.0)

    def test_zero_tracker_all_zeros(self):
        # All non-init frames are 0 → AO=SR50=SR75=0
        ious = np.array([1.0] + [0.0] * 9)
        result = _make_result("T", {"s1": ious})
        report = self.evaluator.evaluate(result)
        assert report.ao == pytest.approx(0.0)
        assert report.sr50 == pytest.approx(0.0)
        assert report.sr75 == pytest.approx(0.0)

    def test_sr75_leq_sr50(self):
        ious = np.array([1.0, 0.8, 0.6, 0.4, 0.9, 0.7])
        result = _make_result("T", {"s1": ious})
        report = self.evaluator.evaluate(result)
        assert report.sr75 <= report.sr50

    def test_per_sequence_count(self):
        result = _make_result("T", {"s1": np.ones(5), "s2": np.ones(5)})
        report = self.evaluator.evaluate(result)
        assert len(report.per_sequence) == 2

    def test_per_sequence_ao_correct(self):
        ious = np.array([1.0, 0.4, 0.6, 0.8])
        result = _make_result("T", {"only": ious})
        report = self.evaluator.evaluate(result)
        expected_ao = np.mean([0.4, 0.6, 0.8])
        assert report.per_sequence[0].ao == pytest.approx(expected_ao)

    def test_empty_result(self):
        result = _FakeBenchmarkResult("T", [])
        report = self.evaluator.evaluate(result)
        assert report.num_sequences == 0
        assert report.ao == pytest.approx(0.0)

    def test_str_representation(self):
        result = _make_result("MOSSE", {"s1": np.ones(10)})
        report = self.evaluator.evaluate(result)
        s = str(report)
        assert "MOSSE" in s
        assert "AO" in s
        assert "SR50" in s

    def test_to_dict_keys(self):
        result = _make_result("T", {"s1": np.ones(8)})
        report = self.evaluator.evaluate(result)
        d = report.to_dict()
        for key in ("tracker", "split", "ao", "sr50", "sr75", "num_sequences", "per_sequence"):
            assert key in d


# ---------------------------------------------------------------------------
# export_submission
# ---------------------------------------------------------------------------

class TestExportSubmission:
    def setup_method(self):
        self.evaluator = GOT10kEvaluator(split="val")

    def test_creates_tracker_directory(self, tmp_path):
        preds = np.tile([10.0, 10.0, 50.0, 50.0], (5, 1))
        seq = _FakeSequenceResult("GOT-10k_Val_000001", np.ones(5), preds)
        result = _FakeBenchmarkResult("MOSSE", [seq])
        report = self.evaluator.evaluate(result)
        submit_dir = self.evaluator.export_submission(report, result, str(tmp_path))
        assert submit_dir.is_dir()
        assert submit_dir.name == "MOSSE"

    def test_creates_per_sequence_txt(self, tmp_path):
        preds = np.tile([10.0, 10.0, 50.0, 50.0], (5, 1))
        seq = _FakeSequenceResult("seq_001", np.ones(5), preds)
        result = _FakeBenchmarkResult("KCF", [seq])
        report = self.evaluator.evaluate(result)
        submit_dir = self.evaluator.export_submission(report, result, str(tmp_path))
        assert (submit_dir / "seq_001.txt").exists()

    def test_txt_skips_init_frame(self, tmp_path):
        preds = np.tile([1.0, 2.0, 10.0, 10.0], (4, 1))
        seq = _FakeSequenceResult("s1", np.ones(4), preds)
        result = _FakeBenchmarkResult("T", [seq])
        report = self.evaluator.evaluate(result)
        submit_dir = self.evaluator.export_submission(report, result, str(tmp_path))
        lines = (submit_dir / "s1.txt").read_text().strip().splitlines()
        # 4 frames → init excluded → 3 lines
        assert len(lines) == 3

    def test_txt_format_comma_separated(self, tmp_path):
        preds = np.array([[10.0, 20.0, 30.0, 40.0]] * 3)
        seq = _FakeSequenceResult("s1", np.ones(3), preds)
        result = _FakeBenchmarkResult("T", [seq])
        report = self.evaluator.evaluate(result)
        submit_dir = self.evaluator.export_submission(report, result, str(tmp_path))
        first_line = (submit_dir / "s1.txt").read_text().strip().splitlines()[0]
        parts = first_line.split(",")
        assert len(parts) == 4
        assert float(parts[0]) == pytest.approx(10.0, abs=0.01)

    def test_creates_json_report(self, tmp_path):
        preds = np.ones((3, 4))
        seq = _FakeSequenceResult("s1", np.ones(3), preds)
        result = _FakeBenchmarkResult("T", [seq])
        report = self.evaluator.evaluate(result)
        submit_dir = self.evaluator.export_submission(report, result, str(tmp_path))
        assert (submit_dir / "got10k_report.json").exists()

    def test_skips_sequence_without_predictions(self, tmp_path):
        seq = _FakeSequenceResult("s1", np.ones(5), predictions=None)
        result = _FakeBenchmarkResult("T", [seq])
        report = self.evaluator.evaluate(result)
        submit_dir = self.evaluator.export_submission(report, result, str(tmp_path))
        assert not (submit_dir / "s1.txt").exists()


# ---------------------------------------------------------------------------
# to_markdown_table
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def setup_method(self):
        self.evaluator = GOT10kEvaluator()

    def _report(self, name, ao, sr50, sr75):
        return GOT10kReport(
            tracker_name=name,
            dataset_split="val",
            ao=ao, sr50=sr50, sr75=sr75,
            num_sequences=5,
        )

    def test_contains_tracker_names(self):
        reports = [self._report("MOSSE", 0.35, 0.30, 0.10),
                   self._report("KCF",   0.42, 0.38, 0.15)]
        table = self.evaluator.to_markdown_table(reports)
        assert "MOSSE" in table
        assert "KCF" in table

    def test_ranked_by_ao(self):
        reports = [self._report("TrackerLow", 0.20, 0.15, 0.05),
                   self._report("TrackerHigh", 0.45, 0.40, 0.20)]
        table = self.evaluator.to_markdown_table(reports)
        # TrackerHigh has higher AO → must appear before TrackerLow
        assert table.index("TrackerHigh") < table.index("TrackerLow")

    def test_empty_reports(self):
        table = self.evaluator.to_markdown_table([])
        assert "|" in table  # header row still present

    def test_single_report(self):
        table = self.evaluator.to_markdown_table([self._report("X", 0.3, 0.25, 0.08)])
        assert "X" in table


# ---------------------------------------------------------------------------
# GOT10kDataset bug regression: load_sequence is public (not _load_sequence)
# ---------------------------------------------------------------------------

class TestGOT10kDatasetLoadSequencePublic:
    """Regression test ensuring __getitem__ can call load_sequence (public)."""

    def test_load_sequence_method_exists(self):
        from eovot.datasets.got10k import GOT10kDataset
        assert hasattr(GOT10kDataset, "load_sequence"), (
            "GOT10kDataset must expose a public load_sequence() method "
            "(was previously _load_sequence, causing __getitem__ to fail)"
        )

    def test_private_method_gone(self):
        from eovot.datasets.got10k import GOT10kDataset
        assert not hasattr(GOT10kDataset, "_load_sequence"), (
            "_load_sequence should be removed after renaming to load_sequence"
        )

    def test_getitem_calls_load_sequence(self, tmp_path):
        """__getitem__ should succeed with a synthetic GOT-10k structure."""
        from eovot.datasets.got10k import GOT10kDataset

        # Build minimal GOT-10k val structure
        val_dir = tmp_path / "val"
        seq_dir = val_dir / "GOT-10k_Val_000001"
        img_dir = seq_dir / "img"
        img_dir.mkdir(parents=True)

        # Write 3 synthetic frames
        for i in range(3):
            frame = np.zeros((60, 80, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"{i + 1:08d}.jpg"), frame)

        # Write groundtruth.txt
        (seq_dir / "groundtruth.txt").write_text(
            "10,20,50,40\n11,20,50,40\n12,20,50,40\n"
        )

        # list.txt
        (val_dir / "list.txt").write_text("GOT-10k_Val_000001\n")

        ds = GOT10kDataset(root=str(tmp_path), split="val")
        assert len(ds) == 1

        # This used to raise AttributeError before the bug fix
        seq = ds[0]
        assert seq.name == "GOT-10k_Val_000001"
        assert len(seq) == 3
        assert seq.ground_truth.shape == (3, 4)
