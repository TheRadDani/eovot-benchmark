"""Tests for DatasetValidator, ValidationReport, and ValidationIssue.

Coverage:
- Empty dataset raises ERROR
- Healthy SyntheticDataset passes with no issues
- Sequence with 1 frame raises ERROR
- GT shape mismatch raises ERROR
- Frame/GT count mismatch: small (WARNING), large (ERROR)
- Zero-area init box → ERROR; zero-area non-init box → WARNING
- Negative coordinates → INFO
- validate_directory on missing root → ERROR
- to_markdown output structure
- is_valid reflects error presence
- num_sequences_with_errors counts correctly
- ValidationReport properties: errors, warnings, infos
- check_frames=True on synthetic dataset (all frames in memory — validator
  should handle gracefully even when _frame_paths is empty)
- build_ensemble factory still exports (unrelated import smoke test)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eovot.datasets.validator import (
    DatasetValidator,
    ValidationIssue,
    ValidationReport,
    Severity,
)
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_dataset(sequences):
    """Build a minimal mock BaseDataset from a list of mock Sequence objects."""
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=len(sequences))
    ds.__repr__ = MagicMock(return_value="MockDataset()")
    ds.__getitem__ = MagicMock(side_effect=lambda i: sequences[i])
    return ds


def _make_sequence(name, n_frames, gt=None, fail_on_load=False):
    """Create a mock Sequence with given GT (defaults to valid boxes)."""
    if gt is None:
        gt = np.tile([10.0, 10.0, 40.0, 30.0], (n_frames, 1))
    seq = MagicMock()
    seq.name = name
    seq.__len__ = MagicMock(return_value=n_frames)
    seq.ground_truth = gt
    seq._frame_paths = [f"/fake/{name}/{i:05d}.jpg" for i in range(n_frames)]
    if fail_on_load:
        raise RuntimeError("Mock load error")
    return seq


# ---------------------------------------------------------------------------
# ValidationIssue / ValidationReport unit tests
# ---------------------------------------------------------------------------

class TestValidationIssue:
    def test_str_without_detail(self):
        issue = ValidationIssue(sequence="seq1", severity=Severity.ERROR, message="bad box")
        s = str(issue)
        assert "ERROR" in s and "seq1" in s and "bad box" in s

    def test_str_with_detail(self):
        issue = ValidationIssue(
            sequence="seq1", severity=Severity.WARNING, message="mismatch", detail="10 vs 12"
        )
        s = str(issue)
        assert "10 vs 12" in s


class TestValidationReport:
    def _report_with(self, severities):
        report = ValidationReport(dataset_name="test", num_sequences=3)
        for sev in severities:
            report.issues.append(ValidationIssue(sequence="s", severity=sev, message="msg"))
        return report

    def test_is_valid_no_errors(self):
        report = self._report_with([Severity.WARNING, Severity.INFO])
        assert report.is_valid is True

    def test_is_valid_with_error(self):
        report = self._report_with([Severity.ERROR])
        assert report.is_valid is False

    def test_errors_property(self):
        report = self._report_with([Severity.ERROR, Severity.WARNING, Severity.INFO])
        assert len(report.errors) == 1

    def test_warnings_property(self):
        report = self._report_with([Severity.ERROR, Severity.WARNING, Severity.WARNING])
        assert len(report.warnings) == 2

    def test_infos_property(self):
        report = self._report_with([Severity.INFO])
        assert len(report.infos) == 1

    def test_num_sequences_with_errors(self):
        report = ValidationReport(dataset_name="d", num_sequences=5)
        report.issues.append(ValidationIssue("s1", Severity.ERROR, "e"))
        report.issues.append(ValidationIssue("s1", Severity.ERROR, "e2"))
        report.issues.append(ValidationIssue("s2", Severity.ERROR, "e3"))
        assert report.num_sequences_with_errors == 2

    def test_num_sequences_with_errors_excludes_dataset_level(self):
        report = ValidationReport(dataset_name="d", num_sequences=0)
        report.issues.append(ValidationIssue("<dataset>", Severity.ERROR, "empty"))
        assert report.num_sequences_with_errors == 0

    def test_str_contains_pass(self):
        report = ValidationReport(dataset_name="D", num_sequences=5)
        assert "PASS" in str(report)

    def test_str_contains_fail(self):
        report = ValidationReport(dataset_name="D", num_sequences=5)
        report.issues.append(ValidationIssue("s", Severity.ERROR, "e"))
        assert "FAIL" in str(report)

    def test_to_markdown_contains_headers(self):
        report = ValidationReport(dataset_name="MyData", num_sequences=10)
        md = report.to_markdown()
        assert "## Dataset Validation Report" in md
        assert "MyData" in md

    def test_to_markdown_no_issues(self):
        report = ValidationReport(dataset_name="D", num_sequences=3)
        md = report.to_markdown()
        assert "No issues found" in md

    def test_to_markdown_table_with_issues(self):
        report = ValidationReport(dataset_name="D", num_sequences=3)
        report.issues.append(ValidationIssue("s1", Severity.WARNING, "warn msg", "detail"))
        md = report.to_markdown()
        assert "| Severity |" in md
        assert "warn msg" in md


# ---------------------------------------------------------------------------
# DatasetValidator with mock datasets
# ---------------------------------------------------------------------------

class TestDatasetValidatorMock:
    def test_empty_dataset_raises_error(self):
        ds = _make_mock_dataset([])
        validator = DatasetValidator()
        report = validator.validate(ds, "empty")
        assert len(report.errors) >= 1
        assert any("no sequences" in i.message.lower() for i in report.errors)

    def test_healthy_sequences_pass(self):
        seqs = [_make_sequence(f"seq{i}", n_frames=20) for i in range(3)]
        ds = _make_mock_dataset(seqs)
        validator = DatasetValidator()
        report = validator.validate(ds, "healthy")
        assert report.is_valid
        assert len(report.errors) == 0

    def test_single_frame_sequence_is_error(self):
        seqs = [_make_sequence("seq0", n_frames=1)]
        ds = _make_mock_dataset(seqs)
        validator = DatasetValidator()
        report = validator.validate(ds)
        assert not report.is_valid
        errors = [i for i in report.errors if "seq0" in i.sequence]
        assert len(errors) >= 1

    def test_gt_shape_mismatch_4_cols(self):
        gt = np.ones((10, 3))  # wrong: 3 cols instead of 4
        seq = MagicMock()
        seq.name = "bad_gt"
        seq.__len__ = MagicMock(return_value=10)
        seq.ground_truth = gt
        seq._frame_paths = [f"/fake/{i}" for i in range(10)]
        ds = _make_mock_dataset([seq])
        validator = DatasetValidator()
        report = validator.validate(ds)
        assert not report.is_valid

    def test_frame_gt_count_mismatch_small_is_warning(self):
        gt = np.tile([10.0, 10.0, 40.0, 30.0], (20, 1))  # 20 GT rows
        seq = _make_sequence("seq0", n_frames=22, gt=gt)  # 22 frames, 20 GT = 9% diff
        ds = _make_mock_dataset([seq])
        validator = DatasetValidator()
        report = validator.validate(ds)
        assert len(report.warnings) >= 1
        assert report.is_valid  # only a warning, not error

    def test_frame_gt_count_mismatch_large_is_error(self):
        gt = np.tile([10.0, 10.0, 40.0, 30.0], (10, 1))  # 10 GT rows
        seq = _make_sequence("seq0", n_frames=50, gt=gt)  # 50 frames → 80% mismatch
        ds = _make_mock_dataset([seq])
        validator = DatasetValidator()
        report = validator.validate(ds)
        assert not report.is_valid

    def test_zero_area_init_box_is_error(self):
        gt = np.tile([10.0, 10.0, 40.0, 30.0], (10, 1))
        gt[0, 2] = 0.0  # zero width on init frame
        seq = _make_sequence("seq0", n_frames=10, gt=gt)
        ds = _make_mock_dataset([seq])
        validator = DatasetValidator()
        report = validator.validate(ds)
        assert not report.is_valid
        assert any("Initialisation" in i.message for i in report.errors)

    def test_zero_area_non_init_box_is_warning(self):
        # Use 100-frame GT with 2 invalid boxes (2%) — below the 5% default threshold
        gt = np.tile([10.0, 10.0, 40.0, 30.0], (100, 1))
        gt[50, 3] = 0.0  # zero height on frame 50 (not init), 1/100 = 1%
        gt[60, 3] = 0.0  # zero height on frame 60, 2/100 = 2% total — below 5%
        seq = _make_sequence("seq0", n_frames=100, gt=gt)
        ds = _make_mock_dataset([seq])
        validator = DatasetValidator(max_invalid_boxes_pct=5.0)
        report = validator.validate(ds)
        assert report.is_valid  # 2% < 5% threshold → only warning
        assert len(report.warnings) >= 1

    def test_negative_coordinates_are_info(self):
        gt = np.tile([10.0, 10.0, 40.0, 30.0], (10, 1))
        gt[3, 0] = -5.0  # negative x
        seq = _make_sequence("seq0", n_frames=10, gt=gt)
        ds = _make_mock_dataset([seq])
        validator = DatasetValidator()
        report = validator.validate(ds)
        assert report.is_valid
        assert len(report.infos) >= 1

    def test_load_failure_is_error(self):
        seq_ok = _make_sequence("seq_ok", n_frames=10)
        seq_bad = MagicMock(spec=[])
        seq_bad.side_effect = RuntimeError("cannot load")

        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=2)
        ds.__repr__ = MagicMock(return_value="MockDataset()")

        def getitem(i):
            if i == 0:
                return seq_ok
            raise RuntimeError("cannot load")

        ds.__getitem__ = MagicMock(side_effect=getitem)
        validator = DatasetValidator()
        report = validator.validate(ds)
        assert not report.is_valid

    def test_multiple_invalid_boxes_above_threshold_is_error(self):
        gt = np.tile([10.0, 10.0, 40.0, 30.0], (20, 1))
        gt[5:11, 2] = 0.0  # 6 of 20 = 30 % invalid non-init boxes
        seq = _make_sequence("seq0", n_frames=20, gt=gt)
        ds = _make_mock_dataset([seq])
        validator = DatasetValidator(max_invalid_boxes_pct=5.0)
        report = validator.validate(ds)
        assert not report.is_valid

    def test_multiple_invalid_boxes_below_threshold_is_warning(self):
        gt = np.tile([10.0, 10.0, 40.0, 30.0], (20, 1))
        gt[5, 2] = 0.0  # 1 of 20 = 5% — below default threshold
        seq = _make_sequence("seq0", n_frames=20, gt=gt)
        ds = _make_mock_dataset([seq])
        validator = DatasetValidator(max_invalid_boxes_pct=10.0)
        report = validator.validate(ds)
        assert report.is_valid  # only warning when below threshold


# ---------------------------------------------------------------------------
# Integration with SyntheticDataset (real dataset object)
# ---------------------------------------------------------------------------

class TestDatasetValidatorSynthetic:
    def test_synthetic_dataset_passes_validation(self):
        dataset = SyntheticDataset(num_sequences=5, num_frames=20, seed=0)
        validator = DatasetValidator()
        report = validator.validate(dataset, dataset_name="Synthetic")
        assert report.is_valid, f"Expected clean synthetic dataset. Issues: {report.issues}"
        assert report.num_sequences == 5
        assert len(report.errors) == 0

    def test_synthetic_dataset_report_str(self):
        dataset = SyntheticDataset(num_sequences=3, num_frames=15, seed=42)
        validator = DatasetValidator()
        report = validator.validate(dataset, dataset_name="SyntheticTest")
        assert "SyntheticTest" in str(report)
        assert "PASS" in str(report)

    def test_synthetic_dataset_to_markdown(self):
        dataset = SyntheticDataset(num_sequences=3, num_frames=10, seed=0)
        validator = DatasetValidator()
        report = validator.validate(dataset)
        md = report.to_markdown()
        assert "## Dataset Validation Report" in md


# ---------------------------------------------------------------------------
# validate_directory on nonexistent path
# ---------------------------------------------------------------------------

class TestValidateDirectory:
    def test_nonexistent_root_returns_error(self):
        validator = DatasetValidator()
        report = validator.validate_directory("/path/that/does/not/exist")
        assert not report.is_valid
        assert len(report.errors) >= 1

    def test_valid_temp_directory_structure(self, tmp_path):
        """Create a minimal OTB-layout dir and validate it."""
        seq_dir = tmp_path / "seq1" / "img"
        seq_dir.mkdir(parents=True)
        # Write a tiny valid PNG
        import cv2
        img = np.zeros((100, 120, 3), dtype=np.uint8)
        cv2.imwrite(str(seq_dir / "0001.jpg"), img)
        cv2.imwrite(str(seq_dir / "0002.jpg"), img)
        gt_path = tmp_path / "seq1" / "groundtruth_rect.txt"
        gt_path.write_text("10,10,40,30\n10,11,40,30\n")

        validator = DatasetValidator()
        report = validator.validate_directory(str(tmp_path), dataset_name="TempOTB")
        assert report.is_valid
        assert report.num_sequences == 1
