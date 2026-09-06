"""Tests for DatasetStatisticsAnalyzer and DatasetStats."""

from __future__ import annotations

import os
import tempfile
from typing import Iterator

import numpy as np
import pytest

from eovot.datasets.base import BaseDataset, Sequence
from eovot.analysis.dataset_stats import (
    DatasetStatisticsAnalyzer,
    DatasetStats,
    SequenceStats,
)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _make_seq(name: str, boxes: np.ndarray) -> Sequence:
    """Build a Sequence from a (N, 4) array of (x, y, w, h) boxes."""
    n = len(boxes)
    return Sequence(name=name, frame_paths=["x"] * n, ground_truth=boxes)


def _static_seq(name: str = "s0", n: int = 10, box=(10.0, 20.0, 50.0, 40.0)) -> Sequence:
    """A sequence where the target never moves."""
    return _make_seq(name, np.tile(box, (n, 1)))


def _moving_seq(name: str = "s0", n: int = 10, dx: float = 5.0) -> Sequence:
    """A sequence where the target translates horizontally by dx each frame."""
    rows = []
    for i in range(n):
        rows.append([10.0 + i * dx, 20.0, 50.0, 40.0])
    return _make_seq(name, np.array(rows))


class _SingleSeqDataset(BaseDataset):
    def __init__(self, seq: Sequence):
        self._seq = seq

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._seq


class _MultiSeqDataset(BaseDataset):
    def __init__(self, seqs):
        self._seqs = seqs

    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, idx):
        return self._seqs[idx]


# ---------------------------------------------------------------------------
# SequenceStats — unit tests via _analyze_sequence
# ---------------------------------------------------------------------------

class TestAnalyzeSequence:
    def setup_method(self):
        self.analyzer = DatasetStatisticsAnalyzer(frame_wh=(100, 100))

    def test_static_motion_is_zero(self):
        seq = _static_seq()
        s = self.analyzer._analyze_sequence(seq)
        assert s.motion_magnitude == pytest.approx(0.0)

    def test_moving_target_motion(self):
        seq = _moving_seq(dx=3.0, n=4)
        s = self.analyzer._analyze_sequence(seq)
        assert s.motion_magnitude == pytest.approx(3.0)

    def test_static_scale_variability_is_zero(self):
        seq = _static_seq()
        s = self.analyzer._analyze_sequence(seq)
        assert s.scale_variability == pytest.approx(0.0, abs=1e-10)

    def test_scale_variability_grows_with_size_change(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 90.0, 90.0]])
        seq = _make_seq("v", boxes)
        s = self.analyzer._analyze_sequence(seq)
        assert s.scale_variability > 0.1

    def test_static_aspect_ratio_stability_is_one(self):
        seq = _static_seq()
        s = self.analyzer._analyze_sequence(seq)
        assert s.aspect_ratio_stability == pytest.approx(1.0, abs=1e-4)

    def test_target_coverage_matches_formula(self):
        # frame 100x100, box 10x10 → coverage = 0.01
        seq = _static_seq(box=(0.0, 0.0, 10.0, 10.0))
        s = self.analyzer._analyze_sequence(seq)
        assert s.target_coverage == pytest.approx(0.01)

    def test_single_frame_motion_is_zero(self):
        seq = _make_seq("one", np.array([[5.0, 5.0, 20.0, 20.0]]))
        s = self.analyzer._analyze_sequence(seq)
        assert s.motion_magnitude == pytest.approx(0.0)

    def test_single_frame_scale_var_is_zero(self):
        seq = _make_seq("one", np.array([[5.0, 5.0, 20.0, 20.0]]))
        s = self.analyzer._analyze_sequence(seq)
        assert s.scale_variability == pytest.approx(0.0)

    def test_single_frame_ar_stability_is_one(self):
        seq = _make_seq("one", np.array([[5.0, 5.0, 20.0, 20.0]]))
        s = self.analyzer._analyze_sequence(seq)
        assert s.aspect_ratio_stability == pytest.approx(1.0)

    def test_n_frames_matches_sequence(self):
        seq = _static_seq(n=17)
        s = self.analyzer._analyze_sequence(seq)
        assert s.n_frames == 17

    def test_sequence_name_preserved(self):
        seq = _static_seq(name="my_seq")
        s = self.analyzer._analyze_sequence(seq)
        assert s.name == "my_seq"

    def test_zero_height_box_does_not_crash(self):
        boxes = np.array([[0.0, 0.0, 10.0, 0.0], [0.0, 0.0, 10.0, 0.0]])
        seq = _make_seq("zeroH", boxes)
        s = self.analyzer._analyze_sequence(seq)
        assert s.aspect_ratio_stability >= 0.0


# ---------------------------------------------------------------------------
# DatasetStatisticsAnalyzer.analyze()
# ---------------------------------------------------------------------------

class TestAnalyze:
    def setup_method(self):
        self.analyzer = DatasetStatisticsAnalyzer(frame_wh=(160, 120))

    def test_returns_dataset_stats(self):
        ds = _SingleSeqDataset(_static_seq())
        result = self.analyzer.analyze(ds, dataset_name="Test")
        assert isinstance(result, DatasetStats)

    def test_n_sequences_correct(self):
        seqs = [_static_seq(f"s{i}") for i in range(4)]
        ds = _MultiSeqDataset(seqs)
        result = self.analyzer.analyze(ds)
        assert result.n_sequences == 4

    def test_total_frames_correct(self):
        seqs = [_static_seq(n=5), _static_seq(n=8)]
        ds = _MultiSeqDataset(seqs)
        result = self.analyzer.analyze(ds)
        assert result.total_frames == 13

    def test_empty_dataset_raises(self):
        ds = _MultiSeqDataset([])
        with pytest.raises(ValueError):
            self.analyzer.analyze(ds)

    def test_dataset_name_propagated(self):
        ds = _SingleSeqDataset(_static_seq())
        result = self.analyzer.analyze(ds, dataset_name="MyDataset")
        assert result.dataset_name == "MyDataset"

    def test_mean_motion_static_is_zero(self):
        seqs = [_static_seq(f"s{i}") for i in range(3)]
        ds = _MultiSeqDataset(seqs)
        result = self.analyzer.analyze(ds)
        assert result.mean_motion_magnitude == pytest.approx(0.0)

    def test_mean_motion_moving_gt_zero(self):
        seqs = [_moving_seq(f"s{i}", dx=4.0) for i in range(3)]
        ds = _MultiSeqDataset(seqs)
        result = self.analyzer.analyze(ds)
        assert result.mean_motion_magnitude > 0.0

    def test_sequence_stats_list_length(self):
        seqs = [_static_seq(f"s{i}") for i in range(5)]
        ds = _MultiSeqDataset(seqs)
        result = self.analyzer.analyze(ds)
        assert len(result.sequence_stats) == 5

    def test_sequence_stats_are_sequence_stats(self):
        ds = _SingleSeqDataset(_static_seq())
        result = self.analyzer.analyze(ds)
        assert all(isinstance(s, SequenceStats) for s in result.sequence_stats)

    def test_str_contains_dataset_name(self):
        ds = _SingleSeqDataset(_static_seq())
        result = self.analyzer.analyze(ds, dataset_name="XYZ")
        assert "XYZ" in str(result)

    def test_mean_target_coverage_in_range(self):
        ds = _SingleSeqDataset(_static_seq())
        result = self.analyzer.analyze(ds)
        assert 0.0 <= result.mean_target_coverage <= 1.0


# ---------------------------------------------------------------------------
# DatasetStats.to_markdown_table()
# ---------------------------------------------------------------------------

class TestToMarkdownTable:
    def setup_method(self):
        analyzer = DatasetStatisticsAnalyzer(frame_wh=(160, 120))
        seqs = [_static_seq("alpha", n=5), _moving_seq("beta", n=8, dx=2.0)]
        ds = _MultiSeqDataset(seqs)
        self.result = analyzer.analyze(ds, dataset_name="DemoDS")

    def test_header_contains_sequence(self):
        table = self.result.to_markdown_table()
        assert "Sequence" in table

    def test_header_contains_motion(self):
        table = self.result.to_markdown_table()
        assert "Motion" in table

    def test_header_contains_frames(self):
        table = self.result.to_markdown_table()
        assert "Frames" in table

    def test_sequence_names_in_table(self):
        table = self.result.to_markdown_table()
        assert "alpha" in table
        assert "beta" in table

    def test_dataset_name_in_summary_row(self):
        table = self.result.to_markdown_table()
        assert "DemoDS" in table

    def test_table_is_multiline(self):
        table = self.result.to_markdown_table()
        assert table.count("\n") >= 3


# ---------------------------------------------------------------------------
# DatasetStats.to_csv()
# ---------------------------------------------------------------------------

class TestToCsv:
    def setup_method(self):
        analyzer = DatasetStatisticsAnalyzer(frame_wh=(160, 120))
        seqs = [_static_seq("alpha", n=5), _moving_seq("beta", n=8, dx=2.0)]
        ds = _MultiSeqDataset(seqs)
        self.result = analyzer.analyze(ds, dataset_name="DemoDS")

    def test_csv_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stats.csv")
            self.result.to_csv(path)
            assert os.path.isfile(path)

    def test_csv_row_count(self):
        import csv as _csv
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stats.csv")
            self.result.to_csv(path)
            with open(path, newline="") as f:
                rows = list(_csv.DictReader(f))
            # One row per sequence
            assert len(rows) == 2

    def test_csv_contains_sequence_names(self):
        import csv as _csv
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stats.csv")
            self.result.to_csv(path)
            with open(path, newline="") as f:
                rows = list(_csv.DictReader(f))
            names = [r["sequence"] for r in rows]
            assert "alpha" in names
            assert "beta" in names

    def test_csv_motion_column_numeric(self):
        import csv as _csv
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stats.csv")
            self.result.to_csv(path)
            with open(path, newline="") as f:
                rows = list(_csv.DictReader(f))
            for r in rows:
                assert float(r["motion_magnitude"]) >= 0.0
