"""Tests for eovot.experiment.prediction_io."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from eovot.experiment.prediction_io import (
    PredictionComparator,
    PredictionExporter,
    PredictionLoader,
    _read_sequence,
    _write_sequence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_preds(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 200, n)
    ys = rng.uniform(0, 150, n)
    ws = rng.uniform(10, 80, n)
    hs = rng.uniform(10, 60, n)
    return np.column_stack([xs, ys, ws, hs]).astype(np.float64)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestInternalIO:
    def test_round_trip_clean_predictions(self, tmp_path: Path) -> None:
        preds = _make_preds(20, seed=1)
        path = _write_sequence(tmp_path, "seq1", preds)
        loaded = _read_sequence(path)
        assert loaded.shape == preds.shape
        np.testing.assert_allclose(loaded, preds, atol=1e-5)

    def test_nan_row_written_and_read(self, tmp_path: Path) -> None:
        preds = _make_preds(5)
        preds[2] = [float("nan")] * 4
        _write_sequence(tmp_path, "nanseq", preds)
        loaded = _read_sequence(tmp_path / "nanseq.txt")
        assert loaded.shape == (5, 4)
        assert all(math.isnan(loaded[2, i]) for i in range(4))

    def test_empty_predictions_creates_file(self, tmp_path: Path) -> None:
        preds = np.empty((0, 4), dtype=np.float64)
        path = _write_sequence(tmp_path, "empty", preds)
        assert path.exists()
        loaded = _read_sequence(path)
        assert loaded.shape == (0, 4)

    def test_file_has_correct_format(self, tmp_path: Path) -> None:
        preds = np.array([[10.0, 20.0, 50.0, 40.0]])
        _write_sequence(tmp_path, "fmt", preds)
        content = (tmp_path / "fmt.txt").read_text()
        # Each line must be 4 comma-separated floats
        lines = [l for l in content.strip().split("\n") if l]
        assert len(lines) == 1
        assert lines[0].count(",") == 3

    def test_six_decimal_places(self, tmp_path: Path) -> None:
        preds = np.array([[1.123456789, 2.0, 3.0, 4.0]])
        _write_sequence(tmp_path, "prec", preds)
        content = (tmp_path / "prec.txt").read_text()
        # Should be written to 6 decimal places
        assert "1.123457" in content or "1.123456" in content


# ---------------------------------------------------------------------------
# PredictionExporter
# ---------------------------------------------------------------------------

class TestPredictionExporter:
    def test_save_sequence_creates_file(self, tmp_path: Path) -> None:
        exporter = PredictionExporter(str(tmp_path))
        preds = _make_preds(15)
        path = exporter.save_sequence("MOSSE", "Basketball", preds)
        assert path.exists()
        assert path.name == "Basketball.txt"
        assert path.parent.name == "MOSSE"

    def test_save_sequence_creates_subdirectory(self, tmp_path: Path) -> None:
        exporter = PredictionExporter(str(tmp_path))
        exporter.save_sequence("KCF", "Bolt", _make_preds())
        assert (tmp_path / "KCF").is_dir()

    def test_save_sequence_roundtrip(self, tmp_path: Path) -> None:
        exporter = PredictionExporter(str(tmp_path))
        preds = _make_preds(30, seed=7)
        exporter.save_sequence("CSRT", "Car4", preds)
        loaded = PredictionLoader(str(tmp_path / "CSRT")).load_sequence("Car4")
        np.testing.assert_allclose(loaded, preds, atol=1e-5)

    def test_save_benchmark_result(self, tmp_path: Path) -> None:
        """Test save() with a minimal BenchmarkResult-like stub."""
        from unittest.mock import MagicMock
        result = MagicMock()
        result.tracker_name = "MOSSE"
        seq1 = MagicMock()
        seq1.sequence_name = "Seq1"
        seq1.predictions = _make_preds(10, seed=0)
        seq2 = MagicMock()
        seq2.sequence_name = "Seq2"
        seq2.predictions = _make_preds(10, seed=1)
        result.sequence_results = [seq1, seq2]

        exporter = PredictionExporter(str(tmp_path))
        written = exporter.save(result)

        assert "Seq1" in written
        assert "Seq2" in written
        assert written["Seq1"].exists()
        assert written["Seq2"].exists()

    def test_save_skips_none_predictions(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock
        result = MagicMock()
        result.tracker_name = "MOSSE"
        seq = MagicMock()
        seq.sequence_name = "NoPreds"
        seq.predictions = None
        result.sequence_results = [seq]

        exporter = PredictionExporter(str(tmp_path))
        written = exporter.save(result)
        assert "NoPreds" not in written

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        exporter = PredictionExporter(str(tmp_path))
        preds1 = np.array([[1.0, 2.0, 3.0, 4.0]])
        preds2 = np.array([[9.0, 8.0, 7.0, 6.0]])
        exporter.save_sequence("T", "S", preds1)
        exporter.save_sequence("T", "S", preds2)
        loaded = PredictionLoader(str(tmp_path / "T")).load_sequence("S")
        np.testing.assert_allclose(loaded[0], [9.0, 8.0, 7.0, 6.0], atol=1e-5)


# ---------------------------------------------------------------------------
# PredictionLoader
# ---------------------------------------------------------------------------

class TestPredictionLoader:
    def test_load_sequence(self, tmp_path: Path) -> None:
        preds = _make_preds(12, seed=3)
        _write_sequence(tmp_path, "Car1", preds)
        loader = PredictionLoader(str(tmp_path))
        loaded = loader.load_sequence("Car1")
        assert loaded.shape == (12, 4)
        np.testing.assert_allclose(loaded, preds, atol=1e-5)

    def test_load_sequence_missing_raises(self, tmp_path: Path) -> None:
        loader = PredictionLoader(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="NonExistent"):
            loader.load_sequence("NonExistent")

    def test_load_all_returns_all_sequences(self, tmp_path: Path) -> None:
        for name in ["SeqA", "SeqB", "SeqC"]:
            _write_sequence(tmp_path, name, _make_preds(5))
        loader = PredictionLoader(str(tmp_path))
        all_preds = loader.load_all()
        assert set(all_preds.keys()) == {"SeqA", "SeqB", "SeqC"}

    def test_load_all_empty_dir(self, tmp_path: Path) -> None:
        loader = PredictionLoader(str(tmp_path))
        assert loader.load_all() == {}

    def test_list_sequences(self, tmp_path: Path) -> None:
        for name in ["Bolt", "Car4", "David"]:
            _write_sequence(tmp_path, name, _make_preds(5))
        loader = PredictionLoader(str(tmp_path))
        names = loader.list_sequences()
        assert names == sorted(["Bolt", "Car4", "David"])

    def test_list_sequences_empty(self, tmp_path: Path) -> None:
        loader = PredictionLoader(str(tmp_path))
        assert loader.list_sequences() == []

    def test_loaded_dtype_is_float64(self, tmp_path: Path) -> None:
        _write_sequence(tmp_path, "Dtype", _make_preds(5))
        loader = PredictionLoader(str(tmp_path))
        loaded = loader.load_sequence("Dtype")
        assert loaded.dtype == np.float64

    def test_nan_preserved_through_roundtrip(self, tmp_path: Path) -> None:
        preds = _make_preds(6)
        preds[3] = np.nan
        _write_sequence(tmp_path, "WithNaN", preds)
        loader = PredictionLoader(str(tmp_path))
        loaded = loader.load_sequence("WithNaN")
        assert all(math.isnan(loaded[3, i]) for i in range(4))
        for i in [0, 1, 2, 4, 5]:
            assert not any(math.isnan(loaded[i, j]) for j in range(4))


# ---------------------------------------------------------------------------
# PredictionComparator
# ---------------------------------------------------------------------------

class TestPredictionComparator:
    def _setup_two_trackers(
        self, tmp_path: Path, sequences: list[str]
    ) -> PredictionComparator:
        dir_a = tmp_path / "TrackA"
        dir_b = tmp_path / "TrackB"
        dir_a.mkdir()
        dir_b.mkdir()
        for seq in sequences:
            _write_sequence(dir_a, seq, _make_preds(10, seed=0))
            _write_sequence(dir_b, seq, _make_preds(10, seed=1))
        return PredictionComparator(str(dir_a), str(dir_b))

    def test_common_sequences(self, tmp_path: Path) -> None:
        cmp = self._setup_two_trackers(tmp_path, ["Seq1", "Seq2", "Seq3"])
        assert cmp.common_sequences() == ["Seq1", "Seq2", "Seq3"]

    def test_common_sequences_partial_overlap(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "A"
        dir_b = tmp_path / "B"
        dir_a.mkdir()
        dir_b.mkdir()
        for seq in ["Seq1", "Seq2"]:
            _write_sequence(dir_a, seq, _make_preds())
        for seq in ["Seq2", "Seq3"]:
            _write_sequence(dir_b, seq, _make_preds())
        cmp = PredictionComparator(str(dir_a), str(dir_b))
        assert cmp.common_sequences() == ["Seq2"]

    def test_spatial_iou_identical_predictions(self, tmp_path: Path) -> None:
        preds = _make_preds(10, seed=42)
        dir_a = tmp_path / "A"
        dir_b = tmp_path / "B"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_sequence(dir_a, "Identical", preds)
        _write_sequence(dir_b, "Identical", preds)
        cmp = PredictionComparator(str(dir_a), str(dir_b))
        ious = cmp.spatial_iou("Identical")
        np.testing.assert_allclose(ious, 1.0, atol=1e-5)

    def test_spatial_iou_non_overlapping(self, tmp_path: Path) -> None:
        preds_a = np.array([[0.0, 0.0, 10.0, 10.0]] * 5)
        preds_b = np.array([[200.0, 200.0, 10.0, 10.0]] * 5)
        dir_a = tmp_path / "A"
        dir_b = tmp_path / "B"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_sequence(dir_a, "NoOverlap", preds_a)
        _write_sequence(dir_b, "NoOverlap", preds_b)
        cmp = PredictionComparator(str(dir_a), str(dir_b))
        ious = cmp.spatial_iou("NoOverlap")
        np.testing.assert_allclose(ious, 0.0, atol=1e-5)

    def test_mean_spatial_iou_range(self, tmp_path: Path) -> None:
        cmp = self._setup_two_trackers(tmp_path, ["S"])
        mean_iou = cmp.mean_spatial_iou("S")
        assert 0.0 <= mean_iou <= 1.0

    def test_agreement_summary_keys(self, tmp_path: Path) -> None:
        seqs = ["Bolt", "Car4", "David"]
        cmp = self._setup_two_trackers(tmp_path, seqs)
        summary = cmp.agreement_summary()
        assert set(summary.keys()) == set(seqs)
        for v in summary.values():
            assert 0.0 <= v <= 1.0
