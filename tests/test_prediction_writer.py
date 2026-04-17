"""Unit tests for eovot.experiments.prediction_writer and engine fixes."""

import os
import tempfile

import numpy as np
import pytest

from eovot.experiments.prediction_writer import PredictionFormat, PredictionWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_preds(n: int = 10) -> np.ndarray:
    """Return (n, 4) array of random bounding boxes."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 200, size=(n, 4))


# ---------------------------------------------------------------------------
# PredictionFormat enum
# ---------------------------------------------------------------------------

class TestPredictionFormat:
    def test_values(self):
        assert PredictionFormat.OTB.value == "otb"
        assert PredictionFormat.GOT10K.value == "got10k"
        assert PredictionFormat.VOT.value == "vot"


# ---------------------------------------------------------------------------
# PredictionWriter construction
# ---------------------------------------------------------------------------

class TestPredictionWriterConstruction:
    def test_default_format_is_otb(self, tmp_path):
        pw = PredictionWriter(str(tmp_path))
        assert pw.fmt == PredictionFormat.OTB

    def test_got10k_format(self, tmp_path):
        pw = PredictionWriter(str(tmp_path), fmt="got10k")
        assert pw.fmt == PredictionFormat.GOT10K
        assert pw.delimiter == ","

    def test_vot_format(self, tmp_path):
        pw = PredictionWriter(str(tmp_path), fmt="vot")
        assert pw.fmt == PredictionFormat.VOT

    def test_invalid_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown prediction format"):
            PredictionWriter(str(tmp_path), fmt="imagenet")

    def test_case_insensitive_format(self, tmp_path):
        pw = PredictionWriter(str(tmp_path), fmt="GOT10K")
        assert pw.fmt == PredictionFormat.GOT10K


# ---------------------------------------------------------------------------
# write_sequence
# ---------------------------------------------------------------------------

class TestWriteSequence:
    def test_creates_file(self, tmp_path):
        pw = PredictionWriter(str(tmp_path))
        preds = _fake_preds(5)
        path = pw.write_sequence("Basketball", preds, tracker_name="MOSSE")
        assert os.path.isfile(path)

    def test_directory_layout(self, tmp_path):
        pw = PredictionWriter(str(tmp_path))
        preds = _fake_preds(3)
        path = pw.write_sequence("CarScale", preds, tracker_name="KCF")
        assert "KCF" in path
        assert "CarScale.txt" in path

    def test_otb_space_delimiter(self, tmp_path):
        pw = PredictionWriter(str(tmp_path), fmt="otb")
        preds = np.array([[10.0, 20.0, 30.0, 40.0]])
        path = pw.write_sequence("Seq1", preds)
        with open(path) as fh:
            line = fh.readline().strip()
        assert " " in line
        assert "," not in line

    def test_got10k_comma_delimiter(self, tmp_path):
        pw = PredictionWriter(str(tmp_path), fmt="got10k")
        preds = np.array([[10.0, 20.0, 30.0, 40.0]])
        path = pw.write_sequence("Seq1", preds)
        with open(path) as fh:
            line = fh.readline().strip()
        assert "," in line

    def test_correct_number_of_lines(self, tmp_path):
        pw = PredictionWriter(str(tmp_path))
        n = 15
        preds = _fake_preds(n)
        path = pw.write_sequence("Test", preds)
        with open(path) as fh:
            lines = [l for l in fh if l.strip()]
        assert len(lines) == n

    def test_invalid_shape_raises(self, tmp_path):
        pw = PredictionWriter(str(tmp_path))
        bad_preds = np.zeros((10, 3))
        with pytest.raises(ValueError, match="shape"):
            pw.write_sequence("Seq", bad_preds)

    def test_1d_array_raises(self, tmp_path):
        pw = PredictionWriter(str(tmp_path))
        with pytest.raises(ValueError, match="shape"):
            pw.write_sequence("Seq", np.zeros(4))


# ---------------------------------------------------------------------------
# load_sequence (round-trip)
# ---------------------------------------------------------------------------

class TestLoadSequence:
    def test_round_trip_otb(self, tmp_path):
        pw = PredictionWriter(str(tmp_path), fmt="otb")
        original = _fake_preds(8)
        path = pw.write_sequence("Seq", original)
        loaded = PredictionWriter.load_sequence(path)
        np.testing.assert_allclose(loaded, original, atol=1e-3)

    def test_round_trip_got10k(self, tmp_path):
        pw = PredictionWriter(str(tmp_path), fmt="got10k")
        original = _fake_preds(8)
        path = pw.write_sequence("Seq", original)
        loaded = PredictionWriter.load_sequence(path)
        np.testing.assert_allclose(loaded, original, atol=1e-3)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            PredictionWriter.load_sequence("/does/not/exist.txt")

    def test_load_returns_float64(self, tmp_path):
        pw = PredictionWriter(str(tmp_path))
        path = pw.write_sequence("S", _fake_preds(3))
        loaded = PredictionWriter.load_sequence(path)
        assert loaded.dtype == np.float64

    def test_load_shape(self, tmp_path):
        pw = PredictionWriter(str(tmp_path))
        n = 12
        path = pw.write_sequence("S", _fake_preds(n))
        loaded = PredictionWriter.load_sequence(path)
        assert loaded.shape == (n, 4)


# ---------------------------------------------------------------------------
# write_result (integration with BenchmarkResult)
# ---------------------------------------------------------------------------

class TestWriteResult:
    def _make_result(self, n_seqs: int = 3, n_frames: int = 10):
        """Build a minimal BenchmarkResult-like object for testing."""
        from unittest.mock import MagicMock
        import numpy as np

        result = MagicMock()
        result.tracker_name = "MOSSE"
        seq_results = []
        for i in range(n_seqs):
            sr = MagicMock()
            sr.sequence_name = f"Seq{i:02d}"
            sr.predictions = _fake_preds(n_frames)
            seq_results.append(sr)
        result.sequence_results = seq_results
        return result

    def test_writes_all_sequences(self, tmp_path):
        pw = PredictionWriter(str(tmp_path))
        result = self._make_result(n_seqs=4)
        written = pw.write_result(result)
        assert len(written) == 4

    def test_raises_if_no_predictions(self, tmp_path):
        from unittest.mock import MagicMock
        pw = PredictionWriter(str(tmp_path))
        result = MagicMock()
        result.tracker_name = "MOSSE"
        sr = MagicMock()
        sr.predictions = None
        result.sequence_results = [sr]
        with pytest.raises(ValueError, match="No prediction arrays"):
            pw.write_result(result)

    def test_files_are_readable(self, tmp_path):
        pw = PredictionWriter(str(tmp_path))
        result = self._make_result(n_seqs=2, n_frames=5)
        written = pw.write_result(result)
        for path in written:
            loaded = PredictionWriter.load_sequence(path)
            assert loaded.shape == (5, 4)


# ---------------------------------------------------------------------------
# Engine fixes — SequenceResult.energy field and BenchmarkResult.to_dict
# ---------------------------------------------------------------------------

class TestEngineEnergyField:
    def test_sequence_result_has_energy_field(self):
        from eovot.benchmark.engine import SequenceResult
        from eovot.profiling.profiler import ProfilingResult

        pr = ProfilingResult(
            tracker_name="MOSSE",
            frame_count=5,
            fps=100.0,
            latency_mean_ms=10.0,
            latency_std_ms=1.0,
            latency_p95_ms=12.0,
            peak_memory_mb=50.0,
        )
        sr = SequenceResult(
            sequence_name="test",
            ious=np.array([0.5, 0.6]),
            profiling=pr,
            energy=None,
        )
        assert sr.energy is None

    def test_benchmark_result_to_dict_no_duplicate(self):
        from eovot.benchmark.engine import BenchmarkResult, SequenceResult
        from eovot.profiling.profiler import ProfilingResult

        pr = ProfilingResult(
            tracker_name="MOSSE",
            frame_count=3,
            fps=120.0,
            latency_mean_ms=8.3,
            latency_std_ms=0.5,
            latency_p95_ms=9.0,
            peak_memory_mb=40.0,
        )
        sr = SequenceResult(
            sequence_name="Basketball",
            ious=np.array([0.4, 0.5, 0.6]),
            profiling=pr,
            predictions=np.zeros((3, 4)),
            ground_truths=np.zeros((3, 4)),
            center_distances=np.array([5.0, 6.0, 7.0]),
            energy=None,
        )
        result = BenchmarkResult(
            tracker_name="MOSSE",
            dataset_name="OTB100",
            sequence_results=[sr],
        )
        d = result.to_dict()
        assert "summary" in d
        assert "sequences" in d
        assert len(d["sequences"]) == 1
        # Verify energy fields absent when energy is None
        assert "energy_j" not in d["sequences"][0]

    def test_benchmark_result_summary_includes_energy(self):
        from eovot.benchmark.engine import BenchmarkResult, SequenceResult
        from eovot.profiling.energy import EnergyResult
        from eovot.profiling.profiler import ProfilingResult

        pr = ProfilingResult(
            tracker_name="KCF",
            frame_count=5,
            fps=200.0,
            latency_mean_ms=5.0,
            latency_std_ms=0.2,
            latency_p95_ms=5.5,
            peak_memory_mb=30.0,
        )
        er = EnergyResult(
            tracker_name="KCF",
            frame_count=5,
            tdp_watts=6.0,
            total_energy_j=0.005,
            mean_power_w=1.5,
            energy_per_frame_mj=1.0,
            peak_cpu_pct=30.0,
            mean_cpu_pct=25.0,
        )
        sr = SequenceResult(
            sequence_name="Seq",
            ious=np.array([0.7, 0.8]),
            profiling=pr,
            energy=er,
        )
        result = BenchmarkResult(
            tracker_name="KCF",
            dataset_name="OTB50",
            sequence_results=[sr],
        )
        s = result.summary()
        assert "total_energy_j" in s
        assert "mean_energy_per_frame_mj" in s
        d = result.to_dict()
        assert d["sequences"][0]["energy_j"] == pytest.approx(0.005, rel=1e-4)
