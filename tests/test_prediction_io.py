"""Tests for eovot.utils.prediction_io."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eovot.utils.prediction_io import (
    PredictionFormat,
    PredictionReader,
    PredictionWriter,
    _sanitize,
    load_predictions_from_benchmark_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PREDS = np.array(
    [
        [10.0, 20.0, 50.0, 40.0],
        [12.0, 22.0, 50.0, 40.0],
        [14.0, 24.0, 50.0, 40.0],
        [16.0, 26.0, 50.0, 40.0],
        [18.0, 28.0, 50.0, 40.0],
    ],
    dtype=np.float64,
)


@pytest.fixture
def benchmark_result_dict():
    """A minimal BenchmarkResult dict (as from BenchmarkResult.to_dict())."""
    return {
        "summary": {
            "tracker": "TestTracker",
            "dataset": "Synthetic",
            "num_sequences": 2,
            "mean_iou": 0.75,
            "mean_fps": 300.0,
            "peak_memory_mb": 100.0,
        },
        "sequences": [
            {
                "sequence_name": "seq_001",
                "mean_iou": 0.80,
                "fps": 310.0,
                "mean_latency_ms": 3.2,
                "peak_memory_mb": 95.0,
                "predictions": SAMPLE_PREDS.tolist(),
                "ious": [0.8, 0.8, 0.8, 0.8, 0.8],
            },
            {
                "sequence_name": "seq_002",
                "mean_iou": 0.70,
                "fps": 290.0,
                "mean_latency_ms": 3.4,
                "peak_memory_mb": 98.0,
                "predictions": (SAMPLE_PREDS + 5.0).tolist(),
                "ious": [0.7, 0.7, 0.7, 0.7, 0.7],
            },
        ],
    }


# ---------------------------------------------------------------------------
# PredictionFormat
# ---------------------------------------------------------------------------

def test_format_values():
    assert PredictionFormat.OTB.value == "otb"
    assert PredictionFormat.GOT10K.value == "got10k"
    assert PredictionFormat.VOT.value == "vot"
    assert PredictionFormat.JSON.value == "json"


def test_format_from_string():
    assert PredictionFormat("otb") == PredictionFormat.OTB
    assert PredictionFormat("json") == PredictionFormat.JSON


# ---------------------------------------------------------------------------
# _sanitize
# ---------------------------------------------------------------------------

def test_sanitize_removes_parens():
    assert _sanitize("AdaptiveRes(MOSSE)") == "AdaptiveRes_MOSSE_"


def test_sanitize_removes_slashes():
    assert "/" not in _sanitize("path/with/slashes")
    assert "\\" not in _sanitize("path\\with\\backslash")


def test_sanitize_plain_name_unchanged():
    assert _sanitize("MOSSE") == "MOSSE"
    assert _sanitize("KCF") == "KCF"


# ---------------------------------------------------------------------------
# PredictionWriter / PredictionReader — OTB format
# ---------------------------------------------------------------------------

def test_otb_roundtrip(tmp_path, benchmark_result_dict):
    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    paths = writer.write_benchmark_result(benchmark_result_dict)

    assert "seq_001" in paths
    assert "seq_002" in paths
    assert paths["seq_001"].name == "predictions.txt"
    assert paths["seq_001"].exists()

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    loaded = reader.read_tracker("TestTracker")

    assert "seq_001" in loaded
    assert "seq_002" in loaded
    np.testing.assert_allclose(loaded["seq_001"], SAMPLE_PREDS, atol=1e-3)
    np.testing.assert_allclose(loaded["seq_002"], SAMPLE_PREDS + 5.0, atol=1e-3)


def test_otb_file_has_four_columns(tmp_path, benchmark_result_dict):
    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    paths = writer.write_benchmark_result(benchmark_result_dict)
    text = paths["seq_001"].read_text()
    for line in text.strip().splitlines():
        assert len(line.split(",")) == 4


def test_otb_single_sequence_write_read(tmp_path):
    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    writer.write_sequence("KCF", "car1", SAMPLE_PREDS)

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    loaded = reader.read_sequence("KCF", "car1")
    np.testing.assert_allclose(loaded, SAMPLE_PREDS, atol=1e-3)


# ---------------------------------------------------------------------------
# GOT-10k format
# ---------------------------------------------------------------------------

def test_got10k_filename(tmp_path, benchmark_result_dict):
    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.GOT10K)
    paths = writer.write_benchmark_result(benchmark_result_dict)
    assert paths["seq_001"].name == "prediction.txt"


def test_got10k_roundtrip(tmp_path, benchmark_result_dict):
    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.GOT10K)
    writer.write_benchmark_result(benchmark_result_dict)

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.GOT10K)
    loaded = reader.read_tracker("TestTracker")

    np.testing.assert_allclose(loaded["seq_001"], SAMPLE_PREDS, atol=1e-3)


# ---------------------------------------------------------------------------
# VOT format
# ---------------------------------------------------------------------------

def test_vot_without_confidence(tmp_path, benchmark_result_dict):
    writer = PredictionWriter(
        output_dir=str(tmp_path), fmt=PredictionFormat.VOT, include_confidence=False
    )
    paths = writer.write_benchmark_result(benchmark_result_dict)
    text = paths["seq_001"].read_text()
    for line in text.strip().splitlines():
        assert len(line.split(",")) == 4  # no confidence column


def test_vot_with_confidence(tmp_path, benchmark_result_dict):
    writer = PredictionWriter(
        output_dir=str(tmp_path), fmt=PredictionFormat.VOT, include_confidence=True
    )
    paths = writer.write_benchmark_result(benchmark_result_dict)
    text = paths["seq_001"].read_text()
    for line in text.strip().splitlines():
        parts = line.split(",")
        assert len(parts) == 5
        assert float(parts[4]) == pytest.approx(1.0)


def test_vot_roundtrip(tmp_path, benchmark_result_dict):
    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.VOT)
    writer.write_benchmark_result(benchmark_result_dict)

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.VOT)
    loaded = reader.read_tracker("TestTracker")
    np.testing.assert_allclose(loaded["seq_001"], SAMPLE_PREDS, atol=1e-3)


# ---------------------------------------------------------------------------
# JSON format
# ---------------------------------------------------------------------------

def test_json_roundtrip(tmp_path, benchmark_result_dict):
    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.JSON)
    writer.write_benchmark_result(benchmark_result_dict)

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.JSON)
    loaded = reader.read_tracker("TestTracker")

    assert "seq_001" in loaded
    np.testing.assert_allclose(loaded["seq_001"], SAMPLE_PREDS, atol=1e-6)


def test_json_combined_file_exists(tmp_path, benchmark_result_dict):
    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.JSON)
    writer.write_benchmark_result(benchmark_result_dict)

    combined = tmp_path / "TestTracker" / "_benchmark_result.json"
    assert combined.exists()


def test_json_single_sequence_read(tmp_path, benchmark_result_dict):
    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.JSON)
    writer.write_benchmark_result(benchmark_result_dict)

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.JSON)
    loaded = reader.read_sequence("TestTracker", "seq_001")
    np.testing.assert_allclose(loaded, SAMPLE_PREDS, atol=1e-6)


# ---------------------------------------------------------------------------
# PredictionReader edge cases
# ---------------------------------------------------------------------------

def test_reader_missing_tracker_raises(tmp_path):
    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    with pytest.raises(FileNotFoundError):
        reader.read_tracker("NonExistent")


def test_reader_missing_sequence_raises(tmp_path, benchmark_result_dict):
    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    writer.write_benchmark_result(benchmark_result_dict)

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    with pytest.raises(FileNotFoundError):
        reader.read_sequence("TestTracker", "nonexistent_seq")


def test_reader_read_all_returns_nested_dict(tmp_path, benchmark_result_dict):
    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    writer.write_benchmark_result(benchmark_result_dict)

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    all_data = reader.read_all()

    assert "TestTracker" in all_data
    assert "seq_001" in all_data["TestTracker"]
    assert "seq_002" in all_data["TestTracker"]


def test_parse_whitespace_delimiter(tmp_path):
    """Reader should parse space-delimited files as well as comma-delimited."""
    seq_dir = tmp_path / "TrackerA" / "seq_ws"
    seq_dir.mkdir(parents=True)
    pred_file = seq_dir / "predictions.txt"
    pred_file.write_text("10.0 20.0 50.0 40.0\n12.0 22.0 50.0 40.0\n")

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    loaded = reader.read_sequence("TrackerA", "seq_ws")
    assert loaded.shape == (2, 4)
    assert loaded[0, 0] == pytest.approx(10.0)


def test_parse_tab_delimiter(tmp_path):
    seq_dir = tmp_path / "TrackerB" / "seq_tab"
    seq_dir.mkdir(parents=True)
    (seq_dir / "predictions.txt").write_text("10.0\t20.0\t50.0\t40.0\n")

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    loaded = reader.read_sequence("TrackerB", "seq_tab")
    assert loaded.shape == (1, 4)


def test_parse_ignores_confidence_column(tmp_path):
    seq_dir = tmp_path / "TrackerC" / "seq_conf"
    seq_dir.mkdir(parents=True)
    (seq_dir / "predictions.txt").write_text("10.0,20.0,50.0,40.0,0.95\n")

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    loaded = reader.read_sequence("TrackerC", "seq_conf")
    assert loaded.shape == (1, 4)


# ---------------------------------------------------------------------------
# load_predictions_from_benchmark_result
# ---------------------------------------------------------------------------

def test_load_from_result_dict(benchmark_result_dict):
    loaded = load_predictions_from_benchmark_result(benchmark_result_dict)
    assert "seq_001" in loaded
    assert "seq_002" in loaded
    np.testing.assert_allclose(loaded["seq_001"], SAMPLE_PREDS, atol=1e-6)


def test_load_from_result_object():
    """Test loading from a live BenchmarkResult object (uses sequence_results attr)."""

    class FakeSeqResult:
        sequence_name = "fake_seq"
        predictions = SAMPLE_PREDS

    class FakeBenchmarkResult:
        sequence_results = [FakeSeqResult()]

    loaded = load_predictions_from_benchmark_result(FakeBenchmarkResult())
    assert "fake_seq" in loaded
    np.testing.assert_allclose(loaded["fake_seq"], SAMPLE_PREDS, atol=1e-6)


def test_load_from_result_object_skips_none_predictions():
    class FakeSeqResult:
        sequence_name = "no_pred_seq"
        predictions = None

    class FakeBenchmarkResult:
        sequence_results = [FakeSeqResult()]

    loaded = load_predictions_from_benchmark_result(FakeBenchmarkResult())
    assert len(loaded) == 0


# ---------------------------------------------------------------------------
# End-to-end integration with BenchmarkEngine
# ---------------------------------------------------------------------------

def test_full_pipeline_otb(tmp_path):
    """Run benchmark → export OTB predictions → load → check shapes match."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker

    dataset = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear", seed=0)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")

    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    writer.write_benchmark_result(result)

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.OTB)
    loaded = reader.read_tracker("MOSSE")

    assert len(loaded) == 2
    for seq_name, preds in loaded.items():
        assert preds.shape[1] == 4
        assert preds.shape[0] == 20  # num_frames predictions per sequence


def test_full_pipeline_json(tmp_path):
    """Run benchmark → export JSON → load via reader → verify prediction count."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.kcf import KCFTracker

    dataset = SyntheticDataset(num_sequences=2, num_frames=15, motion="circular", seed=1)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(KCFTracker(), dataset, dataset_name="Synthetic")

    writer = PredictionWriter(output_dir=str(tmp_path), fmt=PredictionFormat.JSON)
    writer.write_benchmark_result(result)

    reader = PredictionReader(input_dir=str(tmp_path), fmt=PredictionFormat.JSON)
    loaded = reader.read_tracker("KCF")

    assert len(loaded) == 2
    for seq_name, preds in loaded.items():
        assert preds.shape == (15, 4)
