"""Integration tests for eovot.benchmark.engine."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult, SequenceResult
from eovot.datasets.base import OTBDataset, Sequence
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_fake_frame(path: str, h: int = 64, w: int = 64) -> None:
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(path, img)


def _make_otb_dataset(root: str, n_seqs: int = 2, n_frames: int = 8) -> OTBDataset:
    """Build a minimal OTB dataset in *root* and return an OTBDataset."""
    for s in range(n_seqs):
        seq_dir = os.path.join(root, f"seq_{s:02d}")
        img_dir = os.path.join(seq_dir, "img")
        os.makedirs(img_dir)
        for i in range(1, n_frames + 1):
            _make_fake_frame(os.path.join(img_dir, f"{i:04d}.jpg"))
        gt = np.ones((n_frames, 4)) * [5, 5, 20, 20]
        np.savetxt(os.path.join(seq_dir, "groundtruth_rect.txt"), gt, delimiter=",", fmt="%.1f")
    return OTBDataset(root)


class _ConstantTracker(BaseTracker):
    """Dummy tracker that always returns the initialisation bbox."""

    def __init__(self):
        self._bbox = None

    @property
    def name(self) -> str:
        return "ConstantTracker"

    def initialize(self, frame: np.ndarray, bbox) -> None:
        self._bbox = tuple(bbox)

    def update(self, frame: np.ndarray):
        return self._bbox


# ---------------------------------------------------------------------------
# BenchmarkResult tests
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def _make_result(self, n_seqs: int = 3) -> BenchmarkResult:
        from eovot.profiling.profiler import Profiler
        result = BenchmarkResult(tracker_name="TestTracker", dataset_name="TestDataset")
        profiler = Profiler()
        for i in range(n_seqs):
            profiler.reset()
            profiler.start_frame()
            profiler.end_frame()
            seq_result = SequenceResult(
                sequence_name=f"seq_{i}",
                ious=np.array([0.5, 0.6, 0.7]),
                profiling=profiler.summary("TestTracker"),
            )
            result.sequence_results.append(seq_result)
        return result

    def test_mean_iou(self):
        result = self._make_result()
        expected = np.mean([0.5, 0.6, 0.7])
        assert result.mean_iou == pytest.approx(expected)

    def test_mean_fps_positive(self):
        result = self._make_result()
        assert result.mean_fps > 0.0

    def test_peak_memory_positive(self):
        result = self._make_result()
        assert result.peak_memory_mb > 0.0

    def test_summary_keys(self):
        result = self._make_result()
        s = result.summary()
        assert "tracker" in s
        assert "dataset" in s
        assert "mean_iou" in s
        assert "mean_fps" in s
        assert "num_sequences" in s

    def test_to_dict_structure(self):
        result = self._make_result(n_seqs=2)
        d = result.to_dict()
        assert "summary" in d
        assert "sequences" in d
        assert len(d["sequences"]) == 2

    def test_to_dict_summary_fields(self):
        result = self._make_result()
        s = result.to_dict()["summary"]
        assert "tracker_name" in s
        assert "dataset_name" in s
        assert "mean_iou" in s
        assert "mean_fps" in s
        assert "mean_latency_ms" in s
        assert "peak_memory_mb" in s

    def test_to_dict_sequence_fields(self):
        result = self._make_result()
        seq = result.to_dict()["sequences"][0]
        assert "sequence_name" in seq
        assert "mean_iou" in seq
        assert "fps" in seq
        assert "mean_latency_ms" in seq

    def test_str_representation(self):
        result = self._make_result()
        text = str(result)
        assert "TestTracker" in text
        assert "TestDataset" in text


# ---------------------------------------------------------------------------
# BenchmarkEngine integration tests
# ---------------------------------------------------------------------------

class TestBenchmarkEngine:
    def test_run_returns_benchmark_result(self, tmp_path):
        dataset = _make_otb_dataset(str(tmp_path))
        tracker = _ConstantTracker()
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="FakeOTB")
        assert isinstance(result, BenchmarkResult)
        assert result.tracker_name == "ConstantTracker"
        assert result.dataset_name == "FakeOTB"

    def test_run_all_sequences_evaluated(self, tmp_path):
        dataset = _make_otb_dataset(str(tmp_path), n_seqs=3)
        tracker = _ConstantTracker()
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="D")
        assert len(result.sequence_results) == 3

    def test_run_max_sequences(self, tmp_path):
        dataset = _make_otb_dataset(str(tmp_path), n_seqs=5)
        tracker = _ConstantTracker()
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="D", max_sequences=2)
        assert len(result.sequence_results) == 2

    def test_constant_tracker_iou_perfect(self, tmp_path):
        """A tracker that always returns the init bbox gets IoU=1 on the first GT."""
        dataset = _make_otb_dataset(str(tmp_path), n_seqs=1, n_frames=4)
        tracker = _ConstantTracker()
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="D")
        # Frame 0: prediction set to GT[0]=[5,5,20,20], which equals GT[0] → IoU=1
        assert result.sequence_results[0].ious[0] == pytest.approx(1.0)

    def test_run_profiling_collected(self, tmp_path):
        dataset = _make_otb_dataset(str(tmp_path))
        tracker = _ConstantTracker()
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="D")
        for sr in result.sequence_results:
            assert sr.profiling.fps > 0.0
            assert sr.profiling.frame_count > 0
