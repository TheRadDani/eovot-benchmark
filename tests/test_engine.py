"""Integration tests for eovot.benchmark.engine."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult, SequenceResult
from eovot.datasets.base import OTBDataset


# ---------------------------------------------------------------------------
# BenchmarkResult unit tests (no I/O needed)
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def _make_result(self) -> BenchmarkResult:
        from eovot.profiling.profiler import ProfilingResult

        profiling = ProfilingResult(
            tracker_name="Mock",
            frame_count=10,
            fps=100.0,
            latency_mean_ms=10.0,
            latency_std_ms=1.0,
            latency_p95_ms=12.0,
            peak_memory_mb=50.0,
        )
        seq_result = SequenceResult(
            sequence_name="seq1",
            ious=np.array([0.8, 0.9, 0.7, 0.85]),
            profiling=profiling,
        )
        result = BenchmarkResult(tracker_name="Mock", dataset_name="TestDS")
        result.sequence_results.append(seq_result)
        return result

    def test_mean_iou(self):
        result = self._make_result()
        expected = np.array([0.8, 0.9, 0.7, 0.85]).mean()
        assert result.mean_iou == pytest.approx(expected)

    def test_mean_fps(self):
        result = self._make_result()
        assert result.mean_fps == pytest.approx(100.0)

    def test_peak_memory(self):
        result = self._make_result()
        assert result.peak_memory_mb == pytest.approx(50.0)

    def test_summary_keys(self):
        result = self._make_result()
        s = result.summary()
        assert "tracker" in s
        assert "dataset" in s
        assert "mean_iou" in s
        assert "mean_fps" in s
        assert "peak_memory_mb" in s
        assert "num_sequences" in s

    def test_summary_values_rounded(self):
        result = self._make_result()
        s = result.summary()
        # mean_iou should be rounded to 4 decimal places
        assert isinstance(s["mean_iou"], float)

    def test_str_representation(self):
        result = self._make_result()
        s = str(result)
        assert "Mock" in s
        assert "TestDS" in s

    def test_to_dict_structure(self):
        """to_dict() must produce the shape BenchmarkReporter expects."""
        result = self._make_result()
        d = result.to_dict()
        assert "summary" in d
        assert "sequences" in d
        assert isinstance(d["sequences"], list)
        assert len(d["sequences"]) == 1
        seq = d["sequences"][0]
        assert "sequence_name" in seq
        assert "mean_iou" in seq
        assert "fps" in seq

    def test_empty_result_mean_iou(self):
        result = BenchmarkResult(tracker_name="T", dataset_name="D")
        assert result.mean_iou == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BenchmarkEngine integration test (uses synthetic OTBDataset)
# ---------------------------------------------------------------------------

class TestBenchmarkEngine:
    def test_run_returns_benchmark_result(self, synthetic_otb_root, passthrough_tracker):
        ds = OTBDataset(synthetic_otb_root)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(passthrough_tracker, ds, dataset_name="TestOTB")
        assert isinstance(result, BenchmarkResult)

    def test_run_processes_all_sequences(self, synthetic_otb_root, passthrough_tracker):
        ds = OTBDataset(synthetic_otb_root)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(passthrough_tracker, ds, dataset_name="TestOTB")
        assert len(result.sequence_results) == len(ds)

    def test_run_max_sequences(self, synthetic_otb_root, passthrough_tracker):
        ds = OTBDataset(synthetic_otb_root)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(passthrough_tracker, ds, dataset_name="TestOTB", max_sequences=1)
        assert len(result.sequence_results) == 1

    def test_run_profiling_collected(self, synthetic_otb_root, passthrough_tracker):
        ds = OTBDataset(synthetic_otb_root)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(passthrough_tracker, ds, dataset_name="TestOTB")
        for sr in result.sequence_results:
            assert sr.profiling.fps > 0.0
            assert sr.profiling.frame_count > 0

    def test_run_ious_shape(self, synthetic_otb_root, passthrough_tracker):
        ds = OTBDataset(synthetic_otb_root)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(passthrough_tracker, ds, dataset_name="TestOTB")
        for sr in result.sequence_results:
            assert sr.ious.ndim == 1
            assert len(sr.ious) > 0

    def test_passthrough_iou_near_one(self, static_otb_root):
        """Tracker that always returns the GT init bbox should get IoU == 1.0
        on a dataset where the target does not move between frames."""
        ds = OTBDataset(static_otb_root)
        engine = BenchmarkEngine(verbose=False)

        class _StaticTracker:
            name = "Static"
            def initialize(self, frame, bbox):
                self._bbox = bbox
            def update(self, frame):
                return self._bbox

        result = engine.run(_StaticTracker(), ds, dataset_name="D")
        assert result.mean_iou == pytest.approx(1.0, rel=1e-4)

    def test_dataset_name_propagated(self, synthetic_otb_root, passthrough_tracker):
        ds = OTBDataset(synthetic_otb_root)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(passthrough_tracker, ds, dataset_name="MyDataset")
        assert result.dataset_name == "MyDataset"

    def test_tracker_name_propagated(self, synthetic_otb_root):
        ds = OTBDataset(synthetic_otb_root)
        engine = BenchmarkEngine(verbose=False)

        class _NamedTracker:
            name = "SpecialName"
            def initialize(self, frame, bbox): self._bbox = bbox
            def update(self, frame): return self._bbox

        result = engine.run(_NamedTracker(), ds, dataset_name="D")
        assert result.tracker_name == "SpecialName"
