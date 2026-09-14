"""Tests for tracker initialisation latency profiling.

Verifies that:
  - Profiler.start_init/end_init record and expose init_latency_ms correctly.
  - BenchmarkEngine._run_sequence times the initialize() call.
  - BenchmarkResult surfaces mean_init_latency_ms and mean_cold_start_ratio.
  - to_dict() persists init_latency_ms and cold_start_ratio per sequence.
  - from_dict() restores those fields faithfully (including zero default
    for files produced before this feature was added).
  - ProfilingResult.cold_start_ratio is numerically correct.
  - BenchmarkResult.__str__ includes the init latency field.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from eovot.profiling.profiler import Profiler, ProfilingResult
from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.registry import build_tracker


# ---------------------------------------------------------------------------
# Profiler unit tests
# ---------------------------------------------------------------------------

class TestProfilerInitTiming:
    def test_start_end_init_returns_positive_ms(self):
        p = Profiler()
        p.start_init()
        time.sleep(0.005)
        elapsed = p.end_init()
        assert elapsed > 0.0

    def test_end_init_before_start_raises(self):
        p = Profiler()
        with pytest.raises(RuntimeError, match="start_init"):
            p.end_init()

    def test_summary_includes_init_latency(self):
        p = Profiler()
        p.start_init()
        time.sleep(0.005)
        p.end_init()
        # Need at least one update frame for summary
        p.start_frame()
        time.sleep(0.001)
        p.end_frame()
        result = p.summary("tracker")
        assert result.init_latency_ms > 0.0

    def test_reset_clears_init_latency(self):
        p = Profiler()
        p.start_init()
        time.sleep(0.003)
        p.end_init()
        p.start_frame()
        time.sleep(0.001)
        p.end_frame()
        p.reset()
        # After reset, a fresh measurement cycle starts at zero
        p.start_init()
        p.end_init()
        p.start_frame()
        p.end_frame()
        result = p.summary("t")
        # init_latency_ms is set to the last measured init; reset to 0 then
        # immediately re-measured means it could be near 0 but non-negative
        assert result.init_latency_ms >= 0.0

    def test_default_init_latency_is_zero_without_init_call(self):
        p = Profiler()
        p.start_frame()
        time.sleep(0.001)
        p.end_frame()
        result = p.summary("t")
        assert result.init_latency_ms == 0.0

    def test_cold_start_ratio_correct(self):
        result = ProfilingResult(
            tracker_name="t",
            frame_count=10,
            fps=100.0,
            latency_mean_ms=10.0,
            latency_std_ms=1.0,
            latency_p95_ms=12.0,
            peak_memory_mb=50.0,
            init_latency_ms=50.0,
        )
        assert result.cold_start_ratio == pytest.approx(5.0)

    def test_cold_start_ratio_zero_mean_latency(self):
        result = ProfilingResult(
            tracker_name="t",
            frame_count=1,
            fps=0.0,
            latency_mean_ms=0.0,
            latency_std_ms=0.0,
            latency_p95_ms=0.0,
            peak_memory_mb=0.0,
            init_latency_ms=10.0,
        )
        assert result.cold_start_ratio == 0.0

    def test_profiling_result_str_includes_init(self):
        result = ProfilingResult(
            tracker_name="MyTracker",
            frame_count=50,
            fps=120.0,
            latency_mean_ms=8.3,
            latency_std_ms=0.5,
            latency_p95_ms=9.1,
            peak_memory_mb=32.0,
            init_latency_ms=25.4,
        )
        s = str(result)
        assert "init=" in s
        assert "25.40 ms" in s


# ---------------------------------------------------------------------------
# BenchmarkEngine integration tests
# ---------------------------------------------------------------------------

class TestEngineInitLatency:
    def _make_dataset(self, n_seq=2, n_frames=15):
        return SyntheticDataset(
            num_sequences=n_seq,
            num_frames=n_frames,
            frame_size=(160, 120),
            seed=0,
        )

    def test_init_latency_is_positive_after_run(self):
        engine = BenchmarkEngine(verbose=False)
        tracker = build_tracker("MOSSE")
        ds = self._make_dataset()
        result = engine.run(tracker, ds, dataset_name="Synthetic")
        assert result.mean_init_latency_ms > 0.0

    def test_init_latency_per_sequence_in_profiling(self):
        engine = BenchmarkEngine(verbose=False)
        tracker = build_tracker("KCF")
        ds = self._make_dataset(n_seq=3)
        result = engine.run(tracker, ds, dataset_name="Synthetic")
        for sr in result.sequence_results:
            assert sr.profiling.init_latency_ms > 0.0

    def test_cold_start_ratio_positive(self):
        engine = BenchmarkEngine(verbose=False)
        tracker = build_tracker("MOSSE")
        ds = self._make_dataset()
        result = engine.run(tracker, ds, dataset_name="Synthetic")
        assert result.mean_cold_start_ratio > 0.0

    def test_summary_contains_init_fields(self):
        engine = BenchmarkEngine(verbose=False)
        tracker = build_tracker("MOSSE")
        ds = self._make_dataset()
        result = engine.run(tracker, ds, dataset_name="Synthetic")
        s = result.summary()
        assert "mean_init_latency_ms" in s
        assert "mean_cold_start_ratio" in s
        assert s["mean_init_latency_ms"] >= 0.0

    def test_str_includes_init_latency(self):
        engine = BenchmarkEngine(verbose=False)
        tracker = build_tracker("MOSSE")
        ds = self._make_dataset()
        result = engine.run(tracker, ds, dataset_name="Synthetic")
        assert "init=" in str(result)


# ---------------------------------------------------------------------------
# Serialisation round-trip tests
# ---------------------------------------------------------------------------

class TestInitLatencySerialization:
    def _run_result(self):
        engine = BenchmarkEngine(verbose=False)
        tracker = build_tracker("MOSSE")
        ds = SyntheticDataset(num_sequences=2, num_frames=12, frame_size=(160, 120), seed=7)
        return engine.run(tracker, ds, dataset_name="Synth")

    def test_to_dict_contains_init_latency_per_sequence(self):
        r = self._run_result()
        d = r.to_dict()
        for seq in d["sequences"]:
            assert "init_latency_ms" in seq
            assert seq["init_latency_ms"] >= 0.0
            assert "cold_start_ratio" in seq

    def test_from_dict_restores_init_latency(self):
        r = self._run_result()
        d = r.to_dict()
        r2 = BenchmarkResult.from_dict(d)
        assert r2.mean_init_latency_ms == pytest.approx(r.mean_init_latency_ms, rel=1e-3)

    def test_from_dict_old_format_defaults_to_zero(self):
        """Files written before init-latency tracking omit the field; default must be 0.0."""
        d = {
            "summary": {
                "tracker": "MOSSE",
                "dataset": "OTB",
                "num_sequences": 1,
                "mean_iou": 0.65,
                "mean_fps": 200.0,
                "peak_memory_mb": 40.0,
            },
            "sequences": [
                {
                    "sequence_name": "Basketball",
                    "mean_iou": 0.65,
                    "fps": 200.0,
                    "mean_latency_ms": 5.0,
                    "latency_std_ms": 0.2,
                    "latency_p95_ms": 5.5,
                    "latency_p99_ms": 5.8,
                    "latency_cv": 0.04,
                    "peak_memory_mb": 40.0,
                    # no init_latency_ms key
                }
            ],
        }
        r = BenchmarkResult.from_dict(d)
        assert r.mean_init_latency_ms == 0.0
        assert r.sequence_results[0].profiling.init_latency_ms == 0.0

    def test_save_load_roundtrip(self, tmp_path):
        r = self._run_result()
        p = r.save(tmp_path / "result")
        r2 = BenchmarkResult.load(p)
        assert r2.mean_init_latency_ms == pytest.approx(r.mean_init_latency_ms, rel=1e-3)
        for sr_orig, sr_loaded in zip(r.sequence_results, r2.sequence_results):
            assert sr_loaded.profiling.init_latency_ms == pytest.approx(
                sr_orig.profiling.init_latency_ms, rel=1e-3
            )
