"""Tests for cold-start profiling: Profiler init tracking and ColdStartAnalyzer."""

import pytest
import numpy as np

from eovot.profiling.profiler import Profiler, ProfilingResult
from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.analysis.cold_start import ColdStartAnalyzer, ColdStartReport
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Profiler: init timing
# ---------------------------------------------------------------------------

class TestProfilerInitTiming:
    def test_init_latency_recorded(self):
        p = Profiler()
        p.start_init()
        p.end_init()
        p.start_frame()
        p.end_frame()
        result = p.summary("test")
        assert result.init_latency_ms >= 0.0

    def test_init_latency_positive(self):
        import time
        p = Profiler()
        p.start_init()
        time.sleep(0.01)  # 10 ms sleep
        p.end_init()
        p.start_frame()
        p.end_frame()
        result = p.summary("test")
        assert result.init_latency_ms >= 5.0  # conservative lower bound

    def test_init_latency_defaults_zero(self):
        p = Profiler()
        p.start_frame()
        p.end_frame()
        result = p.summary("test")
        assert result.init_latency_ms == 0.0

    def test_end_init_without_start_raises(self):
        p = Profiler()
        with pytest.raises(RuntimeError, match="end_init"):
            p.end_init()

    def test_reset_clears_init(self):
        p = Profiler()
        p.start_init()
        p.end_init()
        p.start_frame()
        p.end_frame()
        p.reset()
        # After reset, start_frame/end_frame must be called again.
        p.start_init()
        p.end_init()
        p.start_frame()
        p.end_frame()
        result = p.summary("test")
        assert result.init_latency_ms >= 0.0

    def test_overhead_ratio(self):
        p = Profiler()
        p.start_init()
        p.end_init()
        p.start_frame()
        p.end_frame()
        result = p.summary("test")
        # When init is 0 and update latency > 0, ratio is 0.
        assert result.init_overhead_ratio >= 0.0

    def test_overhead_ratio_zero_update(self):
        """When mean update latency is 0, overhead ratio should be 0 (no ZeroDivisionError)."""
        r = ProfilingResult(
            tracker_name="test",
            frame_count=1,
            fps=0.0,
            latency_mean_ms=0.0,
            latency_std_ms=0.0,
            latency_p95_ms=0.0,
            peak_memory_mb=10.0,
            init_latency_ms=5.0,
        )
        assert r.init_overhead_ratio == 0.0


# ---------------------------------------------------------------------------
# BenchmarkEngine: init time recorded end-to-end
# ---------------------------------------------------------------------------

class TestEngineInitProfiling:
    def _make_dataset(self, n_seq=3, n_frames=20):
        return SyntheticDataset(num_sequences=n_seq, num_frames=n_frames, motion="linear")

    def test_init_latency_in_result(self):
        ds = self._make_dataset()
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synth")
        for sr in result.sequence_results:
            assert sr.profiling.init_latency_ms >= 0.0

    def test_init_latency_nonzero(self):
        """All sequences should record a positive init time (tracker does work on init)."""
        ds = self._make_dataset()
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synth")
        # At least one sequence should have a positive init time.
        init_times = [sr.profiling.init_latency_ms for sr in result.sequence_results]
        assert any(t > 0.0 for t in init_times)

    def test_init_latency_persisted(self, tmp_path):
        ds = self._make_dataset(n_seq=2, n_frames=10)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synth")
        p = result.save(tmp_path / "result.json")
        loaded = BenchmarkResult.load(p)
        for orig, reloaded in zip(result.sequence_results, loaded.sequence_results):
            assert abs(orig.profiling.init_latency_ms - reloaded.profiling.init_latency_ms) < 0.001


# ---------------------------------------------------------------------------
# ColdStartAnalyzer
# ---------------------------------------------------------------------------

class TestColdStartAnalyzer:
    def _make_result(self, n_seq=4, n_frames=30):
        ds = SyntheticDataset(num_sequences=n_seq, num_frames=n_frames)
        engine = BenchmarkEngine(verbose=False)
        return engine.run(MOSSETracker(), ds, dataset_name="Synthetic")

    def test_analyze_returns_report(self):
        result = self._make_result()
        report = ColdStartAnalyzer().analyze(result)
        assert isinstance(report, ColdStartReport)

    def test_report_fields(self):
        result = self._make_result()
        report = ColdStartAnalyzer().analyze(result)
        assert report.tracker_name == result.tracker_name
        assert report.dataset_name == result.dataset_name
        assert report.n_sequences == len(result.sequence_results)
        assert report.mean_init_ms >= 0.0
        assert report.max_init_ms >= report.min_init_ms
        assert report.mean_update_ms > 0.0
        assert report.init_overhead_ratio >= 0.0

    def test_per_sequence_populated(self):
        result = self._make_result()
        report = ColdStartAnalyzer().analyze(result)
        assert len(report.per_sequence) == len(result.sequence_results)
        for seq_name, d in report.per_sequence.items():
            assert "init_latency_ms" in d
            assert "mean_update_ms" in d
            assert "overhead_ratio" in d

    def test_to_markdown(self):
        result = self._make_result()
        report = ColdStartAnalyzer().analyze(result)
        md = report.to_markdown()
        assert "Cold-Start Analysis" in md
        assert report.tracker_name in md
        for seq_name in report.per_sequence:
            assert seq_name in md

    def test_to_dict_serialisable(self):
        import json
        result = self._make_result()
        report = ColdStartAnalyzer().analyze(result)
        d = report.to_dict()
        json.dumps(d)  # must not raise

    def test_compare_multiple(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=15)
        engine = BenchmarkEngine(verbose=False)
        from eovot.trackers.kcf import KCFTracker
        r1 = engine.run(MOSSETracker(), ds, dataset_name="Synth")
        r2 = engine.run(KCFTracker(), ds, dataset_name="Synth")
        analyzer = ColdStartAnalyzer()
        reports = analyzer.compare([r1, r2])
        assert len(reports) == 2
        md = analyzer.to_comparison_markdown(reports)
        assert "Cold-Start Comparison" in md

    def test_no_init_data_raises(self):
        """Raise ValueError when result has no init-latency data."""
        from unittest.mock import MagicMock
        from eovot.profiling.profiler import ProfilingResult
        from eovot.benchmark.engine import SequenceResult

        mock_prof = ProfilingResult(
            tracker_name="mock",
            frame_count=10,
            fps=30.0,
            latency_mean_ms=33.0,
            latency_std_ms=1.0,
            latency_p95_ms=35.0,
            peak_memory_mb=50.0,
            init_latency_ms=0.0,  # not recorded
        )
        sr = SequenceResult(
            sequence_name="seq0",
            ious=np.array([0.8, 0.7, 0.6]),
            profiling=mock_prof,
        )
        result = BenchmarkResult(tracker_name="mock", dataset_name="mock")
        result.sequence_results = [sr]
        with pytest.raises(ValueError, match="init"):
            ColdStartAnalyzer().analyze(result)
