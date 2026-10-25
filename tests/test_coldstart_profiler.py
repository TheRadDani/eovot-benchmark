"""Tests for ColdStartProfiler (warm-up latency analysis)."""

from __future__ import annotations

import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.profiling.warmup import ColdStartProfiler, ColdStartResult
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kalman import KalmanFilterTracker
from eovot.trackers.kcf import KCFTracker


def make_sequence(num_frames=60, seed=0):
    dataset = SyntheticDataset(num_sequences=1, num_frames=num_frames, seed=seed)
    return dataset[0]


class TestColdStartResult:
    def test_str_repr(self):
        result = ColdStartResult(
            tracker_name="MOSSE",
            sequence_name="seq-0",
            init_latency_ms=1.23,
            per_frame_latencies_ms=[1.0, 1.1, 0.9],
            warmup_frames=2,
            warmup_mean_ms=1.05,
            steady_state_ms=0.95,
            startup_overhead_ms=1.43,
        )
        s = str(result)
        assert "MOSSE" in s
        assert "init=" in s
        assert "warmup=" in s

    def test_to_dict_roundtrip(self):
        result = ColdStartResult(
            tracker_name="KCF",
            sequence_name="test-seq",
            init_latency_ms=2.5,
            warmup_frames=5,
            warmup_mean_ms=3.1,
            steady_state_ms=2.2,
            startup_overhead_ms=6.5,
        )
        d = result.to_dict()
        assert d["tracker_name"] == "KCF"
        assert d["warmup_frames"] == 5
        assert "init_latency_ms" in d


class TestColdStartProfiler:
    def test_basic_profile_mosse(self):
        seq = make_sequence(num_frames=50)
        profiler = ColdStartProfiler(warmup_window=10)
        result = profiler.profile(MOSSETracker(), seq, max_frames=40)

        assert isinstance(result, ColdStartResult)
        assert result.init_latency_ms >= 0.0
        assert result.steady_state_ms >= 0.0
        assert result.tracker_name == "MOSSE"
        assert result.sequence_name == seq.name

    def test_per_frame_latencies_length(self):
        seq = make_sequence(num_frames=30)
        profiler = ColdStartProfiler()
        result = profiler.profile(MOSSETracker(), seq, max_frames=20)
        # Should have max_frames - 1 latency measurements (frame 0 is init).
        assert len(result.per_frame_latencies_ms) == 19

    def test_warmup_frames_within_bounds(self):
        seq = make_sequence(num_frames=80)
        profiler = ColdStartProfiler(warmup_window=15)
        result = profiler.profile(KalmanFilterTracker(), seq, max_frames=60)
        assert 0 <= result.warmup_frames < 60

    def test_startup_overhead_non_negative(self):
        seq = make_sequence(num_frames=50)
        profiler = ColdStartProfiler()
        result = profiler.profile(KCFTracker(), seq)
        assert result.startup_overhead_ms >= 0.0

    def test_profile_trackers_multiple(self):
        seq = make_sequence(num_frames=50)
        profiler = ColdStartProfiler(warmup_window=10)
        trackers = [MOSSETracker(), KalmanFilterTracker()]
        results = profiler.profile_trackers(trackers, seq, max_frames=30)

        assert len(results) == 2
        assert results[0].tracker_name == "MOSSE"
        assert results[1].tracker_name == "KalmanFilter"

    def test_to_markdown_table(self):
        seq = make_sequence(num_frames=40)
        profiler = ColdStartProfiler(warmup_window=10)
        trackers = [MOSSETracker(), KalmanFilterTracker()]
        results = profiler.profile_trackers(trackers, seq, max_frames=25)
        table = profiler.to_markdown_table(results)

        assert "| Tracker |" in table
        assert "MOSSE" in table
        assert "KalmanFilter" in table

    def test_short_sequence_raises(self):
        """A sequence with only 1 frame should raise ValueError."""
        dataset = SyntheticDataset(num_sequences=1, num_frames=1, seed=0)
        seq = dataset[0]
        profiler = ColdStartProfiler()
        with pytest.raises(ValueError, match="at least 2 frames"):
            profiler.profile(MOSSETracker(), seq)

    def test_invalid_warmup_window(self):
        with pytest.raises(ValueError):
            ColdStartProfiler(warmup_window=0)

    def test_invalid_stability_threshold(self):
        with pytest.raises(ValueError):
            ColdStartProfiler(stability_threshold=1.5)

    def test_init_latency_measured_separately(self):
        """init_latency_ms should be the cost of initialize(), not update()."""
        seq = make_sequence(num_frames=30)
        profiler = ColdStartProfiler()
        result = profiler.profile(MOSSETracker(), seq, max_frames=20)
        # Per-frame latencies should NOT include the init call.
        assert len(result.per_frame_latencies_ms) == 19
        assert result.init_latency_ms > 0.0
