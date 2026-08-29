"""Tests for DriftDetector — gradient-based tracker drift early warning."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pytest

from eovot.metrics.drift import DriftDetector, DriftResult
from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profiling(tracker: str = "T") -> ProfilingResult:
    return ProfilingResult(
        tracker_name=tracker,
        frame_count=50,
        fps=30.0,
        latency_mean_ms=33.0,
        latency_std_ms=1.0,
        latency_p95_ms=36.0,
        latency_p99_ms=38.0,
        latency_cv=0.03,
        peak_memory_mb=120.0,
    )


def _make_benchmark_result(
    tracker: str,
    iou_sequences: list[list[float]],
    dataset: str = "TestDS",
) -> BenchmarkResult:
    result = BenchmarkResult(tracker_name=tracker, dataset_name=dataset)
    for i, ious in enumerate(iou_sequences):
        result.sequence_results.append(
            SequenceResult(
                sequence_name=f"seq_{i:02d}",
                ious=np.array(ious, dtype=np.float64),
                profiling=_make_profiling(tracker),
            )
        )
    return result


# ---------------------------------------------------------------------------
# DriftDetector construction
# ---------------------------------------------------------------------------

class TestDriftDetectorInit:
    def test_default_params(self):
        d = DriftDetector()
        assert d.window_size == 10
        assert d.drift_slope_threshold == pytest.approx(-0.005)
        assert d.failure_threshold == pytest.approx(0.1)
        assert d.stride == 1

    def test_custom_params(self):
        d = DriftDetector(window_size=5, drift_slope_threshold=-0.01, stride=2)
        assert d.window_size == 5
        assert d.stride == 2

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            DriftDetector(window_size=1)

    def test_invalid_stride_raises(self):
        with pytest.raises(ValueError, match="stride"):
            DriftDetector(stride=0)


# ---------------------------------------------------------------------------
# _window_slope
# ---------------------------------------------------------------------------

class TestWindowSlope:
    def test_flat_sequence_gives_zero_slope(self):
        d = DriftDetector(window_size=5)
        ious = np.full(5, 0.6)
        assert d._window_slope(ious) == pytest.approx(0.0, abs=1e-10)

    def test_perfectly_declining_gives_negative_slope(self):
        d = DriftDetector(window_size=5)
        # IoU drops 0.1 per frame: [0.8, 0.7, 0.6, 0.5, 0.4]
        ious = np.array([0.8, 0.7, 0.6, 0.5, 0.4])
        slope = d._window_slope(ious)
        assert slope == pytest.approx(-0.1, abs=1e-6)

    def test_perfectly_rising_gives_positive_slope(self):
        d = DriftDetector(window_size=5)
        ious = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        slope = d._window_slope(ious)
        assert slope == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# is_drifting (online API)
# ---------------------------------------------------------------------------

class TestIsDrifting:
    def test_stable_tracker_not_drifting(self):
        d = DriftDetector(window_size=5, drift_slope_threshold=-0.005)
        ious = np.full(5, 0.7)
        assert not d.is_drifting(ious)

    def test_rapidly_declining_tracker_is_drifting(self):
        d = DriftDetector(window_size=5, drift_slope_threshold=-0.005)
        ious = np.array([0.8, 0.7, 0.6, 0.5, 0.4])  # slope = -0.1
        assert d.is_drifting(ious)

    def test_insufficient_data_returns_false(self):
        d = DriftDetector(window_size=10)
        assert not d.is_drifting(np.array([0.5, 0.4]))

    def test_uses_last_window_size_elements(self):
        d = DriftDetector(window_size=3, drift_slope_threshold=-0.005)
        # First elements stable, last 3 declining sharply
        ious = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.5, 0.3])
        assert d.is_drifting(ious)

    def test_online_deque_pattern(self):
        """Simulate the recommended deque usage."""
        d = DriftDetector(window_size=5, drift_slope_threshold=-0.005)
        iou_window: deque = deque(maxlen=d.window_size)

        # Feed a slowly declining sequence
        raw = [0.8] * 3 + [0.78, 0.76, 0.74, 0.72, 0.70]
        warnings = []
        for i, v in enumerate(raw):
            iou_window.append(v)
            if len(iou_window) == d.window_size:
                if d.is_drifting(np.array(iou_window)):
                    warnings.append(i)

        # The last few frames should have triggered a warning
        assert len(warnings) > 0


# ---------------------------------------------------------------------------
# analyze (offline full-sequence API)
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_stable_sequence_zero_drift(self):
        d = DriftDetector(window_size=5)
        ious = np.full(30, 0.7)
        result = d.analyze(ious, "MOSSE", "car1")
        assert result.drift_score == pytest.approx(0.0)
        assert result.drift_onset_frame is None
        assert result.num_drift_windows == 0

    def test_linearly_declining_sequence_detects_drift(self):
        d = DriftDetector(window_size=5, drift_slope_threshold=-0.005)
        # Steep decline: IoU drops ~0.05/frame
        ious = np.linspace(0.9, 0.1, 50)
        result = d.analyze(ious, "MOSSE", "car1")
        assert result.drift_score > 0.0
        assert result.drift_onset_frame is not None
        assert result.trend_slope < 0.0

    def test_drift_onset_frame_is_early(self):
        d = DriftDetector(window_size=5, drift_slope_threshold=-0.01)
        # Stable then sharp decline
        stable = np.full(10, 0.8)
        decline = np.linspace(0.8, 0.1, 30)
        ious = np.concatenate([stable, decline])
        result = d.analyze(ious)
        assert result.drift_onset_frame is not None
        # Onset should be within the declining region, not at frame 0
        assert result.drift_onset_frame >= 5

    def test_time_to_failure_computed_when_failure_follows_onset(self):
        d = DriftDetector(window_size=5, drift_slope_threshold=-0.005, failure_threshold=0.1)
        # Gradual decline ending below failure threshold
        ious = np.concatenate([np.full(10, 0.8), np.linspace(0.7, 0.05, 30)])
        result = d.analyze(ious)
        # A failure must have occurred after drift onset
        if result.drift_onset_frame is not None:
            # Check whether any IoU is below failure threshold
            below = np.any(ious < 0.1)
            if below:
                assert result.time_to_failure_frames is not None
                assert result.time_to_failure_frames >= 0

    def test_no_failure_gives_none_ttf(self):
        d = DriftDetector(window_size=5, failure_threshold=0.0)
        # No IoU ever reaches 0 exactly with strict <
        ious = np.linspace(0.8, 0.05, 50)
        result = d.analyze(ious)
        # With failure_threshold=0.0, IoU must be < 0 which never happens
        assert result.time_to_failure_frames is None

    def test_short_sequence_returns_empty_result(self):
        d = DriftDetector(window_size=15)
        ious = np.array([0.7, 0.6, 0.5])
        result = d.analyze(ious, "T", "s1")
        assert result.num_total_windows == 0
        assert result.drift_score == 0.0
        assert result.warning_frames == []

    def test_drift_score_in_valid_range(self):
        d = DriftDetector(window_size=5)
        ious = np.random.default_rng(42).uniform(0.0, 1.0, size=100)
        result = d.analyze(ious)
        assert 0.0 <= result.drift_score <= 1.0

    def test_result_str_is_informative(self):
        d = DriftDetector(window_size=5)
        ious = np.linspace(0.9, 0.2, 40)
        result = d.analyze(ious, "KCF", "seq_01")
        s = str(result)
        assert "KCF" in s
        assert "seq_01" in s
        assert "score=" in s

    def test_warning_frames_are_subset_of_total_frames(self):
        d = DriftDetector(window_size=5)
        ious = np.linspace(0.9, 0.1, 50)
        result = d.analyze(ious)
        n_frames = len(ious)
        for f in result.warning_frames:
            assert 0 <= f < n_frames

    def test_stride_reduces_window_count(self):
        d1 = DriftDetector(window_size=5, stride=1)
        d2 = DriftDetector(window_size=5, stride=5)
        ious = np.full(50, 0.7)
        r1 = d1.analyze(ious)
        r2 = d2.analyze(ious)
        assert r2.num_total_windows < r1.num_total_windows

    def test_window_slopes_length_matches_total_windows(self):
        d = DriftDetector(window_size=5, stride=1)
        ious = np.full(20, 0.6)
        result = d.analyze(ious)
        assert len(result.window_slopes) == result.num_total_windows


# ---------------------------------------------------------------------------
# analyze_benchmark
# ---------------------------------------------------------------------------

class TestAnalyzeBenchmark:
    def test_aggregate_keys_present(self):
        br = _make_benchmark_result(
            "MOSSE",
            [[0.8] * 30, [0.7] * 30],
        )
        d = DriftDetector(window_size=5)
        out = d.analyze_benchmark(br)
        agg = out["aggregate"]
        assert "tracker_name" in agg
        assert "num_sequences" in agg
        assert "mean_drift_score" in agg
        assert "fraction_sequences_with_drift" in agg
        assert "mean_trend_slope" in agg

    def test_per_sequence_has_entry_for_each_sequence(self):
        br = _make_benchmark_result(
            "KCF",
            [[0.6] * 20, [0.5] * 20, [0.4] * 20],
        )
        d = DriftDetector(window_size=5)
        out = d.analyze_benchmark(br)
        assert len(out["per_sequence"]) == 3

    def test_stable_tracker_gives_zero_mean_drift(self):
        br = _make_benchmark_result(
            "CSRT",
            [np.full(50, 0.75).tolist() for _ in range(4)],
        )
        d = DriftDetector(window_size=5)
        out = d.analyze_benchmark(br)
        assert out["aggregate"]["mean_drift_score"] == pytest.approx(0.0)

    def test_declining_tracker_gives_nonzero_drift_fraction(self):
        br = _make_benchmark_result(
            "MOSSE",
            [np.linspace(0.9, 0.05, 50).tolist() for _ in range(3)],
        )
        d = DriftDetector(window_size=5)
        out = d.analyze_benchmark(br)
        assert out["aggregate"]["fraction_sequences_with_drift"] > 0.0


# ---------------------------------------------------------------------------
# to_markdown_table
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def test_table_has_header(self):
        d = DriftDetector(window_size=5)
        ious = np.linspace(0.8, 0.3, 30)
        results = {
            "seq1": d.analyze(ious, "MOSSE", "seq1"),
            "seq2": d.analyze(ious, "MOSSE", "seq2"),
        }
        table = d.to_markdown_table(results)
        assert "Sequence" in table
        assert "Drift Score" in table
        assert "seq1" in table
        assert "seq2" in table

    def test_table_sorted_by_drift_score_descending(self):
        d = DriftDetector(window_size=5)
        results = {
            "stable": d.analyze(np.full(30, 0.7), "T", "stable"),
            "drifting": d.analyze(np.linspace(0.9, 0.05, 30), "T", "drifting"),
        }
        table = d.to_markdown_table(results)
        lines = table.strip().split("\n")
        # First data row (after header and sep) should be "drifting"
        assert "drifting" in lines[2]
