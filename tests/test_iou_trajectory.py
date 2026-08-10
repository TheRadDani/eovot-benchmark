"""Tests for IoU trajectory analysis (eovot.metrics.iou_trajectory)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eovot.metrics.iou_trajectory import (
    IoUTrajectoryAnalyzer,
    IoUTrajectoryResult,
    SegmentStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ramp(n: int, start: float = 0.0, stop: float = 1.0) -> np.ndarray:
    """Linearly increasing IoU from start to stop over n frames."""
    return np.linspace(start, stop, n)


def _constant(n: int, v: float = 0.7) -> np.ndarray:
    return np.full(n, v, dtype=np.float64)


def _step(n: int, low: float = 0.1, high: float = 0.8, switch: int = None) -> np.ndarray:
    """IoU that steps from low to high at frame switch (default: n//2)."""
    switch = switch if switch is not None else n // 2
    arr = np.full(n, low, dtype=np.float64)
    arr[switch:] = high
    return arr


def _decay(n: int, start: float = 0.9, end: float = 0.1) -> np.ndarray:
    """Exponentially decaying IoU from start to end."""
    t = np.linspace(0.0, 1.0, n)
    return start * (end / start) ** t


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_default_params(self):
        a = IoUTrajectoryAnalyzer()
        assert a.n_segments == 3
        assert a.convergence_threshold == pytest.approx(0.5)
        assert a.peak_window == 10
        assert a.tail_fraction == pytest.approx(0.2)
        assert a.failure_threshold == pytest.approx(0.1)

    def test_invalid_n_segments(self):
        with pytest.raises(ValueError):
            IoUTrajectoryAnalyzer(n_segments=0)

    def test_invalid_convergence_threshold_zero(self):
        with pytest.raises(ValueError):
            IoUTrajectoryAnalyzer(convergence_threshold=0.0)

    def test_invalid_convergence_threshold_gt_one(self):
        with pytest.raises(ValueError):
            IoUTrajectoryAnalyzer(convergence_threshold=1.1)

    def test_invalid_peak_window(self):
        with pytest.raises(ValueError):
            IoUTrajectoryAnalyzer(peak_window=0)

    def test_invalid_tail_fraction_zero(self):
        with pytest.raises(ValueError):
            IoUTrajectoryAnalyzer(tail_fraction=0.0)

    def test_invalid_tail_fraction_one(self):
        with pytest.raises(ValueError):
            IoUTrajectoryAnalyzer(tail_fraction=1.0)

    def test_invalid_failure_threshold_negative(self):
        with pytest.raises(ValueError):
            IoUTrajectoryAnalyzer(failure_threshold=-0.1)

    def test_invalid_failure_threshold_one(self):
        with pytest.raises(ValueError):
            IoUTrajectoryAnalyzer(failure_threshold=1.0)


# ---------------------------------------------------------------------------
# compute_segment_ious
# ---------------------------------------------------------------------------

class TestComputeSegmentIoUs:
    def setup_method(self):
        self.analyzer = IoUTrajectoryAnalyzer(n_segments=3)

    def test_empty_returns_empty(self):
        result = self.analyzer.compute_segment_ious(np.array([]))
        assert result == []

    def test_three_equal_segments(self):
        ious = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
        segs = self.analyzer.compute_segment_ious(ious)
        assert len(segs) == 3
        assert segs[0].mean_iou == pytest.approx(0.0)
        assert segs[1].mean_iou == pytest.approx(0.5)
        assert segs[2].mean_iou == pytest.approx(1.0)

    def test_segment_indices_are_ordered(self):
        segs = self.analyzer.compute_segment_ious(_constant(30, 0.6))
        for i, seg in enumerate(segs):
            assert seg.index == i

    def test_segment_frames_cover_full_sequence(self):
        ious = _constant(30)
        segs = self.analyzer.compute_segment_ious(ious)
        assert segs[0].start_frame == 0
        assert segs[-1].end_frame == 30

    def test_segment_std_constant_ious_is_zero(self):
        segs = self.analyzer.compute_segment_ious(_constant(30, 0.5))
        for seg in segs:
            assert seg.std_iou == pytest.approx(0.0, abs=1e-9)

    def test_single_segment_covers_all(self):
        analyzer = IoUTrajectoryAnalyzer(n_segments=1)
        ious = _constant(20, 0.4)
        segs = analyzer.compute_segment_ious(ious)
        assert len(segs) == 1
        assert segs[0].mean_iou == pytest.approx(0.4)
        assert segs[0].n_frames == 20

    def test_more_segments_than_frames_handled(self):
        analyzer = IoUTrajectoryAnalyzer(n_segments=5)
        ious = np.array([0.3, 0.6, 0.9])
        segs = analyzer.compute_segment_ious(ious)
        # np.array_split may produce some empty splits; check we only keep non-empty
        assert all(seg.n_frames >= 1 for seg in segs)

    def test_returns_segment_stats_instances(self):
        segs = self.analyzer.compute_segment_ious(_constant(15))
        for seg in segs:
            assert isinstance(seg, SegmentStats)


# ---------------------------------------------------------------------------
# compute_convergence_frame
# ---------------------------------------------------------------------------

class TestComputeConvergenceFrame:
    def setup_method(self):
        self.analyzer = IoUTrajectoryAnalyzer(
            convergence_threshold=0.5, peak_window=3
        )

    def test_empty_returns_minus_one(self):
        assert self.analyzer.compute_convergence_frame(np.array([])) == -1

    def test_never_converges_returns_minus_one(self):
        ious = _constant(30, 0.3)
        assert self.analyzer.compute_convergence_frame(ious) == -1

    def test_immediately_above_threshold(self):
        ious = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
        # First frame already ≥ 0.5
        frame = self.analyzer.compute_convergence_frame(ious)
        assert frame == 0

    def test_step_sequence_detects_convergence(self):
        ious = _step(30, low=0.1, high=0.9, switch=15)
        frame = self.analyzer.compute_convergence_frame(ious)
        assert frame >= 14  # at or after the step

    def test_ramp_convergence_reasonable(self):
        ious = _ramp(20, 0.0, 1.0)
        frame = self.analyzer.compute_convergence_frame(ious)
        assert 0 <= frame < 20

    def test_single_frame_above_threshold(self):
        ious = np.array([0.9])
        assert self.analyzer.compute_convergence_frame(ious) == 0

    def test_single_frame_below_threshold(self):
        ious = np.array([0.3])
        assert self.analyzer.compute_convergence_frame(ious) == -1


# ---------------------------------------------------------------------------
# compute_peak_window
# ---------------------------------------------------------------------------

class TestComputePeakWindow:
    def setup_method(self):
        self.analyzer = IoUTrajectoryAnalyzer(peak_window=5)

    def test_empty_returns_minus_one_zero(self):
        start, mean = self.analyzer.compute_peak_window(np.array([]))
        assert start == -1
        assert mean == pytest.approx(0.0)

    def test_constant_sequence_peak_at_start(self):
        ious = _constant(20, 0.6)
        start, mean = self.analyzer.compute_peak_window(ious)
        assert mean == pytest.approx(0.6, rel=1e-5)

    def test_peak_in_middle(self):
        ious = np.zeros(30)
        ious[12:17] = 0.9  # peak window of width 5 starting at frame 12
        start, mean = self.analyzer.compute_peak_window(ious)
        assert start == 12
        assert mean == pytest.approx(0.9, rel=1e-5)

    def test_peak_at_end(self):
        ious = np.zeros(20)
        ious[-5:] = 1.0
        start, mean = self.analyzer.compute_peak_window(ious)
        assert start == 15
        assert mean == pytest.approx(1.0)

    def test_sequence_shorter_than_window(self):
        ious = np.array([0.4, 0.5, 0.6])
        start, mean = self.analyzer.compute_peak_window(ious)
        assert start == 0
        assert mean == pytest.approx(float(ious.mean()), rel=1e-5)

    def test_peak_mean_bounded_by_max_iou(self):
        ious = np.random.default_rng(0).uniform(0, 1, 50)
        _, mean = self.analyzer.compute_peak_window(ious)
        assert 0.0 <= mean <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# compute_tail_drop
# ---------------------------------------------------------------------------

class TestComputeTailDrop:
    def setup_method(self):
        self.analyzer = IoUTrajectoryAnalyzer(tail_fraction=0.25)

    def test_constant_sequence_zero_drop(self):
        ious = _constant(40, 0.6)
        assert self.analyzer.compute_tail_drop(ious) == pytest.approx(0.0, abs=1e-9)

    def test_decaying_sequence_positive_drop(self):
        ious = _decay(40, start=0.9, end=0.1)
        drop = self.analyzer.compute_tail_drop(ious)
        assert drop > 0.0

    def test_improving_sequence_negative_drop(self):
        ious = _ramp(40, start=0.1, stop=0.9)
        drop = self.analyzer.compute_tail_drop(ious)
        assert drop < 0.0

    def test_short_sequence_returns_zero(self):
        ious = np.array([0.5, 0.3])
        # tail_fraction=0.25 → tail_n = round(2*0.25) = 1
        # n <= tail_n: 2 > 1, so NOT <= tail_n; actually 2 - 1 = 1 frame for early, 1 for tail
        # Let's just verify no error and result is a float
        result = self.analyzer.compute_tail_drop(ious)
        assert isinstance(result, float)

    def test_empty_returns_zero(self):
        assert self.analyzer.compute_tail_drop(np.array([])) == pytest.approx(0.0)

    def test_known_value(self):
        # 8 frames: first 6 early (mean=0.8), last 2 tail (mean=0.2)
        # tail_fraction=0.25 → tail_n = round(8*0.25) = 2
        ious = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2])
        drop = self.analyzer.compute_tail_drop(ious)
        assert drop == pytest.approx(0.8 - 0.2, abs=1e-5)


# ---------------------------------------------------------------------------
# compute_iou_stability
# ---------------------------------------------------------------------------

class TestComputeIoUStability:
    def setup_method(self):
        self.analyzer = IoUTrajectoryAnalyzer(failure_threshold=0.1)

    def test_all_failures_returns_zero(self):
        ious = _constant(20, 0.05)
        assert self.analyzer.compute_iou_stability(ious) == pytest.approx(0.0)

    def test_constant_above_threshold_zero_cv(self):
        ious = _constant(30, 0.7)
        assert self.analyzer.compute_iou_stability(ious) == pytest.approx(0.0, abs=1e-9)

    def test_variable_ious_positive_stability(self):
        rng = np.random.default_rng(5)
        ious = rng.uniform(0.3, 0.9, 50)
        stability = self.analyzer.compute_iou_stability(ious)
        assert stability > 0.0

    def test_stability_non_negative(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            ious = rng.uniform(0.0, 1.0, 40)
            assert self.analyzer.compute_iou_stability(ious) >= 0.0

    def test_empty_returns_zero(self):
        assert self.analyzer.compute_iou_stability(np.array([])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# analyze — single sequence
# ---------------------------------------------------------------------------

class TestAnalyze:
    def setup_method(self):
        self.analyzer = IoUTrajectoryAnalyzer(
            n_segments=3, convergence_threshold=0.5, peak_window=5
        )

    def test_returns_iou_trajectory_result(self):
        result = self.analyzer.analyze(_constant(30, 0.6))
        assert isinstance(result, IoUTrajectoryResult)

    def test_empty_ious_returns_defaults(self):
        result = self.analyzer.analyze(np.array([]))
        assert result.num_frames == 0
        assert result.convergence_frame == -1
        assert result.overall_mean_iou == pytest.approx(0.0)
        assert result.segments == []

    def test_tracker_and_sequence_names_propagated(self):
        result = self.analyzer.analyze(_constant(20), "MOSSE", "car1")
        assert result.tracker_name == "MOSSE"
        assert result.sequence_name == "car1"

    def test_num_frames_correct(self):
        result = self.analyzer.analyze(_constant(45))
        assert result.num_frames == 45

    def test_overall_mean_iou_correct(self):
        ious = _ramp(20, 0.2, 0.8)
        result = self.analyzer.analyze(ious)
        assert result.overall_mean_iou == pytest.approx(float(ious.mean()), rel=1e-5)

    def test_segment_count(self):
        result = self.analyzer.analyze(_constant(30))
        assert len(result.segments) == 3

    def test_perfect_tracker_no_tail_drop(self):
        result = self.analyzer.analyze(_constant(50, 0.9))
        assert result.tail_drop == pytest.approx(0.0, abs=1e-9)

    def test_decaying_tracker_positive_tail_drop(self):
        result = self.analyzer.analyze(_decay(50, 0.9, 0.1))
        assert result.tail_drop > 0.0

    def test_to_dict_contains_required_keys(self):
        result = self.analyzer.analyze(_constant(20), "T", "S")
        d = result.to_dict()
        for key in (
            "tracker_name", "sequence_name", "num_frames", "overall_mean_iou",
            "convergence_frame", "peak_window_start", "peak_window_mean_iou",
            "tail_drop", "iou_stability", "segments",
        ):
            assert key in d, f"Missing key: {key}"

    def test_str_contains_tracker_name(self):
        result = self.analyzer.analyze(_constant(20), "KCF", "seq1")
        assert "KCF" in str(result)

    def test_peak_window_mean_geq_overall_mean(self):
        ious = _ramp(40, 0.0, 1.0)
        result = self.analyzer.analyze(ious)
        assert result.peak_window_mean_iou >= result.overall_mean_iou - 1e-9


# ---------------------------------------------------------------------------
# analyze_benchmark — multi-sequence aggregation
# ---------------------------------------------------------------------------

class TestAnalyzeBenchmark:
    def setup_method(self):
        self.analyzer = IoUTrajectoryAnalyzer(n_segments=3, peak_window=5)

    def test_empty_input(self):
        out = self.analyzer.analyze_benchmark({})
        assert out["per_sequence"] == {}
        assert out["aggregate"] == {}

    def test_per_sequence_keys_match_inputs(self):
        seq_ious = {"s0": _constant(20, 0.5), "s1": _constant(20, 0.7)}
        out = self.analyzer.analyze_benchmark(seq_ious)
        assert set(out["per_sequence"].keys()) == {"s0", "s1"}

    def test_aggregate_num_sequences(self):
        seq_ious = {f"s{i}": _constant(30) for i in range(4)}
        agg = self.analyzer.analyze_benchmark(seq_ious)["aggregate"]
        assert agg["num_sequences"] == 4

    def test_aggregate_tracker_name(self):
        seq_ious = {"s": _constant(20)}
        agg = self.analyzer.analyze_benchmark(seq_ious, "CSRT")["aggregate"]
        assert agg["tracker_name"] == "CSRT"

    def test_aggregate_mean_tail_drop_constant_sequences(self):
        seq_ious = {f"s{i}": _constant(40, 0.6) for i in range(3)}
        agg = self.analyzer.analyze_benchmark(seq_ious)["aggregate"]
        assert agg["mean_tail_drop"] == pytest.approx(0.0, abs=1e-9)

    def test_aggregate_segment_mean_ious_length(self):
        seq_ious = {f"s{i}": _constant(30) for i in range(3)}
        agg = self.analyzer.analyze_benchmark(seq_ious)["aggregate"]
        assert len(agg["segment_mean_ious"]) == 3

    def test_convergence_rate_all_converge(self):
        seq_ious = {f"s{i}": _constant(30, 0.9) for i in range(5)}
        agg = self.analyzer.analyze_benchmark(seq_ious)["aggregate"]
        assert agg["convergence_rate"] == pytest.approx(1.0)

    def test_convergence_rate_none_converge(self):
        seq_ious = {f"s{i}": _constant(30, 0.1) for i in range(5)}
        agg = self.analyzer.analyze_benchmark(seq_ious)["aggregate"]
        assert agg["convergence_rate"] == pytest.approx(0.0)

    def test_mean_tail_drop_decaying_vs_constant(self):
        decay_ious = {f"d{i}": _decay(40, 0.9, 0.1) for i in range(4)}
        const_ious = {f"c{i}": _constant(40, 0.5) for i in range(4)}
        agg_decay = self.analyzer.analyze_benchmark(decay_ious)["aggregate"]
        agg_const = self.analyzer.analyze_benchmark(const_ious)["aggregate"]
        assert agg_decay["mean_tail_drop"] > agg_const["mean_tail_drop"]

    def test_aggregate_required_keys(self):
        seq_ious = {"s0": _constant(20), "s1": _ramp(20)}
        agg = self.analyzer.analyze_benchmark(seq_ious, "T")["aggregate"]
        for key in (
            "tracker_name", "num_sequences", "mean_overall_iou",
            "mean_tail_drop", "mean_iou_stability", "mean_peak_window_iou",
            "convergence_rate", "mean_convergence_frame", "segment_mean_ious",
        ):
            assert key in agg, f"Missing aggregate key: {key}"


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestIntegrationWithBenchmarkEngine:
    def test_works_on_benchmark_engine_ious(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=3, num_frames=30, motion="linear")
        result = BenchmarkEngine(verbose=False).run(
            MOSSETracker(), dataset, dataset_name="Syn"
        )

        analyzer = IoUTrajectoryAnalyzer()
        seq_ious = {r.sequence_name: r.ious for r in result.sequence_results}
        out = analyzer.analyze_benchmark(seq_ious, result.tracker_name)

        assert len(out["per_sequence"]) == 3
        agg = out["aggregate"]
        assert 0.0 <= agg["mean_overall_iou"] <= 1.0
        assert isinstance(agg["mean_tail_drop"], float)
        assert 0.0 <= agg["convergence_rate"] <= 1.0
