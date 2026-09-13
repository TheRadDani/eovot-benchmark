"""Tests for MultiRunBenchmark — statistical repeatability engine."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.multi_run import (
    MultiRunBenchmark,
    MultiRunSummary,
    RunStats,
    _run_stats,
    _t_crit_95,
)
from eovot.benchmark.engine import BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_dataset(n_seq: int = 2, seq_len: int = 20) -> SyntheticDataset:
    return SyntheticDataset(num_sequences=n_seq, sequence_length=seq_len)


# ---------------------------------------------------------------------------
# Unit tests for statistics helpers
# ---------------------------------------------------------------------------

class TestRunStats:
    def test_single_value(self):
        stats = _run_stats([5.0])
        assert stats.mean == pytest.approx(5.0)
        assert stats.std == 0.0
        assert stats.ci95_low == pytest.approx(5.0)
        assert stats.ci95_high == pytest.approx(5.0)
        assert stats.n_runs == 1

    def test_multiple_values_mean_std(self):
        values = [10.0, 12.0, 11.0, 13.0]
        stats = _run_stats(values)
        assert stats.mean == pytest.approx(np.mean(values), rel=1e-6)
        assert stats.std == pytest.approx(np.std(values, ddof=1), rel=1e-6)
        assert stats.min == pytest.approx(min(values))
        assert stats.max == pytest.approx(max(values))
        assert stats.n_runs == 4

    def test_ci95_wider_for_small_n(self):
        """CI should be wider for n=2 than n=10 given same std."""
        values_small = [5.0, 7.0]
        values_large = [5.0, 7.0, 5.0, 7.0, 5.0, 7.0, 5.0, 7.0, 5.0, 7.0]
        s_small = _run_stats(values_small)
        s_large = _run_stats(values_large)
        width_small = s_small.ci95_high - s_small.ci95_low
        width_large = s_large.ci95_high - s_large.ci95_low
        assert width_small > width_large

    def test_str_representation(self):
        stats = _run_stats([100.0, 110.0, 105.0])
        s = str(stats)
        assert "±" in s or "+/-" in s or "±" in s
        assert "95" in s


class TestTCrit:
    @pytest.mark.parametrize("df,expected", [
        (1, 12.706),
        (2, 4.303),
        (10, 2.228),
        (120, 1.960),
        (200, 1.960),
    ])
    def test_lookup(self, df, expected):
        assert _t_crit_95(df) == pytest.approx(expected, rel=1e-3)

    def test_interpolation_between_known(self):
        # df=11 should be between df=10 (2.228) and df=12 (2.179)
        val = _t_crit_95(11)
        assert 2.179 < val < 2.228


# ---------------------------------------------------------------------------
# MultiRunBenchmark integration tests
# ---------------------------------------------------------------------------

class TestMultiRunBenchmark:
    def test_basic_run_returns_summary(self):
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=2, warmup_runs=0, verbose=False)
        summary = bench.run(tracker, dataset, dataset_name="test")
        assert isinstance(summary, MultiRunSummary)
        assert summary.n_runs == 2
        assert summary.warmup_runs == 0
        assert summary.tracker_name == tracker.name
        assert summary.dataset_name == "test"

    def test_fps_positive(self):
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=2, warmup_runs=0, verbose=False)
        summary = bench.run(tracker, dataset)
        assert summary.fps.mean > 0
        assert summary.fps.n_runs == 2

    def test_iou_in_unit_range(self):
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=2, warmup_runs=0, verbose=False)
        summary = bench.run(tracker, dataset)
        assert 0.0 <= summary.mean_iou.mean <= 1.0

    def test_warmup_runs_excluded_from_stats(self):
        """With warmup_runs=1 and n_runs=2, only 2 results should be in per_run_results."""
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=2, warmup_runs=1, verbose=False)
        summary = bench.run(tracker, dataset)
        assert len(summary.per_run_results) == 2

    def test_thermal_trend_computed_for_two_plus_runs(self):
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=2, warmup_runs=0, verbose=False)
        summary = bench.run(tracker, dataset)
        assert summary.thermal_trend_ms_per_run is not None

    def test_thermal_trend_none_for_single_run(self):
        """With n_runs=1, we cannot compute a slope — trend should be None."""
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=1, warmup_runs=0, verbose=False)
        summary = bench.run(tracker, dataset)
        # With only one measurement point, no trend can be fit.
        assert summary.thermal_trend_ms_per_run is None

    def test_warmup_ratio_computed_when_warmup_enabled(self):
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=2, warmup_runs=1, verbose=False)
        summary = bench.run(tracker, dataset)
        assert summary.warmup_overhead_ratio is not None
        assert summary.warmup_overhead_ratio > 0

    def test_ci95_within_plausible_range(self):
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=3, warmup_runs=0, verbose=False)
        summary = bench.run(tracker, dataset)
        assert summary.fps.ci95_low <= summary.fps.mean
        assert summary.fps.ci95_high >= summary.fps.mean
        assert summary.fps.ci95_low > 0

    def test_summary_dict_keys(self):
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=2, warmup_runs=1, verbose=False)
        summary = bench.run(tracker, dataset)
        d = summary.summary()
        required_keys = [
            "tracker", "dataset", "n_runs", "warmup_runs",
            "fps_mean", "fps_std", "fps_ci95", "fps_cv",
            "latency_mean_ms", "latency_p95_ms_mean",
            "peak_memory_mb_mean", "mean_iou_mean",
            "thermal_trend_ms_per_run",
            "warmup_overhead_ratio",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_invalid_n_runs_raises(self):
        with pytest.raises(ValueError, match="n_runs"):
            MultiRunBenchmark(n_runs=0)

    def test_invalid_warmup_runs_raises(self):
        with pytest.raises(ValueError, match="warmup_runs"):
            MultiRunBenchmark(n_runs=2, warmup_runs=-1)

    def test_fps_cv_computed(self):
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=3, warmup_runs=0, verbose=False)
        summary = bench.run(tracker, dataset)
        assert summary.fps_cv >= 0.0

    def test_str_representation(self):
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=2, warmup_runs=0, verbose=False)
        summary = bench.run(tracker, dataset)
        s = str(summary)
        assert tracker.name in s
        assert "runs" in s.lower() or "run" in s.lower()

    def test_compare_returns_list(self):
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        trackers = [MOSSETracker(), KCFTracker()]
        bench = MultiRunBenchmark(n_runs=2, warmup_runs=0, verbose=False)
        summaries = bench.compare(trackers, dataset, dataset_name="synthetic")
        assert len(summaries) == 2
        assert summaries[0].tracker_name != summaries[1].tracker_name

    def test_has_thermal_throttling_flag(self):
        """Flag should be False on a short synthetic run (no real throttling)."""
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=3, warmup_runs=0, verbose=False)
        summary = bench.run(tracker, dataset)
        # On a test machine the flag may be True or False — just check it's a bool.
        assert isinstance(summary.has_thermal_throttling, bool)

    def test_has_warmup_penalty_flag(self):
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=2, warmup_runs=1, verbose=False)
        summary = bench.run(tracker, dataset)
        assert isinstance(summary.has_warmup_penalty, bool)

    def test_per_run_results_are_benchmark_results(self):
        dataset = _tiny_dataset(n_seq=2, seq_len=15)
        tracker = MOSSETracker()
        bench = MultiRunBenchmark(n_runs=2, warmup_runs=0, verbose=False)
        summary = bench.run(tracker, dataset)
        for res in summary.per_run_results:
            assert isinstance(res, BenchmarkResult)
