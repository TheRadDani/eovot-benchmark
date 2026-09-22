"""Tests for MultiRunBenchmark stability engine."""

from __future__ import annotations

import pytest
import numpy as np

from eovot.benchmark.stability import (
    MultiRunBenchmark,
    MultiRunResult,
    RunSummary,
    StabilityReport,
)
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=20)


@pytest.fixture
def mosse():
    return MOSSETracker()


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestMultiRunBenchmarkConstruction:
    def test_default_construction(self):
        mb = MultiRunBenchmark()
        assert mb.n_runs == 3
        assert mb.warmup_frames == 5

    def test_custom_params(self):
        mb = MultiRunBenchmark(n_runs=5, warmup_frames=2, verbose=False)
        assert mb.n_runs == 5
        assert mb.warmup_frames == 2

    def test_n_runs_too_small_raises(self):
        with pytest.raises(ValueError, match="n_runs"):
            MultiRunBenchmark(n_runs=1)

    def test_warmup_frames_negative_raises(self):
        with pytest.raises(ValueError, match="warmup_frames"):
            MultiRunBenchmark(n_runs=2, warmup_frames=-1)

    def test_warmup_zero_allowed(self):
        mb = MultiRunBenchmark(n_runs=2, warmup_frames=0)
        assert mb.warmup_frames == 0


# ---------------------------------------------------------------------------
# RunSummary
# ---------------------------------------------------------------------------

class TestRunSummary:
    def test_str_no_auc(self):
        rs = RunSummary(
            run_index=0, mean_iou=0.75, mean_fps=30.0,
            peak_memory_mb=100.0, latency_mean_ms=33.3,
            latency_p95_ms=40.0, wall_time_s=5.0,
        )
        s = str(rs)
        assert "Run 0" in s
        assert "0.7500" in s
        assert "30.0" in s

    def test_str_with_auc(self):
        rs = RunSummary(
            run_index=1, mean_iou=0.65, mean_fps=25.0,
            peak_memory_mb=80.0, latency_mean_ms=40.0,
            latency_p95_ms=55.0, wall_time_s=3.2,
            success_auc=0.61,
        )
        assert "AUC=0.6100" in str(rs)


# ---------------------------------------------------------------------------
# StabilityReport
# ---------------------------------------------------------------------------

class TestStabilityReport:
    def _make_report(self, iou_cv=0.01, fps_cv=0.02):
        return StabilityReport(
            tracker_name="MOSSE",
            dataset_name="synthetic",
            n_runs=3,
            iou_mean=0.70,
            iou_std=iou_cv * 0.70,
            iou_cv=iou_cv,
            fps_mean=50.0,
            fps_std=fps_cv * 50.0,
            fps_cv=fps_cv,
            latency_mean_ms=20.0,
            latency_std_ms=0.5,
            latency_cv=0.025,
        )

    def test_is_stable_within_defaults(self):
        assert self._make_report(iou_cv=0.04, fps_cv=0.09).is_stable()

    def test_is_not_stable_high_iou_cv(self):
        assert not self._make_report(iou_cv=0.06, fps_cv=0.02).is_stable()

    def test_is_not_stable_high_fps_cv(self):
        assert not self._make_report(iou_cv=0.02, fps_cv=0.11).is_stable()

    def test_custom_thresholds(self):
        r = self._make_report(iou_cv=0.06, fps_cv=0.08)
        assert r.is_stable(iou_cv_thresh=0.10, fps_cv_thresh=0.15)

    def test_to_markdown_contains_metrics(self):
        r = self._make_report()
        md = r.to_markdown()
        assert "Stability Report" in md
        assert "mIoU" in md
        assert "FPS" in md
        assert "Latency" in md
        assert "stable" in md.lower()

    def test_to_dict_keys(self):
        d = self._make_report().to_dict()
        for k in ["tracker", "dataset", "n_runs", "iou_mean", "iou_std", "iou_cv",
                   "fps_mean", "fps_std", "fps_cv", "is_stable"]:
            assert k in d

    def test_to_dict_with_auc(self):
        r = self._make_report()
        r.success_auc_mean = 0.65
        r.success_auc_std = 0.01
        d = r.to_dict()
        assert "success_auc_mean" in d
        assert "success_auc_std" in d


# ---------------------------------------------------------------------------
# MultiRunResult
# ---------------------------------------------------------------------------

class TestMultiRunResult:
    def _make_result(self, n_runs=3):
        from eovot.benchmark.engine import BenchmarkResult, SequenceResult
        from eovot.profiling.profiler import ProfilingResult
        import numpy as np

        run_results = []
        run_summaries = []
        for i in range(n_runs):
            ious = np.full(10, 0.70 + i * 0.01)
            prof = ProfilingResult(
                tracker_name="MOSSE",
                frame_count=10,
                fps=50.0 - i,
                latency_mean_ms=20.0 + i * 0.5,
                latency_std_ms=1.0,
                latency_p95_ms=23.0,
                latency_p99_ms=25.0,
                latency_cv=0.05,
                peak_memory_mb=100.0,
            )
            seq_result = SequenceResult(
                sequence_name="seq0",
                ious=ious,
                profiling=prof,
            )
            br = BenchmarkResult(tracker_name="MOSSE", dataset_name="synthetic")
            br.sequence_results = [seq_result]
            run_results.append(br)
            run_summaries.append(RunSummary(
                run_index=i, mean_iou=float(ious.mean()),
                mean_fps=50.0 - i, peak_memory_mb=100.0,
                latency_mean_ms=20.0 + i * 0.5,
                latency_p95_ms=23.0, wall_time_s=1.0,
            ))

        mr = MultiRunResult(
            tracker_name="MOSSE",
            dataset_name="synthetic",
            n_runs=n_runs,
            warmup_frames=2,
        )
        mr.run_results = run_results
        mr.run_summaries = run_summaries
        return mr

    def test_stability_returns_report(self):
        mr = self._make_result()
        s = mr.stability
        assert isinstance(s, StabilityReport)
        assert s.n_runs == 3

    def test_stability_iou_mean_close_to_input(self):
        mr = self._make_result()
        s = mr.stability
        assert abs(s.iou_mean - 0.71) < 0.01

    def test_stability_fps_mean_close_to_input(self):
        mr = self._make_result()
        s = mr.stability
        assert abs(s.fps_mean - 49.0) < 0.5

    def test_summary_is_dict(self):
        mr = self._make_result()
        assert isinstance(mr.summary(), dict)

    def test_to_markdown_contains_per_run(self):
        mr = self._make_result()
        md = mr.to_markdown()
        assert "Per-Run Summary" in md
        assert "Run 0" in md

    def test_str_contains_runs(self):
        mr = self._make_result()
        assert "3 runs" in str(mr)


# ---------------------------------------------------------------------------
# Integration: full run on SyntheticDataset
# ---------------------------------------------------------------------------

class TestMultiRunIntegration:
    def test_run_returns_correct_n_runs(self, small_dataset, mosse):
        mb = MultiRunBenchmark(n_runs=2, warmup_frames=2, verbose=False)
        result = mb.run(mosse, small_dataset, "synthetic")
        assert len(result.run_results) == 2
        assert len(result.run_summaries) == 2

    def test_run_ious_in_valid_range(self, small_dataset, mosse):
        mb = MultiRunBenchmark(n_runs=2, warmup_frames=1, verbose=False)
        result = mb.run(mosse, small_dataset, "synthetic")
        for r in result.run_results:
            assert 0.0 <= r.mean_iou <= 1.0

    def test_run_fps_positive(self, small_dataset, mosse):
        mb = MultiRunBenchmark(n_runs=2, warmup_frames=1, verbose=False)
        result = mb.run(mosse, small_dataset, "synthetic")
        for r in result.run_results:
            assert r.mean_fps > 0

    def test_stability_report_is_stable_on_deterministic_tracker(self, small_dataset, mosse):
        mb = MultiRunBenchmark(n_runs=3, warmup_frames=2, verbose=False)
        result = mb.run(mosse, small_dataset, "synthetic")
        s = result.stability
        # MOSSE on synthetic is deterministic — variance should be near-zero
        assert s.iou_cv < 0.20  # generous threshold for CI

    def test_max_sequences_respected(self, mosse):
        ds = SyntheticDataset(num_sequences=10, num_frames=20)
        mb = MultiRunBenchmark(n_runs=2, warmup_frames=1, verbose=False)
        result = mb.run(mosse, ds, "synthetic", max_sequences=2)
        for r in result.run_results:
            assert r.num_sequences == 2 if hasattr(r, 'num_sequences') else len(r.sequence_results) == 2

    def test_warmup_zero_still_runs(self, small_dataset, mosse):
        mb = MultiRunBenchmark(n_runs=2, warmup_frames=0, verbose=False)
        result = mb.run(mosse, small_dataset, "synthetic")
        assert len(result.run_results) == 2

    def test_kcf_tracker_runs(self, small_dataset):
        kcf = KCFTracker()
        mb = MultiRunBenchmark(n_runs=2, warmup_frames=1, verbose=False)
        result = mb.run(kcf, small_dataset, "synthetic")
        assert result.stability.fps_mean > 0

    def test_summary_dict_round_trips_json(self, small_dataset, mosse):
        import json
        mb = MultiRunBenchmark(n_runs=2, warmup_frames=1, verbose=False)
        result = mb.run(mosse, small_dataset, "synthetic")
        d = result.summary()
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        assert d2["tracker"] == "MOSSE"
        assert d2["n_runs"] == 2

    def test_markdown_output_nonempty(self, small_dataset, mosse):
        mb = MultiRunBenchmark(n_runs=2, warmup_frames=1, verbose=False)
        result = mb.run(mosse, small_dataset, "synthetic")
        md = result.to_markdown()
        assert len(md) > 100

    def test_str_representation(self, small_dataset, mosse):
        mb = MultiRunBenchmark(n_runs=2, warmup_frames=1, verbose=False)
        result = mb.run(mosse, small_dataset, "synthetic")
        s = str(result)
        assert "MultiRunResult" in s
        assert "MOSSE" in s
