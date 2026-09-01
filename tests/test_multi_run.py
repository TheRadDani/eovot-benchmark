"""Tests for eovot.benchmark.multi_run.MultiRunBenchmark."""

import pytest

from eovot.benchmark.multi_run import MultiRunBenchmark, MultiRunEntry, _aggregate
from eovot.benchmark.engine import BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=20, motion="linear", seed=0)


@pytest.fixture
def benchmark():
    return MultiRunBenchmark(num_runs=3, verbose=False)


# ---------------------------------------------------------------------------
# _aggregate helper
# ---------------------------------------------------------------------------


def _make_fake_results(n: int, iou: float = 0.6, fps: float = 100.0) -> list:
    """Build a list of minimal BenchmarkResult stubs for aggregate testing."""
    from eovot.profiling.profiler import ProfilingResult
    from eovot.benchmark.engine import SequenceResult
    import numpy as np

    results = []
    for _ in range(n):
        sr = SequenceResult(
            sequence_name="seq0",
            ious=np.full(50, iou),
            profiling=ProfilingResult(
                tracker_name="Dummy",
                frame_count=50,
                fps=fps,
                latency_mean_ms=1000.0 / fps,
                latency_std_ms=0.5,
                latency_p95_ms=1000.0 / fps + 1.0,
                latency_p99_ms=1000.0 / fps + 2.0,
                latency_cv=0.05,
                peak_memory_mb=64.0,
            ),
        )
        br = BenchmarkResult(tracker_name="Dummy", dataset_name="Fake")
        br.sequence_results = [sr]
        results.append(br)
    return results


class TestAggregate:
    def test_single_run(self):
        runs = _make_fake_results(1)
        entry = _aggregate("Dummy", "Fake", runs)
        assert entry.num_runs == 1
        assert entry.iou_std == 0.0
        assert entry.fps_std == 0.0

    def test_identical_runs(self):
        runs = _make_fake_results(4, iou=0.5, fps=200.0)
        entry = _aggregate("Dummy", "Fake", runs)
        assert entry.iou_mean == pytest.approx(0.5, abs=1e-3)
        assert entry.fps_mean == pytest.approx(200.0, abs=0.1)
        assert entry.iou_cv == pytest.approx(0.0, abs=1e-3)
        assert entry.stability_score == pytest.approx(1.0, abs=1e-3)

    def test_varying_runs(self):
        from eovot.profiling.profiler import ProfilingResult
        from eovot.benchmark.engine import SequenceResult
        import numpy as np

        ious = [0.4, 0.6, 0.8]
        runs = []
        for v in ious:
            sr = SequenceResult(
                sequence_name="seq0",
                ious=np.full(10, v),
                profiling=ProfilingResult(
                    tracker_name="Dummy",
                    frame_count=10,
                    fps=100.0,
                    latency_mean_ms=10.0,
                    latency_std_ms=0.5,
                    latency_p95_ms=11.0,
                    latency_p99_ms=12.0,
                    latency_cv=0.05,
                    peak_memory_mb=64.0,
                ),
            )
            br = BenchmarkResult(tracker_name="Dummy", dataset_name="Fake")
            br.sequence_results = [sr]
            runs.append(br)

        entry = _aggregate("Dummy", "Fake", runs)
        assert entry.iou_mean == pytest.approx(0.6, abs=1e-3)
        assert entry.iou_std > 0.0
        assert entry.stability_score < 1.0

    def test_stability_clamped(self):
        runs = _make_fake_results(3, iou=0.5, fps=100.0)
        entry = _aggregate("Dummy", "Fake", runs)
        assert 0.0 <= entry.stability_score <= 1.0


# ---------------------------------------------------------------------------
# MultiRunBenchmark integration
# ---------------------------------------------------------------------------


class TestMultiRunBenchmark:
    def test_basic_run_single_tracker(self, tiny_dataset, benchmark):
        trackers = [MOSSETracker()]
        entries = benchmark.run(trackers, tiny_dataset, dataset_name="Synthetic")
        assert len(entries) == 1
        e = entries[0]
        assert e.tracker_name == "MOSSE"
        assert e.num_runs == 3
        assert 0.0 <= e.iou_mean <= 1.0
        assert e.fps_mean > 0.0
        assert e.iou_std >= 0.0
        assert 0.0 <= e.stability_score <= 1.0

    def test_basic_run_two_trackers(self, tiny_dataset, benchmark):
        trackers = [MOSSETracker(), KCFTracker()]
        entries = benchmark.run(trackers, tiny_dataset, dataset_name="Synthetic")
        assert len(entries) == 2
        names = {e.tracker_name for e in entries}
        assert "MOSSE" in names
        assert "KCF" in names

    def test_sorted_by_stability(self, tiny_dataset, benchmark):
        trackers = [MOSSETracker(), KCFTracker()]
        entries = benchmark.run(trackers, tiny_dataset, dataset_name="Synthetic")
        stabs = [e.stability_score for e in entries]
        assert stabs == sorted(stabs, reverse=True)

    def test_to_markdown_output(self, tiny_dataset, benchmark):
        trackers = [MOSSETracker()]
        entries = benchmark.run(trackers, tiny_dataset, dataset_name="Synthetic")
        md = benchmark.to_markdown(entries, title="Test Run")
        assert "## Test Run" in md
        assert "MOSSE" in md
        assert "mIoU" in md
        assert "Stability" in md

    def test_to_dict_structure(self, tiny_dataset, benchmark):
        trackers = [MOSSETracker()]
        entries = benchmark.run(trackers, tiny_dataset, dataset_name="Synthetic")
        d = benchmark.to_dict(entries)
        assert "num_runs" in d
        assert "trackers" in d
        assert d["num_runs"] == 3
        t = d["trackers"][0]
        for key in ("tracker_name", "iou_mean", "iou_std", "fps_mean",
                    "fps_cv", "stability_score"):
            assert key in t

    def test_max_sequences_respected(self, benchmark):
        dataset = SyntheticDataset(num_sequences=10, num_frames=20, seed=7)
        entries = benchmark.run(
            [MOSSETracker()], dataset, max_sequences=2, dataset_name="Synthetic"
        )
        assert entries[0].iou_mean >= 0.0

    def test_num_runs_validation(self):
        with pytest.raises(ValueError, match="num_runs must be >= 1"):
            MultiRunBenchmark(num_runs=0)

    def test_str_representation(self, tiny_dataset):
        bm = MultiRunBenchmark(num_runs=2, verbose=False)
        entries = bm.run([MOSSETracker()], tiny_dataset)
        s = str(entries[0])
        assert "MultiRunEntry" in s
        assert "mIoU" in s
        assert "stability" in s
