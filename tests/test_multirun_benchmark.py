"""Tests for MultiRunBenchmark — multi-run latency CI computation."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from eovot.benchmark.multirun import ConfidenceInterval, MultiRunBenchmark, MultiRunResult
from eovot.datasets.base import BaseDataset, Sequence
from eovot.trackers.base import BaseTracker

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

_BBOX = (10.0, 10.0, 50.0, 50.0)
_N_FRAMES = 20
_N_SEQS = 3


class StubTracker(BaseTracker):
    def __init__(self):
        super().__init__(name="StubTracker")

    def initialize(self, frame, bbox):
        pass

    def update(self, frame):
        return _BBOX


class StubSequence(Sequence):
    def __init__(self, name="s0"):
        gt = np.tile(np.array(_BBOX), (_N_FRAMES, 1))
        super().__init__(name, [f"f{i}" for i in range(_N_FRAMES)], gt)

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(_N_FRAMES):
            yield np.zeros((60, 80, 3), dtype=np.uint8)


class StubDataset(BaseDataset):
    def __init__(self):
        self._seqs = [StubSequence(f"s{i}") for i in range(_N_SEQS)]

    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, idx):
        return self._seqs[idx]


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestMultiRunBenchmarkInit:
    def test_num_runs_too_small_raises(self):
        with pytest.raises(ValueError, match="num_runs"):
            MultiRunBenchmark(num_runs=1)

    def test_valid_init(self):
        mr = MultiRunBenchmark(num_runs=3)
        assert mr.num_runs == 3
        assert mr.warmup_run is True


# ---------------------------------------------------------------------------
# Run basic smoke test
# ---------------------------------------------------------------------------

class TestMultiRunBenchmarkRun:
    def setup_method(self):
        self.mr = MultiRunBenchmark(num_runs=3, warmup_run=True, n_bootstrap=100, seed=0)
        self.dataset = StubDataset()
        self.tracker = StubTracker()

    def test_returns_multirun_result(self):
        result = self.mr.run(self.tracker, self.dataset, "Stub")
        assert isinstance(result, MultiRunResult)

    def test_correct_measurement_run_count_with_warmup(self):
        result = self.mr.run(self.tracker, self.dataset, "Stub")
        assert result.num_measurement_runs == 2  # 3 runs - 1 warmup

    def test_correct_measurement_run_count_without_warmup(self):
        mr = MultiRunBenchmark(num_runs=3, warmup_run=False, n_bootstrap=100, seed=0)
        result = mr.run(self.tracker, self.dataset, "Stub")
        assert result.num_measurement_runs == 3

    def test_ci_bounds_ordered(self):
        result = self.mr.run(self.tracker, self.dataset, "Stub")
        for ci in [result.fps_ci, result.latency_mean_ci, result.latency_p95_ci,
                   result.latency_p99_ci, result.mean_iou_ci]:
            assert ci.lower <= ci.mean <= ci.upper, f"CI order violated: {ci}"

    def test_fps_positive(self):
        result = self.mr.run(self.tracker, self.dataset, "Stub")
        assert result.fps_ci.mean > 0

    def test_iou_in_unit_interval(self):
        result = self.mr.run(self.tracker, self.dataset, "Stub")
        assert 0.0 <= result.mean_iou_ci.mean <= 1.0

    def test_success_auc_ci_populated(self):
        result = self.mr.run(self.tracker, self.dataset, "Stub")
        assert result.success_auc_ci is not None
        assert 0.0 <= result.success_auc_ci.mean <= 1.0

    def test_raw_runs_stored(self):
        result = self.mr.run(self.tracker, self.dataset, "Stub")
        assert len(result.raw_runs) == result.num_measurement_runs

    def test_tracker_and_dataset_names_stored(self):
        result = self.mr.run(self.tracker, self.dataset, "MyDataset")
        assert result.tracker_name == "StubTracker"
        assert result.dataset_name == "MyDataset"


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------

class TestMultiRunResultOutput:
    def setup_method(self):
        self.mr = MultiRunBenchmark(num_runs=2, warmup_run=False, n_bootstrap=50, seed=0)
        self.result = self.mr.run(StubTracker(), StubDataset(), "Stub")

    def test_str_contains_tracker_name(self):
        s = str(self.result)
        assert "StubTracker" in s

    def test_to_dict_has_required_keys(self):
        d = self.result.to_dict()
        for key in ["tracker_name", "dataset_name", "num_runs", "fps", "mean_iou",
                    "latency_mean_ms", "latency_p95_ms", "latency_p99_ms"]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_ci_has_mean_lower_upper_std(self):
        d = self.result.to_dict()
        fps_ci = d["fps"]
        for k in ["mean", "lower", "upper", "std"]:
            assert k in fps_ci

    def test_to_markdown_contains_fps_row(self):
        md = self.result.to_markdown()
        assert "FPS" in md

    def test_to_markdown_contains_iou_row(self):
        md = self.result.to_markdown()
        assert "IoU" in md or "iou" in md.lower()
