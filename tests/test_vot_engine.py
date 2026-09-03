"""Tests for the VOT-protocol re-initialization benchmark engine."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.vot_engine import VotBenchmarkEngine, VotBenchmarkResult, VotSequenceResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Helper: a failing tracker that always predicts (0, 0, 1, 1)
# ---------------------------------------------------------------------------

from eovot.trackers.base import BaseTracker, BBox
import numpy as _np


class AlwaysFailTracker(BaseTracker):
    """Tracker that immediately loses the target — always predicts top-left corner."""

    def __init__(self) -> None:
        super().__init__("AlwaysFail")

    def initialize(self, frame: _np.ndarray, bbox: BBox) -> None:
        pass

    def update(self, frame: _np.ndarray) -> BBox:
        return (0.0, 0.0, 1.0, 1.0)


class PerfectTracker(BaseTracker):
    """Oracle tracker that always returns the correct GT bbox (for testing)."""

    def __init__(self, gt: list) -> None:
        super().__init__("PerfectTracker")
        self._gt = gt
        self._frame_idx = 0

    def initialize(self, frame: _np.ndarray, bbox: BBox) -> None:
        self._frame_idx = 0

    def update(self, frame: _np.ndarray) -> BBox:
        self._frame_idx += 1
        if self._frame_idx < len(self._gt):
            g = self._gt[self._frame_idx]
            return (float(g[0]), float(g[1]), float(g[2]), float(g[3]))
        return (0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Basic engine tests
# ---------------------------------------------------------------------------

class TestVotBenchmarkEngine:
    def test_smoke_run(self):
        """Engine runs without errors on a small synthetic dataset."""
        dataset = SyntheticDataset(num_sequences=3, num_frames=30)
        engine = VotBenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")
        assert isinstance(result, VotBenchmarkResult)
        assert len(result.sequence_results) == 3

    def test_result_type(self):
        """Sequence results are VotSequenceResult instances."""
        dataset = SyntheticDataset(num_sequences=2, num_frames=20)
        engine = VotBenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Syn")
        for sr in result.sequence_results:
            assert isinstance(sr, VotSequenceResult)

    def test_iou_first_frame_is_one(self):
        """First frame IoU must be 1.0 by the VOT initialization convention."""
        dataset = SyntheticDataset(num_sequences=2, num_frames=25)
        engine = VotBenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Syn")
        for sr in result.sequence_results:
            assert sr.ious[0] == pytest.approx(1.0)

    def test_ious_length_matches_frames(self):
        """Per-frame IoU array length equals the number of frames in the sequence."""
        n_frames = 40
        dataset = SyntheticDataset(num_sequences=2, num_frames=n_frames)
        engine = VotBenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Syn")
        for sr in result.sequence_results:
            assert len(sr.ious) == n_frames

    def test_max_sequences_limit(self):
        """max_sequences parameter caps the number of evaluated sequences."""
        dataset = SyntheticDataset(num_sequences=10, num_frames=20)
        engine = VotBenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Syn", max_sequences=3)
        assert len(result.sequence_results) == 3

    def test_eao_in_range(self):
        """EAO scalar must be in [0, 1]."""
        dataset = SyntheticDataset(num_sequences=4, num_frames=30)
        engine = VotBenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Syn")
        assert 0.0 <= result.mean_eao <= 1.0
        for sr in result.sequence_results:
            assert 0.0 <= sr.eao <= 1.0


# ---------------------------------------------------------------------------
# Failure detection & grace-period tests
# ---------------------------------------------------------------------------

class TestVotFailureProtocol:
    def test_always_fail_tracker_has_failures(self):
        """AlwaysFailTracker should accumulate failures."""
        dataset = SyntheticDataset(num_sequences=3, num_frames=50)
        engine = VotBenchmarkEngine(failure_threshold=0.1, grace_period_frames=5, verbose=False)
        result = engine.run(AlwaysFailTracker(), dataset, dataset_name="Syn")
        assert result.total_failures > 0

    def test_grace_frames_have_zero_iou(self):
        """Frames inside a grace period must carry IoU = 0."""
        dataset = SyntheticDataset(num_sequences=2, num_frames=50)
        engine = VotBenchmarkEngine(failure_threshold=0.1, grace_period_frames=5, verbose=False)
        result = engine.run(AlwaysFailTracker(), dataset, dataset_name="Syn")
        for sr in result.sequence_results:
            for idx in sr.grace_frames:
                assert sr.ious[idx] == pytest.approx(0.0)

    def test_eao_penalised_for_always_fail_tracker(self):
        """A perfectly failing tracker should have low EAO (zero or near-zero)."""
        dataset = SyntheticDataset(num_sequences=5, num_frames=60)
        engine = VotBenchmarkEngine(failure_threshold=0.1, grace_period_frames=5, verbose=False)
        result = engine.run(AlwaysFailTracker(), dataset, dataset_name="Syn")
        # EAO will be dominated by zeros from grace periods, so should be very low
        assert result.mean_eao < 0.3

    def test_no_grace_frames_for_good_tracker(self):
        """A tracker with no failures should produce no grace frames."""
        dataset = SyntheticDataset(num_sequences=3, num_frames=40)
        # Use perfect tracker that returns exact GT
        seq = dataset[0]
        gt = list(seq.ground_truth)
        tracker = PerfectTracker(gt)

        engine = VotBenchmarkEngine(failure_threshold=0.1, grace_period_frames=5, verbose=False)
        sr = engine._run_sequence(tracker, seq)
        assert sr.num_failures == 0
        assert len(sr.grace_frames) == 0

    def test_grace_period_length(self):
        """After a failure, exactly grace_period_frames frames should be penalised."""
        dataset = SyntheticDataset(num_sequences=1, num_frames=30)
        engine = VotBenchmarkEngine(failure_threshold=0.9, grace_period_frames=4, verbose=False)
        # MOSSE on a short sequence will likely fail with threshold=0.9
        result = engine.run(AlwaysFailTracker(), dataset, dataset_name="Syn")
        for sr in result.sequence_results:
            if sr.num_failures == 0:
                continue
            # Check that consecutive blocks of grace frames are at most grace_period_frames long
            sorted_grace = sorted(sr.grace_frames)
            if sorted_grace:
                block_len = 1
                for a, b in zip(sorted_grace, sorted_grace[1:]):
                    if b == a + 1:
                        block_len += 1
                    else:
                        assert block_len <= 4
                        block_len = 1
                assert block_len <= 4


# ---------------------------------------------------------------------------
# Summary / serialisation tests
# ---------------------------------------------------------------------------

class TestVotBenchmarkResultSummary:
    def test_summary_keys(self):
        """Summary dict must include all required keys."""
        dataset = SyntheticDataset(num_sequences=2, num_frames=20)
        engine = VotBenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Syn")
        summary = result.summary()
        for key in ("tracker", "dataset", "protocol", "mean_eao", "mean_fps",
                    "total_failures", "mean_failures_per_sequence", "num_sequences"):
            assert key in summary, f"Missing key: {key}"

    def test_summary_protocol_label(self):
        dataset = SyntheticDataset(num_sequences=2, num_frames=20)
        engine = VotBenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Syn")
        assert result.summary()["protocol"] == "VOT"

    def test_str_repr(self):
        dataset = SyntheticDataset(num_sequences=2, num_frames=20)
        engine = VotBenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Syn")
        s = str(result)
        assert "VotBenchmarkResult" in s
        assert "EAO=" in s

    def test_total_failures_consistent(self):
        """total_failures must equal sum of per-sequence failures."""
        dataset = SyntheticDataset(num_sequences=4, num_frames=40)
        engine = VotBenchmarkEngine(verbose=False)
        result = engine.run(AlwaysFailTracker(), dataset, dataset_name="Syn")
        expected = sum(sr.num_failures for sr in result.sequence_results)
        assert result.total_failures == expected
