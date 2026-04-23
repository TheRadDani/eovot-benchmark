"""Unit tests for eovot.metrics.stability."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eovot.metrics.stability import IoUStabilityMetrics, compute_stability


class TestComputeStabilityInputValidation:
    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_stability(np.array([]))

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            compute_stability(np.array([0.5, 1.5]))

    def test_negative_iou_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            compute_stability(np.array([-0.1, 0.5]))

    def test_single_frame_works(self):
        result = compute_stability(np.array([0.8]))
        assert result.iou_std == pytest.approx(0.0)
        assert result.failure_rate == pytest.approx(0.0)


class TestIoUStd:
    def test_constant_series_zero_std(self):
        ious = np.full(30, 0.7)
        result = compute_stability(ious)
        assert result.iou_std == pytest.approx(0.0, abs=1e-9)

    def test_std_matches_numpy(self):
        rng = np.random.default_rng(0)
        ious = rng.uniform(0, 1, 50)
        result = compute_stability(ious)
        assert result.iou_std == pytest.approx(float(ious.std()), rel=1e-6)

    def test_high_variance_series_large_std(self):
        ious = np.array([0.0, 1.0] * 20, dtype=float)
        result = compute_stability(ious)
        assert result.iou_std > 0.4


class TestIoURange:
    def test_constant_range_zero(self):
        ious = np.full(10, 0.5)
        result = compute_stability(ious)
        assert result.iou_range == pytest.approx(0.0)

    def test_full_range(self):
        ious = np.array([0.0, 0.5, 1.0])
        result = compute_stability(ious)
        assert result.iou_range == pytest.approx(1.0)


class TestFailureRate:
    def test_no_failures(self):
        ious = np.full(20, 0.5)
        result = compute_stability(ious, failure_threshold=0.1)
        assert result.failure_rate == pytest.approx(0.0)

    def test_all_failures(self):
        ious = np.zeros(20)
        result = compute_stability(ious, failure_threshold=0.1)
        assert result.failure_rate == pytest.approx(1.0)

    def test_half_failures(self):
        ious = np.array([0.0, 0.5] * 10, dtype=float)
        result = compute_stability(ious, failure_threshold=0.1)
        assert result.failure_rate == pytest.approx(0.5)

    def test_custom_threshold(self):
        ious = np.array([0.3, 0.3, 0.8, 0.8])
        result = compute_stability(ious, failure_threshold=0.5)
        assert result.failure_rate == pytest.approx(0.5)


class TestLongestFailureRun:
    def test_no_failures_longest_run_zero(self):
        ious = np.full(10, 0.8)
        result = compute_stability(ious)
        assert result.longest_failure_run == 0

    def test_single_failure_run_of_three(self):
        # Failures at indices 2, 3, 4 (run length = 3)
        ious = np.array([0.8, 0.8, 0.05, 0.05, 0.05, 0.8, 0.8])
        result = compute_stability(ious)
        assert result.longest_failure_run == 3

    def test_longest_run_selected_across_multiple(self):
        # Run of 2 then run of 4
        ious = np.array([0.05, 0.05, 0.9, 0.05, 0.05, 0.05, 0.05, 0.9])
        result = compute_stability(ious)
        assert result.longest_failure_run == 4

    def test_trailing_failure_run_counted(self):
        ious = np.array([0.9, 0.9, 0.05, 0.05, 0.05])
        result = compute_stability(ious)
        assert result.longest_failure_run == 3


class TestRecoveryRate:
    def test_no_failures_recovery_rate_is_nan(self):
        ious = np.full(10, 0.8)
        result = compute_stability(ious)
        assert math.isnan(result.recovery_rate)

    def test_perfect_recovery(self):
        # Single failure run followed by high IoU
        ious = np.array([0.8, 0.05, 0.05, 0.9, 0.9])
        result = compute_stability(ious, failure_threshold=0.1, recovery_threshold=0.5)
        assert result.recovery_rate == pytest.approx(1.0)

    def test_no_recovery(self):
        # Failure run followed by still-low IoU (< recovery_threshold)
        ious = np.array([0.8, 0.05, 0.05, 0.3, 0.3])
        result = compute_stability(ious, failure_threshold=0.1, recovery_threshold=0.5)
        assert result.recovery_rate == pytest.approx(0.0)

    def test_partial_recovery(self):
        # Two failure runs: one recovers, one does not
        ious = np.array([0.05, 0.05, 0.8, 0.05, 0.05, 0.3])
        result = compute_stability(ious, failure_threshold=0.1, recovery_threshold=0.5)
        assert result.recovery_rate == pytest.approx(0.5)


class TestTemporalAutocorr:
    def test_constant_series_autocorr_is_one(self):
        ious = np.full(20, 0.7)
        result = compute_stability(ious)
        assert result.temporal_autocorr == pytest.approx(1.0)

    def test_alternating_series_low_autocorr(self):
        ious = np.array([0.0, 1.0] * 20, dtype=float)
        result = compute_stability(ious)
        assert result.temporal_autocorr < 0.0  # alternating → negative autocorr

    def test_smooth_series_high_autocorr(self):
        # Slowly rising IoU is highly autocorrelated
        ious = np.linspace(0.0, 1.0, 50)
        result = compute_stability(ious)
        assert result.temporal_autocorr > 0.9


class TestToDict:
    def test_to_dict_keys(self):
        ious = np.linspace(0.3, 0.9, 20)
        result = compute_stability(ious)
        d = result.to_dict()
        for key in (
            "iou_std", "iou_range", "failure_rate",
            "longest_failure_run", "recovery_rate", "temporal_autocorr",
        ):
            assert key in d

    def test_to_dict_nan_becomes_none(self):
        ious = np.full(10, 0.8)  # no failures → recovery_rate = nan
        d = compute_stability(ious).to_dict()
        assert d["recovery_rate"] is None


class TestBenchmarkEngineWarmup:
    """Smoke tests verifying warmup_frames integrates with BenchmarkEngine."""

    def test_negative_warmup_raises(self):
        from eovot.benchmark.engine import BenchmarkEngine
        with pytest.raises(ValueError, match="warmup_frames"):
            BenchmarkEngine(warmup_frames=-1)

    def test_warmup_zero_is_default(self):
        from eovot.benchmark.engine import BenchmarkEngine
        engine = BenchmarkEngine()
        assert engine.warmup_frames == 0

    def test_warmup_stored(self):
        from eovot.benchmark.engine import BenchmarkEngine
        engine = BenchmarkEngine(warmup_frames=5)
        assert engine.warmup_frames == 5

    def test_warmup_does_not_change_result_count(self):
        """Running with warmup_frames > 0 must not alter the number of evaluated frames."""
        from typing import Iterator
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.base import BaseDataset, Sequence
        from eovot.trackers.base import BaseTracker

        NUM_FRAMES = 15
        GT_BOX = (10.0, 10.0, 40.0, 40.0)

        class _Tracker(BaseTracker):
            def __init__(self):
                super().__init__("WarmupTestTracker")
            def initialize(self, frame, bbox):
                pass
            def update(self, frame):
                return GT_BOX

        class _Seq(Sequence):
            def __init__(self):
                gt = np.tile(np.array(GT_BOX), (NUM_FRAMES, 1))
                super().__init__("seq", [f"{i}.jpg" for i in range(NUM_FRAMES)], gt)
                self._n = NUM_FRAMES
            def __iter__(self) -> Iterator[np.ndarray]:
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                for _ in range(self._n):
                    yield frame

        class _Dataset(BaseDataset):
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return _Seq()

        engine_no_warmup = BenchmarkEngine(verbose=False, warmup_frames=0)
        engine_warmup = BenchmarkEngine(verbose=False, warmup_frames=5)
        tracker = _Tracker()
        ds = _Dataset()

        r0 = engine_no_warmup.run(tracker, ds, dataset_name="T")
        r5 = engine_warmup.run(tracker, ds, dataset_name="T")

        assert len(r0.sequence_results[0].ious) == len(r5.sequence_results[0].ious)
        assert r0.mean_iou == pytest.approx(r5.mean_iou)
