"""Tests for GOT-10k protocol metrics (AO, SR@0.50, SR@0.75) and MOSSE OOF fix.

Covers:
- MetricsEngine.success_rate_at_threshold()
- AccuracyMetrics.sr_50 / sr_75 populated by compute_all()
- BenchmarkResult.mean_sr_50 / mean_sr_75 and JSON round-trip
- MOSSETracker._extract_patch() for fully out-of-frame boxes (crash fix)
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from eovot.metrics.accuracy import AccuracyMetrics, MetricsEngine
from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# MetricsEngine — success_rate_at_threshold
# ---------------------------------------------------------------------------

class TestSuccessRateAtThreshold:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_all_above_threshold(self):
        ious = np.array([0.8, 0.9, 0.75, 0.85])
        assert self.engine.success_rate_at_threshold(ious, 0.50) == pytest.approx(1.0)

    def test_all_below_threshold(self):
        ious = np.array([0.1, 0.2, 0.3])
        assert self.engine.success_rate_at_threshold(ious, 0.50) == pytest.approx(0.0)

    def test_half_above(self):
        ious = np.array([0.0, 0.0, 1.0, 1.0])
        assert self.engine.success_rate_at_threshold(ious, 0.50) == pytest.approx(0.5)

    def test_empty_array(self):
        assert self.engine.success_rate_at_threshold(np.array([]), 0.50) == 0.0

    def test_boundary_inclusive(self):
        # Both values exactly at their respective thresholds — should count as successes.
        ious_50 = np.array([0.50, 0.50, 0.50])
        assert self.engine.success_rate_at_threshold(ious_50, 0.50) == pytest.approx(1.0)
        ious_75 = np.array([0.75, 0.75])
        assert self.engine.success_rate_at_threshold(ious_75, 0.75) == pytest.approx(1.0)

    def test_sr_50_vs_sr_75_ordering(self):
        ious = np.array([0.6, 0.6, 0.6, 0.8])
        sr50 = self.engine.success_rate_at_threshold(ious, 0.50)
        sr75 = self.engine.success_rate_at_threshold(ious, 0.75)
        assert sr50 >= sr75, "SR@0.50 must be >= SR@0.75 for the same IoU array"


# ---------------------------------------------------------------------------
# AccuracyMetrics — sr_50 / sr_75 from compute_all
# ---------------------------------------------------------------------------

class TestComputeAllSR:
    def setup_method(self):
        self.engine = MetricsEngine()

    def _make_boxes(self, n: int, iou_target: float = 0.8) -> tuple:
        """Create synthetic pred/gt pairs with approximate target IoU."""
        gt = np.tile([100.0, 100.0, 80.0, 60.0], (n, 1))
        # Shift preds to get the desired IoU approximately
        overlap_w = 80.0 * iou_target ** 0.5
        overlap_h = 60.0 * iou_target ** 0.5
        shift = (80.0 - overlap_w) / 2.0
        preds = gt.copy()
        preds[:, 0] += shift
        preds[:, 1] += shift
        return preds, gt

    def test_sr_fields_populated(self):
        preds, gt = self._make_boxes(50, iou_target=0.8)
        result = self.engine.compute_all(preds, gt)
        assert 0.0 <= result.sr_50 <= 1.0
        assert 0.0 <= result.sr_75 <= 1.0

    def test_sr_50_geq_sr_75(self):
        preds, gt = self._make_boxes(100, iou_target=0.6)
        result = self.engine.compute_all(preds, gt)
        assert result.sr_50 >= result.sr_75

    def test_perfect_prediction_full_sr(self):
        gt = np.tile([50.0, 50.0, 100.0, 80.0], (30, 1))
        result = self.engine.compute_all(gt.copy(), gt)
        assert result.sr_50 == pytest.approx(1.0)
        assert result.sr_75 == pytest.approx(1.0)
        assert result.mean_iou == pytest.approx(1.0)

    def test_zero_iou_zero_sr(self):
        gt = np.tile([0.0, 0.0, 50.0, 50.0], (20, 1))
        preds = np.tile([200.0, 200.0, 50.0, 50.0], (20, 1))
        result = self.engine.compute_all(preds, gt)
        assert result.sr_50 == pytest.approx(0.0)
        assert result.sr_75 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BenchmarkResult — aggregate SR properties and JSON round-trip
# ---------------------------------------------------------------------------

def _make_seq_result(name: str, sr_50: float = 0.7, sr_75: float = 0.4) -> SequenceResult:
    ious = np.array([0.8, 0.6, 0.5, 0.3, 0.9])
    profiling = ProfilingResult(
        tracker_name="T",
        frame_count=5,
        fps=30.0,
        latency_mean_ms=33.3,
        latency_std_ms=1.0,
        latency_p95_ms=35.0,
        latency_p99_ms=36.0,
        latency_cv=0.03,
        peak_memory_mb=50.0,
    )
    accuracy = AccuracyMetrics(
        mean_iou=0.62,
        success_auc=0.55,
        precision_auc=0.70,
        normalized_precision_auc=0.65,
        sr_50=sr_50,
        sr_75=sr_75,
    )
    return SequenceResult(
        sequence_name=name,
        ious=ious,
        profiling=profiling,
        accuracy_metrics=accuracy,
    )


class TestBenchmarkResultSR:
    def test_mean_sr_50(self):
        result = BenchmarkResult(tracker_name="T", dataset_name="D")
        result.sequence_results = [
            _make_seq_result("s1", sr_50=0.6, sr_75=0.3),
            _make_seq_result("s2", sr_50=0.8, sr_75=0.5),
        ]
        assert result.mean_sr_50 == pytest.approx(0.7)
        assert result.mean_sr_75 == pytest.approx(0.4)

    def test_sr_in_summary(self):
        result = BenchmarkResult(tracker_name="T", dataset_name="D")
        result.sequence_results = [_make_seq_result("s1", 0.7, 0.4)]
        s = result.summary()
        assert "sr_50" in s
        assert "sr_75" in s
        assert s["sr_50"] == pytest.approx(0.7, abs=1e-3)
        assert s["sr_75"] == pytest.approx(0.4, abs=1e-3)

    def test_json_round_trip(self, tmp_path):
        result = BenchmarkResult(tracker_name="T", dataset_name="D")
        result.sequence_results = [
            _make_seq_result("s1", 0.72, 0.38),
            _make_seq_result("s2", 0.65, 0.50),
        ]
        path = result.save(tmp_path / "result.json")
        loaded = BenchmarkResult.load(path)
        assert loaded.mean_sr_50 == pytest.approx(result.mean_sr_50, abs=1e-3)
        assert loaded.mean_sr_75 == pytest.approx(result.mean_sr_75, abs=1e-3)

    def test_no_accuracy_metrics_returns_none(self):
        ious = np.array([0.5, 0.6])
        profiling = ProfilingResult(
            tracker_name="T", frame_count=2, fps=30.0,
            latency_mean_ms=33.0, latency_std_ms=0.5,
            latency_p95_ms=34.0, latency_p99_ms=34.5,
            latency_cv=0.015, peak_memory_mb=40.0,
        )
        result = BenchmarkResult(tracker_name="T", dataset_name="D")
        result.sequence_results = [
            SequenceResult(sequence_name="s1", ious=ious, profiling=profiling)
        ]
        assert result.mean_sr_50 is None
        assert result.mean_sr_75 is None


# ---------------------------------------------------------------------------
# MOSSE — out-of-frame crash fix
# ---------------------------------------------------------------------------

class TestMOSSEOutOfFrame:
    def test_extract_patch_fully_out_right(self):
        from eovot.trackers.mosse import MOSSETracker
        tracker = MOSSETracker()
        gray = np.zeros((100, 100), dtype=np.uint8)
        patch = tracker._extract_patch(gray, x=200, y=10, w=40, h=30)
        assert patch.shape == (30, 40), "Should return a black patch, not crash"
        assert patch.sum() == 0

    def test_extract_patch_fully_out_bottom(self):
        from eovot.trackers.mosse import MOSSETracker
        tracker = MOSSETracker()
        gray = np.zeros((100, 100), dtype=np.uint8)
        patch = tracker._extract_patch(gray, x=10, y=200, w=40, h=30)
        assert patch.shape == (30, 40)
        assert patch.sum() == 0

    def test_extract_patch_fully_out_left(self):
        from eovot.trackers.mosse import MOSSETracker
        tracker = MOSSETracker()
        gray = np.zeros((100, 100), dtype=np.uint8)
        patch = tracker._extract_patch(gray, x=-60, y=10, w=40, h=30)
        assert patch.shape == (30, 40)
        assert patch.sum() == 0

    def test_extract_patch_fully_out_top(self):
        from eovot.trackers.mosse import MOSSETracker
        tracker = MOSSETracker()
        gray = np.zeros((100, 100), dtype=np.uint8)
        patch = tracker._extract_patch(gray, x=10, y=-50, w=40, h=30)
        assert patch.shape == (30, 40)
        assert patch.sum() == 0

    def test_update_does_not_crash_when_oof(self):
        """Full update() call with a frame where the tracker has drifted OOF."""
        import cv2
        from eovot.trackers.mosse import MOSSETracker

        frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        tracker = MOSSETracker()
        tracker.initialize(frame, (20, 20, 40, 30))

        # Force internal bbox to be fully outside the frame
        tracker._bbox = [500, 500, 40, 30]
        # This should no longer raise cv2 assertion
        bbox = tracker.update(frame)
        assert len(bbox) == 4
