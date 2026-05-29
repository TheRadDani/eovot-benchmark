"""Tests for eovot.analysis — attribute detection and breakdown."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.analysis import (
    AttributeDetector,
    AttributeBreakdown,
    SequenceAttribute,
    TrackerAttributeComparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _linear_gt(n: int = 60, speed: float = 5.0) -> np.ndarray:
    """Ground-truth boxes for a linearly-moving object (no scale change)."""
    boxes = np.zeros((n, 4), dtype=np.float64)
    boxes[:, 0] = np.arange(n) * speed   # x increases
    boxes[:, 1] = 50.0
    boxes[:, 2] = 60.0   # constant width
    boxes[:, 3] = 40.0   # constant height
    return boxes


def _scaling_gt(n: int = 60) -> np.ndarray:
    """Ground-truth boxes that grow rapidly in area — triggers ScaleVariation."""
    boxes = np.zeros((n, 4), dtype=np.float64)
    boxes[:, 0] = 100.0
    boxes[:, 1] = 100.0
    # Large per-frame growth ensures area ratio > 0.25 threshold
    widths = 30.0 + np.arange(n) * 5.0
    boxes[:, 2] = widths
    boxes[:, 3] = widths * 0.75
    return boxes


def _tiny_gt(n: int = 30) -> np.ndarray:
    """Ground-truth boxes with area < 400 px² (low resolution)."""
    boxes = np.zeros((n, 4), dtype=np.float64)
    boxes[:, 0] = 200.0
    boxes[:, 1] = 200.0
    boxes[:, 2] = 15.0
    boxes[:, 3] = 15.0
    return boxes


# ---------------------------------------------------------------------------
# AttributeDetector tests
# ---------------------------------------------------------------------------

class TestAttributeDetector:
    def test_fast_motion_detected(self):
        # speed=50 should trigger FastMotion (normalised displacement > 0.20)
        gt = _linear_gt(n=30, speed=50.0)
        detector = AttributeDetector()
        profiles = detector.detect(gt)
        fm = profiles[SequenceAttribute.FAST_MOTION]
        # Most frames after the first should be flagged
        assert fm.per_frame_flags[1:].mean() > 0.5

    def test_fast_motion_not_triggered_slow(self):
        gt = _linear_gt(n=30, speed=1.0)
        profiles = AttributeDetector().detect(gt)
        fm = profiles[SequenceAttribute.FAST_MOTION]
        # Slow object should not trigger fast motion on most frames
        assert fm.per_frame_flags[1:].mean() < 0.5

    def test_scale_variation_detected(self):
        gt = _scaling_gt(n=60)
        profiles = AttributeDetector().detect(gt)
        sv = profiles[SequenceAttribute.SCALE_VARIATION]
        assert sv.num_frames > 0

    def test_low_resolution_detected(self):
        gt = _tiny_gt(n=30)
        profiles = AttributeDetector().detect(gt)
        lr = profiles[SequenceAttribute.LOW_RESOLUTION]
        # All tiny boxes should be flagged
        assert lr.num_frames == 30

    def test_low_resolution_not_triggered_large(self):
        gt = _linear_gt(n=20)   # 60×40 = 2400 px² — well above 400
        profiles = AttributeDetector().detect(gt)
        lr = profiles[SequenceAttribute.LOW_RESOLUTION]
        assert lr.num_frames == 0

    def test_out_of_view_without_frame_size(self):
        gt = _linear_gt(n=20)
        profiles = AttributeDetector().detect(gt)
        ov = profiles[SequenceAttribute.OUT_OF_VIEW]
        # No frame size → all-False
        assert ov.num_frames == 0

    def test_out_of_view_with_frame_size(self):
        # Object moves off-screen after frame 10
        gt = _linear_gt(n=20, speed=100.0)
        profiles = AttributeDetector(frame_size=(640, 480)).detect(gt)
        ov = profiles[SequenceAttribute.OUT_OF_VIEW]
        assert ov.num_frames > 0

    def test_short_sequence_single_frame(self):
        gt = np.array([[10.0, 20.0, 50.0, 40.0]])
        profiles = AttributeDetector().detect(gt)
        for attr, profile in profiles.items():
            assert len(profile.per_frame_flags) == 1

    def test_prevalence_property(self):
        gt = _tiny_gt(n=40)
        profiles = AttributeDetector().detect(gt)
        lr = profiles[SequenceAttribute.LOW_RESOLUTION]
        assert abs(lr.prevalence - 1.0) < 1e-6

    def test_all_attributes_returned(self):
        gt = _linear_gt(n=30)
        profiles = AttributeDetector().detect(gt)
        expected = set(SequenceAttribute)
        assert expected == set(profiles.keys())

    def test_flags_length_matches_gt(self):
        n = 47
        gt = _linear_gt(n=n)
        profiles = AttributeDetector().detect(gt)
        for attr, profile in profiles.items():
            assert len(profile.per_frame_flags) == n, f"{attr} flag length mismatch"


# ---------------------------------------------------------------------------
# AttributeBreakdown tests
# ---------------------------------------------------------------------------

class TestAttributeBreakdown:
    def test_compute_returns_overall(self):
        gt = _linear_gt(n=50)
        ious = np.random.default_rng(0).uniform(0.3, 0.9, 50)
        results = AttributeBreakdown().compute(gt, ious)
        assert "Overall" in results

    def test_compute_overall_mean_iou(self):
        gt = _linear_gt(n=50)
        ious = np.full(50, 0.6)
        results = AttributeBreakdown().compute(gt, ious)
        assert abs(results["Overall"].mean_iou - 0.6) < 1e-6

    def test_compute_success_auc_perfect(self):
        gt = _linear_gt(n=50)
        ious = np.ones(50)
        results = AttributeBreakdown().compute(gt, ious)
        # Perfect IoU → success rate = 1 at all thresholds up to 1.0 exclusive
        assert results["Overall"].success_auc > 0.95

    def test_compute_success_auc_zero(self):
        gt = _linear_gt(n=50)
        ious = np.zeros(50)
        results = AttributeBreakdown().compute(gt, ious)
        assert results["Overall"].success_auc < 0.05

    def test_attribute_frames_match(self):
        gt = _tiny_gt(n=30)
        ious = np.full(30, 0.5)
        results = AttributeBreakdown(min_frames=1).compute(gt, ious)
        lr = results.get("LowResolution")
        assert lr is not None
        assert lr.num_frames == 30

    def test_min_frames_filter(self):
        gt = _linear_gt(n=20, speed=0.5)   # slow → few FM frames
        ious = np.full(20, 0.5)
        # Set high min_frames threshold so FastMotion might be filtered
        results = AttributeBreakdown(min_frames=100).compute(gt, ious)
        # FastMotion with almost no flagged frames should be absent
        assert "FastMotion" not in results

    def test_compute_gt_ious_length_mismatch(self):
        gt = _linear_gt(n=50)
        ious = np.full(40, 0.7)   # shorter
        results = AttributeBreakdown().compute(gt, ious)
        # Should not raise; uses min(len(gt), len(ious))
        assert results["Overall"].num_frames == 40

    def test_breakdown_result_str(self):
        from eovot.analysis.breakdown import BreakdownResult
        r = BreakdownResult(attribute="FastMotion", num_frames=20, mean_iou=0.5, success_auc=0.45)
        s = str(r)
        assert "FastMotion" in s and "0.5" in s

    def test_from_benchmark_results_overall(self):
        """from_benchmark_results pools IoUs across sequences correctly."""
        from unittest.mock import MagicMock
        from eovot.benchmark.engine import SequenceResult, BenchmarkResult
        from eovot.profiling.profiler import ProfilingResult

        profiling = ProfilingResult(
            tracker_name="MOCK",
            frame_count=30,
            latency_mean_ms=1.0,
            latency_std_ms=0.1,
            latency_p95_ms=2.0,
            fps=100.0,
            peak_memory_mb=10.0,
        )

        gt = _linear_gt(n=30)
        ious = np.full(30, 0.7)
        seq_result = SequenceResult(
            sequence_name="seq1",
            ious=ious,
            profiling=profiling,
            ground_truths=gt,
        )

        bench_result = BenchmarkResult(
            tracker_name="MOSSE",
            dataset_name="Synthetic",
            sequence_results=[seq_result],
        )

        comparison = AttributeBreakdown().from_benchmark_results({"MOSSE": bench_result})
        assert "MOSSE" in comparison.tracker_breakdowns
        overall = comparison.tracker_breakdowns["MOSSE"].get("Overall")
        assert overall is not None
        assert abs(overall.mean_iou - 0.7) < 1e-6

    def test_from_benchmark_results_skips_missing_gt(self):
        """Sequences without ground_truths stored are silently skipped."""
        from eovot.benchmark.engine import SequenceResult, BenchmarkResult
        from eovot.profiling.profiler import ProfilingResult

        profiling = ProfilingResult(
            tracker_name="MOCK",
            frame_count=20,
            latency_mean_ms=1.0,
            latency_std_ms=0.1,
            latency_p95_ms=2.0,
            fps=100.0,
            peak_memory_mb=10.0,
        )
        seq_result = SequenceResult(
            sequence_name="seq_no_gt",
            ious=np.full(20, 0.5),
            profiling=profiling,
            ground_truths=None,  # no GT stored
        )
        bench = BenchmarkResult("T", "D", [seq_result])
        comparison = AttributeBreakdown().from_benchmark_results({"T": bench})
        # Overall key should be absent since no GT was provided
        assert "Overall" not in comparison.tracker_breakdowns.get("T", {})


# ---------------------------------------------------------------------------
# TrackerAttributeComparison tests
# ---------------------------------------------------------------------------

class TestTrackerAttributeComparison:
    def _make_comparison(self) -> TrackerAttributeComparison:
        from eovot.analysis.breakdown import BreakdownResult
        return TrackerAttributeComparison(
            tracker_breakdowns={
                "MOSSE": {
                    "Overall": BreakdownResult("Overall", 100, 0.40, 0.38),
                    "FastMotion": BreakdownResult("FastMotion", 30, 0.25, 0.22),
                },
                "KCF": {
                    "Overall": BreakdownResult("Overall", 100, 0.52, 0.50),
                    "FastMotion": BreakdownResult("FastMotion", 30, 0.38, 0.35),
                    "ScaleVariation": BreakdownResult("ScaleVariation", 15, 0.45, 0.42),
                },
            }
        )

    def test_to_markdown_contains_trackers(self):
        cmp = self._make_comparison()
        md = cmp.to_markdown()
        assert "MOSSE" in md
        assert "KCF" in md

    def test_to_markdown_contains_attributes(self):
        cmp = self._make_comparison()
        md = cmp.to_markdown()
        assert "Overall" in md
        assert "FastMotion" in md
        assert "ScaleVariation" in md

    def test_to_markdown_dash_for_missing(self):
        cmp = self._make_comparison()
        md = cmp.to_markdown()
        # MOSSE has no ScaleVariation entry — should render as "—"
        assert "—" in md

    def test_to_dict_structure(self):
        cmp = self._make_comparison()
        d = cmp.to_dict()
        assert "MOSSE" in d and "KCF" in d
        assert "Overall" in d["MOSSE"]
        assert d["KCF"]["FastMotion"]["success_auc"] == pytest.approx(0.35, abs=1e-3)

    def test_to_markdown_empty(self):
        cmp = TrackerAttributeComparison()
        md = cmp.to_markdown()
        assert md == "_No results._"
