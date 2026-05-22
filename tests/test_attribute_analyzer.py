"""Tests for eovot.analysis.attribute_analyzer."""

import numpy as np
import pytest

from eovot.analysis.attribute_analyzer import (
    AttributeAnalyzer,
    AttributeBreakdown,
    SequenceAttributes,
)
from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.datasets.base import Sequence
from eovot.datasets.synthetic import SyntheticDataset
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profiling() -> ProfilingResult:
    return ProfilingResult(
        tracker_name="test",
        frame_count=10,
        fps=100.0,
        latency_mean_ms=10.0,
        latency_std_ms=1.0,
        latency_p95_ms=12.0,
        peak_memory_mb=50.0,
    )


def _make_sequence(name: str, gt: list) -> Sequence:
    """Build a Sequence from a list of (x, y, w, h) tuples."""
    return Sequence(
        name=name,
        frame_paths=["<mem>"] * len(gt),
        ground_truth=np.array(gt, dtype=np.float64),
    )


def _make_benchmark_result(tracker_name: str, seq_ious: dict) -> BenchmarkResult:
    """Build a minimal BenchmarkResult from a {seq_name: [iou, ...]} dict."""
    br = BenchmarkResult(tracker_name=tracker_name, dataset_name="Test")
    for seq_name, ious in seq_ious.items():
        sr = SequenceResult(
            sequence_name=seq_name,
            ious=np.array(ious, dtype=np.float64),
            profiling=_make_profiling(),
        )
        br.sequence_results.append(sr)
    return br


# ---------------------------------------------------------------------------
# SequenceAttributes
# ---------------------------------------------------------------------------

class TestSequenceAttributesActive:
    def test_no_flags_active(self):
        attrs = SequenceAttributes(name="s")
        assert attrs.active() == []

    def test_single_flag_active(self):
        attrs = SequenceAttributes(name="s", fast_motion=True)
        assert "fast_motion" in attrs.active()

    def test_multiple_flags_active(self):
        attrs = SequenceAttributes(name="s", fast_motion=True, low_resolution=True)
        assert set(attrs.active()) == {"fast_motion", "low_resolution"}


# ---------------------------------------------------------------------------
# AttributeAnalyzer.tag_sequence — individual attribute detection
# ---------------------------------------------------------------------------

class TestFastMotionDetection:
    def test_large_displacement_flagged(self):
        # Box is 40×40 (diagonal ≈ 56.6 px); displacement = 100 px → ratio ≈ 1.77 > 0.20
        gt = [(0.0, 0.0, 40.0, 40.0), (100.0, 100.0, 40.0, 40.0)]
        seq = _make_sequence("fast", gt)
        assert AttributeAnalyzer().tag_sequence(seq).fast_motion is True

    def test_small_displacement_not_flagged(self):
        # Displacement = 1 px → well below 20% of any reasonable diagonal
        gt = [(50.0, 50.0, 40.0, 40.0), (51.0, 50.0, 40.0, 40.0)]
        seq = _make_sequence("slow", gt)
        assert AttributeAnalyzer().tag_sequence(seq).fast_motion is False

    def test_custom_threshold(self):
        # At threshold=0.01 even 1 px displacement should trigger fast_motion
        gt = [(50.0, 50.0, 40.0, 40.0), (51.0, 50.0, 40.0, 40.0)]
        seq = _make_sequence("s", gt)
        assert AttributeAnalyzer(fast_motion_threshold=0.01).tag_sequence(seq).fast_motion is True


class TestScaleVariationDetection:
    def test_large_scale_change_flagged(self):
        # Area: 2×2=4 vs 40×40=1600 → ratio = 400 > 4
        gt = [(0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 40.0, 40.0)]
        seq = _make_sequence("scale", gt)
        assert AttributeAnalyzer().tag_sequence(seq).scale_variation is True

    def test_constant_size_not_flagged(self):
        gt = [(0.0, 0.0, 40.0, 40.0)] * 5
        seq = _make_sequence("const", gt)
        assert AttributeAnalyzer().tag_sequence(seq).scale_variation is False


class TestAspectRatioChangeDetection:
    def test_large_ar_change_flagged(self):
        # AR: 1/10 = 0.1 vs 10/1 = 10 → ratio = 100 > 2
        gt = [(0.0, 0.0, 1.0, 10.0), (0.0, 0.0, 10.0, 1.0)]
        seq = _make_sequence("ar", gt)
        assert AttributeAnalyzer().tag_sequence(seq).aspect_ratio_change is True

    def test_stable_ar_not_flagged(self):
        gt = [(0.0, 0.0, 40.0, 40.0)] * 5
        seq = _make_sequence("stable", gt)
        assert AttributeAnalyzer().tag_sequence(seq).aspect_ratio_change is False


class TestLowResolutionDetection:
    def test_tiny_box_flagged(self):
        # 10×10 = 100 px² < 400 threshold
        gt = [(0.0, 0.0, 10.0, 10.0)] * 3
        seq = _make_sequence("tiny", gt)
        assert AttributeAnalyzer().tag_sequence(seq).low_resolution is True

    def test_normal_box_not_flagged(self):
        gt = [(0.0, 0.0, 40.0, 40.0)] * 3
        seq = _make_sequence("normal", gt)
        assert AttributeAnalyzer().tag_sequence(seq).low_resolution is False


class TestOutOfViewDetection:
    def test_box_outside_frame_flagged(self):
        # Box x+w = 320+40 = 360 > 320 → out of view
        gt = [(300.0, 200.0, 40.0, 40.0)] * 2
        seq = _make_sequence("oov", gt)
        assert AttributeAnalyzer().tag_sequence(seq, frame_size=(320, 240)).out_of_view is True

    def test_box_inside_frame_not_flagged(self):
        gt = [(10.0, 10.0, 40.0, 40.0)] * 2
        seq = _make_sequence("in", gt)
        assert AttributeAnalyzer().tag_sequence(seq, frame_size=(320, 240)).out_of_view is False

    def test_skipped_without_frame_size(self):
        gt = [(300.0, 220.0, 40.0, 40.0)] * 2
        seq = _make_sequence("oov_nosize", gt)
        assert AttributeAnalyzer().tag_sequence(seq).out_of_view is False

    def test_negative_x_flagged(self):
        gt = [(-5.0, 10.0, 40.0, 40.0)] * 2
        seq = _make_sequence("neg", gt)
        assert AttributeAnalyzer().tag_sequence(seq, frame_size=(320, 240)).out_of_view is True


class TestMotionBlurDetection:
    def test_high_velocity_flagged(self):
        # 100 px displacement > 15 px threshold
        gt = [(0.0, 0.0, 40.0, 40.0), (100.0, 0.0, 40.0, 40.0)]
        seq = _make_sequence("blur", gt)
        assert AttributeAnalyzer().tag_sequence(seq).motion_blur is True

    def test_low_velocity_not_flagged(self):
        gt = [(0.0, 0.0, 40.0, 40.0), (1.0, 0.0, 40.0, 40.0)]
        seq = _make_sequence("noblur", gt)
        assert AttributeAnalyzer().tag_sequence(seq).motion_blur is False


class TestSingleFrameEdgeCase:
    def test_single_frame_returns_all_false(self):
        gt = [(10.0, 10.0, 40.0, 40.0)]
        seq = _make_sequence("single", gt)
        attrs = AttributeAnalyzer().tag_sequence(seq)
        assert attrs.active() == []


# ---------------------------------------------------------------------------
# tag_dataset
# ---------------------------------------------------------------------------

class TestTagDataset:
    def test_returns_one_tag_per_sequence(self):
        ds = SyntheticDataset(num_sequences=4, num_frames=15, seed=0)
        tags = AttributeAnalyzer().tag_dataset(ds)
        assert len(tags) == 4

    def test_tag_types(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=10)
        tags = AttributeAnalyzer().tag_dataset(ds)
        for t in tags:
            assert isinstance(t, SequenceAttributes)

    def test_names_match_dataset(self):
        ds = SyntheticDataset(num_sequences=3, num_frames=10)
        tags = AttributeAnalyzer().tag_dataset(ds)
        ds_names = [ds[i].name for i in range(len(ds))]
        assert [t.name for t in tags] == ds_names

    def test_with_frame_size(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=10)
        tags = AttributeAnalyzer().tag_dataset(ds, frame_size=(320, 240))
        assert len(tags) == 2


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------

class TestBreakdown:
    def test_overall_iou_computation(self):
        result = _make_benchmark_result("T", {"s1": [0.8, 0.8], "s2": [0.4, 0.4]})
        tags = [
            SequenceAttributes(name="s1"),
            SequenceAttributes(name="s2"),
        ]
        bd = AttributeAnalyzer().breakdown(result, tags)
        assert abs(bd.overall_iou - 0.6) < 1e-6

    def test_per_attribute_mean_iou(self):
        result = _make_benchmark_result("T", {"s1": [0.7], "s2": [0.3]})
        tags = [
            SequenceAttributes(name="s1", fast_motion=True),
            SequenceAttributes(name="s2", fast_motion=False),
        ]
        bd = AttributeAnalyzer().breakdown(result, tags)
        assert "fast_motion" in bd.attribute_iou
        miou, n = bd.attribute_iou["fast_motion"]
        assert abs(miou - 0.7) < 1e-6
        assert n == 1

    def test_attribute_count(self):
        result = _make_benchmark_result("T", {"s1": [0.5], "s2": [0.5]})
        tags = [
            SequenceAttributes(name="s1", fast_motion=True),
            SequenceAttributes(name="s2", fast_motion=True),
        ]
        bd = AttributeAnalyzer().breakdown(result, tags)
        assert bd.attribute_iou["fast_motion"][1] == 2

    def test_unmatched_sequences_skipped(self):
        result = _make_benchmark_result("T", {"unknown_seq": [0.5]})
        tags = [SequenceAttributes(name="other")]
        bd = AttributeAnalyzer().breakdown(result, tags)
        assert bd.overall_sequences == 1
        assert bd.attribute_iou == {}

    def test_overall_sequences_count(self):
        result = _make_benchmark_result("T", {"s1": [0.9], "s2": [0.1], "s3": [0.5]})
        tags = []
        bd = AttributeAnalyzer().breakdown(result, tags)
        assert bd.overall_sequences == 3


# ---------------------------------------------------------------------------
# report_table
# ---------------------------------------------------------------------------

class TestReportTable:
    def _simple_breakdown(self, tracker: str) -> AttributeBreakdown:
        result = _make_benchmark_result(tracker, {"s1": [0.8]})
        tags = [SequenceAttributes(name="s1", fast_motion=True)]
        return AttributeAnalyzer().breakdown(result, tags)

    def test_output_is_string(self):
        bd = self._simple_breakdown("MOSSE")
        table = AttributeAnalyzer().report_table({"MOSSE": bd})
        assert isinstance(table, str)

    def test_contains_tracker_name(self):
        bd = self._simple_breakdown("KCF")
        table = AttributeAnalyzer().report_table({"KCF": bd})
        assert "KCF" in table

    def test_contains_attribute_label(self):
        bd = self._simple_breakdown("MOSSE")
        table = AttributeAnalyzer().report_table({"MOSSE": bd})
        assert "Fast Motion" in table

    def test_markdown_pipe_separators(self):
        bd = self._simple_breakdown("T")
        table = AttributeAnalyzer().report_table({"T": bd})
        assert "|" in table

    def test_multiple_trackers(self):
        bd1 = self._simple_breakdown("MOSSE")
        bd2 = self._simple_breakdown("KCF")
        table = AttributeAnalyzer().report_table({"MOSSE": bd1, "KCF": bd2})
        assert "MOSSE" in table and "KCF" in table

    def test_missing_attribute_shown_as_dash(self):
        # No fast_motion sequences → attribute_iou is empty
        result = _make_benchmark_result("T", {"s1": [0.5]})
        tags = [SequenceAttributes(name="s1")]  # all flags False
        bd = AttributeAnalyzer().breakdown(result, tags)
        table = AttributeAnalyzer().report_table({"T": bd})
        assert "—" in table


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------

class TestEndToEndIntegration:
    def test_full_pipeline_with_synthetic(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker

        ds = SyntheticDataset(num_sequences=3, num_frames=20, motion="linear", seed=42)
        analyzer = AttributeAnalyzer()
        tags = analyzer.tag_dataset(ds, frame_size=(320, 240))

        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")

        bd = analyzer.breakdown(result, tags)
        table = analyzer.report_table({"MOSSE": bd})

        assert bd.overall_sequences == 3
        assert bd.overall_iou >= 0.0
        assert "MOSSE" in table
