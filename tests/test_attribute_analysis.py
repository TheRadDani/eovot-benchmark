"""Tests for the sequence attribute analysis engine (eovot.metrics.attributes)."""

from __future__ import annotations

from typing import Set

import numpy as np
import pytest

from eovot.metrics.attributes import (
    STANDARD_ATTRIBUTES,
    AttributeAnalyzer,
    AttributeAnnotations,
    AttributeResult,
    auto_annotate_from_gt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gt(
    n: int = 50,
    x: float = 100.0,
    y: float = 100.0,
    w: float = 40.0,
    h: float = 40.0,
    dx: float = 0.0,   # per-frame x displacement
    dy: float = 0.0,
    dw: float = 0.0,   # per-frame width change
    dh: float = 0.0,
) -> np.ndarray:
    """Build a synthetic (n, 4) GT array with linear motion / size change."""
    boxes = []
    for i in range(n):
        boxes.append([x + dx * i, y + dy * i, max(1.0, w + dw * i), max(1.0, h + dh * i)])
    return np.array(boxes, dtype=np.float64)


def _make_benchmark_result(
    tracker_name: str = "DummyTracker",
    dataset_name: str = "Synthetic",
    n_sequences: int = 5,
    n_frames: int = 50,
    mean_iou: float = 0.5,
    fps: float = 100.0,
):
    """Construct a minimal BenchmarkResult for testing."""
    from eovot.benchmark.engine import BenchmarkResult, SequenceResult
    from eovot.metrics.accuracy import AccuracyMetrics
    from eovot.profiling.profiler import ProfilingResult

    profiling = ProfilingResult(
        tracker_name=tracker_name,
        frame_count=n_frames,
        fps=fps,
        latency_mean_ms=1000.0 / fps,
        latency_std_ms=0.1,
        latency_p95_ms=1000.0 / fps * 1.2,
        peak_memory_mb=100.0,
    )
    accuracy = AccuracyMetrics(
        mean_iou=mean_iou,
        success_auc=mean_iou * 0.9,
        precision_auc=mean_iou * 0.85,
    )

    sequence_results = []
    for i in range(n_sequences):
        gt = _make_gt(n_frames)
        ious = np.full(n_frames, mean_iou)
        sr = SequenceResult(
            sequence_name=f"seq_{i:02d}",
            ious=ious,
            profiling=profiling,
            predictions=gt.copy(),
            ground_truths=gt,
            accuracy_metrics=accuracy,
        )
        sequence_results.append(sr)

    result = BenchmarkResult(
        tracker_name=tracker_name,
        dataset_name=dataset_name,
        sequence_results=sequence_results,
    )
    return result


# ---------------------------------------------------------------------------
# auto_annotate_from_gt
# ---------------------------------------------------------------------------

class TestAutoAnnotateFromGt:
    def test_static_target_returns_empty(self):
        gt = _make_gt(50, dx=0, dy=0)
        attrs = auto_annotate_from_gt(gt, frame_size=(640, 480))
        assert isinstance(attrs, set)
        # A static ~40x40 px target with no motion should have no attributes
        assert "fast_motion" not in attrs
        assert "out_of_view" not in attrs

    def test_fast_motion_detected(self):
        gt = _make_gt(50, dx=25.0, dy=0.0)  # 25 px/frame >> threshold of 20
        attrs = auto_annotate_from_gt(gt, frame_size=(1000, 1000))
        assert "fast_motion" in attrs

    def test_slow_motion_not_flagged(self):
        gt = _make_gt(50, dx=2.0, dy=0.0)   # 2 px/frame << threshold
        attrs = auto_annotate_from_gt(gt, frame_size=(640, 480))
        assert "fast_motion" not in attrs

    def test_scale_variation_detected(self):
        # Start at 10x10, grow to 50x50 → area ratio = 25 >> threshold of 4
        gt = _make_gt(50, w=10.0, h=10.0, dw=0.8, dh=0.8)
        attrs = auto_annotate_from_gt(gt, frame_size=(640, 480))
        assert "scale_variation" in attrs

    def test_no_scale_variation_for_constant_size(self):
        gt = _make_gt(50, w=40.0, h=40.0)
        attrs = auto_annotate_from_gt(gt, frame_size=(640, 480))
        assert "scale_variation" not in attrs

    def test_out_of_view_detected(self):
        # Move target off the right edge
        gt = _make_gt(50, x=600.0, dx=15.0)  # centre leaves 640 px boundary
        attrs = auto_annotate_from_gt(gt, frame_size=(640, 480))
        assert "out_of_view" in attrs

    def test_no_out_of_view_when_within_frame(self):
        gt = _make_gt(50, x=100.0, dx=1.0)  # stays well within 640 px
        attrs = auto_annotate_from_gt(gt, frame_size=(640, 480))
        assert "out_of_view" not in attrs

    def test_low_resolution_detected(self):
        gt = _make_gt(50, w=5.0, h=5.0)   # area = 25 px² << threshold 400
        attrs = auto_annotate_from_gt(gt, frame_size=(640, 480))
        assert "low_resolution" in attrs

    def test_no_low_resolution_large_target(self):
        gt = _make_gt(50, w=60.0, h=60.0)  # area = 3600 px² >> threshold
        attrs = auto_annotate_from_gt(gt, frame_size=(640, 480))
        assert "low_resolution" not in attrs

    def test_aspect_ratio_change_detected(self):
        # Oscillate between tall (10x80) and wide (80x10) boxes
        boxes = []
        for i in range(50):
            if i % 2 == 0:
                boxes.append([100.0, 100.0, 10.0, 80.0])
            else:
                boxes.append([100.0, 100.0, 80.0, 10.0])
        gt = np.array(boxes, dtype=np.float64)
        attrs = auto_annotate_from_gt(gt, frame_size=(640, 480))
        assert "aspect_ratio_change" in attrs

    def test_empty_gt_returns_empty_set(self):
        gt = np.empty((0, 4), dtype=np.float64)
        attrs = auto_annotate_from_gt(gt, frame_size=(640, 480))
        assert attrs == set()

    def test_single_frame_returns_empty_set(self):
        gt = _make_gt(1)
        attrs = auto_annotate_from_gt(gt, frame_size=(640, 480))
        assert attrs == set()

    def test_returns_subset_of_standard_attributes(self):
        gt = _make_gt(50, dx=5.0)
        attrs = auto_annotate_from_gt(gt, frame_size=(640, 480))
        for a in attrs:
            assert a in STANDARD_ATTRIBUTES, f"Unexpected attribute: {a!r}"

    def test_custom_fast_motion_threshold(self):
        gt = _make_gt(50, dx=15.0)  # 15 px/frame
        # With default threshold (20), not flagged
        assert "fast_motion" not in auto_annotate_from_gt(gt, fast_motion_px_per_frame=20.0)
        # With lower threshold (10), flagged
        assert "fast_motion" in auto_annotate_from_gt(gt, fast_motion_px_per_frame=10.0)


# ---------------------------------------------------------------------------
# STANDARD_ATTRIBUTES taxonomy
# ---------------------------------------------------------------------------

class TestStandardAttributes:
    def test_all_expected_attributes_present(self):
        expected = {
            "fast_motion", "scale_variation", "occlusion", "illumination_change",
            "aspect_ratio_change", "out_of_view", "low_resolution",
            "background_clutter", "motion_blur", "deformation",
        }
        assert expected.issubset(set(STANDARD_ATTRIBUTES.keys()))

    def test_all_descriptions_are_strings(self):
        for key, desc in STANDARD_ATTRIBUTES.items():
            assert isinstance(desc, str) and len(desc) > 0, (
                f"Attribute '{key}' has empty description."
            )


# ---------------------------------------------------------------------------
# AttributeAnalyzer.analyze
# ---------------------------------------------------------------------------

class TestAttributeAnalyzerAnalyze:
    def setup_method(self):
        self.analyzer = AttributeAnalyzer()
        self.result = _make_benchmark_result(n_sequences=5)
        self.annotations: AttributeAnnotations = {
            "seq_00": {"fast_motion", "scale_variation"},
            "seq_01": {"fast_motion"},
            "seq_02": {"low_resolution"},
            "seq_03": set(),
            "seq_04": {"scale_variation"},
        }

    def test_returns_list_of_attribute_results(self):
        breakdown = self.analyzer.analyze(self.result, self.annotations)
        assert isinstance(breakdown, list)
        for r in breakdown:
            assert isinstance(r, AttributeResult)

    def test_fast_motion_has_two_sequences(self):
        breakdown = self.analyzer.analyze(self.result, self.annotations)
        fm = next((r for r in breakdown if r.attribute == "fast_motion"), None)
        assert fm is not None
        assert fm.num_sequences == 2

    def test_scale_variation_has_two_sequences(self):
        breakdown = self.analyzer.analyze(self.result, self.annotations)
        sv = next((r for r in breakdown if r.attribute == "scale_variation"), None)
        assert sv is not None
        assert sv.num_sequences == 2

    def test_unannotated_sequence_is_skipped(self):
        # seq_03 has no attributes; it should not appear in any result
        breakdown = self.analyzer.analyze(self.result, self.annotations)
        for r in breakdown:
            assert "seq_03" not in r.sequences

    def test_missing_sequence_in_annotations_is_skipped(self):
        partial = {"seq_00": {"fast_motion"}}
        breakdown = self.analyzer.analyze(self.result, partial)
        fm = next((r for r in breakdown if r.attribute == "fast_motion"), None)
        assert fm is not None and fm.num_sequences == 1

    def test_attribute_with_zero_sequences_not_in_output(self):
        breakdown = self.analyzer.analyze(self.result, self.annotations)
        attrs_in_output = {r.attribute for r in breakdown}
        # "occlusion" is not in any annotation
        assert "occlusion" not in attrs_in_output

    def test_mean_iou_is_in_range(self):
        breakdown = self.analyzer.analyze(self.result, self.annotations)
        for r in breakdown:
            assert 0.0 <= r.mean_iou <= 1.0

    def test_tracker_name_propagated(self):
        breakdown = self.analyzer.analyze(self.result, self.annotations)
        for r in breakdown:
            assert r.tracker_name == self.result.tracker_name

    def test_sorted_by_attribute_name(self):
        breakdown = self.analyzer.analyze(self.result, self.annotations)
        names = [r.attribute for r in breakdown]
        assert names == sorted(names)

    def test_empty_annotations_returns_empty_list(self):
        breakdown = self.analyzer.analyze(self.result, {})
        assert breakdown == []

    def test_to_dict_structure(self):
        breakdown = self.analyzer.analyze(self.result, self.annotations)
        for r in breakdown:
            d = r.to_dict()
            assert "attribute" in d
            assert "mean_iou" in d
            assert "num_sequences" in d


# ---------------------------------------------------------------------------
# AttributeAnalyzer.compare_trackers
# ---------------------------------------------------------------------------

class TestCompareTrackers:
    def setup_method(self):
        self.analyzer = AttributeAnalyzer()
        self.mosse = _make_benchmark_result("MOSSE", mean_iou=0.35, n_sequences=4)
        self.kcf = _make_benchmark_result("KCF", mean_iou=0.50, n_sequences=4)
        self.annotations: AttributeAnnotations = {
            "seq_00": {"fast_motion"},
            "seq_01": {"scale_variation"},
            "seq_02": {"fast_motion", "low_resolution"},
            "seq_03": set(),
        }

    def test_returns_dict_keyed_by_tracker_name(self):
        multi = self.analyzer.compare_trackers([self.mosse, self.kcf], self.annotations)
        assert "MOSSE" in multi
        assert "KCF" in multi

    def test_each_value_is_list_of_attribute_results(self):
        multi = self.analyzer.compare_trackers([self.mosse, self.kcf], self.annotations)
        for tracker, breakdown in multi.items():
            assert isinstance(breakdown, list)
            for r in breakdown:
                assert isinstance(r, AttributeResult)

    def test_kcf_higher_iou_than_mosse(self):
        multi = self.analyzer.compare_trackers([self.mosse, self.kcf], self.annotations)
        mosse_fm = next((r for r in multi["MOSSE"] if r.attribute == "fast_motion"), None)
        kcf_fm = next((r for r in multi["KCF"] if r.attribute == "fast_motion"), None)
        if mosse_fm and kcf_fm:
            assert kcf_fm.mean_iou > mosse_fm.mean_iou


# ---------------------------------------------------------------------------
# Markdown table formatting
# ---------------------------------------------------------------------------

class TestMarkdownFormatting:
    def setup_method(self):
        self.analyzer = AttributeAnalyzer()
        self.result = _make_benchmark_result(n_sequences=3)
        self.annotations = {
            "seq_00": {"fast_motion"},
            "seq_01": {"scale_variation"},
            "seq_02": {"fast_motion"},
        }

    def test_to_markdown_table_is_string(self):
        breakdown = self.analyzer.analyze(self.result, self.annotations)
        table = self.analyzer.to_markdown_table(breakdown)
        assert isinstance(table, str)

    def test_to_markdown_table_contains_attribute_names(self):
        breakdown = self.analyzer.analyze(self.result, self.annotations)
        table = self.analyzer.to_markdown_table(breakdown)
        assert "fast_motion" in table
        assert "scale_variation" in table

    def test_to_markdown_table_empty_returns_fallback(self):
        table = self.analyzer.to_markdown_table([])
        assert "No attribute results" in table

    def test_to_comparison_table_contains_tracker_names(self):
        mosse = _make_benchmark_result("MOSSE", n_sequences=3)
        kcf = _make_benchmark_result("KCF", n_sequences=3)
        multi = self.analyzer.compare_trackers([mosse, kcf], self.annotations)
        table = self.analyzer.to_comparison_table(multi)
        assert "MOSSE" in table
        assert "KCF" in table

    def test_to_comparison_table_empty_returns_fallback(self):
        table = self.analyzer.to_comparison_table({})
        assert "No results" in table


# ---------------------------------------------------------------------------
# AttributeAnalyzer constructor validation
# ---------------------------------------------------------------------------

class TestAttributeAnalyzerInit:
    def test_default_attributes(self):
        analyzer = AttributeAnalyzer()
        assert set(analyzer.attributes) == set(STANDARD_ATTRIBUTES.keys())

    def test_custom_attribute_subset(self):
        analyzer = AttributeAnalyzer(attributes=["fast_motion", "occlusion"])
        assert analyzer.attributes == ["fast_motion", "occlusion"]

    def test_unknown_attribute_raises(self):
        with pytest.raises(ValueError, match="Unknown attribute"):
            AttributeAnalyzer(attributes=["nonexistent_challenge"])
