"""Tests for sequence attribute tagging and per-attribute analysis."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.attributes import (
    ATTRIBUTE_CODES,
    ATTRIBUTE_DISPLAY_NAMES,
    DEFAULT_THRESHOLDS,
    AttributeTagger,
    SequenceAttributes,
    TrackingAttribute,
)
from eovot.metrics.attribute_analysis import (
    AttributeAnalyzer,
    AttributeAnalysisReport,
    AttributeSliceResult,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic GT trajectories
# ---------------------------------------------------------------------------


def _make_bbox_array(cx, cy, w=40, h=30):
    """Build an (N, 4) bbox array in (x, y, w, h) format from centre lists."""
    cx = np.asarray(cx, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)
    x = cx - w / 2
    y = cy - h / 2
    return np.column_stack([x, y, np.full_like(cx, w), np.full_like(cy, h)])


@pytest.fixture()
def slow_bboxes():
    """Barely-moving object — no fast-motion attribute."""
    N = 50
    cx = np.linspace(100, 110, N)  # 10px over 50 frames; diagonal≈50
    cy = np.full(N, 200.0)
    return _make_bbox_array(cx, cy, w=40, h=30)


@pytest.fixture()
def fast_bboxes():
    """Fast-moving object — displacement >> 20% of diagonal."""
    N = 50
    cx = np.linspace(50, 550, N)  # 500px over 50 frames; diagonal≈50
    cy = np.linspace(100, 400, N)
    return _make_bbox_array(cx, cy, w=40, h=30)


@pytest.fixture()
def scale_change_bboxes():
    """Object that grows by 5× over the sequence."""
    N = 50
    cx = np.full(N, 200.0)
    cy = np.full(N, 200.0)
    w = np.linspace(10, 150, N)
    h = np.linspace(10, 150, N)
    x = cx - w / 2
    y = cy - h / 2
    return np.column_stack([x, y, w, h])


@pytest.fixture()
def tiny_bboxes():
    """Very small object — triggers low_resolution and small_object."""
    N = 30
    cx = np.full(N, 100.0)
    cy = np.full(N, 100.0)
    return _make_bbox_array(cx, cy, w=10, h=8)  # area = 80 px²


@pytest.fixture()
def tagger():
    return AttributeTagger()


# ---------------------------------------------------------------------------
# AttributeTagger.tag() — input validation
# ---------------------------------------------------------------------------


def test_tag_wrong_shape_raises(tagger):
    with pytest.raises(ValueError, match="shape"):
        tagger.tag(np.zeros((10, 3)), "seq")


def test_tag_single_frame(tagger):
    bboxes = np.array([[100, 100, 40, 30]])
    attrs = tagger.tag(bboxes, "single")
    assert isinstance(attrs, SequenceAttributes)


def test_tag_returns_all_attributes(tagger, slow_bboxes):
    attrs = tagger.tag(slow_bboxes, "s1")
    for attr in TrackingAttribute:
        assert attr in attrs.tags


# ---------------------------------------------------------------------------
# AttributeTagger — fast motion detection
# ---------------------------------------------------------------------------


def test_slow_motion_not_fast(tagger, slow_bboxes):
    attrs = tagger.tag(slow_bboxes, "slow", frame_size=(640, 480))
    assert attrs.tags[TrackingAttribute.FAST_MOTION] is False


def test_fast_motion_detected(tagger, fast_bboxes):
    attrs = tagger.tag(fast_bboxes, "fast", frame_size=(640, 480))
    assert attrs.tags[TrackingAttribute.FAST_MOTION] is True


# ---------------------------------------------------------------------------
# AttributeTagger — scale change detection
# ---------------------------------------------------------------------------


def test_no_scale_change_for_constant_size(tagger, slow_bboxes):
    attrs = tagger.tag(slow_bboxes, "const")
    assert attrs.tags[TrackingAttribute.SCALE_CHANGE] is False


def test_scale_change_detected(tagger, scale_change_bboxes):
    attrs = tagger.tag(scale_change_bboxes, "scale")
    assert attrs.tags[TrackingAttribute.SCALE_CHANGE] is True


# ---------------------------------------------------------------------------
# AttributeTagger — low resolution / small object
# ---------------------------------------------------------------------------


def test_tiny_triggers_low_resolution(tagger, tiny_bboxes):
    attrs = tagger.tag(tiny_bboxes, "tiny")
    assert attrs.tags[TrackingAttribute.LOW_RESOLUTION] is True


def test_tiny_triggers_small_object(tagger, tiny_bboxes):
    attrs = tagger.tag(tiny_bboxes, "tiny")
    assert attrs.tags[TrackingAttribute.SMALL_OBJECT] is True


def test_normal_size_not_low_resolution(tagger, slow_bboxes):
    attrs = tagger.tag(slow_bboxes, "normal")
    assert attrs.tags[TrackingAttribute.LOW_RESOLUTION] is False


# ---------------------------------------------------------------------------
# AttributeTagger — out-of-view detection
# ---------------------------------------------------------------------------


def test_out_of_view_near_edge():
    tagger = AttributeTagger()
    N = 20
    # Object centre at x=5 (very near left edge of 640-wide frame)
    bboxes = _make_bbox_array(np.full(N, 5.0), np.full(N, 200.0), w=20, h=20)
    attrs = tagger.tag(bboxes, "oov", frame_size=(640, 480))
    assert attrs.tags[TrackingAttribute.OUT_OF_VIEW] is True


def test_not_out_of_view_centre():
    tagger = AttributeTagger()
    N = 20
    bboxes = _make_bbox_array(np.full(N, 320.0), np.full(N, 240.0), w=40, h=30)
    attrs = tagger.tag(bboxes, "centre", frame_size=(640, 480))
    assert attrs.tags[TrackingAttribute.OUT_OF_VIEW] is False


def test_out_of_view_false_when_no_frame_size(tagger, fast_bboxes):
    attrs = tagger.tag(fast_bboxes, "nofs", frame_size=None)
    assert attrs.tags[TrackingAttribute.OUT_OF_VIEW] is False


# ---------------------------------------------------------------------------
# AttributeTagger — partial occlusion (abrupt area drop)
# ---------------------------------------------------------------------------


def test_partial_occlusion_detected():
    tagger = AttributeTagger()
    N = 20
    # Sudden area halving at frame 10 → triggers occlusion
    w = np.concatenate([np.full(10, 80.0), np.full(10, 10.0)])  # drops by 87%
    h = np.full(N, 60.0)
    x = np.full(N, 100.0)
    y = np.full(N, 100.0)
    bboxes = np.column_stack([x, y, w, h])
    attrs = tagger.tag(bboxes, "occ")
    assert attrs.tags[TrackingAttribute.PARTIAL_OCCLUSION] is True


def test_no_occlusion_for_stable_size(tagger, slow_bboxes):
    attrs = tagger.tag(slow_bboxes, "stable")
    assert attrs.tags[TrackingAttribute.PARTIAL_OCCLUSION] is False


# ---------------------------------------------------------------------------
# AttributeTagger — degenerate bbox filtering
# ---------------------------------------------------------------------------


def test_zero_area_boxes_filtered(tagger):
    bboxes = np.array([
        [100, 100, 0, 0],   # zero area — filtered
        [100, 100, 40, 30],
        [105, 100, 40, 30],
    ])
    attrs = tagger.tag(bboxes, "zeros")
    assert attrs.n_frames == 2


# ---------------------------------------------------------------------------
# SequenceAttributes helpers
# ---------------------------------------------------------------------------


def test_present_lists_active_attrs(tagger, fast_bboxes):
    attrs = tagger.tag(fast_bboxes, "fast")
    assert TrackingAttribute.FAST_MOTION in attrs.present


def test_absent_lists_inactive_attrs(tagger, slow_bboxes):
    attrs = tagger.tag(slow_bboxes, "slow")
    assert TrackingAttribute.FAST_MOTION in attrs.absent


def test_repr_contains_name(tagger, fast_bboxes):
    attrs = tagger.tag(fast_bboxes, "myseq")
    assert "myseq" in repr(attrs)


# ---------------------------------------------------------------------------
# AttributeTagger.tag_dataset()
# ---------------------------------------------------------------------------


def test_tag_dataset_returns_all_sequences(tagger, slow_bboxes, fast_bboxes):
    seqs = {"slow": slow_bboxes, "fast": fast_bboxes}
    tagged = tagger.tag_dataset(seqs)
    assert set(tagged.keys()) == {"slow", "fast"}


def test_tag_dataset_with_frame_sizes(tagger, fast_bboxes, slow_bboxes):
    seqs = {"fast": fast_bboxes, "slow": slow_bboxes}
    sizes = {"fast": (640, 480), "slow": (640, 480)}
    tagged = tagger.tag_dataset(seqs, frame_sizes=sizes)
    assert tagged["fast"].frame_size == (640, 480)


# ---------------------------------------------------------------------------
# AttributeTagger.attribute_coverage()
# ---------------------------------------------------------------------------


def test_attribute_coverage_between_0_and_1(tagger, slow_bboxes, fast_bboxes):
    seqs = {"slow": slow_bboxes, "fast": fast_bboxes}
    tagged = tagger.tag_dataset(seqs)
    coverage = tagger.attribute_coverage(tagged)
    for val in coverage.values():
        assert 0.0 <= val <= 1.0


def test_attribute_coverage_fast_motion_fraction(tagger, slow_bboxes, fast_bboxes):
    seqs = {"slow": slow_bboxes, "fast": fast_bboxes}
    tagged = tagger.tag_dataset(seqs)
    coverage = tagger.attribute_coverage(tagged)
    # 1 out of 2 sequences has fast motion → 0.5
    assert coverage[TrackingAttribute.FAST_MOTION] == pytest.approx(0.5)


def test_attribute_coverage_empty_returns_empty(tagger):
    assert tagger.attribute_coverage({}) == {}


# ---------------------------------------------------------------------------
# AttributeTagger.coverage_to_markdown()
# ---------------------------------------------------------------------------


def test_coverage_markdown_contains_header(tagger, slow_bboxes, fast_bboxes):
    seqs = {"s1": slow_bboxes, "s2": fast_bboxes}
    tagged = tagger.tag_dataset(seqs)
    md = tagger.coverage_to_markdown(tagged)
    assert "Attribute" in md
    assert "Coverage" in md


# ---------------------------------------------------------------------------
# AttributeAnalyzer — integration with a mock BenchmarkResult
# ---------------------------------------------------------------------------


class _MockSeqResult:
    def __init__(self, name, ious):
        self.sequence_name = name
        self.ious = np.asarray(ious, dtype=np.float64)
        self.center_distances = None


class _MockBenchmarkResult:
    def __init__(self, tracker_name, dataset_name, seq_results):
        self.tracker_name = tracker_name
        self.dataset_name = dataset_name
        self.sequence_results = seq_results


@pytest.fixture()
def mock_result(slow_bboxes, fast_bboxes):
    tagger = AttributeTagger()
    tagged = tagger.tag_dataset({"slow": slow_bboxes, "fast": fast_bboxes})
    seq_results = [
        _MockSeqResult("slow", np.full(50, 0.7)),
        _MockSeqResult("fast", np.full(50, 0.4)),
    ]
    result = _MockBenchmarkResult("MOSSE", "Synthetic", seq_results)
    return tagged, result


def test_analyze_returns_report(mock_result):
    tagged, result = mock_result
    analyzer = AttributeAnalyzer(tagged)
    report = analyzer.analyze(result)
    assert isinstance(report, AttributeAnalysisReport)
    assert report.tracker_name == "MOSSE"


def test_analyze_overall_slice_present(mock_result):
    tagged, result = mock_result
    analyzer = AttributeAnalyzer(tagged)
    report = analyzer.analyze(result)
    assert report.overall is not None
    assert report.overall.attribute == "all"


def test_analyze_overall_n_sequences(mock_result):
    tagged, result = mock_result
    analyzer = AttributeAnalyzer(tagged)
    report = analyzer.analyze(result)
    assert report.overall.n_sequences == 2


def test_analyze_overall_mean_iou_in_range(mock_result):
    tagged, result = mock_result
    analyzer = AttributeAnalyzer(tagged)
    report = analyzer.analyze(result)
    assert 0.0 <= report.overall.mean_iou <= 1.0


def test_analyze_fast_motion_slice(mock_result):
    tagged, result = mock_result
    analyzer = AttributeAnalyzer(tagged)
    report = analyzer.analyze(result)
    fm = report.by_attribute(TrackingAttribute.FAST_MOTION)
    # Only the "fast" sequence should be in this slice.
    assert fm is not None
    assert fm.n_sequences == 1
    assert fm.mean_iou == pytest.approx(0.4, abs=0.01)


def test_analyze_no_sequences_slice_omitted(mock_result):
    tagged, result = mock_result
    analyzer = AttributeAnalyzer(tagged, min_sequences=3)
    report = analyzer.analyze(result)
    # No per-attribute slice should appear (each has < 3 sequences).
    non_all = [s for s in report.slices if s.attribute != "all"]
    assert len(non_all) == 0


# ---------------------------------------------------------------------------
# AttributeSliceResult.to_dict()
# ---------------------------------------------------------------------------


def test_slice_to_dict_keys(mock_result):
    tagged, result = mock_result
    analyzer = AttributeAnalyzer(tagged)
    report = analyzer.analyze(result)
    d = report.overall.to_dict()
    assert "attribute" in d
    assert "n_sequences" in d
    assert "mean_iou" in d
    assert "success_auc" in d
    assert "precision_auc" in d


# ---------------------------------------------------------------------------
# AttributeAnalyzer.to_markdown()
# ---------------------------------------------------------------------------


def test_to_markdown_contains_tracker_name(mock_result):
    tagged, result = mock_result
    analyzer = AttributeAnalyzer(tagged)
    report = analyzer.analyze(result)
    md = analyzer.to_markdown(report)
    assert "MOSSE" in md


def test_to_markdown_contains_all_row(mock_result):
    tagged, result = mock_result
    analyzer = AttributeAnalyzer(tagged)
    report = analyzer.analyze(result)
    md = analyzer.to_markdown(report)
    assert "All Sequences" in md


# ---------------------------------------------------------------------------
# AttributeAnalyzer.compare() + compare_to_markdown()
# ---------------------------------------------------------------------------


def test_compare_returns_dict(mock_result, slow_bboxes, fast_bboxes):
    tagged, result1 = mock_result
    seq_results2 = [
        _MockSeqResult("slow", np.full(50, 0.6)),
        _MockSeqResult("fast", np.full(50, 0.5)),
    ]
    result2 = _MockBenchmarkResult("KCF", "Synthetic", seq_results2)
    analyzer = AttributeAnalyzer(tagged)
    reports = analyzer.compare([result1, result2])
    assert "MOSSE" in reports
    assert "KCF" in reports


def test_compare_to_markdown_contains_trackers(mock_result, slow_bboxes, fast_bboxes):
    tagged, result1 = mock_result
    seq_results2 = [
        _MockSeqResult("slow", np.full(50, 0.6)),
        _MockSeqResult("fast", np.full(50, 0.5)),
    ]
    result2 = _MockBenchmarkResult("KCF", "Synthetic", seq_results2)
    analyzer = AttributeAnalyzer(tagged)
    reports = analyzer.compare([result1, result2])
    md = analyzer.compare_to_markdown(reports)
    assert "MOSSE" in md
    assert "KCF" in md


def test_coverage_matrix_markdown(mock_result, slow_bboxes, fast_bboxes):
    tagged, result1 = mock_result
    seq_results2 = [
        _MockSeqResult("slow", np.full(50, 0.55)),
        _MockSeqResult("fast", np.full(50, 0.45)),
    ]
    result2 = _MockBenchmarkResult("KCF", "Synthetic", seq_results2)
    analyzer = AttributeAnalyzer(tagged)
    reports = analyzer.compare([result1, result2])
    md = analyzer.coverage_to_markdown(reports)
    assert "MOSSE" in md
    assert "KCF" in md


# ---------------------------------------------------------------------------
# ATTRIBUTE_CODES and ATTRIBUTE_DISPLAY_NAMES completeness
# ---------------------------------------------------------------------------


def test_all_attributes_have_display_names():
    for attr in TrackingAttribute:
        assert attr in ATTRIBUTE_DISPLAY_NAMES, f"Missing display name for {attr}"


def test_all_attributes_have_codes():
    for attr in TrackingAttribute:
        assert attr in ATTRIBUTE_CODES, f"Missing code for {attr}"
