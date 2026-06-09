"""Tests for eovot.metrics.difficulty — SequenceDifficultyAnalyzer."""

import numpy as np
import pytest

from eovot.metrics.difficulty import (
    AttributeBreakdown,
    DifficultyReport,
    SequenceDifficultyAnalyzer,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _static_boxes(n: int = 50, x=10.0, y=20.0, w=40.0, h=30.0) -> np.ndarray:
    """Ground-truth boxes that never move or change size."""
    return np.tile([x, y, w, h], (n, 1)).astype(np.float64)


def _linear_motion_boxes(n: int = 50, speed: float = 5.0) -> np.ndarray:
    """Boxes moving linearly to the right by `speed` pixels per frame."""
    boxes = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        boxes[i] = [10.0 + i * speed, 20.0, 40.0, 30.0]
    return boxes


def _growing_boxes(n: int = 50) -> np.ndarray:
    """Boxes with linearly growing width & height (scale change)."""
    boxes = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        size = 20.0 + i * 2.0
        boxes[i] = [10.0, 10.0, size, size]
    return boxes


def _hard_sequence(n: int = 50) -> np.ndarray:
    """Fast motion + highly variable scale/AR → composite score above hard threshold."""
    rng = np.random.default_rng(42)
    boxes = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        # Speed=80px/frame; tiny min size → high normalised displacement
        x = 10.0 + i * 80.0
        w = rng.uniform(5.0, 180.0)   # random scale — high log-ratio variance
        h = w * rng.uniform(0.4, 2.5)  # random AR
        boxes[i] = [x, 10.0, max(w, 1.0), max(h, 1.0)]
    return boxes


# ---------------------------------------------------------------------------
# DifficultyReport
# ---------------------------------------------------------------------------

class TestDifficultyReport:
    def test_to_dict_keys(self):
        report = DifficultyReport(
            sequence_name="test",
            num_frames=50,
            motion_magnitude=0.05,
            scale_change=0.10,
            aspect_ratio_change=0.02,
            size_ratio=1.5,
            overall_score=0.30,
            label="easy",
        )
        d = report.to_dict()
        assert set(d) == {
            "sequence_name", "num_frames", "motion_magnitude", "scale_change",
            "aspect_ratio_change", "size_ratio", "overall_score", "label",
        }

    def test_to_dict_values_rounded(self):
        report = DifficultyReport("s", 10, 0.123456, 0.654321, 0.111111, 2.123456, 0.5, "medium")
        d = report.to_dict()
        assert d["motion_magnitude"] == pytest.approx(0.1235, abs=1e-4)
        assert d["scale_change"] == pytest.approx(0.6543, abs=1e-4)

    def test_str_contains_label(self):
        report = DifficultyReport("seq1", 30, 0.1, 0.2, 0.05, 2.0, 0.4, "medium")
        assert "medium" in str(report)
        assert "seq1" in str(report)


# ---------------------------------------------------------------------------
# SequenceDifficultyAnalyzer — constructor validation
# ---------------------------------------------------------------------------

class TestAnalyzerConstruction:
    def test_default_thresholds(self):
        a = SequenceDifficultyAnalyzer()
        assert a.easy_threshold == pytest.approx(0.35)
        assert a.hard_threshold == pytest.approx(0.65)

    def test_custom_thresholds(self):
        a = SequenceDifficultyAnalyzer(easy_threshold=0.2, hard_threshold=0.8)
        assert a.easy_threshold == pytest.approx(0.2)
        assert a.hard_threshold == pytest.approx(0.8)

    def test_invalid_thresholds_raises(self):
        with pytest.raises(ValueError):
            SequenceDifficultyAnalyzer(easy_threshold=0.7, hard_threshold=0.3)
        with pytest.raises(ValueError):
            SequenceDifficultyAnalyzer(easy_threshold=0.0, hard_threshold=0.5)


# ---------------------------------------------------------------------------
# SequenceDifficultyAnalyzer.analyze — input validation
# ---------------------------------------------------------------------------

class TestAnalyzeInputValidation:
    def setup_method(self):
        self.analyzer = SequenceDifficultyAnalyzer()

    def test_single_frame_raises(self):
        with pytest.raises(ValueError, match="At least 2 frames"):
            self.analyzer.analyze(np.array([[10.0, 20.0, 40.0, 30.0]]))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            self.analyzer.analyze(np.ones((10, 3)))

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            self.analyzer.analyze(np.array([10.0, 20.0, 40.0, 30.0]))


# ---------------------------------------------------------------------------
# SequenceDifficultyAnalyzer.analyze — correctness
# ---------------------------------------------------------------------------

class TestAnalyzeCorrectness:
    def setup_method(self):
        self.analyzer = SequenceDifficultyAnalyzer()

    def test_static_sequence_is_easy(self):
        """A target that never moves should score low difficulty."""
        report = self.analyzer.analyze(_static_boxes(50), "static")
        assert report.label == "easy"
        assert report.motion_magnitude < 1e-6
        assert report.scale_change < 1e-6
        assert report.aspect_ratio_change < 1e-6
        assert report.size_ratio == pytest.approx(1.0, abs=1e-6)

    def test_fast_motion_sequence_scores_high(self):
        """Large per-frame displacement should raise the overall score."""
        slow = self.analyzer.analyze(_linear_motion_boxes(50, speed=1.0), "slow")
        fast = self.analyzer.analyze(_linear_motion_boxes(50, speed=30.0), "fast")
        assert fast.motion_magnitude > slow.motion_magnitude
        assert fast.overall_score > slow.overall_score

    def test_growing_boxes_scale_change(self):
        """Linearly growing boxes should have non-zero scale_change."""
        report = self.analyzer.analyze(_growing_boxes(50), "growing")
        assert report.scale_change > 0.0
        assert report.size_ratio > 1.0

    def test_sequence_name_preserved(self):
        report = self.analyzer.analyze(_static_boxes(), "my_sequence")
        assert report.sequence_name == "my_sequence"

    def test_num_frames_correct(self):
        boxes = _static_boxes(73)
        report = self.analyzer.analyze(boxes)
        assert report.num_frames == 73

    def test_label_easy(self):
        # Static boxes → near-zero score → easy
        report = self.analyzer.analyze(_static_boxes(50))
        assert report.label == "easy"
        assert report.overall_score < self.analyzer.easy_threshold

    def test_label_hard(self):
        # Fast motion + random scale + random AR → all axes contribute → hard
        report = self.analyzer.analyze(_hard_sequence(50), "hard_seq")
        assert report.label == "hard"

    def test_overall_score_bounded(self):
        for speed in [0.1, 1.0, 10.0, 100.0]:
            report = self.analyzer.analyze(_linear_motion_boxes(50, speed=speed))
            assert 0.0 <= report.overall_score <= 1.0

    def test_report_to_dict_roundtrip(self):
        report = self.analyzer.analyze(_growing_boxes(30), "grow")
        d = report.to_dict()
        assert d["sequence_name"] == "grow"
        assert d["num_frames"] == 30
        assert "label" in d
        assert "overall_score" in d

    def test_minimum_two_frames(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 10.0, 10.0]])
        report = self.analyzer.analyze(boxes)
        assert report.num_frames == 2


# ---------------------------------------------------------------------------
# Private axis helpers
# ---------------------------------------------------------------------------

class TestAxisHelpers:
    def test_motion_magnitude_zero_for_static(self):
        gt = _static_boxes(10)
        val = SequenceDifficultyAnalyzer._motion_magnitude(gt)
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_motion_magnitude_positive_for_moving(self):
        gt = _linear_motion_boxes(10, speed=5.0)
        val = SequenceDifficultyAnalyzer._motion_magnitude(gt)
        assert val > 0.0

    def test_scale_change_zero_for_constant(self):
        gt = _static_boxes(10)
        val = SequenceDifficultyAnalyzer._scale_change(gt)
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_scale_change_positive_for_varying(self):
        gt = _growing_boxes(10)
        val = SequenceDifficultyAnalyzer._scale_change(gt)
        assert val > 0.0

    def test_ar_change_zero_for_square(self):
        gt = np.tile([0.0, 0.0, 30.0, 30.0], (10, 1)).astype(float)
        val = SequenceDifficultyAnalyzer._aspect_ratio_change(gt)
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_size_ratio_one_for_constant(self):
        gt = _static_boxes(10)
        assert SequenceDifficultyAnalyzer._size_ratio(gt) == pytest.approx(1.0, abs=1e-9)

    def test_size_ratio_greater_for_varying(self):
        gt = _growing_boxes(10)
        assert SequenceDifficultyAnalyzer._size_ratio(gt) > 1.0


# ---------------------------------------------------------------------------
# analyze_dataset
# ---------------------------------------------------------------------------

class _FakeSeq:
    def __init__(self, name: str, gt: np.ndarray):
        self.name = name
        self.ground_truth = gt


class TestAnalyzeDataset:
    def setup_method(self):
        self.analyzer = SequenceDifficultyAnalyzer()

    def test_returns_one_per_sequence(self):
        seqs = [_FakeSeq(f"s{i}", _static_boxes(20)) for i in range(5)]
        reports = self.analyzer.analyze_dataset(seqs)
        assert len(reports) == 5

    def test_skips_invalid_sequences(self):
        class BadSeq:
            name = "bad"
            ground_truth = np.ones((1, 4))  # single frame → ValueError

        seqs = [_FakeSeq("ok", _static_boxes(10)), BadSeq()]
        reports = self.analyzer.analyze_dataset(seqs)
        assert len(reports) == 1
        assert reports[0].sequence_name == "ok"

    def test_names_preserved(self):
        seqs = [_FakeSeq(f"seq_{i}", _static_boxes(20)) for i in range(3)]
        reports = self.analyzer.analyze_dataset(seqs)
        assert [r.sequence_name for r in reports] == ["seq_0", "seq_1", "seq_2"]


# ---------------------------------------------------------------------------
# performance_by_difficulty
# ---------------------------------------------------------------------------

def _make_benchmark_result(tracker_name: str, seq_names, iou_per_seq):
    """Create a minimal BenchmarkResult-like object for testing."""
    from unittest.mock import MagicMock
    import numpy as np

    result = MagicMock()
    result.tracker_name = tracker_name
    result.dataset_name = "TestDataset"

    sr_list = []
    for name, iou_val in zip(seq_names, iou_per_seq):
        sr = MagicMock()
        sr.sequence_name = name
        sr.mean_iou = iou_val
        sr.ious = np.full(20, iou_val)
        sr_list.append(sr)

    result.sequence_results = sr_list
    return result


class TestPerformanceByDifficulty:
    def setup_method(self):
        self.analyzer = SequenceDifficultyAnalyzer()

    def test_easy_sequences_bucket(self):
        seqs = [_FakeSeq("s1", _static_boxes(30)), _FakeSeq("s2", _static_boxes(30))]
        result = _make_benchmark_result("MOSSE", ["s1", "s2"], [0.7, 0.8])
        breakdown = self.analyzer.performance_by_difficulty(result, seqs)
        assert breakdown.easy is not None
        assert breakdown.easy == pytest.approx((0.7 + 0.8) / 2, abs=1e-6)
        assert breakdown.counts["easy"] == 2

    def test_hard_sequences_bucket(self):
        seqs = [_FakeSeq("hard_seq", _hard_sequence(50))]
        result = _make_benchmark_result("MOSSE", ["hard_seq"], [0.2])
        breakdown = self.analyzer.performance_by_difficulty(result, seqs)
        assert breakdown.hard is not None
        assert breakdown.hard == pytest.approx(0.2, abs=1e-6)
        assert breakdown.easy is None

    def test_missing_sequence_ignored(self):
        seqs = [_FakeSeq("known", _static_boxes(20))]
        result = _make_benchmark_result("KCF", ["known", "unknown"], [0.6, 0.4])
        breakdown = self.analyzer.performance_by_difficulty(result, seqs)
        assert breakdown.counts["easy"] == 1

    def test_to_dict_keys(self):
        seqs = [_FakeSeq("s", _static_boxes(20))]
        result = _make_benchmark_result("T", ["s"], [0.5])
        breakdown = self.analyzer.performance_by_difficulty(result, seqs)
        d = breakdown.to_dict()
        assert "tracker" in d
        assert "mean_iou_easy" in d
        assert "counts" in d

    def test_str_representation(self):
        seqs = [_FakeSeq("s", _static_boxes(20))]
        result = _make_benchmark_result("T", ["s"], [0.5])
        breakdown = self.analyzer.performance_by_difficulty(result, seqs)
        s = str(breakdown)
        assert "T" in s
        assert "easy" in s


# ---------------------------------------------------------------------------
# to_markdown_table
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def test_table_has_header(self):
        breakdowns = [
            AttributeBreakdown("MOSSE", "OTB", 0.7, 0.5, 0.3, {"easy": 30, "medium": 40, "hard": 30}),
            AttributeBreakdown("KCF", "OTB", 0.75, 0.55, 0.35, {"easy": 30, "medium": 40, "hard": 30}),
        ]
        table = SequenceDifficultyAnalyzer.to_markdown_table(breakdowns)
        assert "Tracker" in table
        assert "MOSSE" in table
        assert "KCF" in table

    def test_none_values_shown_as_dash(self):
        breakdowns = [
            AttributeBreakdown("T", "D", None, 0.5, None, {"easy": 0, "medium": 10, "hard": 0}),
        ]
        table = SequenceDifficultyAnalyzer.to_markdown_table(breakdowns)
        assert "—" in table

    def test_empty_list_returns_header_only(self):
        table = SequenceDifficultyAnalyzer.to_markdown_table([])
        assert "Tracker" in table
        lines = [l for l in table.splitlines() if l.strip()]
        assert len(lines) == 2  # header + separator only
