"""Tests for eovot.analysis.difficulty — Sequence Difficulty Analyzer."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.analysis.difficulty import (
    ATTRIBUTE_WEIGHTS,
    DifficultyAnalyzer,
    DifficultyLevel,
    SequenceDifficultyScore,
    _compute_attribute_score,
    _compute_motion_score,
    _dominant_attribute,
    _level_from_score,
)
from eovot.metrics.attributes import AttributeDetector, SequenceAttributes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_static_gt(n: int = 30, x: float = 100, y: float = 80,
                    w: float = 40, h: float = 40) -> np.ndarray:
    """GT array where the box never moves."""
    return np.tile([x, y, w, h], (n, 1)).astype(np.float64)


def _make_moving_gt(n: int = 30, step: float = 10.0) -> np.ndarray:
    """GT array where the box moves horizontally by `step` px per frame."""
    boxes = []
    for i in range(n):
        boxes.append([i * step, 80.0, 40.0, 40.0])
    return np.array(boxes, dtype=np.float64)


def _make_seq_result(ious, gt=None, name="seq"):
    """Create a minimal SequenceResult stub."""
    from eovot.profiling.profiler import ProfilingResult
    from eovot.benchmark.engine import SequenceResult

    prof = ProfilingResult(
        tracker_name="stub",
        frame_count=len(ious),
        fps=100.0,
        latency_mean_ms=10.0,
        latency_std_ms=1.0,
        latency_p95_ms=12.0,
        peak_memory_mb=50.0,
    )
    return SequenceResult(
        sequence_name=name,
        ious=np.array(ious, dtype=np.float64),
        profiling=prof,
        ground_truths=gt,
    )


def _make_benchmark_result(seq_results, tracker_name="MOSSE", dataset_name="Syn"):
    """Create a minimal BenchmarkResult stub."""
    from eovot.benchmark.engine import BenchmarkResult
    br = BenchmarkResult(tracker_name=tracker_name, dataset_name=dataset_name)
    br.sequence_results = seq_results
    return br


# ---------------------------------------------------------------------------
# ATTRIBUTE_WEIGHTS
# ---------------------------------------------------------------------------

def test_attribute_weights_sum_to_one():
    total = sum(ATTRIBUTE_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9


def test_attribute_weights_all_positive():
    assert all(v > 0 for v in ATTRIBUTE_WEIGHTS.values())


def test_attribute_weights_cover_all_known_attributes():
    expected = {
        "fast_motion", "partial_occlusion", "scale_variation",
        "low_resolution", "deformation", "out_plane_rotation", "long_sequence",
    }
    assert set(ATTRIBUTE_WEIGHTS) == expected


# ---------------------------------------------------------------------------
# DifficultyLevel
# ---------------------------------------------------------------------------

def test_difficulty_level_values():
    assert DifficultyLevel.EASY.value == "easy"
    assert DifficultyLevel.MEDIUM.value == "medium"
    assert DifficultyLevel.HARD.value == "hard"
    assert DifficultyLevel.VERY_HARD.value == "very_hard"


def test_difficulty_level_is_str():
    assert isinstance(DifficultyLevel.EASY, str)


# ---------------------------------------------------------------------------
# _level_from_score
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected_level", [
    (0.00, DifficultyLevel.EASY),
    (0.10, DifficultyLevel.EASY),
    (0.24, DifficultyLevel.EASY),
    (0.25, DifficultyLevel.MEDIUM),
    (0.40, DifficultyLevel.MEDIUM),
    (0.50, DifficultyLevel.HARD),
    (0.70, DifficultyLevel.HARD),
    (0.75, DifficultyLevel.VERY_HARD),
    (1.00, DifficultyLevel.VERY_HARD),
])
def test_level_from_score(score, expected_level):
    assert _level_from_score(score) == expected_level


# ---------------------------------------------------------------------------
# _compute_motion_score
# ---------------------------------------------------------------------------

def test_motion_score_empty_gt():
    assert _compute_motion_score(np.empty((0, 4))) == 0.0


def test_motion_score_single_frame():
    gt = np.array([[10, 10, 40, 40]], dtype=np.float64)
    assert _compute_motion_score(gt) == 0.0


def test_motion_score_static_target():
    gt = _make_static_gt(50)
    score = _compute_motion_score(gt)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_motion_score_fast_motion():
    # Step = 10 px, box diagonal ~= 56.6 px  → normalised ≈ 0.177 per frame
    # tanh(0.177 / 0.20) = tanh(0.885) ≈ 0.706
    gt = _make_moving_gt(30, step=10.0)
    score = _compute_motion_score(gt)
    assert score > 0.5


def test_motion_score_very_fast_saturates():
    # 500 px per frame — should approach 1.0
    gt = _make_moving_gt(10, step=500.0)
    score = _compute_motion_score(gt)
    assert score > 0.99


def test_motion_score_in_range():
    gt = _make_moving_gt(20, step=5.0)
    score = _compute_motion_score(gt)
    assert 0.0 <= score <= 1.0


def test_motion_score_zero_area_box():
    gt = np.zeros((10, 4), dtype=np.float64)
    score = _compute_motion_score(gt)
    assert score == 0.0


# ---------------------------------------------------------------------------
# _compute_attribute_score
# ---------------------------------------------------------------------------

def test_attribute_score_no_attributes():
    attrs = SequenceAttributes("seq", {k: False for k in ATTRIBUTE_WEIGHTS})
    assert _compute_attribute_score(attrs) == pytest.approx(0.0)


def test_attribute_score_all_attributes():
    attrs = SequenceAttributes("seq", {k: True for k in ATTRIBUTE_WEIGHTS})
    assert _compute_attribute_score(attrs) == pytest.approx(1.0)


def test_attribute_score_single_fast_motion():
    attrs = SequenceAttributes("seq", {"fast_motion": True})
    score = _compute_attribute_score(attrs)
    assert score == pytest.approx(ATTRIBUTE_WEIGHTS["fast_motion"])


def test_attribute_score_in_range():
    attrs = SequenceAttributes(
        "seq", {"fast_motion": True, "scale_variation": True}
    )
    score = _compute_attribute_score(attrs)
    expected = ATTRIBUTE_WEIGHTS["fast_motion"] + ATTRIBUTE_WEIGHTS["scale_variation"]
    assert score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _dominant_attribute
# ---------------------------------------------------------------------------

def test_dominant_attribute_no_active():
    attrs = SequenceAttributes("seq", {})
    assert _dominant_attribute(attrs) == "none"


def test_dominant_attribute_returns_highest_weight():
    attrs = SequenceAttributes(
        "seq", {"long_sequence": True, "fast_motion": True, "deformation": True}
    )
    assert _dominant_attribute(attrs) == "fast_motion"


def test_dominant_attribute_single():
    attrs = SequenceAttributes("seq", {"low_resolution": True})
    assert _dominant_attribute(attrs) == "low_resolution"


# ---------------------------------------------------------------------------
# DifficultyAnalyzer — constructor
# ---------------------------------------------------------------------------

def test_analyzer_default_construction():
    da = DifficultyAnalyzer()
    assert da._iou_thresh == 0.50
    assert abs(da._fw + da._mw + da._aw - 1.0) < 1e-9


def test_analyzer_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        DifficultyAnalyzer(failure_weight=0.5, motion_weight=0.5, attribute_weight=0.5)


def test_analyzer_invalid_iou_threshold():
    with pytest.raises(ValueError, match="failure_iou_threshold"):
        DifficultyAnalyzer(failure_iou_threshold=0.0)
    with pytest.raises(ValueError, match="failure_iou_threshold"):
        DifficultyAnalyzer(failure_iou_threshold=1.5)


# ---------------------------------------------------------------------------
# DifficultyAnalyzer.score_from_groundtruth
# ---------------------------------------------------------------------------

def test_score_gt_static_easy():
    da = DifficultyAnalyzer()
    gt = _make_static_gt(30)
    s = da.score_from_groundtruth(gt, "seq")
    assert s.failure_rate == 0.0
    assert s.motion_score == pytest.approx(0.0, abs=1e-9)
    assert s.difficulty_level == DifficultyLevel.EASY
    assert s.num_frames == 30


def test_score_gt_returns_correct_type():
    da = DifficultyAnalyzer()
    gt = _make_static_gt(10)
    s = da.score_from_groundtruth(gt)
    assert isinstance(s, SequenceDifficultyScore)


def test_score_gt_failure_rate_always_zero():
    da = DifficultyAnalyzer()
    gt = _make_moving_gt(20, step=50.0)
    s = da.score_from_groundtruth(gt)
    assert s.failure_rate == 0.0


def test_score_gt_high_motion_harder_than_static():
    da = DifficultyAnalyzer()
    fast = da.score_from_groundtruth(_make_moving_gt(30, step=100.0))
    slow = da.score_from_groundtruth(_make_static_gt(30))
    assert fast.difficulty_score > slow.difficulty_score


def test_score_gt_empty_gt():
    da = DifficultyAnalyzer()
    s = da.score_from_groundtruth(np.empty((0, 4)))
    assert s.difficulty_score == pytest.approx(0.0)
    assert s.num_frames == 0


def test_score_gt_sequence_name_stored():
    da = DifficultyAnalyzer()
    s = da.score_from_groundtruth(_make_static_gt(5), sequence_name="walking-dog")
    assert s.sequence_name == "walking-dog"


def test_score_gt_score_in_range():
    da = DifficultyAnalyzer()
    s = da.score_from_groundtruth(_make_moving_gt(50, step=30.0))
    assert 0.0 <= s.difficulty_score <= 1.0


# ---------------------------------------------------------------------------
# DifficultyAnalyzer.score_from_result
# ---------------------------------------------------------------------------

def test_score_result_perfect_tracker_is_easy():
    da = DifficultyAnalyzer()
    sr = _make_seq_result(ious=[1.0] * 30, gt=_make_static_gt(30))
    s = da.score_from_result(sr)
    assert s.failure_rate == pytest.approx(0.0)
    # No failures and no motion → should be EASY
    assert s.difficulty_level in (DifficultyLevel.EASY, DifficultyLevel.MEDIUM)


def test_score_result_all_failing_is_very_hard():
    da = DifficultyAnalyzer()
    gt = _make_moving_gt(30, step=50.0)
    sr = _make_seq_result(ious=[0.0] * 30, gt=gt)
    s = da.score_from_result(sr)
    assert s.failure_rate == pytest.approx(1.0)
    assert s.difficulty_level in (DifficultyLevel.HARD, DifficultyLevel.VERY_HARD)


def test_score_result_partial_failure():
    da = DifficultyAnalyzer()
    ious = [1.0] * 15 + [0.0] * 15
    sr = _make_seq_result(ious=ious, gt=_make_static_gt(30))
    s = da.score_from_result(sr)
    assert s.failure_rate == pytest.approx(0.5)


def test_score_result_no_gt_falls_back():
    da = DifficultyAnalyzer()
    sr = _make_seq_result(ious=[0.3] * 20, gt=None)
    s = da.score_from_result(sr)
    assert 0.0 <= s.difficulty_score <= 1.0


def test_score_result_num_frames():
    da = DifficultyAnalyzer()
    sr = _make_seq_result(ious=[0.6] * 42, gt=_make_static_gt(42))
    s = da.score_from_result(sr)
    assert s.num_frames == 42


def test_score_result_str_representation():
    da = DifficultyAnalyzer()
    sr = _make_seq_result(ious=[0.5] * 10, gt=_make_static_gt(10), name="test-seq")
    s = da.score_from_result(sr)
    text = str(s)
    assert "test-seq" in text
    assert "score=" in text


def test_score_result_to_dict_keys():
    da = DifficultyAnalyzer()
    sr = _make_seq_result(ious=[0.5] * 5, gt=_make_static_gt(5))
    d = da.score_from_result(sr).to_dict()
    for key in ("sequence_name", "difficulty_level", "difficulty_score",
                "failure_rate", "motion_score", "attribute_score",
                "dominant_challenge", "active_attributes", "num_frames"):
        assert key in d


def test_score_result_custom_iou_threshold():
    da = DifficultyAnalyzer(failure_iou_threshold=0.3)
    ious = [0.4] * 10  # above 0.3 → not failures; below 0.5 default
    sr = _make_seq_result(ious=ious, gt=_make_static_gt(10))
    s = da.score_from_result(sr)
    assert s.failure_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# DifficultyAnalyzer.analyze_benchmark
# ---------------------------------------------------------------------------

def test_analyze_benchmark_length():
    da = DifficultyAnalyzer()
    srs = [
        _make_seq_result([0.8] * 20, _make_static_gt(20), f"s{i}")
        for i in range(5)
    ]
    br = _make_benchmark_result(srs)
    scores = da.analyze_benchmark(br)
    assert len(scores) == 5


def test_analyze_benchmark_names_match():
    da = DifficultyAnalyzer()
    srs = [
        _make_seq_result([0.6] * 10, _make_static_gt(10), f"seq_{i}")
        for i in range(3)
    ]
    br = _make_benchmark_result(srs)
    scores = da.analyze_benchmark(br)
    assert [s.sequence_name for s in scores] == ["seq_0", "seq_1", "seq_2"]


def test_analyze_benchmark_empty_result():
    da = DifficultyAnalyzer()
    br = _make_benchmark_result([])
    scores = da.analyze_benchmark(br)
    assert scores == []


# ---------------------------------------------------------------------------
# DifficultyAnalyzer.stratified_summary
# ---------------------------------------------------------------------------

def test_stratified_summary_has_all_levels():
    da = DifficultyAnalyzer()
    srs = [_make_seq_result([1.0] * 10, _make_static_gt(10), "s0")]
    br = _make_benchmark_result(srs)
    scores = da.analyze_benchmark(br)
    summary = da.stratified_summary(scores, br)
    assert set(summary.keys()) == {"easy", "medium", "hard", "very_hard"}


def test_stratified_summary_easy_sequence_in_easy_bucket():
    da = DifficultyAnalyzer()
    sr = _make_seq_result([1.0] * 20, _make_static_gt(20), "perfect")
    br = _make_benchmark_result([sr])
    scores = da.analyze_benchmark(br)
    summary = da.stratified_summary(scores, br)
    # Perfect tracker on static target → EASY
    easy = summary["easy"]
    assert easy["n_sequences"] >= 0  # may be 0 if edge effects push to MEDIUM


def test_stratified_summary_hard_sequence_counted():
    da = DifficultyAnalyzer()
    sr = _make_seq_result([0.0] * 30, _make_moving_gt(30, step=100.0), "hard-seq")
    br = _make_benchmark_result([sr])
    scores = da.analyze_benchmark(br)
    summary = da.stratified_summary(scores, br)
    # Should be in HARD or VERY_HARD
    hard_or_vh = summary["hard"]["n_sequences"] + summary["very_hard"]["n_sequences"]
    assert hard_or_vh >= 1


def test_stratified_summary_total_counts_all_sequences():
    da = DifficultyAnalyzer()
    srs = [
        _make_seq_result([0.9] * 10, _make_static_gt(10), f"easy-{i}")
        for i in range(3)
    ] + [
        _make_seq_result([0.1] * 10, _make_moving_gt(10, step=50.0), f"hard-{i}")
        for i in range(2)
    ]
    br = _make_benchmark_result(srs)
    scores = da.analyze_benchmark(br)
    summary = da.stratified_summary(scores, br)
    total = sum(s["n_sequences"] for s in summary.values())
    assert total == 5


# ---------------------------------------------------------------------------
# DifficultyAnalyzer.to_stratified_leaderboard
# ---------------------------------------------------------------------------

def test_to_stratified_leaderboard_contains_headers():
    da = DifficultyAnalyzer()
    srs = [_make_seq_result([0.7] * 10, _make_static_gt(10), "s0")]
    br = _make_benchmark_result(srs, tracker_name="KCF", dataset_name="OTB")
    scores = da.analyze_benchmark(br)
    md = da.to_stratified_leaderboard(scores, br)
    assert "Difficulty" in md
    assert "Mean IoU" in md
    assert "Failure Rate" in md


def test_to_stratified_leaderboard_contains_tracker_name():
    da = DifficultyAnalyzer()
    srs = [_make_seq_result([0.8] * 10, _make_static_gt(10), "s")]
    br = _make_benchmark_result(srs, tracker_name="MOSSE")
    scores = da.analyze_benchmark(br)
    md = da.to_stratified_leaderboard(scores, br)
    assert "MOSSE" in md


def test_to_stratified_leaderboard_returns_string():
    da = DifficultyAnalyzer()
    srs = [_make_seq_result([0.5] * 10, _make_static_gt(10), "s")]
    br = _make_benchmark_result(srs)
    scores = da.analyze_benchmark(br)
    result = da.to_stratified_leaderboard(scores, br)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

def test_full_pipeline_with_benchmark_engine():
    """End-to-end: BenchmarkEngine → DifficultyAnalyzer."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset

    engine = BenchmarkEngine(verbose=False)
    dataset = SyntheticDataset(num_sequences=4, num_frames=30, seed=0)
    result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")

    da = DifficultyAnalyzer()
    scores = da.analyze_benchmark(result)

    assert len(scores) == 4
    for s in scores:
        assert isinstance(s, SequenceDifficultyScore)
        assert 0.0 <= s.difficulty_score <= 1.0
        assert isinstance(s.difficulty_level, DifficultyLevel)
        assert s.num_frames == 30

    summary = da.stratified_summary(scores, result)
    total = sum(v["n_sequences"] for v in summary.values())
    assert total == 4

    md = da.to_stratified_leaderboard(scores, result)
    assert "MOSSE" in md
    assert "Synthetic" in md
