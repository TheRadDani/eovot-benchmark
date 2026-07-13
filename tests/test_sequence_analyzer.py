"""Tests for SequenceAnalyzer — sequence difficulty analysis module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eovot.analysis.sequence_analyzer import (
    DifficultyTier,
    SequenceAnalyzer,
    SequenceAttributes,
    TierPerformance,
    _compute_attributes,
    _normalize_array,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _linear_gt(n: int, vx: float = 2.0, vy: float = 1.5,
               x0: float = 50, y0: float = 40,
               bw: float = 30, bh: float = 25) -> np.ndarray:
    """Ground-truth boxes for a linearly moving target (constant size)."""
    boxes = []
    cx, cy = x0, y0
    for _ in range(n):
        boxes.append([cx - bw / 2, cy - bh / 2, bw, bh])
        cx += vx
        cy += vy
    return np.array(boxes, dtype=np.float64)


def _growing_gt(n: int, cx: float = 100, cy: float = 80,
                w_start: float = 20, w_end: float = 60) -> np.ndarray:
    """Ground-truth boxes for a growing target (scale change)."""
    boxes = []
    for i in range(n):
        w = w_start + (w_end - w_start) * i / max(n - 1, 1)
        boxes.append([cx - w / 2, cy - w / 2, w, w])
    return np.array(boxes, dtype=np.float64)


def _iou_array(n: int, mean: float = 0.6, seed: int = 0) -> np.ndarray:
    """Synthetic per-frame IoU array with given mean."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(max(0, mean - 0.2), min(1, mean + 0.2), n)
    return np.clip(raw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# _normalize_array
# ---------------------------------------------------------------------------

def test_normalize_all_zeros():
    arr = np.zeros(5)
    result = _normalize_array(arr)
    np.testing.assert_array_equal(result, np.zeros(5))


def test_normalize_constant_nonzero():
    arr = np.full(4, 3.14)
    result = _normalize_array(arr)
    np.testing.assert_array_equal(result, np.zeros(4))


def test_normalize_range():
    arr = np.array([0.0, 1.0, 2.0, 4.0])
    result = _normalize_array(arr)
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)
    assert result[2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _compute_attributes
# ---------------------------------------------------------------------------

def test_compute_attributes_linear():
    gt = _linear_gt(n=30, vx=2.0, vy=0.0)
    attr = _compute_attributes("test_seq", gt)

    assert attr.sequence_name == "test_seq"
    assert attr.num_frames == 30
    assert attr.mean_speed_px == pytest.approx(2.0, abs=1e-6)
    assert attr.max_speed_px == pytest.approx(2.0, abs=1e-6)
    assert attr.scale_change_ratio == pytest.approx(1.0, abs=1e-6)
    assert attr.aspect_ratio_std == pytest.approx(0.0, abs=1e-6)
    assert attr.difficulty_score is None
    assert attr.tier is None


def test_compute_attributes_growing():
    gt = _growing_gt(n=50, w_start=20, w_end=60)
    attr = _compute_attributes("grow_seq", gt)

    # Area goes from 400 to 3600 — ratio = 9
    assert attr.scale_change_ratio == pytest.approx(9.0, rel=0.05)


def test_compute_attributes_single_frame():
    gt = np.array([[10, 20, 30, 40]], dtype=np.float64)
    attr = _compute_attributes("one_frame", gt)
    assert attr.mean_speed_px == pytest.approx(0.0)
    assert attr.max_speed_px == pytest.approx(0.0)


def test_compute_attributes_ar_std():
    """Deforming target: aspect ratio changes each frame."""
    n = 20
    widths = np.linspace(10, 50, n)
    heights = np.full(n, 30)
    boxes = np.stack(
        [np.zeros(n), np.zeros(n), widths, heights], axis=1
    ).astype(np.float64)
    attr = _compute_attributes("deform", boxes)
    assert attr.aspect_ratio_std > 0.3, "Expected non-trivial AR variance"


# ---------------------------------------------------------------------------
# SequenceAnalyzer constructor
# ---------------------------------------------------------------------------

def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        SequenceAnalyzer(speed_weight=0.5, scale_weight=0.5, ar_weight=0.5)


def test_default_weights_sum_to_one():
    a = SequenceAnalyzer()
    total = a.speed_weight + a.scale_weight + a.ar_weight
    assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SequenceAnalyzer.analyze — basic structure
# ---------------------------------------------------------------------------

def _make_gt_iou_dicts(n_seqs: int = 6):
    gt = {}
    iou = {}
    for i in range(n_seqs):
        name = f"seq_{i:03d}"
        speed_factor = i * 3.0
        gt[name] = _linear_gt(n=40, vx=speed_factor, vy=0)
        iou[name] = _iou_array(40, mean=max(0.1, 0.9 - i * 0.1), seed=i)
    return gt, iou


def test_analyze_returns_required_keys():
    gt, iou = _make_gt_iou_dicts(6)
    report = SequenceAnalyzer().analyze(gt, iou, tracker_name="Test")
    assert "attributes" in report
    assert "tier_performance" in report
    assert "tier_summary" in report
    assert "markdown_table" in report


def test_attributes_cover_all_sequences():
    gt, iou = _make_gt_iou_dicts(6)
    report = SequenceAnalyzer().analyze(gt, iou)
    assert set(report["attributes"].keys()) == set(gt.keys())


def test_all_sequences_have_tier():
    gt, iou = _make_gt_iou_dicts(9)
    report = SequenceAnalyzer().analyze(gt, iou)
    for attr in report["attributes"].values():
        assert attr.tier is not None
        assert isinstance(attr.tier, DifficultyTier)


def test_difficulty_scores_in_unit_interval():
    gt, iou = _make_gt_iou_dicts(12)
    report = SequenceAnalyzer().analyze(gt, iou)
    for attr in report["attributes"].values():
        assert 0.0 <= attr.difficulty_score <= 1.0 + 1e-9


def test_tier_performance_keys_cover_all_tiers():
    gt, iou = _make_gt_iou_dicts(9)
    report = SequenceAnalyzer().analyze(gt, iou)
    assert set(report["tier_performance"].keys()) == set(DifficultyTier)


def test_empty_gt_dict():
    report = SequenceAnalyzer().analyze({}, {})
    assert report["attributes"] == {}
    assert report["tier_performance"] == {}


# ---------------------------------------------------------------------------
# Tier assignments — ordering invariant
# ---------------------------------------------------------------------------

def test_fast_sequences_are_harder():
    """Sequences with higher speed should receive harder tiers on average."""
    gt = {
        "slow": _linear_gt(n=50, vx=0.5, vy=0.0),
        "medium": _linear_gt(n=50, vx=5.0, vy=0.0),
        "fast": _linear_gt(n=50, vx=20.0, vy=0.0),
    }
    iou = {k: _iou_array(50, seed=0) for k in gt}
    report = SequenceAnalyzer(
        speed_weight=1.0, scale_weight=0.0, ar_weight=0.0
    ).analyze(gt, iou)
    attrs = report["attributes"]

    # Difficulty score should be monotonically increasing with speed
    assert attrs["slow"].difficulty_score < attrs["medium"].difficulty_score
    assert attrs["medium"].difficulty_score < attrs["fast"].difficulty_score


def test_scale_change_sequences_are_harder():
    """Sequences with more scale change should have higher difficulty scores."""
    gt = {
        "constant": _linear_gt(n=40, vx=0, vy=0),
        "growing": _growing_gt(n=40, w_start=20, w_end=60),
    }
    iou = {k: _iou_array(40, seed=1) for k in gt}
    report = SequenceAnalyzer(
        speed_weight=0.0, scale_weight=1.0, ar_weight=0.0
    ).analyze(gt, iou)
    attrs = report["attributes"]
    assert attrs["growing"].difficulty_score > attrs["constant"].difficulty_score


# ---------------------------------------------------------------------------
# TierPerformance correctness
# ---------------------------------------------------------------------------

def test_tier_performance_mean_iou_in_range():
    gt, iou = _make_gt_iou_dicts(9)
    report = SequenceAnalyzer().analyze(gt, iou)
    for tp in report["tier_performance"].values():
        if tp.num_sequences > 0:
            assert 0.0 <= tp.mean_iou <= 1.0
            assert 0.0 <= tp.mean_success_rate <= 1.0
            assert 0.0 <= tp.mean_survival_rate <= 1.0


def test_tier_sequence_counts_sum_to_total():
    n_seqs = 12
    gt, iou = _make_gt_iou_dicts(n_seqs)
    report = SequenceAnalyzer().analyze(gt, iou)
    total = sum(tp.num_sequences for tp in report["tier_performance"].values())
    assert total == n_seqs


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def test_markdown_contains_tier_names():
    gt, iou = _make_gt_iou_dicts(6)
    report = SequenceAnalyzer().analyze(gt, iou, tracker_name="MOSSE")
    md = report["markdown_table"]
    for tier in DifficultyTier:
        assert tier.value in md


def test_markdown_contains_sequence_names():
    gt, iou = _make_gt_iou_dicts(3)
    report = SequenceAnalyzer().analyze(gt, iou)
    md = report["markdown_table"]
    for name in gt:
        assert name in md


def test_tier_summary_contains_tier_names():
    gt, iou = _make_gt_iou_dicts(9)
    report = SequenceAnalyzer().analyze(gt, iou)
    summary = report["tier_summary"]
    for tier in DifficultyTier:
        assert tier.value.upper() in summary


# ---------------------------------------------------------------------------
# from_benchmark_result integration
# ---------------------------------------------------------------------------

def test_from_benchmark_result():
    """End-to-end: run BenchmarkEngine on SyntheticDataset, then analyze."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker

    dataset = SyntheticDataset(num_sequences=4, num_frames=30, motion="linear", seed=0)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")

    report = SequenceAnalyzer.from_benchmark_result(result)
    assert len(report["attributes"]) == 4
    assert all(tp.num_sequences >= 0 for tp in report["tier_performance"].values())
    assert "Sequence Difficulty Analysis" in report["markdown_table"]


def test_from_benchmark_result_no_gt_raises():
    """If no sequence results have ground_truths, raise ValueError."""
    from unittest.mock import MagicMock

    mock_result = MagicMock()
    mock_result.tracker_name = "FakeTracker"
    sr = MagicMock()
    sr.ground_truths = None
    sr.ious = np.array([0.5, 0.6])
    mock_result.sequence_results = [sr]

    with pytest.raises(ValueError, match="ground-truth arrays"):
        SequenceAnalyzer.from_benchmark_result(mock_result)
