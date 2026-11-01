"""Tests for eovot.metrics.complexity — SequenceComplexityScorer and ComplexityAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.complexity import (
    ComplexityAnalyzer,
    ComplexityResult,
    DifficultyTier,
    SequenceComplexityScorer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _static_gt(n: int = 50, x: float = 100.0, y: float = 80.0,
               w: float = 40.0, h: float = 30.0) -> np.ndarray:
    """GT boxes that never move — should score EASY."""
    return np.tile([x, y, w, h], (n, 1)).astype(np.float64)


def _linear_gt(n: int = 100, speed: float = 5.0) -> np.ndarray:
    """GT boxes moving at a constant speed — moderate motion."""
    boxes = []
    x, y = 100.0, 80.0
    for i in range(n):
        boxes.append([x + i * speed, y, 40.0, 30.0])
    return np.array(boxes, dtype=np.float64)


def _scale_changing_gt(n: int = 60, start_size: float = 20.0,
                       end_size: float = 200.0) -> np.ndarray:
    """GT boxes that grow from start_size to end_size."""
    sizes = np.linspace(start_size, end_size, n)
    return np.column_stack([
        np.full(n, 100.0),
        np.full(n, 80.0),
        sizes,
        sizes,
    ]).astype(np.float64)


def _aspect_varying_gt(n: int = 60) -> np.ndarray:
    """GT boxes with oscillating aspect ratio."""
    boxes = []
    for i in range(n):
        w = 40.0 + 20.0 * np.sin(i * np.pi / 10)
        h = 30.0 + 20.0 * np.cos(i * np.pi / 10)
        boxes.append([100.0, 80.0, w, h])
    return np.array(boxes, dtype=np.float64)


# ---------------------------------------------------------------------------
# DifficultyTier
# ---------------------------------------------------------------------------

class TestDifficultyTier:
    def test_easy_threshold(self):
        assert DifficultyTier.from_score(0.0) == DifficultyTier.EASY
        assert DifficultyTier.from_score(0.24) == DifficultyTier.EASY

    def test_medium_threshold(self):
        assert DifficultyTier.from_score(0.25) == DifficultyTier.MEDIUM
        assert DifficultyTier.from_score(0.54) == DifficultyTier.MEDIUM

    def test_hard_threshold(self):
        assert DifficultyTier.from_score(0.55) == DifficultyTier.HARD
        assert DifficultyTier.from_score(1.0) == DifficultyTier.HARD


# ---------------------------------------------------------------------------
# SequenceComplexityScorer — component metrics
# ---------------------------------------------------------------------------

class TestMotionIntensity:
    def test_static_target_zero_motion(self):
        scorer = SequenceComplexityScorer()
        gt = _static_gt(50)
        assert scorer.motion_intensity(gt) == pytest.approx(0.0, abs=1e-6)

    def test_moving_target_positive_motion(self):
        scorer = SequenceComplexityScorer()
        gt = _linear_gt(50, speed=5.0)
        mi = scorer.motion_intensity(gt)
        assert mi > 0.0

    def test_single_frame_returns_zero(self):
        scorer = SequenceComplexityScorer()
        gt = np.array([[50.0, 50.0, 30.0, 20.0]])
        assert scorer.motion_intensity(gt) == pytest.approx(0.0)

    def test_fast_motion_higher_than_slow(self):
        scorer = SequenceComplexityScorer()
        slow = _linear_gt(50, speed=1.0)
        fast = _linear_gt(50, speed=10.0)
        assert scorer.motion_intensity(fast) > scorer.motion_intensity(slow)


class TestScaleChange:
    def test_constant_size_zero_change(self):
        scorer = SequenceComplexityScorer()
        gt = _static_gt(50)
        assert scorer.scale_change(gt) == pytest.approx(0.0, abs=1e-6)

    def test_growing_target_positive_change(self):
        scorer = SequenceComplexityScorer()
        gt = _scale_changing_gt(50, start_size=20.0, end_size=200.0)
        sc = scorer.scale_change(gt)
        assert sc > 0.0

    def test_100x_change_clips_to_one(self):
        scorer = SequenceComplexityScorer()
        boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 100.0, 100.0]], dtype=np.float64)
        sc = scorer.scale_change(boxes)
        assert sc == pytest.approx(1.0, abs=1e-6)

    def test_scale_change_bounded(self):
        scorer = SequenceComplexityScorer()
        gt = _scale_changing_gt(60, start_size=5.0, end_size=5000.0)
        sc = scorer.scale_change(gt)
        assert 0.0 <= sc <= 1.0


class TestARVariation:
    def test_constant_aspect_ratio_zero(self):
        scorer = SequenceComplexityScorer()
        gt = _static_gt(50)
        assert scorer.ar_variation(gt) == pytest.approx(0.0, abs=1e-6)

    def test_varying_ar_positive(self):
        scorer = SequenceComplexityScorer()
        gt = _aspect_varying_gt(60)
        arv = scorer.ar_variation(gt)
        assert arv > 0.0

    def test_single_valid_box_zero(self):
        scorer = SequenceComplexityScorer()
        gt = np.array([[0.0, 0.0, 40.0, 30.0]])
        assert scorer.ar_variation(gt) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# SequenceComplexityScorer — composite score
# ---------------------------------------------------------------------------

class TestScoreMethod:
    def test_static_sequence_is_easy(self):
        scorer = SequenceComplexityScorer()
        result = scorer.score(_static_gt(50), sequence_name="static")
        assert result.tier == DifficultyTier.EASY
        assert result.complexity_score < 0.25

    def test_fast_moving_scores_higher(self):
        scorer = SequenceComplexityScorer()
        slow = scorer.score(_linear_gt(100, speed=1.0))
        fast = scorer.score(_linear_gt(100, speed=15.0))
        assert fast.complexity_score > slow.complexity_score

    def test_score_in_unit_range(self):
        scorer = SequenceComplexityScorer()
        gts = [_static_gt(50), _linear_gt(100, 8.0), _scale_changing_gt(60)]
        for gt in gts:
            r = scorer.score(gt)
            assert 0.0 <= r.complexity_score <= 1.0

    def test_result_fields_populated(self):
        scorer = SequenceComplexityScorer()
        result = scorer.score(_linear_gt(80, 3.0), sequence_name="test_seq")
        assert result.sequence_name == "test_seq"
        assert result.num_frames == 80
        assert isinstance(result.tier, DifficultyTier)

    def test_combined_hard_sequence(self):
        scorer = SequenceComplexityScorer()
        # Tiny box (strong motion normalization) + fast motion + 100× scale change → HARD
        # size grows from 5 to ~49 (100× area change, SC ≈ 1.0)
        # box moves 20 px/frame → high MI relative to the small starting diagonal
        boxes = []
        for i in range(60):
            size = 5.0 + i * 0.75
            boxes.append([i * 20.0, 80.0, size, size])
        gt = np.array(boxes, dtype=np.float64)
        result = scorer.score(gt)
        assert result.tier == DifficultyTier.HARD


class TestScoreMany:
    def test_returns_sorted_by_score(self):
        scorer = SequenceComplexityScorer()
        seqs = {
            "easy": _static_gt(50),
            "fast": _linear_gt(100, speed=12.0),
            "medium": _linear_gt(100, speed=3.0),
        }
        results = scorer.score_many(seqs)
        scores = [r.complexity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_all_sequences(self):
        scorer = SequenceComplexityScorer()
        seqs = {f"seq_{i}": _linear_gt(50, speed=float(i)) for i in range(5)}
        results = scorer.score_many(seqs)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# SequenceComplexityScorer — weight validation
# ---------------------------------------------------------------------------

class TestWeightValidation:
    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            SequenceComplexityScorer(motion_weight=0.5, scale_weight=0.5, ar_weight=0.5)

    def test_custom_weights_accepted(self):
        scorer = SequenceComplexityScorer(motion_weight=0.6, scale_weight=0.2, ar_weight=0.2)
        result = scorer.score(_linear_gt(50, 5.0))
        assert 0.0 <= result.complexity_score <= 1.0


# ---------------------------------------------------------------------------
# ComplexityResult — serialisation
# ---------------------------------------------------------------------------

class TestComplexityResult:
    def test_to_dict_keys(self):
        scorer = SequenceComplexityScorer()
        result = scorer.score(_linear_gt(60, 4.0), "test")
        d = result.to_dict()
        expected_keys = {
            "sequence_name", "complexity_score", "tier",
            "motion_intensity", "scale_change", "ar_variation", "num_frames",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_tier_is_string(self):
        scorer = SequenceComplexityScorer()
        result = scorer.score(_static_gt(30))
        d = result.to_dict()
        assert isinstance(d["tier"], str)
        assert d["tier"] in ("EASY", "MEDIUM", "HARD")

    def test_str_repr(self):
        scorer = SequenceComplexityScorer()
        result = scorer.score(_linear_gt(50, 5.0), "my_seq")
        s = str(result)
        assert "my_seq" in s
        assert "score=" in s


# ---------------------------------------------------------------------------
# ComplexityAnalyzer — integration with BenchmarkResult
# ---------------------------------------------------------------------------

def _make_fake_benchmark_result():
    """Build a minimal BenchmarkResult-like object for testing."""
    from unittest.mock import MagicMock
    from eovot.benchmark.engine import BenchmarkResult, SequenceResult
    from eovot.profiling.profiler import ProfilingResult
    from eovot.metrics.accuracy import AccuracyMetrics

    seq_results = []
    gts = {
        "seq_static": _static_gt(50),
        "seq_slow": _linear_gt(80, 2.0),
        "seq_fast": _linear_gt(80, 12.0),
        "seq_scale": _scale_changing_gt(60, 20.0, 200.0),
    }
    for name, gt in gts.items():
        prof = ProfilingResult(
            tracker_name="test",
            frame_count=len(gt),
            fps=100.0,
            latency_mean_ms=10.0,
            latency_std_ms=1.0,
            latency_p95_ms=12.0,
            peak_memory_mb=50.0,
        )
        acc = AccuracyMetrics(
            mean_iou=0.6,
            success_auc=0.55,
            precision_auc=0.60,
        )
        sr = SequenceResult(
            sequence_name=name,
            ious=np.full(len(gt), 0.6),
            profiling=prof,
            ground_truths=gt,
            accuracy_metrics=acc,
        )
        seq_results.append(sr)

    result = BenchmarkResult(tracker_name="test_tracker", dataset_name="test_ds")
    result.sequence_results = seq_results
    return result


class TestComplexityAnalyzer:
    def test_breakdown_returns_required_keys(self):
        analyzer = ComplexityAnalyzer()
        result = _make_fake_benchmark_result()
        bd = analyzer.breakdown(result)
        assert "per_sequence" in bd
        assert "tier_summary" in bd
        assert "overall_complexity" in bd

    def test_per_sequence_all_scored(self):
        analyzer = ComplexityAnalyzer()
        result = _make_fake_benchmark_result()
        bd = analyzer.breakdown(result)
        assert len(bd["per_sequence"]) == 4

    def test_tier_summary_has_all_tiers(self):
        analyzer = ComplexityAnalyzer()
        result = _make_fake_benchmark_result()
        bd = analyzer.breakdown(result)
        assert "EASY" in bd["tier_summary"]
        assert "MEDIUM" in bd["tier_summary"]
        assert "HARD" in bd["tier_summary"]

    def test_overall_complexity_in_unit_range(self):
        analyzer = ComplexityAnalyzer()
        result = _make_fake_benchmark_result()
        bd = analyzer.breakdown(result)
        assert 0.0 <= bd["overall_complexity"] <= 1.0

    def test_tier_summary_has_mean_iou(self):
        analyzer = ComplexityAnalyzer()
        result = _make_fake_benchmark_result()
        bd = analyzer.breakdown(result)
        for tier_name, stats in bd["tier_summary"].items():
            if stats.get("count", 0) > 0:
                assert "mean_iou" in stats
                assert 0.0 <= stats["mean_iou"] <= 1.0

    def test_empty_result_handled(self):
        from eovot.benchmark.engine import BenchmarkResult
        analyzer = ComplexityAnalyzer()
        empty = BenchmarkResult(tracker_name="t", dataset_name="d")
        bd = analyzer.breakdown(empty)
        assert bd["per_sequence"] == {}
        assert bd["overall_complexity"] == 0.0

    def test_sequences_without_gt_skipped(self):
        from eovot.benchmark.engine import BenchmarkResult, SequenceResult
        from eovot.profiling.profiler import ProfilingResult

        prof = ProfilingResult(
            tracker_name="t", frame_count=10, fps=50.0,
            latency_mean_ms=20.0, latency_std_ms=1.0,
            latency_p95_ms=22.0, peak_memory_mb=40.0,
        )
        sr_no_gt = SequenceResult(
            sequence_name="no_gt",
            ious=np.array([0.5] * 10),
            profiling=prof,
            ground_truths=None,
        )
        result = BenchmarkResult(tracker_name="t", dataset_name="d")
        result.sequence_results = [sr_no_gt]

        analyzer = ComplexityAnalyzer()
        bd = analyzer.breakdown(result)
        assert len(bd["per_sequence"]) == 0

    def test_select_representative_subset(self):
        analyzer = ComplexityAnalyzer()
        result = _make_fake_benchmark_result()
        subset = analyzer.select_representative_subset(result, n_easy=1, n_medium=1, n_hard=1)
        # At most 3 sequences selected (1 per tier), may be fewer if a tier is empty
        assert len(subset) <= 3
        # All returned names should be valid sequence names
        seq_names = {sr.sequence_name for sr in result.sequence_results}
        for name in subset:
            assert name in seq_names
