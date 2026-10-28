"""Tests for eovot.analysis.difficulty."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pytest

from eovot.analysis.difficulty import (
    DIFFICULTY_TIERS,
    DifficultyResult,
    SequenceDifficultyProfiler,
    _score_to_tier,
    profile_benchmark_result,
)


# ---------------------------------------------------------------------------
# Helpers — minimal stubs so we can test without the full engine
# ---------------------------------------------------------------------------


@dataclass
class _FakeSeqResult:
    sequence_name: str
    ious: np.ndarray = field(default_factory=lambda: np.zeros(10))
    ground_truths: Optional[np.ndarray] = None


@dataclass
class _FakeBenchmarkResult:
    tracker_name: str = "FakeTracker"
    dataset_name: str = "FakeDataset"
    sequence_results: List[_FakeSeqResult] = field(default_factory=list)


def _static_gts(n: int, x=10.0, y=10.0, w=40.0, h=30.0) -> np.ndarray:
    """Generate a static sequence (no motion, no size change)."""
    return np.tile([x, y, w, h], (n, 1)).astype(np.float64)


def _linear_motion_gts(n: int, dx=5.0) -> np.ndarray:
    """Generate a left-to-right linearly moving target."""
    boxes = np.zeros((n, 4), dtype=np.float64)
    boxes[:, 0] = np.arange(n) * dx   # x moves
    boxes[:, 1] = 50.0                 # y static
    boxes[:, 2] = 40.0                 # w static
    boxes[:, 3] = 30.0                 # h static
    return boxes


def _scale_varying_gts(n: int) -> np.ndarray:
    """Generate GT boxes with oscillating size."""
    boxes = np.zeros((n, 4), dtype=np.float64)
    boxes[:, 0] = 50.0
    boxes[:, 1] = 50.0
    # Width oscillates between 20 and 80
    boxes[:, 2] = 20.0 + 60.0 * (0.5 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, n)))
    boxes[:, 3] = 30.0
    return boxes


def _aspect_deforming_gts(n: int) -> np.ndarray:
    """Generate GT boxes with wildly changing aspect ratio."""
    boxes = np.zeros((n, 4), dtype=np.float64)
    boxes[:, 0] = 50.0
    boxes[:, 1] = 50.0
    boxes[:, 2] = 10.0 + 70.0 * np.abs(np.sin(np.linspace(0, 8 * np.pi, n)))
    boxes[:, 3] = 30.0
    return boxes


def _occluded_gts(n: int) -> np.ndarray:
    """Generate GT with periodic abrupt area drops."""
    boxes = _static_gts(n, w=60.0, h=60.0)
    # Every 10 frames: area drops by 60 % (abrupt occlusion proxy)
    for i in range(10, n, 10):
        boxes[i, 2] = 10.0
        boxes[i, 3] = 10.0
    return boxes


# ---------------------------------------------------------------------------
# _score_to_tier
# ---------------------------------------------------------------------------


class TestScoreToTier:
    def test_zero_is_easy(self):
        assert _score_to_tier(0.0) == "easy"

    def test_just_below_medium(self):
        assert _score_to_tier(0.249) == "easy"

    def test_at_medium(self):
        assert _score_to_tier(0.25) == "medium"

    def test_just_below_hard(self):
        assert _score_to_tier(0.499) == "medium"

    def test_at_hard(self):
        assert _score_to_tier(0.50) == "hard"

    def test_at_extreme(self):
        assert _score_to_tier(0.75) == "extreme"

    def test_one_is_extreme(self):
        assert _score_to_tier(1.0) == "extreme"


# ---------------------------------------------------------------------------
# SequenceDifficultyProfiler construction
# ---------------------------------------------------------------------------


class TestProfilerConstruction:
    def test_default_weights_sum_to_one(self):
        p = SequenceDifficultyProfiler()
        assert abs(p._weights.sum() - 1.0) < 1e-9

    def test_custom_weights_normalised(self):
        p = SequenceDifficultyProfiler(
            motion_weight=2.0, scale_weight=1.0, deform_weight=1.0, occlusion_weight=0.0
        )
        assert abs(p._weights.sum() - 1.0) < 1e-9

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            SequenceDifficultyProfiler(motion_weight=-1.0)

    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError):
            SequenceDifficultyProfiler(
                motion_weight=0.0, scale_weight=0.0,
                deform_weight=0.0, occlusion_weight=0.0,
            )


# ---------------------------------------------------------------------------
# profile_from_gts — input validation
# ---------------------------------------------------------------------------


class TestProfileFromGtsValidation:
    def test_single_row_raises(self):
        gts = np.array([[10.0, 20.0, 30.0, 40.0]])
        with pytest.raises(ValueError):
            SequenceDifficultyProfiler().profile_from_gts(gts)

    def test_wrong_shape_raises(self):
        gts = np.ones((10, 3))
        with pytest.raises(ValueError):
            SequenceDifficultyProfiler().profile_from_gts(gts)

    def test_two_rows_ok(self):
        gts = np.array([[10.0, 20.0, 30.0, 40.0], [15.0, 20.0, 30.0, 40.0]])
        result = SequenceDifficultyProfiler().profile_from_gts(gts)
        assert isinstance(result, DifficultyResult)


# ---------------------------------------------------------------------------
# Motion speed metric
# ---------------------------------------------------------------------------


class TestMotionSpeed:
    def test_static_sequence_has_zero_motion(self):
        gts = _static_gts(50)
        p = SequenceDifficultyProfiler()
        assert p._compute_motion_speed(gts) == pytest.approx(0.0, abs=1e-9)

    def test_fast_motion_higher_than_slow(self):
        slow = _linear_motion_gts(50, dx=1.0)
        fast = _linear_motion_gts(50, dx=20.0)
        p = SequenceDifficultyProfiler()
        assert p._compute_motion_speed(fast) > p._compute_motion_speed(slow)

    def test_motion_nonnegative(self):
        gts = _linear_motion_gts(100, dx=3.0)
        p = SequenceDifficultyProfiler()
        assert p._compute_motion_speed(gts) >= 0.0


# ---------------------------------------------------------------------------
# Scale volatility metric
# ---------------------------------------------------------------------------


class TestScaleVolatility:
    def test_constant_size_has_zero_cv(self):
        gts = _static_gts(50)
        p = SequenceDifficultyProfiler()
        assert p._compute_scale_volatility(gts) == pytest.approx(0.0, abs=1e-9)

    def test_varying_size_has_nonzero_cv(self):
        gts = _scale_varying_gts(100)
        p = SequenceDifficultyProfiler()
        assert p._compute_scale_volatility(gts) > 0.05


# ---------------------------------------------------------------------------
# Aspect deformation metric
# ---------------------------------------------------------------------------


class TestAspectDeformation:
    def test_constant_ratio_has_zero_std(self):
        gts = _static_gts(50)
        p = SequenceDifficultyProfiler()
        assert p._compute_aspect_deformation(gts) == pytest.approx(0.0, abs=1e-9)

    def test_deforming_has_nonzero_std(self):
        gts = _aspect_deforming_gts(100)
        p = SequenceDifficultyProfiler()
        assert p._compute_aspect_deformation(gts) > 0.1


# ---------------------------------------------------------------------------
# Occlusion events metric
# ---------------------------------------------------------------------------


class TestOcclusionEvents:
    def test_static_has_no_occlusion(self):
        gts = _static_gts(100)
        p = SequenceDifficultyProfiler()
        assert p._compute_occlusion_events(gts) == pytest.approx(0.0)

    def test_periodic_drops_detected(self):
        gts = _occluded_gts(100)
        p = SequenceDifficultyProfiler()
        rate = p._compute_occlusion_events(gts)
        assert rate > 0.0


# ---------------------------------------------------------------------------
# profile_from_gts — end-to-end
# ---------------------------------------------------------------------------


class TestProfileFromGts:
    def test_static_is_easy(self):
        gts = _static_gts(200)
        result = SequenceDifficultyProfiler().profile_from_gts(gts, "static")
        assert result.difficulty_tier == "easy"
        assert result.difficulty_score < 0.25

    def test_result_fields_populated(self):
        gts = _linear_motion_gts(80, dx=5.0)
        result = SequenceDifficultyProfiler().profile_from_gts(gts, "seq1")
        assert result.sequence_name == "seq1"
        assert result.num_frames == 80
        assert result.difficulty_score >= 0.0
        assert result.difficulty_score <= 1.0
        assert result.difficulty_tier in {"easy", "medium", "hard", "extreme"}

    def test_fast_motion_harder_than_static(self):
        static = SequenceDifficultyProfiler().profile_from_gts(_static_gts(100), "s")
        moving = SequenceDifficultyProfiler().profile_from_gts(_linear_motion_gts(100, dx=15.0), "m")
        assert moving.difficulty_score > static.difficulty_score

    def test_str_representation(self):
        gts = _static_gts(20)
        result = SequenceDifficultyProfiler().profile_from_gts(gts, "test_seq")
        s = str(result)
        assert "test_seq" in s
        assert "score=" in s

    def test_to_dict_keys(self):
        gts = _static_gts(30)
        result = SequenceDifficultyProfiler().profile_from_gts(gts, "seq")
        d = result.to_dict()
        for key in ("sequence_name", "num_frames", "difficulty_score", "difficulty_tier",
                    "motion_speed", "scale_volatility", "aspect_deformation",
                    "occlusion_events_per_100fr"):
            assert key in d


# ---------------------------------------------------------------------------
# profile_benchmark
# ---------------------------------------------------------------------------


class TestProfileBenchmark:
    def _make_br(self, seqs):
        br = _FakeBenchmarkResult()
        for name, gts in seqs:
            br.sequence_results.append(_FakeSeqResult(sequence_name=name, ground_truths=gts))
        return br

    def test_basic(self):
        br = self._make_br([
            ("seq1", _static_gts(50)),
            ("seq2", _linear_motion_gts(50, dx=8.0)),
        ])
        results = SequenceDifficultyProfiler().profile_benchmark(br)
        assert set(results) == {"seq1", "seq2"}

    def test_sequence_without_gts_skipped(self):
        br = self._make_br([("seq_no_gt", None), ("seq_ok", _static_gts(20))])
        results = SequenceDifficultyProfiler().profile_benchmark(br)
        assert "seq_no_gt" not in results
        assert "seq_ok" in results

    def test_short_sequence_skipped(self):
        single = np.array([[10.0, 20.0, 30.0, 40.0]])   # only 1 box → skipped
        br = self._make_br([("short", single), ("ok", _static_gts(10))])
        results = SequenceDifficultyProfiler().profile_benchmark(br)
        assert "short" not in results

    def test_empty_benchmark_returns_empty(self):
        br = _FakeBenchmarkResult()
        results = SequenceDifficultyProfiler().profile_benchmark(br)
        assert results == {}


# ---------------------------------------------------------------------------
# aggregate_stats
# ---------------------------------------------------------------------------


class TestAggregateStats:
    def test_empty_returns_empty(self):
        assert SequenceDifficultyProfiler().aggregate_stats({}) == {}

    def test_keys_present(self):
        p = SequenceDifficultyProfiler()
        gts = _static_gts(50)
        results = {
            "s1": p.profile_from_gts(gts, "s1"),
            "s2": p.profile_from_gts(_linear_motion_gts(50), "s2"),
        }
        stats = p.aggregate_stats(results)
        for key in ("num_sequences", "mean_difficulty_score", "std_difficulty_score",
                    "mean_motion_speed", "tier_distribution"):
            assert key in stats
        assert stats["num_sequences"] == 2

    def test_tier_counts_sum_to_total(self):
        p = SequenceDifficultyProfiler()
        gts_list = [
            ("a", _static_gts(100)),
            ("b", _linear_motion_gts(100, dx=10.0)),
            ("c", _scale_varying_gts(100)),
        ]
        results = {name: p.profile_from_gts(gts, name) for name, gts in gts_list}
        stats = p.aggregate_stats(results)
        tier_total = sum(stats["tier_distribution"].values())
        assert tier_total == 3


# ---------------------------------------------------------------------------
# to_markdown_table
# ---------------------------------------------------------------------------


class TestToMarkdownTable:
    def test_contains_sequence_names(self):
        p = SequenceDifficultyProfiler()
        results = {
            "alpha": p.profile_from_gts(_static_gts(50), "alpha"),
            "beta": p.profile_from_gts(_linear_motion_gts(50), "beta"),
        }
        table = p.to_markdown_table(results)
        assert "alpha" in table
        assert "beta" in table

    def test_sorted_hardest_first(self):
        p = SequenceDifficultyProfiler()
        easy = p.profile_from_gts(_static_gts(100), "easy_seq")
        hard = p.profile_from_gts(_linear_motion_gts(100, dx=20.0), "hard_seq")
        results = {"easy_seq": easy, "hard_seq": hard}
        table = p.to_markdown_table(results)
        lines = [l for l in table.split("\n") if "|" in l and "Sequence" not in l and "---" not in l]
        if lines and hard.difficulty_score > easy.difficulty_score:
            assert "hard_seq" in lines[0]


# ---------------------------------------------------------------------------
# profile_benchmark_result convenience function
# ---------------------------------------------------------------------------


class TestProfileBenchmarkResultConvenienceFn:
    def test_uses_default_profiler(self):
        br = _FakeBenchmarkResult()
        br.sequence_results.append(
            _FakeSeqResult("s1", ground_truths=_static_gts(20))
        )
        results = profile_benchmark_result(br)
        assert "s1" in results

    def test_accepts_custom_profiler(self):
        custom = SequenceDifficultyProfiler(motion_weight=1.0, scale_weight=0.0,
                                             deform_weight=0.0, occlusion_weight=0.0)
        br = _FakeBenchmarkResult()
        br.sequence_results.append(
            _FakeSeqResult("s1", ground_truths=_linear_motion_gts(30))
        )
        results = profile_benchmark_result(br, profiler=custom)
        assert "s1" in results
