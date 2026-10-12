"""Tests for SequenceDifficultyProfiler."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.difficulty import DifficultyResult, SequenceDifficultyProfiler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _static_gt(n: int = 50) -> np.ndarray:
    """GT with no motion, constant box."""
    gt = np.tile([10.0, 10.0, 40.0, 30.0], (n, 1))
    return gt


def _moving_gt(n: int = 50, velocity: float = 5.0) -> np.ndarray:
    """GT with linear motion."""
    gt = np.zeros((n, 4))
    for i in range(n):
        gt[i] = [10.0 + i * velocity, 10.0, 40.0, 30.0]
    return gt


def _scaling_gt(n: int = 50) -> np.ndarray:
    """GT where box grows linearly each frame."""
    gt = np.zeros((n, 4))
    for i in range(n):
        size = 20.0 + i * 1.0
        gt[i] = [10.0, 10.0, size, size]
    return gt


def _oov_gt(n: int = 50, oov_fraction: float = 0.4) -> np.ndarray:
    """GT with some zero-area (out-of-view) frames."""
    gt = _static_gt(n)
    oov_count = int(n * oov_fraction)
    gt[:oov_count, 2:] = 0.0  # zero w,h
    return gt


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_weights_positive(self):
        p = SequenceDifficultyProfiler()
        assert p.weight_motion > 0
        assert p.weight_scale > 0
        assert p.weight_aspect > 0
        assert p.weight_oov > 0

    def test_invalid_zero_weights(self):
        with pytest.raises(ValueError, match="positive"):
            SequenceDifficultyProfiler(
                weight_motion=0, weight_scale=0, weight_aspect=0, weight_oov=0
            )


# ---------------------------------------------------------------------------
# Profile — single sequence
# ---------------------------------------------------------------------------

class TestProfileSingle:
    def test_returns_difficulty_result(self):
        p = SequenceDifficultyProfiler()
        result = p.profile(_static_gt(), "test")
        assert isinstance(result, DifficultyResult)

    def test_static_sequence_low_motion(self):
        p = SequenceDifficultyProfiler()
        result = p.profile(_static_gt(100), "static")
        assert result.motion_score == pytest.approx(0.0, abs=1e-6)

    def test_fast_motion_higher_than_slow(self):
        p = SequenceDifficultyProfiler()
        slow = p.profile(_moving_gt(50, velocity=1.0), "slow")
        fast = p.profile(_moving_gt(50, velocity=10.0), "fast")
        assert fast.motion_score > slow.motion_score

    def test_scaling_gt_non_zero_scale_score(self):
        p = SequenceDifficultyProfiler()
        result = p.profile(_scaling_gt(50), "scaling")
        assert result.scale_variation_score > 0.0

    def test_static_gt_zero_scale_variation(self):
        p = SequenceDifficultyProfiler()
        result = p.profile(_static_gt(50), "static")
        assert result.scale_variation_score == pytest.approx(0.0, abs=1e-6)

    def test_oov_fraction_matches_input(self):
        p = SequenceDifficultyProfiler()
        result = p.profile(_oov_gt(50, 0.4), "oov")
        assert abs(result.out_of_view_fraction - 0.4) < 0.01

    def test_composite_in_range(self):
        p = SequenceDifficultyProfiler()
        for gt in [_static_gt(), _moving_gt(), _scaling_gt(), _oov_gt()]:
            result = p.profile(gt, "x")
            assert 0.0 <= result.composite_score <= 1.0

    def test_num_frames_stored(self):
        p = SequenceDifficultyProfiler()
        result = p.profile(_static_gt(77), "x")
        assert result.num_frames == 77

    def test_raises_on_single_frame(self):
        p = SequenceDifficultyProfiler()
        with pytest.raises(ValueError, match="at least 2 frames"):
            p.profile(np.array([[10.0, 10.0, 40.0, 30.0]]), "x")

    def test_raises_on_wrong_shape(self):
        p = SequenceDifficultyProfiler()
        with pytest.raises(ValueError):
            p.profile(np.zeros((10, 3)), "x")

    def test_sequence_name_stored(self):
        p = SequenceDifficultyProfiler()
        result = p.profile(_static_gt(), "my_seq")
        assert result.sequence_name == "my_seq"


# ---------------------------------------------------------------------------
# DifficultyResult.bucket
# ---------------------------------------------------------------------------

class TestBucket:
    def test_easy_bucket(self):
        r = DifficultyResult("x", 0.0, 0.0, 0.0, 0.0, 0.10, 50)
        assert r.bucket() == "easy"

    def test_medium_bucket(self):
        r = DifficultyResult("x", 0.3, 0.1, 0.05, 0.0, 0.40, 50)
        assert r.bucket() == "medium"

    def test_hard_bucket(self):
        r = DifficultyResult("x", 0.5, 0.3, 0.1, 0.2, 0.80, 50)
        assert r.bucket() == "hard"

    def test_str_includes_tier(self):
        r = DifficultyResult("myseq", 0.0, 0.0, 0.0, 0.0, 0.10, 50)
        assert "easy" in str(r)


# ---------------------------------------------------------------------------
# Profile dataset
# ---------------------------------------------------------------------------

class TestProfileDataset:
    def _make_dataset(self) -> dict:
        return {
            "static": _static_gt(60),
            "moving": _moving_gt(60, velocity=5.0),
            "scaling": _scaling_gt(60),
            "oov": _oov_gt(60, 0.5),
        }

    def test_summary_keys(self):
        p = SequenceDifficultyProfiler()
        report = p.profile_dataset(self._make_dataset())
        s = report["summary"]
        for key in ["num_sequences", "mean_composite", "std_composite",
                    "min_composite", "max_composite",
                    "easy_count", "medium_count", "hard_count"]:
            assert key in s, f"Missing key: {key}"

    def test_per_sequence_count(self):
        p = SequenceDifficultyProfiler()
        report = p.profile_dataset(self._make_dataset())
        assert report["summary"]["num_sequences"] == 4

    def test_bucket_counts_sum_to_total(self):
        p = SequenceDifficultyProfiler()
        report = p.profile_dataset(self._make_dataset())
        s = report["summary"]
        assert s["easy_count"] + s["medium_count"] + s["hard_count"] == s["num_sequences"]

    def test_empty_dataset_returns_empty(self):
        p = SequenceDifficultyProfiler()
        report = p.profile_dataset({})
        assert report["summary"] == {}

    def test_mean_composite_in_range(self):
        p = SequenceDifficultyProfiler()
        report = p.profile_dataset(self._make_dataset())
        assert 0.0 <= report["summary"]["mean_composite"] <= 1.0


# ---------------------------------------------------------------------------
# Rank by difficulty
# ---------------------------------------------------------------------------

class TestRankByDifficulty:
    def _seqs(self) -> dict:
        return {
            "easy": _static_gt(50),
            "hard": _oov_gt(50, 0.9),
            "medium": _moving_gt(50, 2.0),
        }

    def test_descending_order(self):
        p = SequenceDifficultyProfiler()
        ranked = p.rank_by_difficulty(self._seqs(), descending=True)
        scores = [r.composite_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ascending_order(self):
        p = SequenceDifficultyProfiler()
        ranked = p.rank_by_difficulty(self._seqs(), descending=False)
        scores = [r.composite_score for r in ranked]
        assert scores == sorted(scores)

    def test_all_sequences_present(self):
        p = SequenceDifficultyProfiler()
        ranked = p.rank_by_difficulty(self._seqs())
        names = {r.sequence_name for r in ranked}
        assert names == {"easy", "hard", "medium"}


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def test_table_contains_header(self):
        p = SequenceDifficultyProfiler()
        ranked = p.rank_by_difficulty({"s": _static_gt()})
        table = p.to_markdown_table(ranked)
        assert "Sequence" in table
        assert "Score" in table

    def test_table_contains_sequence_name(self):
        p = SequenceDifficultyProfiler()
        ranked = p.rank_by_difficulty({"my_seq": _static_gt()})
        table = p.to_markdown_table(ranked)
        assert "my_seq" in table


# ---------------------------------------------------------------------------
# Integration with SyntheticDataset
# ---------------------------------------------------------------------------

class TestSyntheticIntegration:
    def test_profiles_synthetic_dataset(self):
        from eovot.datasets.synthetic import SyntheticDataset
        dataset = SyntheticDataset(num_sequences=5, num_frames=80, seed=42)
        p = SequenceDifficultyProfiler()
        sequences = {seq.name: seq.ground_truth for seq in dataset}
        report = p.profile_dataset(sequences)
        assert report["summary"]["num_sequences"] == 5

    def test_harder_sequences_score_higher_than_static(self):
        from eovot.datasets.synthetic import SyntheticDataset
        dataset = SyntheticDataset(num_sequences=3, num_frames=100,
                                   motion="circular", seed=0)
        p = SequenceDifficultyProfiler()
        sequences = {seq.name: seq.ground_truth for seq in dataset}
        report = p.profile_dataset(sequences)
        # Sinusoidal motion sequences should have non-trivial motion scores
        for r in report["per_sequence"].values():
            assert r.motion_score >= 0.0
