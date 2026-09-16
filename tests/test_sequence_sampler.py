"""Tests for eovot.datasets.sampler."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.datasets.sampler import (
    ScoredSequence,
    SequenceDifficultyScorer,
    StratifiedSampler,
)


def _make_gt(n=50, speed=1.0, scale_var=0.0, seed=0):
    """Synthetic ground-truth array with controllable difficulty properties."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, speed, n)) + 160.0
    y = np.cumsum(rng.normal(0, speed, n)) + 120.0
    w = 40.0 + rng.normal(0, scale_var, n)
    h = 40.0 + rng.normal(0, scale_var, n)
    return np.stack([x, y, np.maximum(w, 5.0), np.maximum(h, 5.0)], axis=1)


def _make_mock_seq(name, gt):
    seq = MagicMock()
    seq.name = name
    seq.ground_truth = gt
    return seq


class TestSequenceDifficultyScorer:
    def setup_method(self):
        self.scorer = SequenceDifficultyScorer()

    def test_score_in_range(self):
        gt = _make_gt()
        result = self.scorer.score_sequence(gt, "test")
        assert 0.0 <= result.difficulty_score <= 1.0

    def test_fast_sequence_harder_than_slow(self):
        slow = self.scorer.score_sequence(_make_gt(speed=0.1), "slow")
        fast = self.scorer.score_sequence(_make_gt(speed=50.0), "fast")
        assert fast.difficulty_score > slow.difficulty_score

    def test_high_scale_variability_harder(self):
        stable   = self.scorer.score_sequence(_make_gt(scale_var=0.0),  "stable")
        variable = self.scorer.score_sequence(_make_gt(scale_var=30.0), "variable")
        assert variable.difficulty_score > stable.difficulty_score

    def test_num_frames_recorded(self):
        result = self.scorer.score_sequence(_make_gt(n=75), "seq")
        assert result.num_frames == 75

    def test_all_components_present(self):
        result = self.scorer.score_sequence(_make_gt(), "seq")
        for comp in ("motion_speed", "scale_variability", "aspect_change", "target_smallness"):
            assert comp in result.components

    def test_single_frame_no_crash(self):
        result = self.scorer.score_sequence(_make_gt(n=1), "one")
        assert 0.0 <= result.difficulty_score <= 1.0

    def test_two_frames_no_crash(self):
        result = self.scorer.score_sequence(_make_gt(n=2), "two")
        assert 0.0 <= result.difficulty_score <= 1.0

    def test_score_dataset_sorted_ascending(self):
        class FakeDataset:
            def __len__(self): return 3
            def __getitem__(self, idx):
                speeds = [0.1, 50.0, 5.0]
                return _make_mock_seq(f"seq{idx}", _make_gt(speed=speeds[idx]))

        scored = self.scorer.score_dataset(FakeDataset())
        diffs = [s.difficulty_score for s in scored]
        assert diffs == sorted(diffs)

    def test_invalid_all_zero_weights_raises(self):
        with pytest.raises(ValueError):
            SequenceDifficultyScorer(weights={"motion_speed": 0.0})

    def test_custom_weights_accepted(self):
        scorer = SequenceDifficultyScorer(
            weights={"motion_speed": 2.0, "scale_variability": 1.0}
        )
        result = scorer.score_sequence(_make_gt(), "seq")
        assert 0.0 <= result.difficulty_score <= 1.0

    def test_small_target_harder_than_large(self):
        small_gt = np.array([[100.0, 100.0, 5.0, 5.0]] * 20)
        large_gt = np.array([[100.0, 100.0, 200.0, 200.0]] * 20)
        small = self.scorer.score_sequence(small_gt, "small")
        large = self.scorer.score_sequence(large_gt, "large")
        assert small.difficulty_score > large.difficulty_score

    def test_stationary_target_has_low_motion_score(self):
        static_gt = np.array([[100.0, 100.0, 40.0, 40.0]] * 30)
        result = self.scorer.score_sequence(static_gt, "static")
        assert result.components["motion_speed"] < 0.05


class TestStratifiedSampler:
    def _make_scored(self, n=20):
        return [
            ScoredSequence(name=f"seq{i:03d}", difficulty_score=i / (n - 1), num_frames=50)
            for i in range(n)
        ]

    def setup_method(self):
        self.sampler = StratifiedSampler(n_bins=5, seed=42)

    def test_sample_count(self):
        scored = self._make_scored(20)
        result = self.sampler.sample(scored, n=10)
        assert len(result) == 10

    def test_sample_all(self):
        scored = self._make_scored(10)
        result = self.sampler.sample(scored, n=10)
        assert len(result) == 10

    def test_sample_more_than_available_clamped(self):
        scored = self._make_scored(5)
        result = self.sampler.sample(scored, n=20)
        assert len(result) == 5

    def test_sample_empty_list(self):
        assert self.sampler.sample([], n=10) == []

    def test_result_sorted_by_difficulty(self):
        scored = self._make_scored(20)
        result = self.sampler.sample(scored, n=10)
        diffs = [s.difficulty_score for s in result]
        assert diffs == sorted(diffs)

    def test_sample_names_returns_strings(self):
        scored = self._make_scored(10)
        names = self.sampler.sample_names(scored, n=5)
        assert len(names) == 5
        assert all(isinstance(nm, str) for nm in names)

    def test_stratified_spans_difficulty_range(self):
        scored = self._make_scored(20)
        result = self.sampler.sample(scored, n=10)
        min_diff = min(s.difficulty_score for s in result)
        max_diff = max(s.difficulty_score for s in result)
        assert max_diff - min_diff > 0.4

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError):
            StratifiedSampler(n_bins=0)

    def test_reproducible_with_seed(self):
        scored = self._make_scored(20)
        r1 = [s.name for s in StratifiedSampler(n_bins=5, seed=123).sample(scored, n=8)]
        r2 = [s.name for s in StratifiedSampler(n_bins=5, seed=123).sample(scored, n=8)]
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        scored = self._make_scored(20)
        r1 = [s.name for s in StratifiedSampler(n_bins=5, seed=1).sample(scored, n=8)]
        r2 = [s.name for s in StratifiedSampler(n_bins=5, seed=99).sample(scored, n=8)]
        # Very unlikely to be identical with different seeds on 20 sequences
        assert r1 != r2

    def test_single_bin_returns_correct_count(self):
        scored = self._make_scored(10)
        result = StratifiedSampler(n_bins=1, seed=0).sample(scored, n=5)
        assert len(result) == 5
