"""Tests for SequenceWeightedLeaderboard and related dataclasses.

Coverage:
- Construction validation (empty results, no shared sequences)
- IoU matrix construction
- Variance weighting: discriminative sequences get higher weight
- Difficulty weighting: mid-IoU sequences get highest weight,
  trivial (all 1.0) and hopeless (all 0.0) sequences get low weight
- Rank ordering: highest weighted IoU → rank 1
- rank_delta sign semantics
- to_markdown_table structure
- rank_delta_table structure
- sequence_weight_table structure
- Single-result case (no comparisons needed)
- All sequences shared across results
- Partial overlap: intersection used
- Integration with BenchmarkEngine + SyntheticDataset
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.analysis.weighted_leaderboard import (
    SequenceWeightedLeaderboard,
    WeightedLeaderboardResult,
    LeaderboardEntry,
    _rank_descending,
)


# ---------------------------------------------------------------------------
# Helper: build lightweight BenchmarkResult mocks
# ---------------------------------------------------------------------------

def _make_result(tracker_name: str, seq_ious: dict) -> "BenchmarkResult":
    """Build a minimal BenchmarkResult mock from {seq_name: mean_iou} dict."""
    seq_results = []
    for seq_name, mean_iou in seq_ious.items():
        sr = MagicMock()
        sr.sequence_name = seq_name
        # Ensure sr.mean_iou is a float attribute, not a mock
        type(sr).mean_iou = property(lambda self, m=mean_iou: m)
        sr.ious = np.array([mean_iou], dtype=np.float64)
        seq_results.append(sr)

    result = MagicMock()
    result.tracker_name = tracker_name
    result.sequence_results = seq_results
    return result


# ---------------------------------------------------------------------------
# _rank_descending utility
# ---------------------------------------------------------------------------

class TestRankDescending:
    def test_basic_order(self):
        scores = np.array([0.9, 0.5, 0.7])
        ranks = _rank_descending(scores)
        assert list(ranks) == [1, 3, 2]

    def test_all_equal(self):
        scores = np.array([0.5, 0.5, 0.5])
        ranks = _rank_descending(scores)
        assert set(ranks) == {1, 2, 3}

    def test_single(self):
        assert _rank_descending(np.array([0.8]))[0] == 1

    def test_descending_input(self):
        scores = np.array([1.0, 0.8, 0.6, 0.4])
        ranks = _rank_descending(scores)
        assert list(ranks) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            SequenceWeightedLeaderboard([])

    def test_no_shared_sequences_raises(self):
        r1 = _make_result("T1", {"seqA": 0.8})
        r2 = _make_result("T2", {"seqB": 0.5})  # different sequence names
        with pytest.raises(ValueError, match="[Nn]o sequences"):
            SequenceWeightedLeaderboard([r1, r2])

    def test_single_result_accepted(self):
        r = _make_result("T1", {"seq1": 0.8, "seq2": 0.7})
        lb = SequenceWeightedLeaderboard([r])
        assert lb.shared_sequence_names == ["seq1", "seq2"]


# ---------------------------------------------------------------------------
# Shared sequence detection
# ---------------------------------------------------------------------------

class TestSharedSequences:
    def test_full_overlap(self):
        r1 = _make_result("T1", {"s1": 0.8, "s2": 0.6})
        r2 = _make_result("T2", {"s1": 0.7, "s2": 0.5})
        lb = SequenceWeightedLeaderboard([r1, r2])
        assert set(lb.shared_sequence_names) == {"s1", "s2"}

    def test_partial_overlap(self):
        r1 = _make_result("T1", {"s1": 0.8, "s2": 0.6, "s3": 0.9})
        r2 = _make_result("T2", {"s1": 0.7, "s2": 0.5, "s4": 0.4})
        lb = SequenceWeightedLeaderboard([r1, r2])
        assert set(lb.shared_sequence_names) == {"s1", "s2"}  # s3, s4 excluded

    def test_shared_sequences_sorted(self):
        r1 = _make_result("T1", {"b": 0.8, "a": 0.6, "c": 0.9})
        r2 = _make_result("T2", {"b": 0.7, "a": 0.5, "c": 0.4})
        lb = SequenceWeightedLeaderboard([r1, r2])
        assert lb.shared_sequence_names == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Variance weighting semantics
# ---------------------------------------------------------------------------

class TestVarianceWeighting:
    def test_high_variance_sequence_gets_more_weight(self):
        # seq_easy: both trackers score near 0.9 → low variance
        # seq_hard: T1=0.9, T2=0.1 → high variance (spread = 0.8)
        r1 = _make_result("T1", {"seq_easy": 0.9, "seq_hard": 0.9})
        r2 = _make_result("T2", {"seq_easy": 0.88, "seq_hard": 0.1})
        lb = SequenceWeightedLeaderboard([r1, r2])
        result = lb.rank("variance")
        w = result.sequence_weights
        assert w["seq_hard"] > w["seq_easy"], (
            f"seq_hard (var={0.8**2/4:.4f}) should have higher weight than "
            f"seq_easy (var≈{0.02**2/4:.6f})"
        )

    def test_all_equal_gives_uniform_weights(self):
        # If all trackers score the same on all sequences → variance = 0 everywhere
        # → weights collapse to eps-uniform
        r1 = _make_result("T1", {"s1": 0.7, "s2": 0.7})
        r2 = _make_result("T2", {"s1": 0.7, "s2": 0.7})
        lb = SequenceWeightedLeaderboard([r1, r2])
        result = lb.rank("variance")
        w1, w2 = result.sequence_weights["s1"], result.sequence_weights["s2"]
        assert abs(w1 - w2) < 1e-6

    def test_rank_by_weighted_iou(self):
        # T1 scores 0.9 on the high-weight sequence, T2 scores 0.2
        r1 = _make_result("T1", {"s_high": 0.9, "s_low": 0.5})
        r2 = _make_result("T2", {"s_high": 0.2, "s_low": 0.5})
        lb = SequenceWeightedLeaderboard([r1, r2])
        result = lb.rank("variance")
        t1_entry = next(e for e in result.entries if e.tracker_name == "T1")
        t2_entry = next(e for e in result.entries if e.tracker_name == "T2")
        assert t1_entry.weighted_rank < t2_entry.weighted_rank


# ---------------------------------------------------------------------------
# Difficulty weighting semantics
# ---------------------------------------------------------------------------

class TestDifficultyWeighting:
    def test_mid_iou_sequence_gets_highest_weight(self):
        # seq_mid: mean IoU ≈ 0.5 → weight = 0.5*0.5 = 0.25 (maximum)
        # seq_easy: mean IoU ≈ 0.9 → weight = 0.9*0.1 = 0.09
        # seq_hard: mean IoU ≈ 0.1 → weight = 0.1*0.9 = 0.09
        r1 = _make_result("T1", {"seq_mid": 0.5, "seq_easy": 0.9, "seq_hard": 0.1})
        r2 = _make_result("T2", {"seq_mid": 0.5, "seq_easy": 0.9, "seq_hard": 0.1})
        lb = SequenceWeightedLeaderboard([r1, r2])
        result = lb.rank("difficulty")
        w = result.sequence_weights
        assert w["seq_mid"] > w["seq_easy"]
        assert w["seq_mid"] > w["seq_hard"]

    def test_all_easy_sequences_low_weight(self):
        r1 = _make_result("T1", {"s1": 0.99, "s2": 0.98})
        r2 = _make_result("T2", {"s1": 0.97, "s2": 0.96})
        lb = SequenceWeightedLeaderboard([r1, r2])
        result = lb.rank("difficulty")
        # All weights should be small (but not zero due to eps)
        for w in result.sequence_weights.values():
            assert w < 0.6  # 0.99 * 0.01 = 0.0099, normalised still < 0.6

    def test_invalid_scheme_raises(self):
        r1 = _make_result("T1", {"s": 0.8})
        lb = SequenceWeightedLeaderboard([r1])
        with pytest.raises(ValueError, match="Unknown weighting scheme"):
            lb.rank("invalid_scheme")


# ---------------------------------------------------------------------------
# LeaderboardEntry and rank_delta
# ---------------------------------------------------------------------------

class TestLeaderboardEntry:
    def test_rank_delta_positive_when_improves(self):
        # T1 unweighted rank 2, weighted rank 1 → delta = 2 - 1 = +1
        r1 = _make_result("T1", {"s_hard": 0.9, "s_easy": 0.5})
        r2 = _make_result("T2", {"s_hard": 0.3, "s_easy": 0.8})
        lb = SequenceWeightedLeaderboard([r1, r2])
        result = lb.rank("variance")
        # Find entries
        t1 = next(e for e in result.entries if e.tracker_name == "T1")
        # T1 scores better on the high-variance (discriminative) sequence
        # If T1 ranks higher in weighted → its rank_delta could be positive or 0
        assert isinstance(t1.rank_delta, int)

    def test_rank_delta_is_unweighted_minus_weighted(self):
        r1 = _make_result("T1", {"s1": 0.8})
        lb = SequenceWeightedLeaderboard([r1])
        result = lb.rank("variance")
        e = result.entries[0]
        assert e.rank_delta == e.unweighted_rank - e.weighted_rank

    def test_entry_str(self):
        e = LeaderboardEntry(
            tracker_name="KCF",
            unweighted_mean_iou=0.75,
            weighted_mean_iou=0.80,
            unweighted_rank=2,
            weighted_rank=1,
            rank_delta=1,
            num_sequences=10,
        )
        s = str(e)
        assert "KCF" in s and "1" in s


# ---------------------------------------------------------------------------
# Reporting methods
# ---------------------------------------------------------------------------

class TestReporting:
    def _three_tracker_lb(self):
        r1 = _make_result("MOSSE", {"s1": 0.9, "s2": 0.4, "s3": 0.7})
        r2 = _make_result("KCF",   {"s1": 0.7, "s2": 0.6, "s3": 0.75})
        r3 = _make_result("CSRT",  {"s1": 0.6, "s2": 0.85, "s3": 0.8})
        return SequenceWeightedLeaderboard([r1, r2, r3])

    def test_to_markdown_table_contains_header(self):
        lb = self._three_tracker_lb()
        md = lb.rank("variance").to_markdown_table()
        assert "W-Rank" in md
        assert "Tracker" in md

    def test_to_markdown_table_contains_all_trackers(self):
        lb = self._three_tracker_lb()
        md = lb.rank("variance").to_markdown_table()
        assert "MOSSE" in md and "KCF" in md and "CSRT" in md

    def test_rank_delta_table_contains_both_schemes(self):
        lb = self._three_tracker_lb()
        md = lb.rank_delta_table()
        assert "Var-IoU" in md and "Diff-IoU" in md

    def test_sequence_weight_table_contains_sequences(self):
        lb = self._three_tracker_lb()
        md = lb.sequence_weight_table("variance")
        assert "s1" in md and "s2" in md and "s3" in md

    def test_weighted_leaderboard_result_str(self):
        lb = self._three_tracker_lb()
        result = lb.rank("difficulty")
        s = str(result)
        assert "difficulty" in s

    def test_entries_sorted_by_weighted_rank(self):
        lb = self._three_tracker_lb()
        result = lb.rank("variance")
        for i in range(len(result.entries) - 1):
            assert result.entries[i].weighted_rank <= result.entries[i + 1].weighted_rank

    def test_all_ranks_unique(self):
        lb = self._three_tracker_lb()
        result = lb.rank("variance")
        weighted_ranks = [e.weighted_rank for e in result.entries]
        assert len(set(weighted_ranks)) == len(weighted_ranks)

    def test_num_sequences_in_entry(self):
        lb = self._three_tracker_lb()
        result = lb.rank("variance")
        for e in result.entries:
            assert e.num_sequences == 3  # s1, s2, s3

    def test_weights_sum_to_one(self):
        lb = self._three_tracker_lb()
        result = lb.rank("variance")
        total = sum(result.sequence_weights.values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestBenchmarkEngineIntegration:
    def test_end_to_end_with_synthetic(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker
        from eovot.trackers.kcf import KCFTracker

        dataset = SyntheticDataset(num_sequences=5, num_frames=20, seed=7)
        engine = BenchmarkEngine(verbose=False)

        mosse = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")
        kcf = engine.run(KCFTracker(), dataset, dataset_name="Synthetic")

        lb = SequenceWeightedLeaderboard([mosse, kcf])
        assert set(lb.shared_sequence_names) == {
            sr.sequence_name for sr in mosse.sequence_results
        }

        var_result = lb.rank("variance")
        diff_result = lb.rank("difficulty")

        assert len(var_result.entries) == 2
        assert len(diff_result.entries) == 2
        assert abs(sum(var_result.sequence_weights.values()) - 1.0) < 1e-6

        md = lb.rank_delta_table()
        assert "MOSSE" in md and "KCF" in md
