"""Tests for SequenceDifficulty and DifficultyAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.difficulty import DifficultyAnalyzer, SequenceDifficulty
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockSequence:
    """Minimal Sequence stand-in for tests that don't need frame data."""

    def __init__(self, name: str, ground_truth: np.ndarray) -> None:
        self.name = name
        self.ground_truth = ground_truth

    def __len__(self) -> int:
        return len(self.ground_truth)


def _seq(name: str, positions, w: float = 40.0, h: float = 40.0) -> _MockSequence:
    """Build a mock sequence from (cx, cy) positions."""
    boxes = np.array(
        [(cx - w / 2, cy - h / 2, w, h) for cx, cy in positions],
        dtype=np.float64,
    )
    return _MockSequence(name, boxes)


# ---------------------------------------------------------------------------
# DifficultyAnalyzer — motion_speed
# ---------------------------------------------------------------------------


class TestMotionSpeed:
    def setup_method(self):
        self.analyzer = DifficultyAnalyzer()

    def test_static_sequence_zero_speed(self):
        s = _seq("static", [(100, 100)] * 30)
        d = self.analyzer.analyze(s)
        assert d.motion_speed == pytest.approx(0.0, abs=1e-6)

    def test_faster_motion_higher_speed(self):
        slow = _seq("slow", [(100 + i * 1, 100) for i in range(30)])
        fast = _seq("fast", [(100 + i * 10, 100) for i in range(30)])
        d_slow = self.analyzer.analyze(slow)
        d_fast = self.analyzer.analyze(fast)
        assert d_fast.motion_speed > d_slow.motion_speed

    def test_single_frame_zero_speed(self):
        s = _seq("single", [(160, 120)])
        d = self.analyzer.analyze(s)
        assert d.motion_speed == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# DifficultyAnalyzer — scale_variation
# ---------------------------------------------------------------------------


class TestScaleVariation:
    def setup_method(self):
        self.analyzer = DifficultyAnalyzer()

    def test_constant_size_zero_variation(self):
        s = _seq("const", [(100 + i, 100) for i in range(20)])
        d = self.analyzer.analyze(s)
        assert d.scale_variation == pytest.approx(0.0, abs=1e-6)

    def test_growing_box_nonzero_variation(self):
        boxes = np.array(
            [(100.0, 100.0, 20.0 + i * 2, 20.0 + i * 2) for i in range(20)],
            dtype=np.float64,
        )
        s = _MockSequence("growing", boxes)
        d = self.analyzer.analyze(s)
        assert d.scale_variation > 0.0

    def test_single_frame_zero_variation(self):
        s = _seq("single", [(100, 100)])
        d = self.analyzer.analyze(s)
        assert d.scale_variation == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# DifficultyAnalyzer — out_of_view_ratio
# ---------------------------------------------------------------------------


class TestOutOfViewRatio:
    def test_no_frame_size_always_zero(self):
        analyzer = DifficultyAnalyzer(frame_size=None)
        s = _seq("near_edge", [(5, 5)] * 20)
        d = analyzer.analyze(s)
        assert d.out_of_view_ratio == pytest.approx(0.0, abs=1e-6)

    def test_centroid_near_boundary_detected(self):
        analyzer = DifficultyAnalyzer(frame_size=(320, 240), boundary_margin=15)
        # All centroids at (5, 120) — well inside the left boundary margin.
        s = _seq("near_left", [(5, 120)] * 20)
        d = analyzer.analyze(s)
        assert d.out_of_view_ratio == pytest.approx(1.0, abs=1e-6)

    def test_centroid_in_centre_zero_ratio(self):
        analyzer = DifficultyAnalyzer(frame_size=(320, 240), boundary_margin=10)
        s = _seq("centre", [(160, 120)] * 20)
        d = analyzer.analyze(s)
        assert d.out_of_view_ratio == pytest.approx(0.0, abs=1e-6)

    def test_half_near_boundary(self):
        analyzer = DifficultyAnalyzer(frame_size=(320, 240), boundary_margin=15)
        positions = [(5, 120)] * 10 + [(160, 120)] * 10
        s = _seq("half_near", positions)
        d = analyzer.analyze(s)
        assert d.out_of_view_ratio == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# DifficultyAnalyzer — overall_score range and monotonicity
# ---------------------------------------------------------------------------


class TestOverallScore:
    def setup_method(self):
        self.analyzer = DifficultyAnalyzer(frame_size=(320, 240))

    def test_score_in_unit_interval(self):
        ds = SyntheticDataset(num_sequences=5, num_frames=50)
        for seq in ds:
            d = self.analyzer.analyze(seq)
            assert 0.0 <= d.overall_score <= 1.0

    def test_harder_sequence_higher_score(self):
        slow = _seq("slow", [(100 + i, 100) for i in range(50)])
        fast = _seq("fast", [(100 + i * 15, 100) for i in range(50)])
        d_slow = self.analyzer.analyze(slow)
        d_fast = self.analyzer.analyze(fast)
        assert d_fast.overall_score > d_slow.overall_score

    def test_static_centre_sequence_low_score(self):
        s = _seq("static_centre", [(160, 120)] * 50)
        d = self.analyzer.analyze(s)
        assert d.overall_score < 0.1


# ---------------------------------------------------------------------------
# DifficultyAnalyzer — rank and filter
# ---------------------------------------------------------------------------


class TestRankAndFilter:
    def setup_method(self):
        self.analyzer = DifficultyAnalyzer(frame_size=(320, 240))
        ds = SyntheticDataset(num_sequences=8, num_frames=50, motion="linear")
        self.difficulties = self.analyzer.analyze_dataset(ds)

    def test_rank_descending_by_default(self):
        ranked = self.analyzer.rank(self.difficulties)
        scores = [d.overall_score for d in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_ascending(self):
        ranked = self.analyzer.rank(self.difficulties, ascending=True)
        scores = [d.overall_score for d in ranked]
        assert scores == sorted(scores)

    def test_rank_does_not_mutate_input(self):
        original_order = [d.name for d in self.difficulties]
        self.analyzer.rank(self.difficulties)
        assert [d.name for d in self.difficulties] == original_order

    def test_filter_min_score(self):
        min_s = 0.2
        filtered = self.analyzer.filter(self.difficulties, min_score=min_s)
        assert all(d.overall_score >= min_s for d in filtered)

    def test_filter_max_score(self):
        max_s = 0.5
        filtered = self.analyzer.filter(self.difficulties, max_score=max_s)
        assert all(d.overall_score <= max_s for d in filtered)

    def test_filter_returns_subset(self):
        filtered = self.analyzer.filter(self.difficulties, min_score=0.0, max_score=1.0)
        assert len(filtered) == len(self.difficulties)


# ---------------------------------------------------------------------------
# DifficultyAnalyzer — summary statistics
# ---------------------------------------------------------------------------


class TestSummary:
    def setup_method(self):
        self.analyzer = DifficultyAnalyzer()
        ds = SyntheticDataset(num_sequences=5, num_frames=50)
        self.difficulties = self.analyzer.analyze_dataset(ds)

    def test_summary_keys_present(self):
        stats = self.analyzer.summary(self.difficulties)
        for attr in ["motion_speed", "scale_variation", "aspect_ratio_change",
                     "out_of_view_ratio", "overall_score"]:
            for stat in ["mean", "std", "min", "max"]:
                assert f"{attr}_{stat}" in stats

    def test_summary_empty_list(self):
        stats = self.analyzer.summary([])
        assert stats == {}

    def test_mean_within_min_max(self):
        stats = self.analyzer.summary(self.difficulties)
        for attr in ["overall_score"]:
            assert stats[f"{attr}_min"] <= stats[f"{attr}_mean"] <= stats[f"{attr}_max"]


# ---------------------------------------------------------------------------
# SequenceDifficulty — to_dict
# ---------------------------------------------------------------------------


class TestToDictMethod:
    def test_to_dict_has_all_keys(self):
        d = SequenceDifficulty(
            name="test",
            num_frames=50,
            motion_speed=3.5,
            scale_variation=0.05,
            aspect_ratio_change=0.02,
            out_of_view_ratio=0.1,
            overall_score=0.42,
        )
        result = d.to_dict()
        expected_keys = {
            "name", "num_frames", "motion_speed", "scale_variation",
            "aspect_ratio_change", "out_of_view_ratio", "overall_score",
        }
        assert set(result.keys()) == expected_keys

    def test_to_dict_values_are_rounded(self):
        d = SequenceDifficulty(
            name="test",
            num_frames=100,
            motion_speed=1.23456789,
            scale_variation=0.12345678,
            aspect_ratio_change=0.01234567,
            out_of_view_ratio=0.5,
            overall_score=0.33333333,
        )
        result = d.to_dict()
        assert result["motion_speed"] == pytest.approx(1.2346, abs=1e-4)
        assert result["overall_score"] == pytest.approx(0.3333, abs=1e-4)


# ---------------------------------------------------------------------------
# Integration: analyze_dataset with SyntheticDataset
# ---------------------------------------------------------------------------


class TestAnalyzeDatasetIntegration:
    def test_returns_one_result_per_sequence(self):
        analyzer = DifficultyAnalyzer()
        ds = SyntheticDataset(num_sequences=6, num_frames=40)
        results = analyzer.analyze_dataset(ds)
        assert len(results) == 6

    def test_names_match_sequences(self):
        analyzer = DifficultyAnalyzer()
        ds = SyntheticDataset(num_sequences=4, num_frames=30)
        results = analyzer.analyze_dataset(ds)
        for i, d in enumerate(results):
            assert d.name == ds[i].name

    def test_different_motions_produce_different_scores(self):
        analyzer = DifficultyAnalyzer(frame_size=(320, 240))
        linear_ds = SyntheticDataset(num_sequences=3, num_frames=50, motion="linear")
        random_ds = SyntheticDataset(num_sequences=3, num_frames=50, motion="random")
        linear_scores = [analyzer.analyze(linear_ds[i]).overall_score for i in range(3)]
        random_scores = [analyzer.analyze(random_ds[i]).overall_score for i in range(3)]
        # Both should be valid floats in [0, 1]; not all equal to each other.
        assert not (linear_scores == random_scores)
