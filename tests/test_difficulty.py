"""Unit tests for eovot.metrics.difficulty."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.difficulty import (
    DifficultyFactors,
    DifficultyReport,
    SequenceDifficultyEntry,
    SequenceDifficultyScorer,
    score_dataset,
)
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _static_gt(
    n: int = 20,
    x: float = 50.0,
    y: float = 50.0,
    w: float = 40.0,
    h: float = 30.0,
) -> np.ndarray:
    """GT with no motion and no scale change."""
    return np.tile([x, y, w, h], (n, 1)).astype(np.float64)


def _fast_motion_gt(n: int = 30) -> np.ndarray:
    """Target moves 20 px/frame horizontally."""
    return np.array(
        [[i * 20.0, 50.0, 40.0, 30.0] for i in range(n)], dtype=np.float64
    )


def _scale_change_gt(n: int = 30) -> np.ndarray:
    """Target area grows linearly: side 10 px -> 10 + 5*(n-1) px."""
    return np.array(
        [[50.0, 50.0, 10.0 + i * 5.0, 10.0 + i * 5.0] for i in range(n)],
        dtype=np.float64,
    )


def _non_overlapping_gt(n: int = 10) -> np.ndarray:
    """Each consecutive box has zero overlap (max deformation)."""
    return np.array(
        [[i * 60.0, 0.0, 40.0, 40.0] for i in range(n)], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# DifficultyFactors
# ---------------------------------------------------------------------------


class TestDifficultyFactors:
    def test_default_values_are_zero(self):
        f = DifficultyFactors()
        assert f.difficulty_score == 0.0
        assert f.motion_speed == 0.0

    def test_to_dict_has_seven_keys(self):
        f = DifficultyFactors(motion_speed=0.5, difficulty_score=0.3)
        d = f.to_dict()
        assert len(d) == 7
        assert "motion_speed" in d
        assert "difficulty_score" in d

    def test_to_dict_rounds_to_4dp(self):
        f = DifficultyFactors(motion_speed=0.123456789)
        assert f.to_dict()["motion_speed"] == 0.1235

    def test_str_contains_score(self):
        f = DifficultyFactors(difficulty_score=0.75)
        assert "0.750" in str(f)

    def test_all_fields_float(self):
        f = DifficultyFactors(motion_speed=0.1, scale_change=0.2)
        d = f.to_dict()
        for v in d.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# SequenceDifficultyScorer
# ---------------------------------------------------------------------------


class TestSequenceDifficultyScorer:
    def test_static_target_scores_low(self):
        gt = _static_gt()
        scorer = SequenceDifficultyScorer(frame_size=(480, 640))
        f = scorer.score(gt)
        assert f.difficulty_score < 0.3

    def test_fast_motion_raises_motion_factor(self):
        scorer = SequenceDifficultyScorer(frame_size=(480, 640))
        f_static = scorer.score(_static_gt())
        f_fast = scorer.score(_fast_motion_gt())
        assert f_fast.motion_speed > f_static.motion_speed

    def test_scale_change_raises_scale_factor(self):
        scorer = SequenceDifficultyScorer()
        f_static = scorer.score(_static_gt())
        f_scale = scorer.score(_scale_change_gt())
        assert f_scale.scale_change > f_static.scale_change

    def test_difficulty_score_in_unit_interval(self):
        scorer = SequenceDifficultyScorer(frame_size=(480, 640))
        for gt in [_static_gt(), _fast_motion_gt(), _scale_change_gt()]:
            f = scorer.score(gt)
            assert 0.0 <= f.difficulty_score <= 1.0

    def test_requires_at_least_two_frames(self):
        scorer = SequenceDifficultyScorer()
        with pytest.raises(ValueError, match="at least 2"):
            scorer.score(np.array([[10, 10, 30, 30]], dtype=np.float64))

    def test_wrong_shape_raises_value_error(self):
        scorer = SequenceDifficultyScorer()
        with pytest.raises(ValueError, match="shape"):
            scorer.score(np.zeros((10, 3)))

    def test_oov_zero_without_frame_size(self):
        scorer = SequenceDifficultyScorer(frame_size=None)
        f = scorer.score(_fast_motion_gt())
        assert f.out_of_view_risk == 0.0

    def test_oov_detected_at_boundary(self):
        gt = np.zeros((10, 4), dtype=np.float64)
        gt[:, 0] = 0.0  # left edge
        gt[:, 1] = 100.0
        gt[:, 2] = 30.0
        gt[:, 3] = 30.0
        scorer = SequenceDifficultyScorer(frame_size=(240, 320))
        f = scorer.score(gt)
        assert f.out_of_view_risk > 0.0

    def test_custom_weights_normalised(self):
        scorer = SequenceDifficultyScorer(
            weights={"motion": 1.0, "scale": 0.0, "aspect": 0.0,
                     "small": 0.0, "oov": 0.0, "deform": 0.0}
        )
        f = scorer.score(_fast_motion_gt())
        assert f.difficulty_score > 0.0

    def test_unknown_weight_key_raises(self):
        with pytest.raises(ValueError, match="Unknown weight"):
            SequenceDifficultyScorer(weights={"bad_key": 1.0})

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError, match="positive"):
            SequenceDifficultyScorer(
                weights={"motion": 0, "scale": 0, "aspect": 0,
                         "small": 0, "oov": 0, "deform": 0}
            )

    def test_large_target_small_target_factor_is_zero(self):
        gt = np.tile([0.0, 0.0, 200.0, 200.0], (10, 1)).astype(np.float64)
        scorer = SequenceDifficultyScorer(frame_size=(240, 320))
        f = scorer.score(gt)
        assert f.small_target == 0.0

    def test_tiny_target_small_target_factor_is_one(self):
        gt = np.tile([100.0, 100.0, 2.0, 2.0], (10, 1)).astype(np.float64)
        scorer = SequenceDifficultyScorer(frame_size=(1080, 1920))
        f = scorer.score(gt)
        assert f.small_target == 1.0

    def test_deformation_zero_for_identical_boxes(self):
        gt = _static_gt()
        scorer = SequenceDifficultyScorer()
        f = scorer.score(gt)
        assert f.deformation == pytest.approx(0.0, abs=1e-3)

    def test_deformation_high_for_non_overlapping_boxes(self):
        gt = _non_overlapping_gt()
        scorer = SequenceDifficultyScorer()
        f = scorer.score(gt)
        assert f.deformation > 0.5

    def test_aspect_ratio_zero_for_constant_shape(self):
        gt = _static_gt()
        scorer = SequenceDifficultyScorer()
        f = scorer.score(gt)
        assert f.aspect_ratio_change == pytest.approx(0.0, abs=1e-4)

    def test_two_frame_minimum(self):
        gt = np.array([[0, 0, 20, 20], [10, 0, 20, 20]], dtype=np.float64)
        scorer = SequenceDifficultyScorer()
        f = scorer.score(gt)
        assert isinstance(f.difficulty_score, float)

    def test_without_frame_size_no_error(self):
        scorer = SequenceDifficultyScorer(frame_size=None)
        gt = _fast_motion_gt()
        f = scorer.score(gt)
        assert 0.0 <= f.difficulty_score <= 1.0


# ---------------------------------------------------------------------------
# SequenceDifficultyEntry / DifficultyReport
# ---------------------------------------------------------------------------


class TestDifficultyReport:
    def _make_entries(self, n: int = 4) -> List[SequenceDifficultyEntry]:
        return [
            SequenceDifficultyEntry(
                f"seq_{i:02d}",
                DifficultyFactors(difficulty_score=0.1 * i),
            )
            for i in range(n)
        ]

    def test_entries_sorted_descending(self):
        report = DifficultyReport(self._make_entries(5))
        scores = [e.difficulty_score for e in report.entries]
        assert scores == sorted(scores, reverse=True)

    def test_hardest_returns_top_n(self):
        report = DifficultyReport(self._make_entries(6))
        hard = report.hardest(3)
        assert len(hard) == 3
        assert hard[0].difficulty_score >= hard[-1].difficulty_score

    def test_easiest_returns_bottom_n(self):
        report = DifficultyReport(self._make_entries(6))
        easy = report.easiest(2)
        assert len(easy) == 2
        assert easy[0].difficulty_score <= easy[1].difficulty_score

    def test_to_markdown_contains_header_keywords(self):
        report = DifficultyReport(self._make_entries(3))
        md = report.to_markdown()
        for kw in ("Rank", "Sequence", "Score", "Motion"):
            assert kw in md, f"{kw!r} missing from Markdown output"

    def test_to_dict_length(self):
        n = 4
        report = DifficultyReport(self._make_entries(n))
        assert len(report.to_dict()) == n

    def test_to_dict_has_required_keys(self):
        report = DifficultyReport(self._make_entries(2))
        row = report.to_dict()[0]
        assert "sequence" in row
        assert "difficulty_score" in row

    def test_repr_contains_count(self):
        report = DifficultyReport(self._make_entries(3))
        assert "3" in repr(report)

    def test_len(self):
        report = DifficultyReport(self._make_entries(7))
        assert len(report) == 7

    def test_iter_yields_all_entries(self):
        report = DifficultyReport(self._make_entries(3))
        assert sum(1 for _ in report) == 3

    def test_empty_report_repr(self):
        report = DifficultyReport([])
        assert "empty" in repr(report).lower()


# ---------------------------------------------------------------------------
# score_dataset
# ---------------------------------------------------------------------------


class TestScoreDataset:
    def test_scores_all_sequences(self):
        ds = SyntheticDataset(num_sequences=5, num_frames=20)
        report = score_dataset(ds, frame_size=(240, 320))
        assert len(report) == 5

    def test_all_scores_in_unit_interval(self):
        ds = SyntheticDataset(num_sequences=3, num_frames=20)
        for entry in score_dataset(ds):
            assert 0.0 <= entry.difficulty_score <= 1.0

    def test_custom_scorer_accepted(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=10)
        custom = SequenceDifficultyScorer(
            weights={"motion": 1, "scale": 0, "aspect": 0,
                     "small": 0, "oov": 0, "deform": 0}
        )
        report = score_dataset(ds, scorer=custom)
        assert len(report) == 2

    def test_nonzero_scores_on_dynamic_sequences(self):
        ds = SyntheticDataset(num_sequences=5, num_frames=30, motion="random")
        scorer = SequenceDifficultyScorer(frame_size=(240, 320))
        report = score_dataset(ds, scorer=scorer)
        assert any(e.difficulty_score > 0 for e in report)


# ---------------------------------------------------------------------------
# Type alias for List used in TestDifficultyReport methods
# ---------------------------------------------------------------------------

from typing import List  # noqa: E402 (after test class definitions)
