"""Tests for eovot.analysis.sequence_profiler."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.analysis.sequence_profiler import (
    SequenceDifficulty,
    SequenceDifficultyProfiler,
    _composite_score,
    _compute_metrics,
)


# ---------------------------------------------------------------------------
# Minimal Sequence stub — no disk I/O
# ---------------------------------------------------------------------------

class _MockSequence:
    """Minimal sequence stub for profiler tests.

    Yields synthetic numpy frames without touching the filesystem.
    """

    def __init__(
        self,
        gt: np.ndarray,
        frame_w: int = 320,
        frame_h: int = 240,
        name: str = "mock_seq",
    ) -> None:
        self.ground_truth = gt
        self.name = name
        self._frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    def __iter__(self):
        for _ in range(len(self.ground_truth)):
            yield self._frame

    def __len__(self) -> int:
        return len(self.ground_truth)

    @property
    def init_bbox(self):
        return tuple(self.ground_truth[0])


# ---------------------------------------------------------------------------
# GT factories
# ---------------------------------------------------------------------------

def _static_gt(n: int = 50, x=50, y=50, w=40, h=30) -> np.ndarray:
    """Target stationary at a fixed location and size."""
    return np.tile([x, y, w, h], (n, 1)).astype(np.float64)


def _moving_gt(n: int = 50, speed: float = 5.0) -> np.ndarray:
    """Target moving right at constant speed."""
    gt = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        gt[i] = [50 + speed * i, 50, 40, 30]
    return gt


def _erratic_gt(n: int = 50, seed: int = 0) -> np.ndarray:
    """Target moving with random per-frame displacements."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((n, 4), dtype=np.float64)
    x, y = 100.0, 100.0
    for i in range(n):
        x += rng.uniform(-20, 20)
        y += rng.uniform(-20, 20)
        gt[i] = [max(0, x), max(0, y), 40, 30]
    return gt


def _scaling_gt(n: int = 50) -> np.ndarray:
    """Target that grows 5% per frame (scale change)."""
    gt = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        scale = 1.0 + 0.05 * i
        gt[i] = [50, 50, 40 * scale, 30 * scale]
    return gt


def _deforming_gt(n: int = 50) -> np.ndarray:
    """Target that changes aspect ratio each frame (deformation)."""
    gt = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        w = 20 + 2 * (i % 20)    # oscillates 20 → 60
        h = 60 - 2 * (i % 20)    # oscillates 60 → 20
        gt[i] = [50, 50, max(1, w), max(1, h)]
    return gt


# ---------------------------------------------------------------------------
# TestComputeMetrics — internal helper
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_static_target_zero_motion(self):
        m = _compute_metrics(_static_gt(), 320, 240)
        assert m["motion_mean"] == pytest.approx(0.0)

    def test_static_target_zero_scale_change(self):
        m = _compute_metrics(_static_gt(), 320, 240)
        assert m["target_area_cv"] == pytest.approx(0.0)

    def test_static_target_zero_deformation(self):
        m = _compute_metrics(_static_gt(), 320, 240)
        assert m["aspect_ratio_std"] == pytest.approx(0.0)

    def test_target_area_normalised_by_frame(self):
        # 40×30 box in a 320×240 frame → 1200 / 76800 = 0.015625
        m = _compute_metrics(_static_gt(w=40, h=30), 320, 240)
        assert m["target_area_mean"] == pytest.approx(1200.0 / 76800.0, rel=1e-4)

    def test_moving_target_nonzero_motion(self):
        m = _compute_metrics(_moving_gt(speed=10.0), 320, 240)
        assert m["motion_mean"] > 0.0

    def test_scaling_target_nonzero_area_cv(self):
        m = _compute_metrics(_scaling_gt(), 640, 480)
        assert m["target_area_cv"] > 0.0

    def test_deforming_target_nonzero_ar_std(self):
        m = _compute_metrics(_deforming_gt(), 320, 240)
        assert m["aspect_ratio_std"] > 0.0

    def test_single_frame_zero_motion(self):
        m = _compute_metrics(_static_gt(n=1), 320, 240)
        assert m["motion_mean"] == pytest.approx(0.0)
        assert m["motion_cv"] == pytest.approx(0.0)

    def test_all_zero_area_returns_zeros(self):
        gt = np.zeros((10, 4), dtype=np.float64)
        m = _compute_metrics(gt, 320, 240)
        assert m["target_area_mean"] == pytest.approx(0.0)
        assert m["motion_mean"] == pytest.approx(0.0)

    def test_motion_normalised_by_diagonal(self):
        # Speed 10px/frame, diagonal = sqrt(320²+240²) = 400
        m = _compute_metrics(_moving_gt(speed=10.0), 320, 240)
        assert m["motion_mean"] == pytest.approx(10.0 / 400.0, rel=1e-4)


# ---------------------------------------------------------------------------
# TestCompositeScore
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_all_zero_gives_zero(self):
        m = {"target_area_cv": 0.0, "motion_mean": 0.0,
             "motion_cv": 0.0, "aspect_ratio_std": 0.0}
        assert _composite_score(m) == pytest.approx(0.0)

    def test_score_in_unit_interval_random(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            m = {
                "target_area_cv":   rng.uniform(0, 3),
                "motion_mean":      rng.uniform(0, 0.5),
                "motion_cv":        rng.uniform(0, 3),
                "aspect_ratio_std": rng.uniform(0, 2),
            }
            s = _composite_score(m)
            assert 0.0 <= s <= 1.0, f"score {s} out of [0, 1] for metrics {m}"

    def test_max_inputs_give_one(self):
        m = {"target_area_cv": 10.0, "motion_mean": 10.0,
             "motion_cv": 10.0, "aspect_ratio_std": 10.0}
        assert _composite_score(m) == pytest.approx(1.0)

    def test_motion_dominated_sequences_score_higher(self):
        fast   = {"target_area_cv": 0.0, "motion_mean": 0.4,
                  "motion_cv": 0.0, "aspect_ratio_std": 0.0}
        slow   = {"target_area_cv": 0.0, "motion_mean": 0.01,
                  "motion_cv": 0.0, "aspect_ratio_std": 0.0}
        assert _composite_score(fast) > _composite_score(slow)


# ---------------------------------------------------------------------------
# TestSequenceDifficultyProfiler
# ---------------------------------------------------------------------------

class TestSequenceDifficultyProfiler:
    def test_profile_sequence_returns_dataclass(self):
        seq = _MockSequence(_static_gt())
        d = SequenceDifficultyProfiler().profile_sequence(seq)
        assert isinstance(d, SequenceDifficulty)

    def test_profile_sequence_name_preserved(self):
        seq = _MockSequence(_static_gt(), name="ball")
        d = SequenceDifficultyProfiler().profile_sequence(seq)
        assert d.sequence_name == "ball"

    def test_profile_sequence_frame_count(self):
        seq = _MockSequence(_static_gt(n=75))
        d = SequenceDifficultyProfiler().profile_sequence(seq)
        assert d.num_frames == 75

    def test_static_sequence_low_score(self):
        seq = _MockSequence(_static_gt())
        d = SequenceDifficultyProfiler().profile_sequence(seq)
        assert d.difficulty_score == pytest.approx(0.0)

    def test_fast_motion_higher_than_static(self):
        static  = _MockSequence(_static_gt())
        moving  = _MockSequence(_moving_gt(speed=30.0))
        p = SequenceDifficultyProfiler()
        assert p.profile_sequence(moving).difficulty_score > \
               p.profile_sequence(static).difficulty_score

    def test_scale_change_higher_than_static(self):
        static  = _MockSequence(_static_gt())
        scaling = _MockSequence(_scaling_gt())
        p = SequenceDifficultyProfiler()
        assert p.profile_sequence(scaling).difficulty_score > \
               p.profile_sequence(static).difficulty_score

    def test_deformation_higher_than_static(self):
        static   = _MockSequence(_static_gt())
        deforming = _MockSequence(_deforming_gt())
        p = SequenceDifficultyProfiler()
        assert p.profile_sequence(deforming).difficulty_score > \
               p.profile_sequence(static).difficulty_score

    def test_sort_hardest_first(self):
        p = SequenceDifficultyProfiler()
        seqs = [
            _MockSequence(_static_gt(),        name="easy"),
            _MockSequence(_moving_gt(speed=30), name="hard"),
        ]
        diffs  = [p.profile_sequence(s) for s in seqs]
        ranked = p.sort_by_difficulty(diffs, descending=True)
        assert ranked[0].sequence_name == "hard"
        assert ranked[-1].sequence_name == "easy"

    def test_sort_easiest_first(self):
        p = SequenceDifficultyProfiler()
        seqs = [
            _MockSequence(_static_gt(),        name="easy"),
            _MockSequence(_moving_gt(speed=30), name="hard"),
        ]
        diffs  = [p.profile_sequence(s) for s in seqs]
        ranked = p.sort_by_difficulty(diffs, descending=False)
        assert ranked[0].sequence_name == "easy"

    def test_sort_does_not_modify_original(self):
        p = SequenceDifficultyProfiler()
        diffs = [
            p.profile_sequence(_MockSequence(_static_gt(),        name="easy")),
            p.profile_sequence(_MockSequence(_moving_gt(speed=30), name="hard")),
        ]
        original_order = [d.sequence_name for d in diffs]
        p.sort_by_difficulty(diffs)
        assert [d.sequence_name for d in diffs] == original_order

    def test_summary_stats_keys(self):
        p = SequenceDifficultyProfiler()
        diffs = [
            p.profile_sequence(_MockSequence(_static_gt())),
            p.profile_sequence(_MockSequence(_moving_gt())),
        ]
        stats = p.summary_stats(diffs)
        assert stats["num_sequences"] == 2
        for key in ("difficulty_score", "motion_mean", "target_area_mean"):
            assert key in stats
            assert "mean" in stats[key] and "std" in stats[key]

    def test_summary_stats_empty_input(self):
        assert SequenceDifficultyProfiler().summary_stats([]) == {}

    def test_to_dict_is_serialisable(self):
        seq = _MockSequence(_moving_gt())
        d = SequenceDifficultyProfiler().profile_sequence(seq)
        dd = d.to_dict()
        assert isinstance(dd, dict)
        assert "difficulty_score" in dd
        assert "sequence_name"    in dd
        assert isinstance(dd["num_frames"], int)

    def test_difficulty_score_bounded(self):
        """Regardless of the input, score must stay in [0, 1]."""
        p = SequenceDifficultyProfiler()
        for gt_factory in [_static_gt, _moving_gt, _scaling_gt, _erratic_gt, _deforming_gt]:
            seq = _MockSequence(gt_factory())
            d   = p.profile_sequence(seq)
            assert 0.0 <= d.difficulty_score <= 1.0, (
                f"score {d.difficulty_score} out of [0, 1] for {gt_factory.__name__}"
            )
