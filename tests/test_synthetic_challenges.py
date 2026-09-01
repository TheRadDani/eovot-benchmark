"""Tests for eovot.datasets.synthetic_challenges.ChallengeSyntheticDataset."""

import numpy as np
import pytest

from eovot.datasets.synthetic_challenges import (
    ChallengeSyntheticDataset,
    ChallengeType,
)
from eovot.benchmark.engine import BenchmarkEngine
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Parametrize over every challenge type
# ---------------------------------------------------------------------------


@pytest.fixture(params=list(ChallengeType))
def challenge(request):
    return request.param


@pytest.fixture
def tiny_ds(challenge):
    return ChallengeSyntheticDataset(
        challenge=challenge,
        num_sequences=3,
        num_frames=30,
        frame_size=(160, 120),
        bbox_size=(20, 20),
        seed=0,
    )


# ---------------------------------------------------------------------------
# Dataset interface tests
# ---------------------------------------------------------------------------


class TestDatasetInterface:
    def test_len(self, tiny_ds):
        assert len(tiny_ds) == 3

    def test_getitem_returns_sequence(self, tiny_ds):
        seq = tiny_ds[0]
        assert seq is not None
        assert seq.ground_truth.shape == (30, 4)

    def test_sequence_names_contain_challenge(self, tiny_ds, challenge):
        seq = tiny_ds[0]
        assert challenge.value in seq.name

    def test_index_out_of_range(self, tiny_ds):
        with pytest.raises(IndexError):
            tiny_ds[99]

    def test_sequences_are_distinct(self, tiny_ds):
        s0 = tiny_ds[0]
        s1 = tiny_ds[1]
        # Different sequences have different GT boxes.
        assert not np.allclose(s0.ground_truth, s1.ground_truth)

    def test_caching(self, tiny_ds):
        s0a = tiny_ds[0]
        s0b = tiny_ds[0]
        assert s0a is s0b

    def test_frame_iteration(self, tiny_ds):
        seq = tiny_ds[0]
        frames = list(seq)
        assert len(frames) == 30
        for f in frames:
            assert f.shape == (120, 160, 3)
            assert f.dtype == np.uint8

    def test_gt_dtype(self, tiny_ds):
        seq = tiny_ds[0]
        assert seq.ground_truth.dtype == np.float64

    def test_repr(self, tiny_ds, challenge):
        r = repr(tiny_ds)
        assert challenge.value in r
        assert "ChallengeSyntheticDataset" in r


# ---------------------------------------------------------------------------
# Challenge-specific content tests
# ---------------------------------------------------------------------------


class TestOcclusion:
    def test_frames_rendered(self):
        ds = ChallengeSyntheticDataset(
            ChallengeType.OCCLUSION, num_sequences=1, num_frames=60, seed=1
        )
        seq = ds[0]
        frames = list(seq)
        # During occlusion window, a grey block should appear.
        # At least one frame should differ from the next (dynamics happen).
        diffs = [
            not np.array_equal(frames[i], frames[i + 1])
            for i in range(len(frames) - 1)
        ]
        assert any(diffs), "All frames identical — dynamics not applied."

    def test_gt_unaffected_by_occlusion(self):
        """GT boxes should be valid (positive width/height) even during occlusion."""
        ds = ChallengeSyntheticDataset(
            ChallengeType.OCCLUSION, num_sequences=1, num_frames=60, seed=2
        )
        gt = ds[0].ground_truth
        assert (gt[:, 2] > 0).all(), "GT widths must be positive."
        assert (gt[:, 3] > 0).all(), "GT heights must be positive."


class TestIllumination:
    def test_brightness_varies(self):
        ds = ChallengeSyntheticDataset(
            ChallengeType.ILLUMINATION_CHANGE, num_sequences=1, num_frames=60, seed=3
        )
        frames = list(ds[0])
        means = [f.mean() for f in frames]
        # Mean pixel value must vary across frames (brightness changes).
        assert max(means) - min(means) > 5.0, "Brightness variation too small."

    def test_gt_unchanged_by_illumination(self):
        ds = ChallengeSyntheticDataset(
            ChallengeType.ILLUMINATION_CHANGE, num_sequences=1, num_frames=30, seed=4
        )
        gt = ds[0].ground_truth
        assert (gt[:, 2] > 0).all()
        assert (gt[:, 3] > 0).all()


class TestScaleChange:
    def test_gt_width_varies(self):
        ds = ChallengeSyntheticDataset(
            ChallengeType.SCALE_CHANGE,
            num_sequences=1,
            num_frames=100,
            bbox_size=(40, 40),
            scale_min=0.5,
            scale_max=2.0,
            seed=5,
        )
        gt = ds[0].ground_truth
        widths = gt[:, 2]
        # Width must vary across the sequence.
        assert widths.max() > widths.min() + 1.0, "Scale not varying."

    def test_scale_bounds_respected(self):
        bw = 40.0
        ds = ChallengeSyntheticDataset(
            ChallengeType.SCALE_CHANGE,
            num_sequences=1,
            num_frames=100,
            bbox_size=(int(bw), 40),
            scale_min=0.5,
            scale_max=2.0,
            seed=6,
        )
        gt = ds[0].ground_truth
        widths = gt[:, 2]
        assert widths.min() >= bw * 0.5 - 1.0
        assert widths.max() <= bw * 2.0 + 1.0


class TestBackgroundClutter:
    def test_distractors_present(self):
        ds = ChallengeSyntheticDataset(
            ChallengeType.BACKGROUND_CLUTTER,
            num_sequences=1,
            num_frames=30,
            num_distractors=5,
            seed=7,
        )
        frames = list(ds[0])
        # With distractors, frames must differ from a plain background.
        bg_only = np.zeros_like(frames[0])
        assert not np.array_equal(frames[0], bg_only)

    def test_gt_is_target_not_distractor(self):
        """GT must have positive, non-trivially-small boxes."""
        ds = ChallengeSyntheticDataset(
            ChallengeType.BACKGROUND_CLUTTER, num_sequences=1, num_frames=20, seed=8
        )
        gt = ds[0].ground_truth
        assert (gt[:, 2] > 5).all()
        assert (gt[:, 3] > 5).all()


# ---------------------------------------------------------------------------
# ChallengeType enum tests
# ---------------------------------------------------------------------------


class TestChallengeType:
    def test_all_types_defined(self):
        names = {c.value for c in ChallengeType}
        assert "occlusion" in names
        assert "illumination_change" in names
        assert "scale_change" in names
        assert "background_clutter" in names

    def test_string_construction(self):
        ds = ChallengeSyntheticDataset(
            challenge="illumination_change", num_sequences=1, num_frames=10
        )
        assert ds.challenge == ChallengeType.ILLUMINATION_CHANGE

    def test_invalid_challenge_raises(self):
        with pytest.raises(ValueError):
            ChallengeSyntheticDataset(challenge="bad_challenge")


# ---------------------------------------------------------------------------
# Full benchmark integration
# ---------------------------------------------------------------------------


class TestBenchmarkIntegration:
    def test_mosse_on_all_challenges(self, challenge):
        ds = ChallengeSyntheticDataset(
            challenge=challenge,
            num_sequences=2,
            num_frames=30,
            frame_size=(160, 120),
            seed=99,
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name=f"Synth-{challenge.value}")
        assert result.mean_fps > 0.0
        assert 0.0 <= result.mean_iou <= 1.0
        assert len(result.sequence_results) == 2

    def test_reproducibility(self, challenge):
        def make_result():
            ds = ChallengeSyntheticDataset(
                challenge=challenge, num_sequences=1, num_frames=20, seed=123
            )
            frames = list(ds[0])
            return np.stack(frames)

        r1 = make_result()
        r2 = make_result()
        assert np.array_equal(r1, r2), "Same seed must produce identical frames."
