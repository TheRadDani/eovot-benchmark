"""Tests for eovot.datasets.challenge.ChallengeDataset."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.challenge import (
    ChallengeDataset,
    ChallengeSequence,
    _VALID_CHALLENGES,
    _TAG,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=list(sorted(_VALID_CHALLENGES)))
def challenge_ds(request):
    """One ChallengeDataset per challenge type."""
    return ChallengeDataset(
        challenge=request.param,
        num_sequences=3,
        num_frames=30,
        frame_size=(120, 90),
        bbox_size=(20, 20),
        seed=0,
    )


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_challenges(self):
        for name in _VALID_CHALLENGES:
            ds = ChallengeDataset(name, num_sequences=1, num_frames=10)
            assert len(ds) == 1

    def test_invalid_challenge_raises(self):
        with pytest.raises(ValueError, match="Unknown challenge"):
            ChallengeDataset("nonexistent_challenge")

    def test_repr(self):
        ds = ChallengeDataset("occlusion", num_sequences=2)
        r = repr(ds)
        assert "ChallengeDataset" in r
        assert "occlusion" in r

    def test_all_challenges_returns_sorted_list(self):
        lst = ChallengeDataset.all_challenges()
        assert sorted(lst) == lst
        assert set(lst) == _VALID_CHALLENGES

    def test_tag_for_known_challenges(self):
        for challenge, tag in _TAG.items():
            assert ChallengeDataset.tag_for(challenge) == tag

    def test_tag_for_unknown_raises(self):
        with pytest.raises(KeyError):
            ChallengeDataset.tag_for("not_a_challenge")


# ---------------------------------------------------------------------------
# Sequence interface
# ---------------------------------------------------------------------------

class TestSequenceInterface:
    def test_len_matches(self, challenge_ds):
        assert len(challenge_ds) == challenge_ds.num_sequences

    def test_getitem_returns_challenge_sequence(self, challenge_ds):
        seq = challenge_ds[0]
        assert isinstance(seq, ChallengeSequence)

    def test_index_out_of_range(self, challenge_ds):
        with pytest.raises(IndexError):
            challenge_ds[challenge_ds.num_sequences]

    def test_negative_index_raises(self, challenge_ds):
        with pytest.raises(IndexError):
            challenge_ds[-1]

    def test_iter_yields_all_sequences(self, challenge_ds):
        seqs = list(challenge_ds)
        assert len(seqs) == challenge_ds.num_sequences
        for seq in seqs:
            assert isinstance(seq, ChallengeSequence)

    def test_caching(self, challenge_ds):
        seq_a = challenge_ds[0]
        seq_b = challenge_ds[0]
        assert seq_a is seq_b  # same object from cache


# ---------------------------------------------------------------------------
# Generated sequence structure
# ---------------------------------------------------------------------------

class TestGeneratedSequence:
    def test_name_contains_challenge(self, challenge_ds):
        seq = challenge_ds[0]
        assert challenge_ds.challenge.replace("_", "-") in seq.name

    def test_name_has_index_suffix(self, challenge_ds):
        for i in range(challenge_ds.num_sequences):
            seq = challenge_ds[i]
            assert seq.name.endswith(f"_{i:03d}")

    def test_frame_count(self, challenge_ds):
        seq = challenge_ds[0]
        assert len(seq) == challenge_ds.num_frames

    def test_ground_truth_shape(self, challenge_ds):
        seq = challenge_ds[0]
        assert seq.ground_truth.shape == (challenge_ds.num_frames, 4)

    def test_ground_truth_dtype(self, challenge_ds):
        seq = challenge_ds[0]
        assert seq.ground_truth.dtype == np.float64

    def test_per_frame_attributes_length(self, challenge_ds):
        seq = challenge_ds[0]
        assert len(seq.per_frame_attributes) == challenge_ds.num_frames

    def test_attributes_set_nonempty(self, challenge_ds):
        seq = challenge_ds[0]
        assert len(seq.attributes) >= 1

    def test_tag_in_attributes(self, challenge_ds):
        expected_tag = _TAG[challenge_ds.challenge]
        seq = challenge_ds[0]
        assert expected_tag in seq.attributes


# ---------------------------------------------------------------------------
# Frame iteration
# ---------------------------------------------------------------------------

class TestFrameIteration:
    def test_frames_are_numpy_arrays(self, challenge_ds):
        seq = challenge_ds[0]
        for frame in seq:
            assert isinstance(frame, np.ndarray)

    def test_frame_shape(self, challenge_ds):
        W, H = challenge_ds.frame_size
        seq = challenge_ds[0]
        for frame in seq:
            assert frame.shape == (H, W, 3)

    def test_frame_dtype(self, challenge_ds):
        seq = challenge_ds[0]
        for frame in seq:
            assert frame.dtype == np.uint8

    def test_correct_frame_count(self, challenge_ds):
        seq = challenge_ds[0]
        frames = list(seq)
        assert len(frames) == challenge_ds.num_frames


# ---------------------------------------------------------------------------
# Ground-truth validity
# ---------------------------------------------------------------------------

class TestGroundTruth:
    def test_init_bbox_is_first_gt(self, challenge_ds):
        seq = challenge_ds[0]
        init = seq.init_bbox
        assert len(init) == 4
        np.testing.assert_array_almost_equal(init, seq.ground_truth[0])

    def test_gt_width_positive(self, challenge_ds):
        seq = challenge_ds[0]
        # Width and height should be positive for all non-degenerate sequences
        assert np.all(seq.ground_truth[:, 2] > 0)
        assert np.all(seq.ground_truth[:, 3] > 0)


# ---------------------------------------------------------------------------
# Attribute-sliced IoU helper
# ---------------------------------------------------------------------------

class TestIoUByAttribute:
    def _mock_ious(self, n: int, val: float = 0.7) -> np.ndarray:
        return np.full(n, val, dtype=np.float64)

    def test_iou_by_attribute_tagged_frames(self):
        ds = ChallengeDataset("occlusion", num_sequences=1, num_frames=50, seed=10)
        seq = ds[0]
        ious = self._mock_ious(len(seq), val=0.5)
        result = seq.iou_by_attribute(ious, "OCC")
        # There should be occluded frames in this sequence
        occluded = [i for i, tags in enumerate(seq.per_frame_attributes) if "OCC" in tags]
        if occluded:
            assert result is not None
            assert 0.0 <= result <= 1.0
        else:
            assert result is None

    def test_iou_by_attribute_unknown_tag_returns_none(self):
        ds = ChallengeDataset("occlusion", num_sequences=1, num_frames=20, seed=5)
        seq = ds[0]
        ious = self._mock_ious(len(seq))
        assert seq.iou_by_attribute(ious, "UNKNOWN_TAG") is None

    def test_challenging_frame_indices_subset(self):
        ds = ChallengeDataset("occlusion", num_sequences=1, num_frames=60, seed=7)
        seq = ds[0]
        indices = seq.challenging_frame_indices()
        assert isinstance(indices, list)
        for idx in indices:
            assert 0 <= idx < len(seq)
            assert len(seq.per_frame_attributes[idx]) > 0


# ---------------------------------------------------------------------------
# Per-frame attributes consistency
# ---------------------------------------------------------------------------

class TestPerFrameAttributes:
    def test_all_frozensets(self, challenge_ds):
        seq = challenge_ds[0]
        for tags in seq.per_frame_attributes:
            assert isinstance(tags, frozenset)

    def test_tags_are_strings(self, challenge_ds):
        seq = challenge_ds[0]
        for tags in seq.per_frame_attributes:
            for t in tags:
                assert isinstance(t, str)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_frames(self):
        ds1 = ChallengeDataset("fast_motion", num_sequences=2, num_frames=20, seed=99)
        ds2 = ChallengeDataset("fast_motion", num_sequences=2, num_frames=20, seed=99)
        for i in range(2):
            frames1 = list(ds1[i])
            frames2 = list(ds2[i])
            for f1, f2 in zip(frames1, frames2):
                np.testing.assert_array_equal(f1, f2)

    def test_different_seed_different_frames(self):
        ds1 = ChallengeDataset("fast_motion", num_sequences=1, num_frames=20, seed=1)
        ds2 = ChallengeDataset("fast_motion", num_sequences=1, num_frames=20, seed=2)
        frames1 = list(ds1[0])
        frames2 = list(ds2[0])
        # At least some frames should differ
        diffs = [not np.array_equal(f1, f2) for f1, f2 in zip(frames1, frames2)]
        assert any(diffs)

    def test_sequences_differ_within_dataset(self):
        ds = ChallengeDataset("linear" if False else "occlusion",
                              num_sequences=3, num_frames=20, seed=42)
        frames_0 = list(ds[0])
        frames_1 = list(ds[1])
        # Sequences with different seeds should produce different content
        diffs = [not np.array_equal(f0, f1) for f0, f1 in zip(frames_0, frames_1)]
        assert any(diffs)


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:
    def test_engine_runs_without_error(self):
        """Verify the BenchmarkEngine can run a tracker over a ChallengeDataset.

        Uses fast_motion (no occlusion-induced off-frame drift) with a
        generous frame size so the tracker's patch stays in bounds.
        """
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker

        ds = ChallengeDataset(
            "fast_motion",
            num_sequences=2,
            num_frames=20,
            frame_size=(160, 120),
            bbox_size=(20, 20),
            seed=0,
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="test-fm")
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0

    def test_engine_result_ious_in_range(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker

        ds = ChallengeDataset(
            "scale_variation",
            num_sequences=1,
            num_frames=15,
            frame_size=(160, 120),
            bbox_size=(20, 20),
            seed=1,
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="test-sv")
        for sr in result.sequence_results:
            assert np.all(sr.ious >= 0.0)
            assert np.all(sr.ious <= 1.0)


# ---------------------------------------------------------------------------
# Distractors-specific
# ---------------------------------------------------------------------------

class TestDistractors:
    def test_n_distractors_param(self):
        ds = ChallengeDataset(
            "distractors",
            num_sequences=1,
            num_frames=20,
            n_distractors=5,
            seed=0,
        )
        seq = ds[0]
        assert len(seq) == 20
        # DIS tag must be in sequence attributes
        assert "DIS" in seq.attributes
