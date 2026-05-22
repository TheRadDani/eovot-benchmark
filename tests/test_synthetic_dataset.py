"""Tests for eovot.datasets.synthetic.SyntheticDataset."""

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.datasets.base import Sequence


class TestSyntheticDatasetConstruction:
    def test_default_construction(self):
        ds = SyntheticDataset()
        assert len(ds) == 10

    def test_custom_size(self):
        ds = SyntheticDataset(num_sequences=5, num_frames=30)
        assert len(ds) == 5

    def test_invalid_motion_raises(self):
        with pytest.raises(ValueError, match="Unknown motion pattern"):
            SyntheticDataset(motion="teleport")

    def test_repr_contains_motion(self):
        ds = SyntheticDataset(motion="circular")
        assert "circular" in repr(ds)

    def test_index_out_of_range_raises(self):
        ds = SyntheticDataset(num_sequences=3)
        with pytest.raises(IndexError):
            _ = ds[5]


class TestSyntheticSequenceProperties:
    @pytest.fixture
    def seq(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=20, seed=0)
        return ds[0]

    def test_sequence_type(self, seq):
        assert isinstance(seq, Sequence)

    def test_frame_count(self, seq):
        frames = list(seq)
        assert len(frames) == 20

    def test_ground_truth_shape(self, seq):
        assert seq.ground_truth.shape == (20, 4)

    def test_ground_truth_positive_dimensions(self, seq):
        w = seq.ground_truth[:, 2]
        h = seq.ground_truth[:, 3]
        assert np.all(w > 0) and np.all(h > 0)

    def test_init_bbox_matches_first_gt(self, seq):
        assert seq.init_bbox == tuple(seq.ground_truth[0])

    def test_frame_shape(self, seq):
        frame = next(iter(seq))
        assert frame.ndim == 3 and frame.shape[2] == 3

    def test_frame_dtype(self, seq):
        frame = next(iter(seq))
        assert frame.dtype == np.uint8

    def test_sequence_name_contains_motion(self):
        ds = SyntheticDataset(num_sequences=1, motion="circular")
        assert "circular" in ds[0].name


class TestMotionPatterns:
    @pytest.mark.parametrize("motion", ["linear", "circular", "random"])
    def test_all_patterns_produce_valid_sequences(self, motion):
        ds = SyntheticDataset(num_sequences=2, num_frames=15, motion=motion, seed=7)
        for seq in ds:
            assert seq.ground_truth.shape[1] == 4
            assert len(list(seq)) == 15

    def test_reproducibility(self):
        ds1 = SyntheticDataset(num_sequences=2, seed=42)
        ds2 = SyntheticDataset(num_sequences=2, seed=42)
        np.testing.assert_array_equal(ds1[0].ground_truth, ds2[0].ground_truth)

    def test_different_seeds_differ(self):
        ds1 = SyntheticDataset(num_sequences=1, seed=0)
        ds2 = SyntheticDataset(num_sequences=1, seed=1)
        assert not np.array_equal(ds1[0].ground_truth, ds2[0].ground_truth)

    def test_caching_returns_same_object(self):
        ds = SyntheticDataset(num_sequences=1)
        assert ds[0] is ds[0]


class TestSyntheticDatasetIteration:
    def test_iteration_count(self):
        ds = SyntheticDataset(num_sequences=4, num_frames=10)
        seqs = list(ds)
        assert len(seqs) == 4

    def test_sequence_names_unique(self):
        ds = SyntheticDataset(num_sequences=5)
        names = [seq.name for seq in ds]
        assert len(set(names)) == 5
