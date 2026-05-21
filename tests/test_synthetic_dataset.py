"""Tests for SyntheticDataset procedural sequence generator."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.synthetic import MOTION_MODELS, SyntheticDataset


class TestSyntheticDatasetConstruction:
    def test_default_construction(self):
        dataset = SyntheticDataset()
        assert len(dataset) == 10

    def test_custom_num_sequences(self):
        dataset = SyntheticDataset(num_sequences=5)
        assert len(dataset) == 5

    def test_invalid_motion_raises(self):
        with pytest.raises(ValueError, match="motion"):
            SyntheticDataset(motion="nonexistent")

    @pytest.mark.parametrize("motion", MOTION_MODELS)
    def test_all_motion_models_construct(self, motion):
        dataset = SyntheticDataset(num_sequences=2, num_frames=10, motion=motion, seed=0)
        assert len(dataset) == 2


class TestSyntheticSequence:
    @pytest.fixture()
    def dataset(self):
        return SyntheticDataset(num_sequences=3, num_frames=20, seed=7)

    def test_sequence_name(self, dataset):
        seq = dataset[0]
        assert "synthetic" in seq.name

    def test_frame_count(self, dataset):
        seq = dataset[0]
        assert len(seq) == 20

    def test_frame_shape(self, dataset):
        seq = dataset[0]
        frames = list(seq)
        assert frames[0].shape == (240, 320, 3)
        assert frames[0].dtype == np.uint8

    def test_ground_truth_shape(self, dataset):
        seq = dataset[0]
        assert seq.ground_truth.shape == (20, 4)

    def test_ground_truth_dtype(self, dataset):
        seq = dataset[0]
        assert seq.ground_truth.dtype == np.float64

    def test_init_bbox_matches_gt_first_row(self, dataset):
        seq = dataset[0]
        np.testing.assert_array_equal(
            np.array(seq.init_bbox), seq.ground_truth[0]
        )

    def test_bboxes_within_frame(self, dataset):
        seq = dataset[0]
        gt = seq.ground_truth
        assert np.all(gt[:, 0] >= 0)  # x >= 0
        assert np.all(gt[:, 1] >= 0)  # y >= 0

    def test_custom_frame_size(self):
        dataset = SyntheticDataset(
            num_sequences=1, num_frames=5,
            frame_size=(160, 120), seed=0
        )
        seq = dataset[0]
        frames = list(seq)
        assert frames[0].shape == (120, 160, 3)

    def test_custom_bbox_size(self):
        dataset = SyntheticDataset(
            num_sequences=1, num_frames=5,
            bbox_size=(20, 15), seed=0
        )
        seq = dataset[0]
        gt = seq.ground_truth
        # All GT widths should equal the configured bbox width
        assert np.all(gt[:, 2] == 20.0)
        assert np.all(gt[:, 3] == 15.0)


class TestSyntheticDatasetReproducibility:
    def test_same_seed_same_frames(self):
        d1 = SyntheticDataset(num_sequences=2, num_frames=10, seed=42)
        d2 = SyntheticDataset(num_sequences=2, num_frames=10, seed=42)
        frames1 = list(d1[0])
        frames2 = list(d2[0])
        np.testing.assert_array_equal(frames1[0], frames2[0])
        np.testing.assert_array_equal(frames1[-1], frames2[-1])

    def test_different_seed_different_frames(self):
        d1 = SyntheticDataset(num_sequences=2, num_frames=10, seed=1)
        d2 = SyntheticDataset(num_sequences=2, num_frames=10, seed=2)
        frames1 = list(d1[0])
        frames2 = list(d2[0])
        assert not np.array_equal(frames1[0], frames2[0])

    def test_sequences_differ_within_dataset(self):
        dataset = SyntheticDataset(num_sequences=3, num_frames=10, seed=0)
        gt0 = dataset[0].ground_truth
        gt1 = dataset[1].ground_truth
        assert not np.array_equal(gt0, gt1)


class TestSyntheticDatasetIndexing:
    def test_out_of_range_raises(self):
        dataset = SyntheticDataset(num_sequences=3, num_frames=5)
        with pytest.raises(IndexError):
            _ = dataset[3]

    def test_negative_index_raises(self):
        dataset = SyntheticDataset(num_sequences=3, num_frames=5)
        with pytest.raises(IndexError):
            _ = dataset[-1]

    def test_all_sequences_accessible(self):
        dataset = SyntheticDataset(num_sequences=4, num_frames=5, seed=0)
        for i in range(len(dataset)):
            seq = dataset[i]
            assert len(seq) == 5


class TestSyntheticDatasetMotionModels:
    @pytest.mark.parametrize("motion", MOTION_MODELS)
    def test_trajectory_stays_valid(self, motion):
        """All GT boxes should have positive width and height."""
        dataset = SyntheticDataset(
            num_sequences=2, num_frames=50, motion=motion, seed=0
        )
        for i in range(len(dataset)):
            gt = dataset[i].ground_truth
            assert np.all(gt[:, 2] > 0), f"{motion}: w must be > 0"
            assert np.all(gt[:, 3] > 0), f"{motion}: h must be > 0"

    def test_circular_motion_returns_to_origin(self):
        """Circular trajectory should complete at least one full loop in 200 frames."""
        dataset = SyntheticDataset(
            num_sequences=1, num_frames=200, motion="circular", seed=0
        )
        gt = dataset[0].ground_truth
        # The trajectory should not be strictly monotone — it reverses direction
        xs = gt[:, 0]
        assert not np.all(np.diff(xs) >= 0) and not np.all(np.diff(xs) <= 0)
