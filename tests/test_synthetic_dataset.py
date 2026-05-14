"""Unit tests for the SyntheticDataset generator."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset, SyntheticSequence


class TestSyntheticSequence:
    def _make(self, n: int = 10) -> SyntheticSequence:
        gt = np.column_stack([
            np.linspace(0, 50, n),
            np.linspace(0, 30, n),
            np.full(n, 40.0),
            np.full(n, 30.0),
        ])
        return SyntheticSequence(
            name="test_seq",
            ground_truth=gt,
            frame_size=(120, 160),
        )

    def test_len(self):
        seq = self._make(15)
        assert len(seq) == 15

    def test_iter_yields_correct_frame_count(self):
        seq = self._make(8)
        frames = list(seq)
        assert len(frames) == 8

    def test_frame_shape(self):
        seq = self._make(5)
        for frame in seq:
            assert frame.shape == (120, 160, 3)
            assert frame.dtype == np.uint8

    def test_init_bbox(self):
        seq = self._make(10)
        x, y, w, h = seq.init_bbox
        assert w == pytest.approx(40.0)
        assert h == pytest.approx(30.0)

    def test_target_color_painted(self):
        gt = np.array([[10.0, 10.0, 40.0, 30.0]])
        seq = SyntheticSequence(
            name="color_test",
            ground_truth=gt,
            frame_size=(100, 100),
            target_color=(0, 0, 255),
            background_color=(128, 128, 128),
        )
        frame = next(iter(seq))
        # Centre of the target should be painted red (BGR: 0, 0, 255)
        cy, cx = 10 + 15, 10 + 20
        np.testing.assert_array_equal(frame[cy, cx], [0, 0, 255])

    def test_invalid_ground_truth_shape(self):
        with pytest.raises(ValueError):
            SyntheticSequence(
                name="bad",
                ground_truth=np.zeros((10, 3)),  # wrong shape
                frame_size=(100, 100),
            )


class TestSyntheticDatasetCreation:
    @pytest.mark.parametrize("motion", ["linear", "sinusoidal", "random_walk"])
    def test_motion_types(self, motion):
        ds = SyntheticDataset(num_sequences=2, sequence_length=20, motion=motion, seed=1)
        assert len(ds) == 2
        for seq in ds:
            assert len(seq) == 20

    def test_invalid_motion_raises(self):
        with pytest.raises(ValueError, match="motion must be one of"):
            SyntheticDataset(motion="bounce")

    def test_invalid_num_sequences(self):
        with pytest.raises(ValueError):
            SyntheticDataset(num_sequences=0)

    def test_invalid_sequence_length(self):
        with pytest.raises(ValueError):
            SyntheticDataset(sequence_length=1)

    def test_len(self):
        ds = SyntheticDataset(num_sequences=7)
        assert len(ds) == 7

    def test_getitem(self):
        ds = SyntheticDataset(num_sequences=3)
        seq = ds[1]
        assert isinstance(seq, SyntheticSequence)

    def test_repr(self):
        ds = SyntheticDataset(motion="sinusoidal")
        r = repr(ds)
        assert "sinusoidal" in r


class TestTrajectoryProperties:
    def _gt(self, motion: str, seed: int = 0) -> np.ndarray:
        ds = SyntheticDataset(
            num_sequences=1,
            sequence_length=50,
            frame_size=(240, 320),
            target_size=(40, 30),
            motion=motion,
            seed=seed,
        )
        return ds[0].ground_truth

    def test_gt_shape(self):
        gt = self._gt("linear")
        assert gt.shape == (50, 4)

    def test_target_size_constant(self):
        for motion in ("linear", "sinusoidal", "random_walk"):
            gt = self._gt(motion)
            # w and h must stay fixed at target_size = (40, 30)
            np.testing.assert_allclose(gt[:, 2], 40.0)
            np.testing.assert_allclose(gt[:, 3], 30.0)

    def test_boxes_inside_frame(self):
        fh, fw = 240, 320
        tw, th = 40, 30
        for motion in ("linear", "sinusoidal", "random_walk"):
            gt = self._gt(motion)
            assert np.all(gt[:, 0] >= 0), f"{motion}: x < 0"
            assert np.all(gt[:, 1] >= 0), f"{motion}: y < 0"
            assert np.all(gt[:, 0] <= fw - tw), f"{motion}: x + w > frame"
            assert np.all(gt[:, 1] <= fh - th), f"{motion}: y + h > frame"

    def test_reproducibility_same_seed(self):
        ds1 = SyntheticDataset(num_sequences=2, seed=42, motion="random_walk")
        ds2 = SyntheticDataset(num_sequences=2, seed=42, motion="random_walk")
        np.testing.assert_array_equal(ds1[0].ground_truth, ds2[0].ground_truth)
        np.testing.assert_array_equal(ds1[1].ground_truth, ds2[1].ground_truth)

    def test_different_seeds_produce_different_trajectories(self):
        ds1 = SyntheticDataset(num_sequences=1, seed=0, motion="random_walk")
        ds2 = SyntheticDataset(num_sequences=1, seed=999, motion="random_walk")
        assert not np.allclose(ds1[0].ground_truth, ds2[0].ground_truth)

    def test_sequence_names_are_unique(self):
        ds = SyntheticDataset(num_sequences=5, motion="linear")
        names = [ds[i].name for i in range(5)]
        assert len(set(names)) == 5


class TestSyntheticDatasetIteration:
    def test_iter_over_dataset(self):
        ds = SyntheticDataset(num_sequences=3, sequence_length=10)
        count = sum(1 for _ in ds)
        assert count == 3

    def test_frames_are_numpy_arrays(self):
        ds = SyntheticDataset(num_sequences=1, sequence_length=5, frame_size=(80, 120))
        seq = ds[0]
        for frame in seq:
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3

    def test_perfect_tracker_iou(self):
        """A tracker that always returns ground truth achieves IoU = 1."""
        from eovot.metrics.accuracy import MetricsEngine

        ds = SyntheticDataset(num_sequences=1, sequence_length=30, motion="sinusoidal")
        seq = ds[0]
        gt = seq.ground_truth
        engine = MetricsEngine()
        result = engine.compute_all(gt, gt)
        assert result.mean_iou == pytest.approx(1.0)
