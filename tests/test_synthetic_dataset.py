"""Tests for eovot.datasets.synthetic.SyntheticDataset.

All tests run entirely in memory — no real video or annotation files needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset, SyntheticSequence
from eovot.datasets.base import Sequence, BaseDataset


# ---------------------------------------------------------------------------
# SyntheticDataset construction
# ---------------------------------------------------------------------------

class TestSyntheticDatasetConstruction:
    def test_default_construction(self):
        ds = SyntheticDataset()
        assert isinstance(ds, BaseDataset)
        assert len(ds) == 5

    def test_custom_n_sequences(self):
        ds = SyntheticDataset(n_sequences=8)
        assert len(ds) == 8

    def test_invalid_motion_raises(self):
        with pytest.raises(ValueError, match="motion must be one of"):
            SyntheticDataset(motion="zigzag")  # type: ignore[arg-type]

    def test_invalid_speed_raises(self):
        with pytest.raises(ValueError, match="speed must be positive"):
            SyntheticDataset(speed=-1.0)

    @pytest.mark.parametrize("motion", ["linear", "circular", "sinusoidal", "random"])
    def test_all_motion_patterns_construct(self, motion):
        ds = SyntheticDataset(n_sequences=2, n_frames=10, motion=motion, seed=0)
        assert len(ds) == 2


# ---------------------------------------------------------------------------
# Sequence retrieval and structure
# ---------------------------------------------------------------------------

class TestSyntheticDatasetGetItem:
    def setup_method(self):
        self.ds = SyntheticDataset(n_sequences=4, n_frames=20, motion="linear", seed=7)

    def test_getitem_returns_sequence(self):
        seq = self.ds[0]
        assert isinstance(seq, Sequence)

    def test_getitem_out_of_range_raises(self):
        with pytest.raises(IndexError):
            _ = self.ds[99]

    def test_iteration_yields_all_sequences(self):
        seqs = list(self.ds)
        assert len(seqs) == 4

    def test_sequence_names_are_unique(self):
        names = [seq.name for seq in self.ds]
        assert len(names) == len(set(names))

    def test_sequence_length(self):
        seq = self.ds[0]
        assert len(seq) == 20

    def test_ground_truth_shape(self):
        seq = self.ds[0]
        assert seq.ground_truth.shape == (20, 4)

    def test_ground_truth_dtype(self):
        seq = self.ds[0]
        assert seq.ground_truth.dtype == np.float64

    def test_ground_truth_positive_size(self):
        seq = self.ds[0]
        gt = seq.ground_truth
        assert np.all(gt[:, 2] > 0), "Width must be positive"
        assert np.all(gt[:, 3] > 0), "Height must be positive"

    def test_init_bbox_valid(self):
        seq = self.ds[0]
        x, y, w, h = seq.init_bbox
        assert w > 0 and h > 0


# ---------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------

class TestSyntheticSequenceFrames:
    def setup_method(self):
        self.ds = SyntheticDataset(n_sequences=2, n_frames=15, motion="circular", seed=3)

    def test_frame_count(self):
        seq = self.ds[0]
        frames = list(seq)
        assert len(frames) == 15

    def test_frame_shape(self):
        seq = self.ds[0]
        frame = next(iter(seq))
        # default frame_size=(320, 240) → shape (240, 320, 3)
        assert frame.shape == (240, 320, 3)

    def test_frame_dtype(self):
        seq = self.ds[0]
        frame = next(iter(seq))
        assert frame.dtype == np.uint8

    def test_frame_not_entirely_black(self):
        seq = self.ds[0]
        frame = next(iter(seq))
        assert frame.max() > 0, "Frame should contain a visible target"

    def test_reproducibility(self):
        """Same seed should produce identical frame pixel values."""
        ds1 = SyntheticDataset(n_sequences=1, n_frames=5, motion="linear", seed=99)
        ds2 = SyntheticDataset(n_sequences=1, n_frames=5, motion="linear", seed=99)
        frames1 = list(ds1[0])
        frames2 = list(ds2[0])
        for f1, f2 in zip(frames1, frames2):
            np.testing.assert_array_equal(f1, f2)

    def test_different_seeds_differ(self):
        """Different seeds should produce distinct trajectories."""
        ds1 = SyntheticDataset(n_sequences=1, n_frames=20, motion="random", seed=1)
        ds2 = SyntheticDataset(n_sequences=1, n_frames=20, motion="random", seed=2)
        gt1 = ds1[0].ground_truth
        gt2 = ds2[0].ground_truth
        assert not np.allclose(gt1, gt2), "Different seeds should produce different trajectories"


# ---------------------------------------------------------------------------
# Motion pattern sanity checks
# ---------------------------------------------------------------------------

class TestMotionPatterns:
    @pytest.mark.parametrize("motion", ["linear", "circular", "sinusoidal", "random"])
    def test_boxes_stay_within_frame(self, motion):
        fw, fh = 320, 240
        tw, th = 40, 30
        ds = SyntheticDataset(
            n_sequences=3, n_frames=100, motion=motion,
            frame_size=(fw, fh), target_size=(tw, th), seed=0,
        )
        for seq in ds:
            gt = seq.ground_truth
            assert np.all(gt[:, 0] >= 0), f"{motion}: x < 0"
            assert np.all(gt[:, 1] >= 0), f"{motion}: y < 0"
            assert np.all(gt[:, 0] + gt[:, 2] <= fw), f"{motion}: box exceeds frame width"
            assert np.all(gt[:, 1] + gt[:, 3] <= fh), f"{motion}: box exceeds frame height"

    def test_circular_motion_varies(self):
        ds = SyntheticDataset(n_sequences=1, n_frames=60, motion="circular", seed=5)
        gt = ds[0].ground_truth
        # Circular motion should move the centre significantly.
        cx = gt[:, 0] + gt[:, 2] / 2
        assert cx.max() - cx.min() > 10, "Circular motion should produce visible displacement"

    def test_sinusoidal_motion_oscillates(self):
        ds = SyntheticDataset(n_sequences=1, n_frames=100, motion="sinusoidal", seed=5)
        gt = ds[0].ground_truth
        cx = gt[:, 0] + gt[:, 2] / 2
        # A sinusoidal pattern should span a decent range on the x-axis.
        assert cx.max() - cx.min() > 5, "Sinusoidal motion should oscillate on x-axis"


# ---------------------------------------------------------------------------
# Dataset interface compliance
# ---------------------------------------------------------------------------

class TestBaseDatasetCompliance:
    """Verify SyntheticDataset fully satisfies the BaseDataset contract."""

    def test_iter_protocol(self):
        """BaseDataset.__iter__ delegates to __getitem__; verify it works."""
        ds = SyntheticDataset(n_sequences=3, n_frames=10)
        seqs = list(iter(ds))
        assert len(seqs) == 3
        for seq in seqs:
            assert isinstance(seq, Sequence)

    def test_repr_contains_motion(self):
        ds = SyntheticDataset(motion="sinusoidal")
        assert "sinusoidal" in repr(ds)

    def test_target_size_reflected_in_gt(self):
        tw, th = 60, 45
        ds = SyntheticDataset(
            n_sequences=1, n_frames=20,
            target_size=(tw, th), seed=0,
        )
        gt = ds[0].ground_truth
        np.testing.assert_allclose(gt[:, 2], tw, err_msg="GT width should match target_size[0]")
        np.testing.assert_allclose(gt[:, 3], th, err_msg="GT height should match target_size[1]")

    def test_custom_frame_size(self):
        ds = SyntheticDataset(n_sequences=1, n_frames=5, frame_size=(160, 120))
        frame = next(iter(ds[0]))
        assert frame.shape == (120, 160, 3)
