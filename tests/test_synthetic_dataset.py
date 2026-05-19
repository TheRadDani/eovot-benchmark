"""Unit tests for eovot.datasets.synthetic."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset, SyntheticSequence


# ---------------------------------------------------------------------------
# SyntheticDataset — construction and basic properties
# ---------------------------------------------------------------------------

class TestSyntheticDatasetInit:
    def test_default_construction(self):
        ds = SyntheticDataset()
        assert len(ds) == 10

    def test_custom_num_sequences(self):
        ds = SyntheticDataset(num_sequences=5)
        assert len(ds) == 5

    def test_getitem_returns_synthetic_sequence(self):
        ds = SyntheticDataset(num_sequences=3)
        seq = ds[0]
        assert isinstance(seq, SyntheticSequence)

    def test_all_sequences_unique_names(self):
        ds = SyntheticDataset(num_sequences=4, motion="linear")
        names = [ds[i].name for i in range(4)]
        assert len(set(names)) == 4

    def test_bbox_larger_than_frame_raises(self):
        with pytest.raises(ValueError, match="bbox_size"):
            SyntheticDataset(frame_size=(100, 80), bbox_size=(150, 40))

    def test_zero_sequences(self):
        ds = SyntheticDataset(num_sequences=0)
        assert len(ds) == 0

    def test_iteration_over_dataset(self):
        ds = SyntheticDataset(num_sequences=3)
        seqs = [ds[i] for i in range(len(ds))]
        assert len(seqs) == 3


# ---------------------------------------------------------------------------
# SyntheticSequence — ground truth shape and validity
# ---------------------------------------------------------------------------

class TestSyntheticSequenceGroundTruth:
    @pytest.fixture
    def seq(self):
        return SyntheticSequence(
            name="test_linear",
            num_frames=50,
            frame_size=(320, 240),
            bbox_size=(40, 30),
            motion="linear",
            seed=7,
        )

    def test_ground_truth_shape(self, seq):
        gt = seq.ground_truth
        assert gt.shape == (50, 4)

    def test_ground_truth_dtype(self, seq):
        assert seq.ground_truth.dtype == np.float64

    def test_init_bbox_matches_first_gt_row(self, seq):
        gt = seq.ground_truth
        expected = tuple(gt[0])
        assert seq.init_bbox == expected

    def test_bbox_width_height_constant(self, seq):
        gt = seq.ground_truth
        # Width and height should stay at bbox_size throughout
        np.testing.assert_allclose(gt[:, 2], 40.0)
        np.testing.assert_allclose(gt[:, 3], 30.0)

    def test_all_boxes_within_frame(self, seq):
        gt = seq.ground_truth
        W, H = 320, 240
        bw, bh = 40, 30
        assert (gt[:, 0] >= 0).all(), "x must be >= 0"
        assert (gt[:, 1] >= 0).all(), "y must be >= 0"
        assert (gt[:, 0] + gt[:, 2] <= W).all(), "x+w must be <= W"
        assert (gt[:, 1] + gt[:, 3] <= H).all(), "y+h must be <= H"

    def test_non_negative_coordinates(self, seq):
        gt = seq.ground_truth
        assert (gt[:, 0] >= 0).all()
        assert (gt[:, 1] >= 0).all()


# ---------------------------------------------------------------------------
# Motion patterns
# ---------------------------------------------------------------------------

class TestMotionPatterns:
    @pytest.mark.parametrize("motion", ["linear", "circular", "random"])
    def test_all_motion_patterns_run(self, motion):
        seq = SyntheticSequence(
            name=f"test_{motion}",
            num_frames=30,
            frame_size=(160, 120),
            bbox_size=(20, 20),
            motion=motion,
            seed=0,
        )
        gt = seq.ground_truth
        assert gt.shape == (30, 4)

    def test_invalid_motion_raises(self):
        with pytest.raises(ValueError, match="Unknown motion"):
            SyntheticSequence(
                name="bad",
                num_frames=10,
                frame_size=(160, 120),
                bbox_size=(20, 20),
                motion="teleport",
                seed=0,
            )

    def test_linear_motion_is_not_stationary(self):
        seq = SyntheticSequence(
            name="t", num_frames=40, frame_size=(320, 240),
            bbox_size=(40, 40), motion="linear", seed=1,
        )
        gt = seq.ground_truth
        # The target should have moved over 40 frames
        displacements = np.diff(gt[:, :2], axis=0)
        assert (np.abs(displacements) > 0).any()

    def test_circular_motion_returns_to_start(self):
        """After one full orbit the target should be near its starting position."""
        seq = SyntheticSequence(
            name="t", num_frames=100, frame_size=(320, 240),
            bbox_size=(20, 20), motion="circular", seed=0,
        )
        gt = seq.ground_truth
        start = gt[0, :2]
        end = gt[-1, :2]
        # Last frame is near frame 0 but not equal (99/100 of the orbit)
        dist = float(np.linalg.norm(end - start))
        assert dist < 30.0  # within 30 px of start after ~full orbit

    def test_different_seeds_different_trajectories(self):
        s0 = SyntheticSequence("a", 30, (160, 120), (20, 20), "random", seed=0)
        s1 = SyntheticSequence("b", 30, (160, 120), (20, 20), "random", seed=99)
        assert not np.allclose(s0.ground_truth, s1.ground_truth)


# ---------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------

class TestFrameGeneration:
    def test_frame_count(self):
        seq = SyntheticSequence("t", 25, (160, 120), (20, 20), "linear", seed=5)
        frames = list(seq)
        assert len(frames) == 25

    def test_frame_shape(self):
        seq = SyntheticSequence("t", 5, (160, 120), (20, 20), "linear", seed=5)
        frames = list(seq)
        for frame in frames:
            assert frame.shape == (120, 160, 3)

    def test_frame_dtype(self):
        seq = SyntheticSequence("t", 5, (160, 120), (20, 20), "linear", seed=5)
        frames = list(seq)
        for frame in frames:
            assert frame.dtype == np.uint8

    def test_frames_are_not_identical(self):
        seq = SyntheticSequence("t", 10, (160, 120), (20, 20), "linear", seed=5)
        frames = list(seq)
        assert not all(np.array_equal(frames[0], f) for f in frames[1:])

    def test_deterministic_generation(self):
        seq1 = SyntheticSequence("t", 5, (160, 120), (20, 20), "linear", seed=42)
        seq2 = SyntheticSequence("t", 5, (160, 120), (20, 20), "linear", seed=42)
        for f1, f2 in zip(seq1, seq2):
            np.testing.assert_array_equal(f1, f2)

    def test_different_seeds_different_frames(self):
        seq1 = SyntheticSequence("t", 5, (160, 120), (20, 20), "linear", seed=1)
        seq2 = SyntheticSequence("t", 5, (160, 120), (20, 20), "linear", seed=2)
        frames1 = list(seq1)
        frames2 = list(seq2)
        assert not all(np.array_equal(f1, f2) for f1, f2 in zip(frames1, frames2))


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestSyntheticDatasetIntegration:
    def test_benchmark_engine_runs_on_synthetic(self):
        """BenchmarkEngine must be able to run a tracker on SyntheticDataset."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=0)
        tracker = MOSSETracker()
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic-Test")
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0

    def test_mean_iou_nonnegative_on_synthetic(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=3, num_frames=30, motion="linear", seed=1)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, "Synthetic")
        assert result.mean_iou >= 0.0
        assert result.mean_iou <= 1.0
