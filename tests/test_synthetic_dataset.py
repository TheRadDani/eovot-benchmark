"""Unit and integration tests for eovot.datasets.synthetic.

Tests cover:
- SyntheticDataset construction and validation
- Per-sequence frame count, shape, dtype, and GT alignment
- All three motion patterns
- Reproducibility (same seed → same frames)
- Lazy caching behaviour
- Full benchmark pipeline integration (no external data required)
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset, _InMemorySequence
from eovot.metrics.accuracy import MetricsEngine
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

class TestSyntheticDatasetConstruction:

    def test_default_length(self):
        ds = SyntheticDataset()
        assert len(ds) == 10

    def test_custom_length(self):
        ds = SyntheticDataset(num_sequences=7)
        assert len(ds) == 7

    def test_invalid_motion_raises_at_construction(self):
        with pytest.raises(ValueError, match="Unknown motion"):
            SyntheticDataset(motion="teleport")  # type: ignore[arg-type]

    @pytest.mark.parametrize("motion", ["linear", "circular", "random"])
    def test_valid_motions_accepted(self, motion):
        ds = SyntheticDataset(num_sequences=1, motion=motion)
        assert ds.motion == motion

    def test_repr_contains_class_name(self):
        ds = SyntheticDataset(num_sequences=5, motion="circular")
        r = repr(ds)
        assert "SyntheticDataset" in r
        assert "circular" in r


# ---------------------------------------------------------------------------
# Index access
# ---------------------------------------------------------------------------

class TestSyntheticDatasetIndexing:

    def test_getitem_returns_sequence(self):
        ds = SyntheticDataset(num_sequences=3, num_frames=5)
        seq = ds[0]
        assert seq is not None

    def test_getitem_positive_out_of_range_raises(self):
        ds = SyntheticDataset(num_sequences=3)
        with pytest.raises(IndexError):
            _ = ds[10]

    def test_getitem_negative_index_raises(self):
        ds = SyntheticDataset(num_sequences=3)
        with pytest.raises(IndexError):
            _ = ds[-1]

    def test_iteration_yields_all_sequences(self):
        ds = SyntheticDataset(num_sequences=4, num_frames=5)
        seqs = list(ds)
        assert len(seqs) == 4

    def test_caching_returns_same_object(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=5)
        assert ds[0] is ds[0]

    def test_sequence_names(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=5, motion="linear")
        assert ds[0].name == "synth_linear_000"
        assert ds[1].name == "synth_linear_001"


# ---------------------------------------------------------------------------
# Frame properties
# ---------------------------------------------------------------------------

class TestSyntheticSequenceFrames:

    def test_frame_count_matches_num_frames(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=30)
        seq = ds[0]
        assert len(seq) == 30
        assert len(list(seq)) == 30

    def test_frame_shape(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=5, frame_size=(80, 60), bbox_size=(20, 20))
        for frame in ds[0]:
            assert frame.shape == (60, 80, 3)

    def test_frame_dtype(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=5)
        for frame in ds[0]:
            assert frame.dtype == np.uint8

    def test_frames_are_not_all_identical(self):
        """Frames should differ since the target moves."""
        ds = SyntheticDataset(num_sequences=1, num_frames=10, motion="linear")
        frames = list(ds[0])
        assert not all(np.array_equal(frames[0], f) for f in frames[1:])


# ---------------------------------------------------------------------------
# Ground-truth properties
# ---------------------------------------------------------------------------

class TestSyntheticSequenceGroundTruth:

    def test_gt_shape(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=20)
        assert ds[0].ground_truth.shape == (20, 4)

    def test_gt_dtype(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=5)
        assert ds[0].ground_truth.dtype == np.float64

    def test_gt_bbox_size_constant(self):
        bw, bh = 50, 50
        ds = SyntheticDataset(num_sequences=1, num_frames=10, bbox_size=(bw, bh))
        gt = ds[0].ground_truth
        np.testing.assert_array_equal(gt[:, 2], bw)
        np.testing.assert_array_equal(gt[:, 3], bh)

    def test_init_bbox_matches_first_gt(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=10)
        seq = ds[0]
        assert tuple(seq.init_bbox) == pytest.approx(tuple(seq.ground_truth[0]))


# ---------------------------------------------------------------------------
# Motion patterns
# ---------------------------------------------------------------------------

class TestMotionPatterns:

    @pytest.mark.parametrize("motion", ["linear", "circular", "random"])
    def test_motion_produces_correct_frame_count(self, motion):
        ds = SyntheticDataset(num_sequences=1, num_frames=15, motion=motion)
        assert len(list(ds[0])) == 15

    def test_circular_target_returns_to_start(self):
        """After a full number of rotations the target should be near its start."""
        # With 3 full rotations and num_frames divisible by them,
        # position at frame 0 and last frame should be close.
        ds = SyntheticDataset(
            num_sequences=1, num_frames=31, motion="circular", frame_size=(200, 200)
        )
        gt = ds[0].ground_truth
        # Centre of first and last bounding box
        cx0 = gt[0, 0] + gt[0, 2] / 2
        cy0 = gt[0, 1] + gt[0, 3] / 2
        cxN = gt[-1, 0] + gt[-1, 2] / 2
        cyN = gt[-1, 1] + gt[-1, 3] / 2
        dist = np.sqrt((cx0 - cxN) ** 2 + (cy0 - cyN) ** 2)
        assert dist < 20.0  # should be close to starting position

    def test_linear_target_stays_inside_frame(self):
        W, H = 200, 150
        bw, bh = 30, 30
        ds = SyntheticDataset(
            num_sequences=1, num_frames=200, motion="linear",
            frame_size=(W, H), bbox_size=(bw, bh), seed=0,
        )
        # At least 90% of GT centre positions should be inside the frame.
        gt = ds[0].ground_truth
        cx = gt[:, 0] + gt[:, 2] / 2
        cy = gt[:, 1] + gt[:, 3] / 2
        inside = ((cx >= 0) & (cx < W) & (cy >= 0) & (cy < H))
        assert inside.mean() > 0.9


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_seed_same_frames(self):
        ds1 = SyntheticDataset(num_sequences=1, num_frames=5, seed=99)
        ds2 = SyntheticDataset(num_sequences=1, num_frames=5, seed=99)
        for f1, f2 in zip(ds1[0], ds2[0]):
            np.testing.assert_array_equal(f1, f2)

    def test_different_seeds_different_frames(self):
        ds1 = SyntheticDataset(num_sequences=1, num_frames=5, seed=1)
        ds2 = SyntheticDataset(num_sequences=1, num_frames=5, seed=2)
        frames1, frames2 = list(ds1[0]), list(ds2[0])
        assert any(not np.array_equal(f1, f2) for f1, f2 in zip(frames1, frames2))

    def test_different_sequences_different_frames(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=5, seed=0)
        frames0 = list(ds[0])
        frames1 = list(ds[1])
        assert any(not np.array_equal(f0, f1) for f0, f1 in zip(frames0, frames1))


# ---------------------------------------------------------------------------
# Full benchmark pipeline integration
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:

    def test_mosse_runs_on_synthetic(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=15, seed=7)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")
        assert result.tracker_name == "MOSSE"
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0

    def test_kcf_runs_on_synthetic(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=15, seed=7)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KCFTracker(), ds, dataset_name="Synthetic")
        assert result.tracker_name == "KCF"
        assert result.mean_fps > 0

    def test_iou_in_valid_range(self):
        ds = SyntheticDataset(num_sequences=3, num_frames=20, seed=0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")
        assert 0.0 <= result.mean_iou <= 1.0

    def test_result_serialisation(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=10, seed=0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")
        d = result.to_dict()
        assert "summary" in d
        assert "sequences" in d
        assert len(d["sequences"]) == 2

    def test_center_distances_stored(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=10, seed=0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")
        for sr in result.sequence_results:
            assert sr.center_distances is not None
            assert len(sr.center_distances) == len(sr.ious)

    def test_max_sequences_limit(self):
        ds = SyntheticDataset(num_sequences=10, num_frames=5)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic", max_sequences=3)
        assert len(result.sequence_results) == 3

    @pytest.mark.parametrize("motion", ["linear", "circular", "random"])
    def test_all_motion_patterns_complete(self, motion):
        ds = SyntheticDataset(num_sequences=1, num_frames=10, motion=motion, seed=5)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name=f"Synthetic-{motion}")
        assert len(result.sequence_results) == 1

    def test_metrics_engine_on_synthetic_results(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=20, seed=0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")
        me = MetricsEngine()
        for sr in result.sequence_results:
            assert sr.predictions is not None
            assert sr.ground_truths is not None
            acc = me.compute_all(sr.predictions, sr.ground_truths)
            assert 0.0 <= acc.mean_iou <= 1.0
            assert 0.0 <= acc.success_auc <= 1.0
