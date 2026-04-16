"""Unit tests for eovot.datasets.synthetic — SyntheticDataset and friends."""

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset, SyntheticInMemorySequence


# ---------------------------------------------------------------------------
# SyntheticDataset construction
# ---------------------------------------------------------------------------


class TestSyntheticDatasetConstruction:
    def test_default_construction(self):
        ds = SyntheticDataset()
        assert len(ds) == 10

    def test_custom_length(self):
        ds = SyntheticDataset(num_sequences=5)
        assert len(ds) == 5

    def test_repr_contains_params(self):
        ds = SyntheticDataset(num_sequences=3, seq_len=50, seed=7)
        r = repr(ds)
        assert "3" in r
        assert "50" in r
        assert "7" in r

    def test_invalid_num_sequences(self):
        with pytest.raises(ValueError):
            SyntheticDataset(num_sequences=0)

    def test_invalid_seq_len(self):
        with pytest.raises(ValueError):
            SyntheticDataset(seq_len=1)

    def test_invalid_frame_size(self):
        with pytest.raises(ValueError):
            SyntheticDataset(frame_size=(0, 240))

    def test_invalid_target_size(self):
        with pytest.raises(ValueError):
            SyntheticDataset(target_size=(40, 0))


# ---------------------------------------------------------------------------
# Indexing and iteration
# ---------------------------------------------------------------------------


class TestSyntheticDatasetAccess:
    def setup_method(self):
        self.ds = SyntheticDataset(num_sequences=4, seq_len=20, seed=0)

    def test_getitem_returns_sequence(self):
        seq = self.ds[0]
        assert isinstance(seq, SyntheticInMemorySequence)

    def test_getitem_negative_raises(self):
        with pytest.raises(IndexError):
            _ = self.ds[-1]

    def test_getitem_out_of_range_raises(self):
        with pytest.raises(IndexError):
            _ = self.ds[100]

    def test_iter_yields_all_sequences(self):
        seqs = list(self.ds)
        assert len(seqs) == 4

    def test_sequence_names_unique(self):
        names = [seq.name for seq in self.ds]
        assert len(set(names)) == len(names)

    def test_sequence_names_sorted(self):
        names = [seq.name for seq in self.ds]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Sequence properties
# ---------------------------------------------------------------------------


class TestSyntheticInMemorySequence:
    def setup_method(self):
        self.ds = SyntheticDataset(num_sequences=2, seq_len=30, seed=1)
        self.seq = self.ds[0]

    def test_len_matches_seq_len(self):
        assert len(self.seq) == 30

    def test_ground_truth_shape(self):
        gt = self.seq.ground_truth
        assert gt.ndim == 2
        assert gt.shape[1] == 4
        assert gt.shape[0] == 30

    def test_init_bbox_positive_wh(self):
        x, y, w, h = self.seq.init_bbox
        assert w > 0
        assert h > 0

    def test_ground_truth_positive_wh(self):
        gt = self.seq.ground_truth
        assert np.all(gt[:, 2] > 0), "All widths must be positive"
        assert np.all(gt[:, 3] > 0), "All heights must be positive"

    def test_ground_truth_within_frame(self):
        ds = SyntheticDataset(
            num_sequences=1, seq_len=50,
            frame_size=(320, 240), target_size=(40, 30),
            margin=20, seed=42,
        )
        gt = ds[0].ground_truth
        # x >= 0, y >= 0, x+w <= width, y+h <= height (with some slack for boundary reflection)
        assert np.all(gt[:, 0] >= -5), "Some x coords very negative"
        assert np.all(gt[:, 1] >= -5), "Some y coords very negative"
        assert np.all(gt[:, 0] + gt[:, 2] <= 325), "Some boxes exceed frame width"
        assert np.all(gt[:, 1] + gt[:, 3] <= 245), "Some boxes exceed frame height"


# ---------------------------------------------------------------------------
# Frame iteration
# ---------------------------------------------------------------------------


class TestFrameIteration:
    def setup_method(self):
        self.seq = SyntheticDataset(
            num_sequences=1, seq_len=15,
            frame_size=(160, 120), target_size=(20, 15), seed=0,
        )[0]

    def test_frame_count(self):
        frames = list(self.seq)
        assert len(frames) == 15

    def test_frame_dtype(self):
        frame = next(iter(self.seq))
        assert frame.dtype == np.uint8

    def test_frame_shape(self):
        frame = next(iter(self.seq))
        assert frame.shape == (120, 160, 3)

    def test_frame_value_range(self):
        frame = next(iter(self.seq))
        assert frame.min() >= 0
        assert frame.max() <= 255

    def test_iteration_reproducible(self):
        frames_a = list(self.seq)
        frames_b = list(self.seq)
        for fa, fb in zip(frames_a, frames_b):
            np.testing.assert_array_equal(fa, fb)

    def test_different_seeds_different_frames(self):
        seq_a = SyntheticDataset(num_sequences=1, seq_len=10, seed=0)[0]
        seq_b = SyntheticDataset(num_sequences=1, seq_len=10, seed=99)[0]
        fa = next(iter(seq_a))
        fb = next(iter(seq_b))
        assert not np.array_equal(fa, fb), "Different seeds should produce different frames"

    def test_gt_and_frame_count_aligned(self):
        frames = list(self.seq)
        assert len(frames) == len(self.seq.ground_truth)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_gt(self):
        ds1 = SyntheticDataset(num_sequences=3, seq_len=20, seed=42)
        ds2 = SyntheticDataset(num_sequences=3, seq_len=20, seed=42)
        for i in range(3):
            np.testing.assert_array_equal(
                ds1[i].ground_truth,
                ds2[i].ground_truth,
                err_msg=f"Sequence {i} GT differs between same-seed datasets",
            )

    def test_different_seed_different_gt(self):
        ds1 = SyntheticDataset(num_sequences=1, seq_len=20, seed=1)
        ds2 = SyntheticDataset(num_sequences=1, seq_len=20, seed=2)
        assert not np.array_equal(ds1[0].ground_truth, ds2[0].ground_truth)

    def test_sequences_within_dataset_differ(self):
        ds = SyntheticDataset(num_sequences=3, seq_len=30, seed=0)
        gt0 = ds[0].ground_truth
        gt1 = ds[1].ground_truth
        assert not np.array_equal(gt0, gt1), "Sequences in the same dataset must differ"


# ---------------------------------------------------------------------------
# Integration: full pipeline with BenchmarkEngine
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_benchmark_engine_runs(self):
        """SyntheticDataset plugs into BenchmarkEngine without errors."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=2, seq_len=15, seed=0)
        tracker = MOSSETracker()
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")

        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0
        assert result.peak_memory_mb > 0.0
        assert len(result.sequence_results) == 2

    def test_result_to_dict_structure(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=1, seq_len=10, seed=0)
        tracker = MOSSETracker()
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        d = result.to_dict()

        assert "summary" in d
        assert "sequences" in d
        assert d["summary"]["num_sequences"] == 1
        assert len(d["sequences"]) == 1
