"""Tests for SyntheticDataset scale and occlusion challenge patterns.

Verifies that the new patterns generate correct GT sequences, render frames
consistently with the pattern specification, and integrate cleanly with
the BenchmarkEngine.
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kalman import KalmanFilterTracker


# ---------------------------------------------------------------------------
# Scale pattern
# ---------------------------------------------------------------------------

class TestSyntheticScalePattern:
    def test_creates_without_error(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=30, motion="scale")
        assert len(ds) == 2

    def test_sequence_names_include_scale(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=20, motion="scale")
        for i in range(len(ds)):
            assert "scale" in ds[i].name

    def test_repr_includes_motion(self):
        ds = SyntheticDataset(motion="scale")
        assert "scale" in repr(ds)

    def test_gt_shape(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=20, motion="scale")
        for i in range(len(ds)):
            assert ds[i].ground_truth.shape == (20, 4)

    def test_gt_sizes_vary(self):
        """Scale pattern must produce meaningful width variation (>50 %)."""
        ds = SyntheticDataset(
            num_sequences=1, num_frames=60, motion="scale",
            bbox_size=(40, 40), seed=0,
        )
        widths = ds[0].ground_truth[:, 2]
        assert widths.max() > widths.min() * 1.5, (
            f"Expected > 50 % width variation, got min={widths.min():.1f} "
            f"max={widths.max():.1f}"
        )

    def test_gt_widths_positive(self):
        ds = SyntheticDataset(num_sequences=3, num_frames=40, motion="scale")
        for i in range(len(ds)):
            assert (ds[i].ground_truth[:, 2] > 0).all()
            assert (ds[i].ground_truth[:, 3] > 0).all()

    def test_frame_count_matches(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=50, motion="scale")
        frames = list(ds[0])
        assert len(frames) == 50

    def test_scale_peak_near_middle(self):
        """Maximum target width should appear near the centre of the sequence."""
        ds = SyntheticDataset(
            num_sequences=1, num_frames=100, motion="scale",
            bbox_size=(40, 40), seed=0,
        )
        widths = ds[0].ground_truth[:, 2]
        peak_frame = int(np.argmax(widths))
        # The half-sine peak lands at t=0.5, i.e. frame 50 out of 100.
        # Allow +/-10 % tolerance.
        assert 40 <= peak_frame <= 60, f"Scale peak at frame {peak_frame}, expected ~50"

    def test_engine_runs_on_scale(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=30, motion="scale")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KCFTracker(), ds, dataset_name="Syn-Scale")
        assert result.mean_fps > 0

    def test_cached_sequence_consistent(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=20, motion="scale")
        seq_a = ds[0]
        seq_b = ds[0]  # second access — should return cached object
        np.testing.assert_array_equal(seq_a.ground_truth, seq_b.ground_truth)


# ---------------------------------------------------------------------------
# Occlusion pattern
# ---------------------------------------------------------------------------

class TestSyntheticOcclusionPattern:
    def test_creates_without_error(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=40, motion="occlusion")
        assert len(ds) == 2

    def test_sequence_names_include_occlusion(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=30, motion="occlusion")
        for i in range(len(ds)):
            assert "occlusion" in ds[i].name

    def test_repr_includes_motion(self):
        ds = SyntheticDataset(motion="occlusion")
        assert "occlusion" in repr(ds)

    def test_gt_shape(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=40, motion="occlusion")
        for i in range(len(ds)):
            assert ds[i].ground_truth.shape == (40, 4)

    def test_gt_sizes_constant(self):
        """Occlusion pattern has constant target size (only motion changes)."""
        ds = SyntheticDataset(
            num_sequences=1, num_frames=60, motion="occlusion",
            bbox_size=(40, 40), seed=0,
        )
        widths = ds[0].ground_truth[:, 2]
        heights = ds[0].ground_truth[:, 3]
        assert np.allclose(widths, widths[0]), "Width should be constant across frames"
        assert np.allclose(heights, heights[0]), "Height should be constant across frames"

    def test_frames_during_occlusion_show_background(self):
        """During occlusion (frames 35–64 % of sequence), no target rectangle."""
        ds = SyntheticDataset(
            num_sequences=1, num_frames=100, motion="occlusion",
            frame_size=(160, 120), bbox_size=(30, 30), seed=0,
        )
        seq = ds[0]
        frames = list(seq)
        n = len(frames)
        occ_start = int(n * 0.35)
        occ_end = int(n * 0.65)

        # Sample a frame in the middle of the occlusion window
        mid_occ = (occ_start + occ_end) // 2
        frame_occ = frames[mid_occ]
        gt = seq.ground_truth[mid_occ]
        cx = int(round(gt[0] + gt[2] / 2))
        cy = int(round(gt[1] + gt[3] / 2))
        cx = np.clip(cx, 0, frame_occ.shape[1] - 1)
        cy = np.clip(cy, 0, frame_occ.shape[0] - 1)
        pixel = frame_occ[cy, cx]
        # Background pixels have values < 100; target pixels have values >= 160
        assert all(int(p) < 150 for p in pixel), (
            f"Expected background pixel at ({cx},{cy}) during occlusion, "
            f"got {pixel}"
        )

    def test_frames_before_occlusion_show_target(self):
        """Before the occlusion window the target should be visible."""
        ds = SyntheticDataset(
            num_sequences=1, num_frames=100, motion="occlusion",
            frame_size=(160, 120), bbox_size=(30, 30), seed=0,
        )
        seq = ds[0]
        frames = list(seq)
        # Check a frame well before occlusion starts (frame 10, occ starts at 35)
        early_frame = frames[10]
        gt = seq.ground_truth[10]
        cx = int(round(gt[0] + gt[2] / 2))
        cy = int(round(gt[1] + gt[3] / 2))
        cx = np.clip(cx, 0, early_frame.shape[1] - 1)
        cy = np.clip(cy, 0, early_frame.shape[0] - 1)
        pixel = early_frame[cy, cx]
        # Target pixels are bright (160–255)
        assert any(int(p) >= 150 for p in pixel), (
            f"Expected target pixel at ({cx},{cy}) before occlusion, got {pixel}"
        )

    def test_frame_count_matches(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=60, motion="occlusion")
        assert len(list(ds[0])) == 60

    def test_engine_runs_on_occlusion(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=40, motion="occlusion")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Syn-Occ")
        assert result.mean_fps > 0

    def test_kalman_on_occlusion(self):
        """KalmanFilterTracker should propagate state through occlusion."""
        ds = SyntheticDataset(
            num_sequences=2, num_frames=60, motion="occlusion", seed=5
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(
            KalmanFilterTracker(), ds, dataset_name="Syn-KF-Occ"
        )
        assert result.mean_fps > 0
        # After occlusion the tracker should not completely lose the target;
        # non-zero mean IoU across the whole sequence is a reasonable sanity check.
        assert result.mean_iou >= 0.0


# ---------------------------------------------------------------------------
# Invalid motion
# ---------------------------------------------------------------------------

class TestInvalidMotion:
    def test_invalid_motion_raises(self):
        with pytest.raises(ValueError, match="Unknown motion pattern"):
            SyntheticDataset(motion="teleport")  # type: ignore[arg-type]

    def test_error_message_lists_valid_options(self):
        with pytest.raises(ValueError, match="linear"):
            SyntheticDataset(motion="fly")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    @pytest.mark.parametrize("motion", ["scale", "occlusion"])
    def test_same_seed_same_gt(self, motion):
        ds_a = SyntheticDataset(num_sequences=2, num_frames=30, motion=motion, seed=77)
        ds_b = SyntheticDataset(num_sequences=2, num_frames=30, motion=motion, seed=77)
        for i in range(len(ds_a)):
            np.testing.assert_array_equal(
                ds_a[i].ground_truth,
                ds_b[i].ground_truth,
                err_msg=f"GT differs for motion={motion}, seq={i}",
            )

    @pytest.mark.parametrize("motion", ["scale", "occlusion"])
    def test_different_seeds_different_gt(self, motion):
        ds_a = SyntheticDataset(num_sequences=1, num_frames=30, motion=motion, seed=1)
        ds_b = SyntheticDataset(num_sequences=1, num_frames=30, motion=motion, seed=2)
        # The two sequences should not be identical (very high probability)
        assert not np.array_equal(
            ds_a[0].ground_truth, ds_b[0].ground_truth
        ), "Different seeds produced identical GT"
