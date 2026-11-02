"""Tests for SyntheticDataset scale_change and occlusion motion modes."""

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset


@pytest.mark.parametrize("motion", ["scale_change", "occlusion"])
def test_new_motion_creates_sequences(motion):
    ds = SyntheticDataset(num_sequences=3, num_frames=30, motion=motion, seed=0)
    assert len(ds) == 3
    for i in range(3):
        seq = ds[i]
        assert seq.name == f"synth_{motion}_{i:03d}"
        assert len(seq) == 30
        assert seq.ground_truth.shape == (30, 4)


@pytest.mark.parametrize("motion", ["scale_change", "occlusion"])
def test_frames_are_correct_shape(motion):
    ds = SyntheticDataset(
        num_sequences=2, num_frames=20,
        frame_size=(160, 120), motion=motion, seed=1,
    )
    for seq in ds:
        for frame in seq:
            assert frame.shape == (120, 160, 3)
            assert frame.dtype == np.uint8


def test_scale_change_varies_bbox_size():
    """GT box width/height must vary across frames for scale_change."""
    ds = SyntheticDataset(
        num_sequences=1, num_frames=50, bbox_size=(40, 40),
        motion="scale_change", seed=42,
    )
    gt = ds[0].ground_truth  # (50, 4) — columns: x, y, w, h
    # Width column should not be constant (sinusoidal scaling).
    assert gt[:, 2].std() > 1.0, "scale_change should produce varying GT widths"
    assert gt[:, 3].std() > 1.0, "scale_change should produce varying GT heights"


def test_occlusion_frames_differ():
    """Occlusion overlay must modify at least some rendered frames."""
    ds_occ = SyntheticDataset(
        num_sequences=1, num_frames=60, motion="occlusion", seed=7,
    )
    ds_ref = SyntheticDataset(
        num_sequences=1, num_frames=60, motion="random", seed=7,
    )
    occ_frames = list(ds_occ[0])
    ref_frames = list(ds_ref[0])
    # At least one frame should look different due to the gray patch.
    diffs = [
        not np.array_equal(f_occ, f_ref)
        for f_occ, f_ref in zip(occ_frames, ref_frames)
    ]
    assert any(diffs), "Occlusion patches should modify at least some frames"


def test_invalid_motion_raises():
    with pytest.raises(ValueError, match="Unknown motion pattern"):
        SyntheticDataset(motion="teleport")  # type: ignore[arg-type]


@pytest.mark.parametrize("motion", SyntheticDataset._VALID_MOTIONS)
def test_all_motions_work_end_to_end(motion):
    """All five motion modes must produce valid sequences without error."""
    ds = SyntheticDataset(num_sequences=2, num_frames=15, motion=motion, seed=42)
    for seq in ds:
        assert seq.ground_truth.shape[1] == 4
        frames = list(seq)
        assert len(frames) == 15


def test_scale_change_scale_range():
    """Scale factors should cover the expected [0.25, 1.75] range across frames."""
    ds = SyntheticDataset(
        num_sequences=1, num_frames=200, bbox_size=(40, 40),
        motion="scale_change", seed=0,
    )
    gt = ds[0].ground_truth
    widths = gt[:, 2]
    # At scale 0.25 -> width ~= 10; at scale 1.75 -> width ~= 70.
    assert widths.min() < 20, f"Expected small widths near min scale; got min={widths.min()}"
    assert widths.max() > 50, f"Expected large widths near max scale; got max={widths.max()}"


def test_occlusion_period_count():
    """At least one occlusion burst must appear in a 100-frame sequence."""
    ds_occ = SyntheticDataset(
        num_sequences=1, num_frames=100, motion="occlusion", seed=3,
    )
    ds_ref = SyntheticDataset(
        num_sequences=1, num_frames=100, motion="random", seed=3,
    )
    occ_frames = list(ds_occ[0])
    ref_frames = list(ds_ref[0])
    n_diff = sum(
        not np.array_equal(a, b) for a, b in zip(occ_frames[1:], ref_frames[1:])
    )
    assert n_diff > 0, "Expected at least some occluded frames in 100-frame sequence"
