"""Unit tests for eovot.datasets.base (OTBDataset) and got10k (GOT10kDataset)."""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from eovot.datasets.base import OTBDataset, Sequence


# ---------------------------------------------------------------------------
# Fixtures — synthetic dataset directories
# ---------------------------------------------------------------------------

def _create_otb_sequence(root: Path, seq_name: str, n_frames: int = 5):
    """Create a minimal OTB-style sequence with synthetic JPEG frames."""
    seq_dir = root / seq_name
    img_dir = seq_dir / "img"
    img_dir.mkdir(parents=True)

    # Write ground truth (comma-delimited)
    gt_lines = [f"{10 + i},{10},{50},{40}\n" for i in range(n_frames)]
    (seq_dir / "groundtruth_rect.txt").write_text("".join(gt_lines))

    # Write tiny synthetic JPEG frames
    for i in range(n_frames):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[:, :, i % 3] = 128  # add some colour variation
        cv2.imwrite(str(img_dir / f"{i + 1:04d}.jpg"), frame)


@pytest.fixture()
def otb_root(tmp_path):
    """Return path to a synthetic OTB-style dataset with 3 sequences."""
    for name in ["Basketball", "BlurBody", "Car4"]:
        _create_otb_sequence(tmp_path, name, n_frames=4)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Sequence tests
# ---------------------------------------------------------------------------

class TestSequence:
    def test_len(self, tmp_path):
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for i in range(3):
            cv2.imwrite(str(img_dir / f"{i}.jpg"), frame)
        paths = sorted(str(p) for p in img_dir.glob("*.jpg"))
        gt = np.array([[0, 0, 5, 5]] * 3, dtype=float)
        seq = Sequence(name="test", frame_paths=paths, ground_truth=gt)
        assert len(seq) == 3

    def test_init_bbox(self, tmp_path):
        paths = [str(tmp_path / "f.jpg")]
        cv2.imwrite(paths[0], np.zeros((10, 10, 3), dtype=np.uint8))
        gt = np.array([[5.0, 10.0, 20.0, 30.0]], dtype=float)
        seq = Sequence(name="s", frame_paths=paths, ground_truth=gt)
        assert seq.init_bbox == pytest.approx((5.0, 10.0, 20.0, 30.0))

    def test_repr(self, tmp_path):
        paths = [str(tmp_path / "f.jpg")]
        cv2.imwrite(paths[0], np.zeros((10, 10, 3), dtype=np.uint8))
        gt = np.array([[0, 0, 5, 5]], dtype=float)
        seq = Sequence(name="demo", frame_paths=paths, ground_truth=gt)
        assert "demo" in repr(seq)

    def test_invalid_gt_shape_raises(self, tmp_path):
        paths = [str(tmp_path / "f.jpg")]
        cv2.imwrite(paths[0], np.zeros((10, 10, 3), dtype=np.uint8))
        bad_gt = np.array([[0, 0, 5]], dtype=float)  # only 3 columns
        with pytest.raises(ValueError):
            Sequence(name="s", frame_paths=paths, ground_truth=bad_gt)

    def test_iterate_loads_frames(self, tmp_path):
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        for i in range(3):
            cv2.imwrite(str(img_dir / f"{i}.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
        paths = sorted(str(p) for p in img_dir.glob("*.jpg"))
        gt = np.array([[0, 0, 5, 5]] * 3, dtype=float)
        seq = Sequence(name="s", frame_paths=paths, ground_truth=gt)
        frames = list(seq)
        assert len(frames) == 3
        for f in frames:
            assert f.shape == (10, 10, 3)


# ---------------------------------------------------------------------------
# OTBDataset tests
# ---------------------------------------------------------------------------

class TestOTBDataset:
    def test_len(self, otb_root):
        ds = OTBDataset(otb_root)
        assert len(ds) == 3

    def test_getitem_returns_sequence(self, otb_root):
        ds = OTBDataset(otb_root)
        seq = ds[0]
        assert isinstance(seq, Sequence)

    def test_sequence_names_sorted(self, otb_root):
        ds = OTBDataset(otb_root)
        names = [ds[i].name for i in range(len(ds))]
        assert names == sorted(names)

    def test_ground_truth_shape(self, otb_root):
        ds = OTBDataset(otb_root)
        seq = ds[0]
        assert seq.ground_truth.ndim == 2
        assert seq.ground_truth.shape[1] == 4

    def test_frame_count_matches_gt(self, otb_root):
        ds = OTBDataset(otb_root)
        for seq in ds:
            assert len(seq) == len(seq.ground_truth)

    def test_iter_protocol(self, otb_root):
        ds = OTBDataset(otb_root)
        seqs = list(ds)
        assert len(seqs) == 3

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            OTBDataset(str(tmp_path / "nonexistent"))

    def test_repr(self, otb_root):
        ds = OTBDataset(otb_root)
        assert "OTBDataset" in repr(ds)

    def test_whitespace_delimited_gt(self, tmp_path):
        """OTBDataset should handle whitespace-delimited groundtruth files."""
        seq_dir = tmp_path / "TestSeq"
        img_dir = seq_dir / "img"
        img_dir.mkdir(parents=True)
        cv2.imwrite(str(img_dir / "0001.jpg"), np.zeros((30, 40, 3), dtype=np.uint8))
        # whitespace-delimited ground truth
        (seq_dir / "groundtruth_rect.txt").write_text("5 10 20 30\n")
        ds = OTBDataset(str(tmp_path))
        seq = ds[0]
        np.testing.assert_array_equal(seq.ground_truth[0], [5, 10, 20, 30])
