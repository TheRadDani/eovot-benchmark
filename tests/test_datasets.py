"""Unit tests for eovot.datasets (Sequence, OTBDataset, GOT10kDataset)."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from eovot.datasets.base import OTBDataset, Sequence
from eovot.datasets.got10k import GOT10kDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_frame(path: str, h: int = 64, w: int = 64) -> None:
    """Write a small random BGR image to *path*."""
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(path, img)


def _make_otb_sequence(root: str, name: str, n_frames: int = 5) -> str:
    """Create a minimal OTB-style sequence directory and return its path."""
    seq_dir = os.path.join(root, name)
    img_dir = os.path.join(seq_dir, "img")
    os.makedirs(img_dir)

    for i in range(1, n_frames + 1):
        _make_fake_frame(os.path.join(img_dir, f"{i:04d}.jpg"))

    gt_path = os.path.join(seq_dir, "groundtruth_rect.txt")
    boxes = np.random.uniform(1, 30, (n_frames, 4))
    np.savetxt(gt_path, boxes, delimiter=",", fmt="%.1f")
    return seq_dir


def _make_got10k_sequence(split_dir: str, name: str, n_frames: int = 5) -> str:
    """Create a minimal GOT-10k-style sequence directory."""
    seq_dir = os.path.join(split_dir, name)
    img_dir = os.path.join(seq_dir, "img")
    os.makedirs(img_dir)

    for i in range(1, n_frames + 1):
        _make_fake_frame(os.path.join(img_dir, f"{i:08d}.jpg"))

    gt_path = os.path.join(seq_dir, "groundtruth.txt")
    boxes = np.random.uniform(1, 30, (n_frames, 4))
    with open(gt_path, "w") as fh:
        for box in boxes:
            fh.write(",".join(f"{v:.2f}" for v in box) + "\n")
    return seq_dir


# ---------------------------------------------------------------------------
# Sequence tests
# ---------------------------------------------------------------------------

class TestSequence:
    def test_init_and_len(self, tmp_path):
        img_path = str(tmp_path / "frame.jpg")
        _make_fake_frame(img_path)
        gt = np.array([[1.0, 2.0, 10.0, 15.0]])
        seq = Sequence(name="test", frame_paths=[img_path], ground_truth=gt)
        assert len(seq) == 1
        assert seq.name == "test"

    def test_init_bbox(self, tmp_path):
        img_path = str(tmp_path / "frame.jpg")
        _make_fake_frame(img_path)
        gt = np.array([[5.0, 10.0, 20.0, 30.0]])
        seq = Sequence(name="s", frame_paths=[img_path], ground_truth=gt)
        assert seq.init_bbox == (5.0, 10.0, 20.0, 30.0)

    def test_iteration_yields_bgr_frames(self, tmp_path):
        paths = []
        for i in range(3):
            p = str(tmp_path / f"{i}.jpg")
            _make_fake_frame(p)
            paths.append(p)
        gt = np.ones((3, 4))
        seq = Sequence(name="s", frame_paths=paths, ground_truth=gt)
        frames = list(seq)
        assert len(frames) == 3
        for frame in frames:
            assert frame.shape == (64, 64, 3)

    def test_invalid_ground_truth_shape(self, tmp_path):
        img_path = str(tmp_path / "frame.jpg")
        _make_fake_frame(img_path)
        bad_gt = np.array([1.0, 2.0, 3.0])  # 1-D, not (N,4)
        with pytest.raises(ValueError):
            Sequence(name="s", frame_paths=[img_path], ground_truth=bad_gt)

    def test_missing_frame_raises(self, tmp_path):
        gt = np.ones((1, 4))
        seq = Sequence(name="s", frame_paths=["/nonexistent/frame.jpg"], ground_truth=gt)
        with pytest.raises(FileNotFoundError):
            list(seq)

    def test_repr(self, tmp_path):
        img_path = str(tmp_path / "frame.jpg")
        _make_fake_frame(img_path)
        gt = np.ones((1, 4))
        seq = Sequence(name="myseq", frame_paths=[img_path], ground_truth=gt)
        assert "myseq" in repr(seq)


# ---------------------------------------------------------------------------
# OTBDataset tests
# ---------------------------------------------------------------------------

class TestOTBDataset:
    def test_len_and_getitem(self, tmp_path):
        root = str(tmp_path)
        _make_otb_sequence(root, "seq_a", n_frames=4)
        _make_otb_sequence(root, "seq_b", n_frames=6)
        dataset = OTBDataset(root)
        assert len(dataset) == 2
        seq = dataset[0]
        assert isinstance(seq, Sequence)

    def test_iteration(self, tmp_path):
        root = str(tmp_path)
        for name in ["alpha", "beta", "gamma"]:
            _make_otb_sequence(root, name, n_frames=3)
        dataset = OTBDataset(root)
        names = [seq.name for seq in dataset]
        assert set(names) == {"alpha", "beta", "gamma"}

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            OTBDataset("/totally/nonexistent/path")

    def test_ground_truth_shape(self, tmp_path):
        root = str(tmp_path)
        _make_otb_sequence(root, "seq", n_frames=5)
        dataset = OTBDataset(root)
        seq = dataset[0]
        assert seq.ground_truth.shape == (5, 4)


# ---------------------------------------------------------------------------
# GOT10kDataset tests
# ---------------------------------------------------------------------------

class TestGOT10kDataset:
    def _make_dataset(self, tmp_path, n_seqs: int = 3, n_frames: int = 5):
        root = str(tmp_path)
        split_dir = os.path.join(root, "val")
        os.makedirs(split_dir)
        list_file = os.path.join(split_dir, "list.txt")
        names = [f"GOT-10k_Val_{i:06d}" for i in range(1, n_seqs + 1)]
        with open(list_file, "w") as fh:
            fh.write("\n".join(names) + "\n")
        for name in names:
            _make_got10k_sequence(split_dir, name, n_frames=n_frames)
        return root, names

    def test_len(self, tmp_path):
        root, _ = self._make_dataset(tmp_path, n_seqs=4)
        dataset = GOT10kDataset(root, split="val")
        assert len(dataset) == 4

    def test_max_sequences(self, tmp_path):
        root, _ = self._make_dataset(tmp_path, n_seqs=6)
        dataset = GOT10kDataset(root, split="val", max_sequences=3)
        assert len(dataset) == 3

    def test_getitem_returns_sequence(self, tmp_path):
        root, _ = self._make_dataset(tmp_path)
        dataset = GOT10kDataset(root, split="val")
        seq = dataset[0]
        assert isinstance(seq, Sequence)
        assert seq.ground_truth.ndim == 2
        assert seq.ground_truth.shape[1] == 4

    def test_getitem_out_of_range(self, tmp_path):
        root, _ = self._make_dataset(tmp_path, n_seqs=2)
        dataset = GOT10kDataset(root, split="val")
        with pytest.raises(IndexError):
            dataset[99]

    def test_invalid_split_raises(self, tmp_path):
        with pytest.raises(ValueError):
            GOT10kDataset(str(tmp_path), split="invalid")

    def test_iteration(self, tmp_path):
        root, names = self._make_dataset(tmp_path, n_seqs=3)
        dataset = GOT10kDataset(root, split="val")
        seqs = list(dataset)
        assert len(seqs) == 3
        for seq in seqs:
            assert isinstance(seq, Sequence)

    def test_name_property(self, tmp_path):
        root, _ = self._make_dataset(tmp_path)
        dataset = GOT10kDataset(root, split="val")
        assert dataset.name == "GOT-10k-val"

    def test_list_sequences_cached(self, tmp_path):
        root, _ = self._make_dataset(tmp_path, n_seqs=2)
        dataset = GOT10kDataset(root, split="val")
        first = dataset.list_sequences()
        second = dataset.list_sequences()
        assert first is second  # same list object (cached)

    def test_gt_whitespace_delimited(self, tmp_path):
        """GOT-10k GT files sometimes use spaces instead of commas."""
        root = str(tmp_path)
        split_dir = os.path.join(root, "val")
        seq_dir = os.path.join(split_dir, "seq1")
        img_dir = os.path.join(seq_dir, "img")
        os.makedirs(img_dir)
        _make_fake_frame(os.path.join(img_dir, "00000001.jpg"))
        with open(os.path.join(seq_dir, "groundtruth.txt"), "w") as fh:
            fh.write("10 20 30 40\n")

        dataset = GOT10kDataset(root, split="val")
        seq = dataset.load_sequence("seq1")
        np.testing.assert_allclose(seq.ground_truth[0], [10, 20, 30, 40])
