"""Unit tests for GOT10kDataset.

Creates a minimal synthetic GOT-10k directory layout in a temporary
directory and verifies that:

- ``__len__`` returns the correct number of sequences.
- ``__getitem__`` returns a properly formed :class:`~eovot.datasets.base.Sequence`.
- ``__iter__`` yields sequences in order.
- ``max_sequences`` cap is respected.
- ``list_sequences`` reads ``list.txt`` when present.
- ``list_sequences`` falls back to directory enumeration.
- Ground-truth is parsed correctly for both comma and space delimiters.
- A missing ``groundtruth.txt`` raises :exc:`FileNotFoundError`.
- A missing ``img/`` directory raises :exc:`FileNotFoundError`.
- ``IndexError`` is raised for out-of-range indices.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.base import Sequence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fake_frame(path: str, width: int = 64, height: int = 64) -> None:
    """Write a random BGR JPEG to *path*."""
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(path, img)


def _make_sequence_dir(
    split_dir: Path,
    seq_name: str,
    n_frames: int = 5,
    gt_delimiter: str = ",",
) -> Path:
    """Create a minimal GOT-10k sequence directory."""
    seq_dir = split_dir / seq_name
    img_dir = seq_dir / "img"
    img_dir.mkdir(parents=True)

    # Write synthetic frames
    for i in range(1, n_frames + 1):
        _make_fake_frame(str(img_dir / f"{i:08d}.jpg"))

    # Write ground-truth file
    gt_path = seq_dir / "groundtruth.txt"
    with open(gt_path, "w") as fh:
        for i in range(n_frames):
            x, y, w, h = 10 + i, 20, 30, 40
            if gt_delimiter == ",":
                fh.write(f"{x},{y},{w},{h}\n")
            else:
                fh.write(f"{x} {y} {w} {h}\n")

    return seq_dir


@pytest.fixture
def fake_got10k_root(tmp_path: Path):
    """Return (root_str, split_dir, seq_names) for a synthetic GOT-10k val split."""
    split_dir = tmp_path / "val"
    split_dir.mkdir()

    seq_names = ["GOT-10k_Val_000001", "GOT-10k_Val_000002", "GOT-10k_Val_000003"]
    for name in seq_names:
        _make_sequence_dir(split_dir, name, n_frames=5)

    return str(tmp_path), split_dir, seq_names


# ---------------------------------------------------------------------------
# Basic interface tests
# ---------------------------------------------------------------------------

class TestGOT10kDatasetLen:
    def test_len_all(self, fake_got10k_root):
        root, _, seq_names = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        assert len(ds) == len(seq_names)

    def test_len_with_max_sequences(self, fake_got10k_root):
        root, _, _ = fake_got10k_root
        ds = GOT10kDataset(root, split="val", max_sequences=2)
        assert len(ds) == 2

    def test_len_max_sequences_larger_than_available(self, fake_got10k_root):
        root, _, seq_names = fake_got10k_root
        ds = GOT10kDataset(root, split="val", max_sequences=100)
        assert len(ds) == len(seq_names)


class TestGOT10kDatasetGetItem:
    def test_getitem_returns_sequence(self, fake_got10k_root):
        root, _, seq_names = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        seq = ds[0]
        assert isinstance(seq, Sequence)
        assert seq.name == seq_names[0]

    def test_getitem_sequence_length(self, fake_got10k_root):
        root, _, _ = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        seq = ds[0]
        assert len(seq) == 5  # 5 frames per synthetic sequence

    def test_getitem_ground_truth_shape(self, fake_got10k_root):
        root, _, _ = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        seq = ds[0]
        assert seq.ground_truth.ndim == 2
        assert seq.ground_truth.shape[1] == 4

    def test_getitem_ground_truth_values(self, fake_got10k_root):
        root, _, _ = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        seq = ds[0]
        # First box: x=10, y=20, w=30, h=40
        np.testing.assert_array_equal(seq.ground_truth[0], [10, 20, 30, 40])

    def test_getitem_index_error(self, fake_got10k_root):
        root, _, seq_names = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        with pytest.raises(IndexError):
            _ = ds[len(seq_names)]

    def test_getitem_all_indices(self, fake_got10k_root):
        root, _, seq_names = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        names = [ds[i].name for i in range(len(ds))]
        assert names == seq_names


class TestGOT10kDatasetIter:
    def test_iter_yields_sequences(self, fake_got10k_root):
        root, _, seq_names = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        seqs = list(ds)
        assert len(seqs) == len(seq_names)
        assert all(isinstance(s, Sequence) for s in seqs)

    def test_iter_order_matches_list_sequences(self, fake_got10k_root):
        root, _, _ = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        iter_names = [s.name for s in ds]
        assert iter_names == ds.list_sequences()


# ---------------------------------------------------------------------------
# Sequence content tests
# ---------------------------------------------------------------------------

class TestSequenceContent:
    def test_frames_loadable(self, fake_got10k_root):
        """All frames returned by the iterator should be valid BGR images."""
        root, _, _ = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        seq = ds[0]
        frames = list(seq)
        assert len(frames) == 5
        for f in frames:
            assert isinstance(f, np.ndarray)
            assert f.ndim == 3
            assert f.shape[2] == 3

    def test_init_bbox_matches_gt(self, fake_got10k_root):
        root, _, _ = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        seq = ds[0]
        assert seq.init_bbox == tuple(seq.ground_truth[0])


# ---------------------------------------------------------------------------
# list_sequences and list.txt
# ---------------------------------------------------------------------------

class TestListSequences:
    def test_list_sequences_without_list_txt(self, fake_got10k_root):
        root, _, seq_names = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        assert ds.list_sequences() == seq_names

    def test_list_sequences_with_list_txt(self, fake_got10k_root):
        root, split_dir, seq_names = fake_got10k_root
        # Write a list.txt with only the first two sequences
        list_txt = split_dir / "list.txt"
        list_txt.write_text("\n".join(seq_names[:2]) + "\n")
        ds = GOT10kDataset(root, split="val")
        assert ds.list_sequences() == seq_names[:2]

    def test_list_sequences_cached(self, fake_got10k_root):
        root, _, _ = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        names1 = ds.list_sequences()
        names2 = ds.list_sequences()
        assert names1 is names2  # same object — cached


# ---------------------------------------------------------------------------
# Ground-truth delimiter tests
# ---------------------------------------------------------------------------

class TestGroundTruthParsing:
    def test_space_delimited_gt(self, tmp_path):
        split_dir = tmp_path / "val"
        split_dir.mkdir()
        _make_sequence_dir(split_dir, "seq01", n_frames=3, gt_delimiter=" ")
        ds = GOT10kDataset(str(tmp_path), split="val")
        seq = ds[0]
        assert seq.ground_truth.shape == (3, 4)
        np.testing.assert_array_equal(seq.ground_truth[0], [10, 20, 30, 40])

    def test_comma_delimited_gt(self, tmp_path):
        split_dir = tmp_path / "val"
        split_dir.mkdir()
        _make_sequence_dir(split_dir, "seq01", n_frames=3, gt_delimiter=",")
        ds = GOT10kDataset(str(tmp_path), split="val")
        seq = ds[0]
        assert seq.ground_truth.shape == (3, 4)
        np.testing.assert_array_equal(seq.ground_truth[0], [10, 20, 30, 40])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_invalid_split_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="split must be one of"):
            GOT10kDataset(str(tmp_path), split="invalid")

    def test_missing_groundtruth_raises(self, tmp_path):
        split_dir = tmp_path / "val"
        seq_dir = split_dir / "seq01"
        img_dir = seq_dir / "img"
        img_dir.mkdir(parents=True)
        _make_fake_frame(str(img_dir / "00000001.jpg"))
        # No groundtruth.txt written
        ds = GOT10kDataset(str(tmp_path), split="val")
        with pytest.raises(FileNotFoundError, match="groundtruth.txt"):
            ds.load_sequence("seq01")

    def test_missing_img_dir_raises(self, tmp_path):
        split_dir = tmp_path / "val"
        seq_dir = split_dir / "seq01"
        seq_dir.mkdir(parents=True)
        gt_path = seq_dir / "groundtruth.txt"
        gt_path.write_text("10,20,30,40\n")
        # No img/ directory
        ds = GOT10kDataset(str(tmp_path), split="val")
        with pytest.raises(FileNotFoundError, match="img/"):
            ds.load_sequence("seq01")

    def test_name_property(self, fake_got10k_root):
        root, _, _ = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        assert ds.name == "GOT-10k-val"

    def test_repr(self, fake_got10k_root):
        root, _, seq_names = fake_got10k_root
        ds = GOT10kDataset(root, split="val")
        r = repr(ds)
        assert str(len(seq_names)) in r
