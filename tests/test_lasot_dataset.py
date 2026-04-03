"""Unit tests for LaSOTDataset using a temporary synthetic dataset tree."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from eovot.datasets.lasot import LaSOTDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_groundtruth(gt_file: Path, n_frames: int) -> None:
    with open(gt_file, "w") as fh:
        for i in range(n_frames):
            fh.write(f"{10 + i},{10 + i},50,50\n")


def _write_image(img_path: Path) -> None:
    """Write a tiny valid JPEG-like file (just needs to exist for path tests)."""
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)  # minimal JPEG header


def _make_sequence(seq_dir: Path, n_frames: int = 5) -> None:
    img_dir = seq_dir / "img"
    img_dir.mkdir(parents=True)
    for i in range(1, n_frames + 1):
        _write_image(img_dir / f"{i:08d}.jpg")
    _write_groundtruth(seq_dir / "groundtruth.txt", n_frames)


@pytest.fixture()
def lasot_root(tmp_path: Path) -> Path:
    """Synthetic LaSOT root with 2 categories × 3 sequences each (6 total)."""
    for cat in ("airplane", "basketball"):
        for seq_idx in range(1, 4):
            seq_name = f"{cat}-{seq_idx}"
            _make_sequence(tmp_path / cat / seq_name, n_frames=8)
    return tmp_path


@pytest.fixture()
def lasot_root_with_split(lasot_root: Path) -> Path:
    """Same root but with a testing_set.txt marking 2 sequences as test."""
    testing = lasot_root / "testing_set.txt"
    testing.write_text("airplane-1\nbasketball-1\n")
    return lasot_root


# ---------------------------------------------------------------------------
# Tests — no split file
# ---------------------------------------------------------------------------

class TestLaSOTDatasetNoSplitFile:
    def test_len_all(self, lasot_root):
        ds = LaSOTDataset(str(lasot_root), split="all")
        assert len(ds) == 6

    def test_len_test_without_split_file(self, lasot_root):
        """Without testing_set.txt, test split falls back to all sequences."""
        ds = LaSOTDataset(str(lasot_root), split="test")
        assert len(ds) == 6

    def test_getitem_returns_sequence(self, lasot_root):
        ds = LaSOTDataset(str(lasot_root), split="all")
        seq = ds[0]
        assert seq.name is not None
        assert len(seq) == 8

    def test_getitem_out_of_range(self, lasot_root):
        ds = LaSOTDataset(str(lasot_root), split="all")
        with pytest.raises(IndexError):
            _ = ds[100]

    def test_max_sequences(self, lasot_root):
        ds = LaSOTDataset(str(lasot_root), split="all", max_sequences=3)
        assert len(ds) == 3

    def test_ground_truth_shape(self, lasot_root):
        ds = LaSOTDataset(str(lasot_root), split="all")
        seq = ds[0]
        assert seq.ground_truth.ndim == 2
        assert seq.ground_truth.shape[1] == 4

    def test_categories_property(self, lasot_root):
        ds = LaSOTDataset(str(lasot_root), split="all")
        cats = ds.categories
        assert "airplane" in cats
        assert "basketball" in cats

    def test_name_property(self, lasot_root):
        ds = LaSOTDataset(str(lasot_root), split="all")
        assert ds.name == "LaSOT-all"

    def test_iteration(self, lasot_root):
        ds = LaSOTDataset(str(lasot_root), split="all")
        names = [seq.name for seq in ds]
        assert len(names) == 6

    def test_root_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LaSOTDataset(str(tmp_path / "does_not_exist"))

    def test_invalid_split_raises(self, lasot_root):
        with pytest.raises(ValueError, match="split must be one of"):
            LaSOTDataset(str(lasot_root), split="invalid")


# ---------------------------------------------------------------------------
# Tests — with split file
# ---------------------------------------------------------------------------

class TestLaSOTDatasetWithSplitFile:
    def test_test_split_count(self, lasot_root_with_split):
        ds = LaSOTDataset(str(lasot_root_with_split), split="test")
        assert len(ds) == 2

    def test_train_split_count(self, lasot_root_with_split):
        ds = LaSOTDataset(str(lasot_root_with_split), split="train")
        assert len(ds) == 4

    def test_test_sequence_names(self, lasot_root_with_split):
        ds = LaSOTDataset(str(lasot_root_with_split), split="test")
        names = {seq.name for seq in ds}
        assert names == {"airplane-1", "basketball-1"}

    def test_train_does_not_contain_test_sequences(self, lasot_root_with_split):
        test_ds  = LaSOTDataset(str(lasot_root_with_split), split="test")
        train_ds = LaSOTDataset(str(lasot_root_with_split), split="train")
        test_names  = {seq.name for seq in test_ds}
        train_names = {seq.name for seq in train_ds}
        assert test_names.isdisjoint(train_names)

    def test_all_split_ignores_split_file(self, lasot_root_with_split):
        ds = LaSOTDataset(str(lasot_root_with_split), split="all")
        assert len(ds) == 6
