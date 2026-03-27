"""Unit tests for eovot.datasets.base (Sequence, OTBDataset)."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from eovot.datasets.base import OTBDataset, Sequence


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

class TestSequence:
    def _make_seq(self, tmp_path, n_frames: int = 4):
        """Helper: write n_frames to tmp_path and return (Sequence, gt)."""
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        paths = []
        for i in range(n_frames):
            p = str(img_dir / f"{i:04d}.jpg")
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame[5:15, 5:15] = i * 20  # distinct per frame
            cv2.imwrite(p, frame)
            paths.append(p)
        gt = np.array([[5.0, 5.0, 10.0, 10.0]] * n_frames)
        return Sequence(name="test_seq", frame_paths=paths, ground_truth=gt), gt

    def test_len(self, tmp_path):
        seq, _ = self._make_seq(tmp_path)
        assert len(seq) == 4

    def test_repr(self, tmp_path):
        seq, _ = self._make_seq(tmp_path)
        r = repr(seq)
        assert "test_seq" in r
        assert "4" in r

    def test_init_bbox(self, tmp_path):
        seq, gt = self._make_seq(tmp_path)
        assert seq.init_bbox == pytest.approx(tuple(gt[0]))

    def test_iteration_yields_frames(self, tmp_path):
        seq, _ = self._make_seq(tmp_path, n_frames=3)
        frames = list(seq)
        assert len(frames) == 3
        for f in frames:
            assert isinstance(f, np.ndarray)
            assert f.shape == (32, 32, 3)

    def test_wrong_gt_shape_raises(self, tmp_path):
        img_dir = tmp_path / "imgs2"
        img_dir.mkdir()
        p = str(img_dir / "0000.jpg")
        cv2.imwrite(p, np.zeros((32, 32, 3), dtype=np.uint8))
        bad_gt = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3) — should be (N, 4)
        with pytest.raises(ValueError, match="ground_truth must be shape"):
            Sequence("bad", [p], bad_gt)

    def test_missing_frame_raises(self, tmp_path):
        img_dir = tmp_path / "imgs3"
        img_dir.mkdir()
        gt = np.array([[0.0, 0.0, 10.0, 10.0]])
        seq = Sequence("missing", [str(img_dir / "ghost.jpg")], gt)
        with pytest.raises(FileNotFoundError):
            list(seq)


# ---------------------------------------------------------------------------
# OTBDataset
# ---------------------------------------------------------------------------

class TestOTBDataset:
    def test_len(self, synthetic_otb_root):
        ds = OTBDataset(synthetic_otb_root)
        assert len(ds) == 2

    def test_getitem_returns_sequence(self, synthetic_otb_root):
        ds = OTBDataset(synthetic_otb_root)
        seq = ds[0]
        assert isinstance(seq, Sequence)

    def test_getitem_sequence_len(self, synthetic_otb_root):
        ds = OTBDataset(synthetic_otb_root)
        assert len(ds[0]) == 5

    def test_gt_shape(self, synthetic_otb_root):
        ds = OTBDataset(synthetic_otb_root)
        seq = ds[0]
        assert seq.ground_truth.shape == (5, 4)

    def test_iter_yields_all_sequences(self, synthetic_otb_root):
        ds = OTBDataset(synthetic_otb_root)
        seqs = list(ds)
        assert len(seqs) == 2
        assert all(isinstance(s, Sequence) for s in seqs)

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            OTBDataset("/nonexistent/path/dataset")

    def test_repr(self, synthetic_otb_root):
        ds = OTBDataset(synthetic_otb_root)
        r = repr(ds)
        assert "OTBDataset" in r
        assert "2" in r

    def test_sequences_have_distinct_names(self, synthetic_otb_root):
        ds = OTBDataset(synthetic_otb_root)
        names = [ds[i].name for i in range(len(ds))]
        assert len(set(names)) == len(names)

    def test_frames_loadable(self, synthetic_otb_root):
        ds = OTBDataset(synthetic_otb_root)
        for frame in ds[0]:
            assert frame is not None
            assert frame.dtype == np.uint8
