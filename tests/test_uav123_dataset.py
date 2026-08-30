"""Tests for UAV123Dataset.

All tests work without a real UAV123 download: a minimal synthetic fixture is
created in a temporary directory that mirrors the expected on-disk layout.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np
import pytest

from eovot.datasets.uav123 import (
    UAV123Dataset,
    SEQUENCE_ATTRIBUTES,
    _load_groundtruth,
    _discover_frames,
    _UAV20L_SEQUENCES,
)
from eovot.datasets.base import Sequence


# ---------------------------------------------------------------------------
# Fixtures — minimal on-disk UAV123 layout
# ---------------------------------------------------------------------------

_GT_ROWS = [(10, 20, 50, 60), (12, 22, 50, 60), (14, 24, 51, 61)]


def _write_gt(path: Path, rows=_GT_ROWS, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(delimiter.join(str(v) for v in row) + "\n")


def _write_frame(path: Path, w: int = 16, h: int = 16) -> None:
    """Write a tiny valid JPEG so cv2.imread succeeds."""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture()
def uav123_root(tmp_path: Path) -> Path:
    """Build a minimal UAV123 tree with two sequences (seq_a, seq_b)."""
    for seq in ("seq_a", "seq_b"):
        # Annotation
        _write_gt(tmp_path / "anno" / "UAV123" / f"{seq}.txt")
        # Frames
        for i in range(1, 4):
            _write_frame(tmp_path / "data_seq" / "UAV123" / seq / f"img{i:05d}.jpg")
    return tmp_path


@pytest.fixture()
def dataset(uav123_root: Path) -> UAV123Dataset:
    return UAV123Dataset(str(uav123_root))


# ---------------------------------------------------------------------------
# Construction and split validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_uav123_split(self, uav123_root: Path):
        ds = UAV123Dataset(str(uav123_root), split="UAV123")
        assert ds.split == "UAV123"

    def test_invalid_split_raises(self, uav123_root: Path):
        with pytest.raises(ValueError, match="split"):
            UAV123Dataset(str(uav123_root), split="OTB100")

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            UAV123Dataset("/nonexistent/path/to/uav123")

    def test_name_property(self, uav123_root: Path):
        ds = UAV123Dataset(str(uav123_root), split="UAV123")
        assert ds.name == "UAV123-UAV123"

    def test_repr(self, dataset: UAV123Dataset):
        assert "UAV123Dataset" in repr(dataset)


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------

class TestSequenceDiscovery:
    def test_len_matches_annotation_files(self, dataset: UAV123Dataset):
        assert len(dataset) == 2

    def test_list_sequences_sorted(self, dataset: UAV123Dataset):
        names = dataset.list_sequences()
        assert names == sorted(names)

    def test_list_sequences_cached(self, dataset: UAV123Dataset):
        n1 = dataset.list_sequences()
        n2 = dataset.list_sequences()
        assert n1 is n2

    def test_max_sequences_limit(self, uav123_root: Path):
        ds = UAV123Dataset(str(uav123_root), max_sequences=1)
        assert len(ds) == 1

    def test_missing_anno_dir_raises(self, tmp_path: Path):
        # Root exists but anno/UAV123 does not
        (tmp_path / "data_seq" / "UAV123").mkdir(parents=True, exist_ok=True)
        ds = UAV123Dataset(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Annotation directory"):
            ds.list_sequences()


# ---------------------------------------------------------------------------
# Sequence loading
# ---------------------------------------------------------------------------

class TestSequenceLoading:
    def test_getitem_returns_sequence(self, dataset: UAV123Dataset):
        seq = dataset[0]
        assert isinstance(seq, Sequence)

    def test_getitem_out_of_range(self, dataset: UAV123Dataset):
        with pytest.raises(IndexError):
            _ = dataset[100]

    def test_sequence_length_matches_gt(self, dataset: UAV123Dataset):
        seq = dataset[0]
        assert len(seq) == len(_GT_ROWS)
        assert seq.ground_truth.shape == (len(_GT_ROWS), 4)

    def test_ground_truth_values(self, dataset: UAV123Dataset):
        seq = dataset[0]
        expected = np.array(_GT_ROWS, dtype=np.float64)
        np.testing.assert_array_equal(seq.ground_truth, expected)

    def test_init_bbox(self, dataset: UAV123Dataset):
        seq = dataset[0]
        assert seq.init_bbox == (10.0, 20.0, 50.0, 60.0)

    def test_iter_yields_all_sequences(self, dataset: UAV123Dataset):
        seqs = list(dataset)
        assert len(seqs) == 2

    def test_missing_sequence_raises(self, uav123_root: Path):
        ds = UAV123Dataset(str(uav123_root))
        ds._seq_names = ["nonexistent_seq"]
        with pytest.raises(FileNotFoundError):
            ds[0]


# ---------------------------------------------------------------------------
# Ground-truth format tolerance
# ---------------------------------------------------------------------------

class TestGroundTruthParsing:
    def test_comma_delimited(self, tmp_path: Path):
        gt = tmp_path / "gt.txt"
        gt.write_text("10,20,50,60\n12,22,50,60\n")
        boxes = _load_groundtruth(gt)
        assert len(boxes) == 2
        assert boxes[0] == (10.0, 20.0, 50.0, 60.0)

    def test_space_delimited(self, tmp_path: Path):
        gt = tmp_path / "gt.txt"
        gt.write_text("10 20 50 60\n12 22 50 60\n")
        boxes = _load_groundtruth(gt)
        assert len(boxes) == 2

    def test_tab_delimited(self, tmp_path: Path):
        gt = tmp_path / "gt.txt"
        gt.write_text("10\t20\t50\t60\n12\t22\t50\t60\n")
        boxes = _load_groundtruth(gt)
        assert len(boxes) == 2

    def test_blank_lines_skipped(self, tmp_path: Path):
        gt = tmp_path / "gt.txt"
        gt.write_text("\n10,20,50,60\n\n12,22,50,60\n\n")
        boxes = _load_groundtruth(gt)
        assert len(boxes) == 2

    def test_incomplete_lines_skipped(self, tmp_path: Path):
        gt = tmp_path / "gt.txt"
        gt.write_text("10,20,50,60\n10,20\n12,22,50,60\n")
        boxes = _load_groundtruth(gt)
        assert len(boxes) == 2

    def test_float_values_preserved(self, tmp_path: Path):
        gt = tmp_path / "gt.txt"
        gt.write_text("10.5,20.3,50.1,60.9\n")
        boxes = _load_groundtruth(gt)
        assert boxes[0][0] == pytest.approx(10.5)

    def test_returns_list_of_tuples(self, tmp_path: Path):
        gt = tmp_path / "gt.txt"
        gt.write_text("1,2,3,4\n")
        boxes = _load_groundtruth(gt)
        assert isinstance(boxes, list)
        assert len(boxes[0]) == 4


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------

class TestFrameDiscovery:
    def test_finds_jpg_frames(self, tmp_path: Path):
        for i in range(3):
            (tmp_path / f"img{i:05d}.jpg").touch()
        paths = _discover_frames(tmp_path)
        assert len(paths) == 3

    def test_finds_png_frames(self, tmp_path: Path):
        for i in range(2):
            (tmp_path / f"img{i:05d}.png").touch()
        paths = _discover_frames(tmp_path)
        assert len(paths) == 2

    def test_sorted_order(self, tmp_path: Path):
        for i in (5, 1, 3):
            (tmp_path / f"img{i:05d}.jpg").touch()
        paths = _discover_frames(tmp_path)
        assert paths == sorted(paths)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path):
        paths = _discover_frames(tmp_path)
        assert paths == []

    def test_no_duplicates(self, tmp_path: Path):
        (tmp_path / "img00001.jpg").touch()
        (tmp_path / "img00001.JPG").touch()
        paths = _discover_frames(tmp_path)
        assert len(paths) == len(set(paths))


# ---------------------------------------------------------------------------
# Attribute filtering
# ---------------------------------------------------------------------------

class TestAttributeFiltering:
    def test_attributes_property_non_empty(self, dataset: UAV123Dataset):
        attrs = dataset.attributes
        assert len(attrs) > 0
        assert "fast_motion" in attrs
        assert "partial_occlusion" in attrs

    def test_attributes_sorted(self, dataset: UAV123Dataset):
        attrs = dataset.attributes
        assert attrs == sorted(attrs)

    def test_filter_by_unknown_attribute_raises(self, dataset: UAV123Dataset):
        with pytest.raises(ValueError, match="Unknown attribute"):
            dataset.filter_by_attribute("not_a_real_attribute")

    def test_filter_empty_args_raises(self, dataset: UAV123Dataset):
        with pytest.raises(ValueError, match="at least one attribute"):
            dataset.filter_by_attribute()

    def test_filter_by_attribute_returns_dataset(self, dataset: UAV123Dataset):
        view = dataset.filter_by_attribute("aerial_view")
        assert isinstance(view, UAV123Dataset)

    def test_filter_result_is_subset_of_known_sequences(self, dataset: UAV123Dataset):
        # Our fixture sequences (seq_a, seq_b) are not in the attribute map,
        # so the view for a specific attribute should have 0 or a proper subset.
        view = dataset.filter_by_attribute("fast_motion")
        assert len(view) <= len(dataset)

    def test_sequence_attributes_for_bike1(self, dataset: UAV123Dataset):
        attrs = dataset.sequence_attributes("bike1")
        assert "aerial_view" in attrs
        assert "fast_motion" in attrs

    def test_sequence_attributes_unknown_sequence(self, dataset: UAV123Dataset):
        attrs = dataset.sequence_attributes("nonexistent_seq_xyz")
        assert attrs == []

    def test_filter_view_has_correct_split(self, dataset: UAV123Dataset):
        view = dataset.filter_by_attribute("fast_motion")
        assert view.split == dataset.split


# ---------------------------------------------------------------------------
# Attribute catalogue consistency
# ---------------------------------------------------------------------------

class TestAttributeCatalogue:
    def test_all_attribute_values_are_sets(self):
        for attr, seq_set in SEQUENCE_ATTRIBUTES.items():
            assert isinstance(seq_set, set), f"{attr} is not a set"

    def test_uav20l_sequences_are_known_uav123_names(self):
        # Every long sequence should appear in the aerial_view attribute
        # (all UAV123 sequences are aerial by definition)
        for seq in _UAV20L_SEQUENCES:
            assert seq in SEQUENCE_ATTRIBUTES["aerial_view"], (
                f"UAV20L sequence '{seq}' missing from aerial_view attribute"
            )

    def test_no_empty_attribute_sets(self):
        for attr, seq_set in SEQUENCE_ATTRIBUTES.items():
            assert len(seq_set) > 0, f"Attribute '{attr}' has an empty sequence set"
