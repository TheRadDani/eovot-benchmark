"""Tests for the full-featured OTBDataset (eovot.datasets.otb)."""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from eovot.datasets.otb import (
    OTBDataset,
    OTB_ATTRIBUTES,
    OTB_ATTRIBUTE_NAMES,
    OTB100_ATTRIBUTES,
    OTB50_SEQUENCES,
)
from eovot.datasets.base import OTBDataset as OTBDatasetCompat  # backward compat re-export


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sequence(root: Path, name: str, n_frames: int = 4) -> None:
    """Write a minimal OTB-style sequence directory."""
    img_dir = root / name / "img"
    img_dir.mkdir(parents=True)
    gt_lines = [f"{10 + i},{20},{50},{40}\n" for i in range(n_frames)]
    (root / name / "groundtruth_rect.txt").write_text("".join(gt_lines))
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    for i in range(n_frames):
        cv2.imwrite(str(img_dir / f"{i + 1:04d}.jpg"), frame)


@pytest.fixture()
def otb_root(tmp_path: Path) -> str:
    """Return a synthetic OTB root with sequences matching real OTB-100 names."""
    # Use real OTB-100 names so attribute lookups work correctly.
    for name in ["Basketball", "Car4", "BlurBody", "FaceOcc1", "Jumping"]:
        _make_sequence(tmp_path, name)
    return str(tmp_path)


@pytest.fixture()
def otb_root_with_extra(tmp_path: Path) -> str:
    """Root that contains both OTB-50 and OTB-100-only sequences."""
    for name in ["Basketball", "Car4"]:          # both are in OTB-50
        _make_sequence(tmp_path, name)
    for name in ["BlurCar1", "BlurCar3"]:        # OTB-100 only
        _make_sequence(tmp_path, name)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestOTBDatasetBasic:
    def test_len(self, otb_root):
        ds = OTBDataset(otb_root)
        assert len(ds) == 5

    def test_getitem_returns_sequence(self, otb_root):
        from eovot.datasets.base import Sequence
        ds = OTBDataset(otb_root)
        assert isinstance(ds[0], Sequence)

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
        assert len(list(ds)) == 5

    def test_index_out_of_range_raises(self, otb_root):
        ds = OTBDataset(otb_root)
        with pytest.raises(IndexError):
            _ = ds[999]

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            OTBDataset(str(tmp_path / "no_such_dir"))

    def test_invalid_version_raises(self, otb_root):
        with pytest.raises(ValueError, match="version"):
            OTBDataset(otb_root, version="200")

    def test_invalid_attribute_raises(self, otb_root):
        with pytest.raises(ValueError, match="Unknown attribute"):
            OTBDataset(otb_root, attributes=["INVALID"])

    def test_whitespace_gt_parsed(self, tmp_path):
        seq_dir = tmp_path / "TestSeq"
        img_dir = seq_dir / "img"
        img_dir.mkdir(parents=True)
        cv2.imwrite(str(img_dir / "0001.jpg"), np.zeros((30, 40, 3), dtype=np.uint8))
        (seq_dir / "groundtruth_rect.txt").write_text("5 10 20 30\n")
        ds = OTBDataset(str(tmp_path))
        seq = ds[0]
        np.testing.assert_array_equal(seq.ground_truth[0], [5, 10, 20, 30])


# ---------------------------------------------------------------------------
# Version filtering (OTB-50 vs OTB-100)
# ---------------------------------------------------------------------------

class TestVersionFilter:
    def test_version_100_returns_all(self, otb_root_with_extra):
        ds = OTBDataset(otb_root_with_extra, version="100")
        # All 4 sequences should be visible.
        assert len(ds) == 4

    def test_version_50_filters_non_otb50(self, otb_root_with_extra):
        ds = OTBDataset(otb_root_with_extra, version="50")
        # Basketball and Car4 are in OTB-50; BlurCar1 and BlurCar3 are not.
        assert len(ds) == 2
        names = ds.sequence_names()
        assert "Basketball" in names
        assert "Car4" in names
        assert "BlurCar1" not in names
        assert "BlurCar3" not in names


# ---------------------------------------------------------------------------
# max_sequences cap
# ---------------------------------------------------------------------------

class TestMaxSequences:
    def test_max_sequences_limits(self, otb_root):
        ds = OTBDataset(otb_root, max_sequences=2)
        assert len(ds) == 2

    def test_max_sequences_zero(self, otb_root):
        ds = OTBDataset(otb_root, max_sequences=0)
        assert len(ds) == 0

    def test_max_sequences_larger_than_total(self, otb_root):
        ds = OTBDataset(otb_root, max_sequences=999)
        assert len(ds) == 5


# ---------------------------------------------------------------------------
# Attribute filtering
# ---------------------------------------------------------------------------

class TestAttributeFilter:
    def test_attribute_iv_filters_correctly(self, otb_root):
        # Basketball has IV; Car4 has IV; BlurBody does not; FaceOcc1 does not; Jumping does not.
        ds = OTBDataset(otb_root, attributes=["IV"])
        names = ds.sequence_names()
        assert "Basketball" in names
        assert "Car4" in names
        assert "BlurBody" not in names

    def test_multi_attribute_filter_requires_all(self, otb_root):
        # Only sequences with BOTH OCC and IV
        # Basketball: {IV, OCC, DEF, BC} → yes
        # Car4: {IV, SV} → no (no OCC)
        ds = OTBDataset(otb_root, attributes=["IV", "OCC"])
        names = ds.sequence_names()
        assert "Basketball" in names
        assert "Car4" not in names

    def test_no_match_returns_empty(self, otb_root):
        # LR not in any of our 5 fixture sequences' attribute sets
        ds = OTBDataset(otb_root, attributes=["LR"])
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# name property
# ---------------------------------------------------------------------------

class TestNameProperty:
    def test_name_v100_no_filter(self, otb_root):
        ds = OTBDataset(otb_root, version="100")
        assert ds.name == "OTB-100"

    def test_name_v50_no_filter(self, otb_root):
        ds = OTBDataset(otb_root, version="50")
        assert ds.name == "OTB-50"

    def test_name_with_single_attribute(self, otb_root):
        ds = OTBDataset(otb_root, attributes=["FM"])
        assert "FM" in ds.name
        assert "OTB-100" in ds.name

    def test_name_with_multiple_attributes_sorted(self, otb_root):
        ds = OTBDataset(otb_root, attributes=["OCC", "IV"])
        # Attributes in the name should be sorted.
        assert "IV+OCC" in ds.name


# ---------------------------------------------------------------------------
# Attribute metadata API
# ---------------------------------------------------------------------------

class TestAttributeAPI:
    def test_get_attributes_basketball(self, otb_root):
        ds = OTBDataset(otb_root)
        attrs = ds.get_attributes("Basketball")
        assert "IV" in attrs
        assert "OCC" in attrs

    def test_get_attributes_unknown_sequence_returns_empty(self, otb_root):
        ds = OTBDataset(otb_root)
        attrs = ds.get_attributes("__not_a_real_sequence__")
        assert attrs == frozenset()

    def test_attribute_map_keys_match_sequence_names(self, otb_root):
        ds = OTBDataset(otb_root)
        amap = ds.attribute_map()
        assert set(amap.keys()) == set(ds.sequence_names())

    def test_sequences_by_attribute_returns_sorted(self, otb_root):
        ds = OTBDataset(otb_root)
        seq_with_iv = ds.sequences_by_attribute("IV")
        assert seq_with_iv == sorted(seq_with_iv)

    def test_sequences_by_attribute_unknown_raises(self, otb_root):
        ds = OTBDataset(otb_root)
        with pytest.raises(ValueError, match="Unknown attribute"):
            ds.sequences_by_attribute("XYZ")

    def test_sequences_by_attribute_correct_results(self, otb_root):
        ds = OTBDataset(otb_root)
        # FaceOcc1 has only OCC; Jumping has MB+FM; neither has IV.
        seq_with_fm = ds.sequences_by_attribute("FM")
        assert "Jumping" in seq_with_fm
        assert "FaceOcc1" not in seq_with_fm


# ---------------------------------------------------------------------------
# Annotation table sanity checks
# ---------------------------------------------------------------------------

class TestAnnotationTable:
    def test_all_attributes_known(self):
        for seq_name, attrs in OTB100_ATTRIBUTES.items():
            unknown = attrs - OTB_ATTRIBUTES
            assert not unknown, f"{seq_name} has unknown attributes: {unknown}"

    def test_otb50_subset_of_otb100(self):
        assert OTB50_SEQUENCES.issubset(set(OTB100_ATTRIBUTES.keys()))

    def test_attribute_names_complete(self):
        assert set(OTB_ATTRIBUTE_NAMES.keys()) == OTB_ATTRIBUTES

    def test_otb50_has_50_sequences(self):
        assert len(OTB50_SEQUENCES) == 50

    def test_otb100_has_at_least_100_sequences(self):
        # The annotation table covers 100+ entries because some OTB-100 videos
        # are split into numbered sub-sequences (e.g. Jogging_1/2, Skating2_1/2).
        assert len(OTB100_ATTRIBUTES) >= 100


# ---------------------------------------------------------------------------
# Backward compatibility: import from base.py still works
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_import_from_base(self, otb_root):
        ds = OTBDatasetCompat(otb_root)
        assert len(ds) == 5

    def test_is_same_class(self):
        assert OTBDataset is OTBDatasetCompat
