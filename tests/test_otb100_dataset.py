"""Tests for the OTB-50 / OTB-100 dataset loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from eovot.datasets.otb import (
    ATTRIBUTE_DESCRIPTIONS,
    OTB50_SEQUENCES,
    OTB100_SEQUENCES,
    SEQUENCE_ATTRIBUTES,
    VALID_ATTRIBUTES,
    OTB100Dataset,
    _OTB100_EXTRA,
)


# ---------------------------------------------------------------------------
# Constants integrity
# ---------------------------------------------------------------------------

class TestConstants:
    def test_otb50_count(self):
        assert len(OTB50_SEQUENCES) == 50

    def test_otb100_extra_count(self):
        assert len(_OTB100_EXTRA) == 50

    def test_otb100_total_count(self):
        assert len(OTB100_SEQUENCES) == 100

    def test_otb100_is_superset(self):
        assert OTB50_SEQUENCES.issubset(OTB100_SEQUENCES)

    def test_no_overlap_between_splits(self):
        assert OTB50_SEQUENCES.isdisjoint(_OTB100_EXTRA)

    def test_valid_attributes_count(self):
        assert len(VALID_ATTRIBUTES) == 11

    def test_attribute_descriptions_complete(self):
        assert VALID_ATTRIBUTES == set(ATTRIBUTE_DESCRIPTIONS.keys())

    def test_sequence_attributes_coverage(self):
        missing = OTB100_SEQUENCES - set(SEQUENCE_ATTRIBUTES.keys())
        assert not missing, f"Sequences missing from SEQUENCE_ATTRIBUTES: {missing}"

    def test_all_attribute_values_valid(self):
        for seq, attrs in SEQUENCE_ATTRIBUTES.items():
            invalid = attrs - VALID_ATTRIBUTES
            assert not invalid, f"{seq} has unknown attributes: {invalid}"

    def test_known_otb50_sequences_present(self):
        for seq in ("Basketball", "Bolt", "David", "Dudek", "Woman"):
            assert seq in OTB50_SEQUENCES

    def test_known_otb100_extra_present(self):
        for seq in ("Bird2", "Dog", "FleetFace", "Polo", "Yo-Yo"):
            assert seq in _OTB100_EXTRA


# ---------------------------------------------------------------------------
# Fixture: minimal on-disk OTB layout
# ---------------------------------------------------------------------------

def _write_gt(seq_dir: Path, bboxes: list[tuple]) -> None:
    gt_path = seq_dir / "groundtruth_rect.txt"
    lines = [f"{x},{y},{w},{h}" for x, y, w, h in bboxes]
    gt_path.write_text("\n".join(lines))


def _write_frames(seq_dir: Path, n: int) -> None:
    img_dir = seq_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        import cv2
        for i in range(1, n + 1):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"{i:04d}.jpg"), frame)
    except Exception:
        for i in range(1, n + 1):
            (img_dir / f"{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")


@pytest.fixture()
def otb_root(tmp_path: Path) -> Path:
    """Build a tiny OTB layout with 3 known sequences."""
    for seq_name in ("Basketball", "Bolt", "Bird2"):
        seq_dir = tmp_path / seq_name
        seq_dir.mkdir()
        n_frames = 5
        bboxes = [(10 + i, 20 + i, 50, 60) for i in range(n_frames)]
        _write_gt(seq_dir, bboxes)
        _write_frames(seq_dir, n_frames)
    return tmp_path


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_otb100_split(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="otb100")
        assert len(ds) >= 1

    def test_otb50_split(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="otb50")
        seqs = ds.sequence_names
        # Basketball is in OTB50; Bird2 is not
        assert "Basketball" in seqs
        assert "Bird2" not in seqs

    def test_all_split(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        assert len(ds) == 3

    def test_max_sequences(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all", max_sequences=2)
        assert len(ds) == 2

    def test_invalid_split(self, otb_root):
        with pytest.raises(ValueError, match="split"):
            OTB100Dataset(root=otb_root, split="badvalue")

    def test_nonexistent_root(self, tmp_path):
        with pytest.raises((FileNotFoundError, NotADirectoryError, ValueError)):
            OTB100Dataset(root=tmp_path / "no_such_dir")

    def test_empty_root(self, tmp_path):
        with pytest.raises(ValueError, match="[Nn]o.*sequence"):
            OTB100Dataset(root=tmp_path, split="all")


# ---------------------------------------------------------------------------
# Name and metadata
# ---------------------------------------------------------------------------

class TestName:
    def test_default_name_otb100(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="otb100")
        assert "OTB" in ds.name or "otb" in ds.name.lower()

    def test_default_name_otb50(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="otb50")
        assert "50" in ds.name or "otb50" in ds.name.lower()

    def test_custom_name(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all", name="MyTest")
        assert ds.name == "MyTest"

    def test_sequence_names_type(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        names = ds.sequence_names
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_categories(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        cats = ds.categories
        assert isinstance(cats, list)


# ---------------------------------------------------------------------------
# Sequence loading
# ---------------------------------------------------------------------------

class TestSequenceLoading:
    def test_len_matches_sequence_names(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        assert len(ds) == len(ds.sequence_names)

    def test_iteration(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        seqs = list(ds)
        assert len(seqs) == len(ds)

    def test_sequence_has_frames(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        seq = next(iter(ds))
        assert len(seq) == 5

    def test_sequence_has_gt(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        seq = next(iter(ds))
        assert len(seq.ground_truth) == 5

    def test_gt_bbox_values(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        for seq in ds:
            if seq.name == "Basketball":
                x, y, w, h = seq.ground_truth[0]
                assert (x, y, w, h) == (10.0, 20.0, 50.0, 60.0)
                break


# ---------------------------------------------------------------------------
# Attribute API
# ---------------------------------------------------------------------------

class TestAttributeAPI:
    def test_get_attributes_known_sequence(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        attrs = ds.get_attributes("Basketball")
        assert isinstance(attrs, frozenset)
        assert attrs.issubset(VALID_ATTRIBUTES)

    def test_get_attributes_unknown_sequence(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        attrs = ds.get_attributes("__nonexistent__")
        assert attrs == frozenset()

    def test_filter_by_attribute_returns_subset(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        basketball_attrs = ds.get_attributes("Basketball")
        if basketball_attrs:
            attr = next(iter(basketball_attrs))
            filtered = ds.filter_by_attribute(attr)
            assert "Basketball" in filtered.sequence_names

    def test_filter_by_impossible_combination(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        # Filtering by all 11 attributes simultaneously is very restrictive
        filtered = ds.filter_by_attribute(*list(VALID_ATTRIBUTES))
        assert len(filtered) <= len(ds)

    def test_sequences_with_attribute(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        result = ds.sequences_with_attribute("IV")
        assert isinstance(result, list)

    def test_attribute_coverage(self, otb_root):
        ds = OTB100Dataset(root=otb_root, split="all")
        cov = ds.attribute_coverage()
        assert isinstance(cov, dict)
        for attr, count in cov.items():
            assert attr in VALID_ATTRIBUTES
            assert isinstance(count, int)
            assert count >= 0
