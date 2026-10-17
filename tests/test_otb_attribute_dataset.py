"""Tests for eovot.datasets.otb (AttributeAwareOTBDataset).

All tests use a synthetic on-disk fixture that mirrors the OTB directory
layout; no real OTB download is required.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pytest

from eovot.datasets.otb import (
    OTB_ATTRIBUTES,
    OTB50_SEQUENCES,
    SEQUENCE_ATTRIBUTES,
    AttributeAwareOTBDataset,
    OTBTaggedSequence,
    get_sequence_attributes,
)


# ── fixture helpers ────────────────────────────────────────────────────────────

def _write_minimal_jpg(path: Path) -> None:
    """Write a minimal valid JPEG so cv2.imread can open it."""
    path.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01"
        b"\x00\x00\xff\xd9"
    )


def _make_sequence(
    root: Path,
    name: str,
    n_frames: int = 5,
    gt_sep: str = ",",
) -> None:
    """Create a minimal OTB-style sequence directory under *root*."""
    img_dir = root / name / "img"
    img_dir.mkdir(parents=True)

    for i in range(1, n_frames + 1):
        _write_minimal_jpg(img_dir / f"{i:04d}.jpg")

    rows = [[10 + i, 20 + i, 50, 50] for i in range(n_frames)]
    sep = "," if gt_sep == "," else " "
    gt_lines = [sep.join(str(v) for v in r) for r in rows]
    (root / name / "groundtruth_rect.txt").write_text("\n".join(gt_lines) + "\n")


@pytest.fixture()
def otb_root(tmp_path):
    """OTB root with four sequences covering several attributes."""
    # Basketball: IV, OCC, IPR, OPR  (in OTB50)
    _make_sequence(tmp_path, "Basketball", n_frames=6)
    # Bolt: OCC, DEF, FM  (in OTB50)
    _make_sequence(tmp_path, "Bolt", n_frames=4)
    # Dog: SV, OPR, DEF  (OTB100 only)
    _make_sequence(tmp_path, "Dog", n_frames=5)
    # Man: IV only
    _make_sequence(tmp_path, "Man", n_frames=3)
    return tmp_path


# ── get_sequence_attributes ───────────────────────────────────────────────────

class TestGetSequenceAttributes:
    def test_known_sequence(self):
        attrs = get_sequence_attributes("Basketball")
        assert "occlusion" in attrs
        assert "illumination_variation" in attrs

    def test_unknown_sequence_returns_empty(self):
        assert get_sequence_attributes("ThisDoesNotExist") == []

    def test_returns_list(self):
        assert isinstance(get_sequence_attributes("Bolt"), list)


# ── SEQUENCE_ATTRIBUTES data integrity ───────────────────────────────────────

class TestSequenceAttributesData:
    def test_all_attribute_values_are_valid(self):
        for seq_name, attrs in SEQUENCE_ATTRIBUTES.items():
            for attr in attrs:
                assert attr in OTB_ATTRIBUTES, (
                    f"{seq_name}: unknown attribute '{attr}'"
                )

    def test_no_empty_attribute_lists(self):
        for seq_name, attrs in SEQUENCE_ATTRIBUTES.items():
            assert len(attrs) > 0, f"{seq_name} has an empty attribute list"

    def test_otb50_entries_have_valid_attrs(self):
        for name in OTB50_SEQUENCES:
            if name in SEQUENCE_ATTRIBUTES:
                for attr in SEQUENCE_ATTRIBUTES[name]:
                    assert attr in OTB_ATTRIBUTES


# ── AttributeAwareOTBDataset – loading ────────────────────────────────────────

class TestDatasetLoading:
    def test_loads_all_sequences(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        assert len(ds) == 4

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            AttributeAwareOTBDataset("/this/does/not/exist")

    def test_invalid_subset_raises(self, otb_root):
        with pytest.raises(ValueError, match="subset"):
            AttributeAwareOTBDataset(str(otb_root), subset="OTB200")

    def test_otb50_subset_excludes_otb100_only(self, otb_root):
        ds50 = AttributeAwareOTBDataset(str(otb_root), subset="OTB50")
        for seq in ds50:
            assert seq.name in OTB50_SEQUENCES

    def test_otb50_is_subset_of_otb100(self, otb_root):
        ds50 = AttributeAwareOTBDataset(str(otb_root), subset="OTB50")
        ds100 = AttributeAwareOTBDataset(str(otb_root), subset="OTB100")
        assert len(ds50) <= len(ds100)

    def test_max_sequences(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root), max_sequences=2)
        assert len(ds) == 2

    def test_name_property_otb100(self, otb_root):
        assert AttributeAwareOTBDataset(str(otb_root)).name == "OTB100"

    def test_name_property_otb50(self, otb_root):
        assert AttributeAwareOTBDataset(str(otb_root), subset="OTB50").name == "OTB50"

    def test_subset_property(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root), subset="OTB50")
        assert ds.subset == "OTB50"

    def test_iter_yields_all(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        seqs = list(ds)
        assert len(seqs) == 4

    def test_getitem_returns_tagged_sequence(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        seq = ds[0]
        assert isinstance(seq, OTBTaggedSequence)


# ── OTBTaggedSequence properties ─────────────────────────────────────────────

class TestOTBTaggedSequence:
    def test_sequence_names_nonempty(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        for seq in ds:
            assert isinstance(seq.name, str) and len(seq.name) > 0

    def test_ground_truth_shape(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        for seq in ds:
            assert seq.ground_truth.ndim == 2
            assert seq.ground_truth.shape[1] == 4

    def test_init_bbox_matches_gt_row0(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        for seq in ds:
            assert list(seq.init_bbox) == seq.ground_truth[0].tolist()

    def test_gt_length_matches_frames(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        for seq in ds:
            assert len(seq) == len(seq.ground_truth)

    def test_attributes_are_valid_otb_attrs(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        for seq in ds:
            for attr in seq.attributes:
                assert attr in OTB_ATTRIBUTES

    def test_has_attribute_true(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        basketball = next(s for s in ds if s.name == "Basketball")
        assert basketball.has_attribute("occlusion")

    def test_has_attribute_false(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        basketball = next(s for s in ds if s.name == "Basketball")
        assert not basketball.has_attribute("low_resolution")

    def test_attributes_property_returns_copy(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        seq = ds[0]
        orig = seq.attributes
        orig.append("fake_attr")
        assert "fake_attr" not in seq.attributes

    def test_ground_truth_dtype(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        for seq in ds:
            assert seq.ground_truth.dtype == np.float64


# ── attribute filtering ───────────────────────────────────────────────────────

class TestAttributeFiltering:
    def test_filter_reduces_count(self, otb_root):
        ds_all = AttributeAwareOTBDataset(str(otb_root))
        ds_occ = AttributeAwareOTBDataset(str(otb_root), attributes=["occlusion"])
        assert len(ds_occ) <= len(ds_all)

    def test_unknown_attribute_in_filter_gives_empty_dataset(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root), attributes=["not_a_real_attr"])
        assert len(ds) == 0

    def test_filter_illumination_variation(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root), attributes=["illumination_variation"])
        for seq in ds:
            assert seq.has_attribute("illumination_variation")

    def test_multi_attribute_filter_is_union(self, otb_root):
        ds = AttributeAwareOTBDataset(
            str(otb_root), attributes=["occlusion", "illumination_variation"]
        )
        for seq in ds:
            has = seq.has_attribute("occlusion") or seq.has_attribute("illumination_variation")
            assert has


# ── sequences_by_attribute ────────────────────────────────────────────────────

class TestSequencesByAttribute:
    def test_returns_list(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        result = ds.sequences_by_attribute("occlusion")
        assert isinstance(result, list)

    def test_all_returned_have_attribute(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        for seq in ds.sequences_by_attribute("scale_variation"):
            assert seq.has_attribute("scale_variation")

    def test_invalid_attribute_raises(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        with pytest.raises(ValueError, match="Unknown attribute"):
            ds.sequences_by_attribute("invalid_attr")

    def test_consistent_with_manual_filter(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        attr = "fast_motion"
        by_method = ds.sequences_by_attribute(attr)
        by_hand = [s for s in ds if s.has_attribute(attr)]
        assert {s.name for s in by_method} == {s.name for s in by_hand}


# ── attribute_distribution ────────────────────────────────────────────────────

class TestAttributeDistribution:
    def test_returns_dict(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        dist = ds.attribute_distribution()
        assert isinstance(dist, dict)

    def test_all_attributes_present(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        dist = ds.attribute_distribution()
        for attr in OTB_ATTRIBUTES:
            assert attr in dist

    def test_counts_non_negative(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        for count in ds.attribute_distribution().values():
            assert count >= 0

    def test_sorted_descending(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        counts = list(ds.attribute_distribution().values())
        assert counts == sorted(counts, reverse=True)

    def test_sum_consistent_with_tag_counts(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        dist = ds.attribute_distribution()
        expected = sum(len(s.attributes) for s in ds)
        assert sum(dist.values()) == expected


# ── attribute_breakdown_summary ───────────────────────────────────────────────

class TestAttributeBreakdownSummary:
    def test_returns_string(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        s = ds.attribute_breakdown_summary()
        assert isinstance(s, str)

    def test_contains_attribute_names(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        s = ds.attribute_breakdown_summary()
        assert "occlusion" in s
        assert "scale_variation" in s

    def test_contains_subset_name(self, otb_root):
        ds = AttributeAwareOTBDataset(str(otb_root))
        s = ds.attribute_breakdown_summary()
        assert "OTB100" in s


# ── GT parsing edge cases ─────────────────────────────────────────────────────

class TestGTParsing:
    def test_space_separated_gt(self, tmp_path):
        _make_sequence(tmp_path, "SpaceSeq", n_frames=3, gt_sep=" ")
        ds = AttributeAwareOTBDataset(str(tmp_path))
        seq = ds[0]
        assert seq.ground_truth.shape == (3, 4)

    def test_comma_separated_gt(self, tmp_path):
        _make_sequence(tmp_path, "CommaSeq", n_frames=3, gt_sep=",")
        ds = AttributeAwareOTBDataset(str(tmp_path))
        seq = ds[0]
        assert seq.ground_truth.dtype == np.float64
