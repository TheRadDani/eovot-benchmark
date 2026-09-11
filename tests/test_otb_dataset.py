"""Tests for the OTB-100/50 dataset loader with sequence attributes."""

import os
import tempfile

import numpy as np
import pytest

from eovot.datasets.otb import (
    ALL_ATTRIBUTES,
    ATTRIBUTE_NAMES,
    OTB100Dataset,
    OTB50Dataset,
    AttributedSequence,
    FM,
    IV,
    OCC,
    SV,
    _OTB100_ATTRIBUTES,
    _OTB50_NAMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_otb_sequence(root: str, name: str, n_frames: int = 5) -> None:
    """Write a minimal OTB-style sequence directory under *root*."""
    seq_dir = os.path.join(root, name)
    img_dir = os.path.join(seq_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    import cv2
    for i in range(1, n_frames + 1):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(img_dir, f"{i:04d}.jpg"), frame)
    gt = np.array([[10.0, 10.0, 20.0, 20.0]] * n_frames, dtype=np.float32)
    np.savetxt(
        os.path.join(seq_dir, "groundtruth_rect.txt"),
        gt,
        delimiter=",",
        fmt="%g",
    )


# ---------------------------------------------------------------------------
# Attribute constant tests
# ---------------------------------------------------------------------------

def test_all_attributes_count():
    assert len(ALL_ATTRIBUTES) == 11


def test_attribute_names_complete():
    for attr in ALL_ATTRIBUTES:
        assert attr in ATTRIBUTE_NAMES, f"Missing name for attribute {attr!r}"


# ---------------------------------------------------------------------------
# OTB-100 attribute table integrity
# ---------------------------------------------------------------------------

def test_attribute_values_are_subsets_of_all():
    for seq_name, attrs in _OTB100_ATTRIBUTES.items():
        unknown = attrs - ALL_ATTRIBUTES
        assert not unknown, f"Sequence {seq_name!r} has unknown attributes: {unknown}"


def test_otb50_names_present_in_otb100_table():
    for name in _OTB50_NAMES:
        assert name in _OTB100_ATTRIBUTES, f"{name!r} is in OTB-50 but not in OTB-100 attribute table"


def test_otb50_count_is_50():
    assert len(_OTB50_NAMES) == 50


def test_otb100_table_has_at_least_100_entries():
    assert len(_OTB100_ATTRIBUTES) >= 100


# ---------------------------------------------------------------------------
# AttributedSequence
# ---------------------------------------------------------------------------

def test_attributed_sequence_has_attribute():
    seq = AttributedSequence(
        name="TestSeq",
        frame_paths=["a.jpg"],
        ground_truth=np.array([[0.0, 0.0, 10.0, 10.0]]),
        attributes=frozenset({FM, IV}),
    )
    assert seq.has_attribute(FM)
    assert seq.has_attribute(IV)
    assert not seq.has_attribute(OCC)


def test_attributed_sequence_repr_contains_name_and_attrs():
    seq = AttributedSequence(
        name="MySeq",
        frame_paths=["x.jpg"],
        ground_truth=np.array([[0.0, 0.0, 5.0, 5.0]]),
        attributes=frozenset({SV}),
    )
    r = repr(seq)
    assert "MySeq" in r
    assert "SV" in r


# ---------------------------------------------------------------------------
# OTB100Dataset
# ---------------------------------------------------------------------------

def test_otb100_loads_discovered_sequences():
    with tempfile.TemporaryDirectory() as root:
        _make_otb_sequence(root, "Basketball", n_frames=3)
        _make_otb_sequence(root, "Bolt", n_frames=4)
        ds = OTB100Dataset(root)
        assert len(ds) == 2


def test_otb100_returns_attributed_sequence_type():
    with tempfile.TemporaryDirectory() as root:
        _make_otb_sequence(root, "Basketball")
        ds = OTB100Dataset(root)
        seq = ds[0]
        assert isinstance(seq, AttributedSequence)


def test_otb100_basketball_has_correct_attributes():
    with tempfile.TemporaryDirectory() as root:
        _make_otb_sequence(root, "Basketball")
        ds = OTB100Dataset(root)
        seq = ds[0]
        # Basketball: {IV, OCC, DEF, OPR, BC}
        assert IV in seq.attributes
        assert OCC in seq.attributes


def test_otb100_unknown_sequence_gets_empty_attributes():
    with tempfile.TemporaryDirectory() as root:
        _make_otb_sequence(root, "CustomSeqXYZ")
        ds = OTB100Dataset(root)
        seq = ds[0]
        assert isinstance(seq.attributes, frozenset)
        assert len(seq.attributes) == 0


def test_filter_by_attribute_returns_matching_sequences():
    with tempfile.TemporaryDirectory() as root:
        _make_otb_sequence(root, "Basketball")   # has IV
        _make_otb_sequence(root, "Jumping")      # MB, FM — no IV
        ds = OTB100Dataset(root)
        iv_seqs = ds.filter_by_attribute(IV)
        names = [s.name for s in iv_seqs]
        assert "Basketball" in names
        assert "Jumping" not in names


def test_filter_by_attribute_invalid_raises_value_error():
    with tempfile.TemporaryDirectory() as root:
        _make_otb_sequence(root, "Basketball")
        ds = OTB100Dataset(root)
        with pytest.raises(ValueError, match="Unknown attribute"):
            ds.filter_by_attribute("NOTREAL")


def test_attribute_summary_has_all_keys():
    with tempfile.TemporaryDirectory() as root:
        _make_otb_sequence(root, "Basketball")
        ds = OTB100Dataset(root)
        summary = ds.attribute_summary()
        assert set(summary.keys()) == ALL_ATTRIBUTES


def test_attribute_summary_counts_correctly():
    with tempfile.TemporaryDirectory() as root:
        # Basketball has IV, OCC, DEF, OPR, BC
        _make_otb_sequence(root, "Basketball")
        ds = OTB100Dataset(root)
        summary = ds.attribute_summary()
        assert summary[IV] == 1
        assert summary[OCC] == 1
        assert summary[FM] == 0   # Basketball has no FM


# ---------------------------------------------------------------------------
# OTB50Dataset
# ---------------------------------------------------------------------------

def test_otb50_excludes_non_otb50_sequences():
    with tempfile.TemporaryDirectory() as root:
        _make_otb_sequence(root, "Basketball")   # in OTB-50
        _make_otb_sequence(root, "Biker")         # NOT in OTB-50
        _make_otb_sequence(root, "Bolt")          # in OTB-50
        ds = OTB50Dataset(root)
        names = {ds[i].name for i in range(len(ds))}
        assert "Basketball" in names
        assert "Bolt" in names
        assert "Biker" not in names


def test_otb50_returns_attributed_sequence_type():
    with tempfile.TemporaryDirectory() as root:
        _make_otb_sequence(root, "Basketball")
        ds = OTB50Dataset(root)
        assert len(ds) == 1
        assert isinstance(ds[0], AttributedSequence)


def test_otb50_filter_by_attribute_works():
    with tempfile.TemporaryDirectory() as root:
        # Shaking: IV, SV, IPR, OPR, BC — no OCC
        _make_otb_sequence(root, "Shaking")
        # Matrix: IV, SV, OCC, MB, FM, IPR, OPR, BC
        _make_otb_sequence(root, "Matrix")
        ds = OTB50Dataset(root)
        occ_seqs = ds.filter_by_attribute(OCC)
        names = [s.name for s in occ_seqs]
        assert "Matrix" in names
        assert "Shaking" not in names
