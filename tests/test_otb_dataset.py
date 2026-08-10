"""Tests for OTB100Dataset — attribute-aware OTB-100 loader.

All tests use a temporary directory with synthetic OTB-style sequences so
no real OTB dataset download is required.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from eovot.datasets.otb import (
    ATTRIBUTE_DESCRIPTIONS,
    VALID_ATTRIBUTES,
    AttributedSequence,
    OTB100Dataset,
    _ATTRIBUTES,
    _OTBSubset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sequence(root: str, name: str, n_frames: int = 5) -> None:
    """Write a minimal valid OTB sequence directory under *root*."""
    seq_dir = os.path.join(root, name)
    img_dir = os.path.join(seq_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    rng = np.random.default_rng(abs(hash(name)) & 0xFFFF)
    gt = rng.integers(5, 50, (n_frames, 4)).astype(float)
    np.savetxt(os.path.join(seq_dir, "groundtruth_rect.txt"), gt,
               delimiter=",", fmt="%.1f")

    for i in range(1, n_frames + 1):
        frame = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(img_dir, f"{i:04d}.jpg"), frame)


def _ds(*seq_names: str, **kwargs) -> "tuple[str, OTB100Dataset]":
    """Create a temp directory, populate it, return (tmpdir, dataset)."""
    td = tempfile.mkdtemp()
    for name in seq_names:
        _make_sequence(td, name)
    return td, OTB100Dataset(td, **kwargs)


# ---------------------------------------------------------------------------
# Attribute table integrity
# ---------------------------------------------------------------------------

class TestAttributeTable:

    def test_all_attribute_codes_are_valid(self):
        for seq, attrs in _ATTRIBUTES.items():
            bad = attrs - VALID_ATTRIBUTES
            assert not bad, f"{seq}: unknown codes {bad}"

    def test_valid_attributes_has_eleven_entries(self):
        assert len(VALID_ATTRIBUTES) == 11

    def test_descriptions_cover_all_valid_attributes(self):
        assert set(ATTRIBUTE_DESCRIPTIONS.keys()) == VALID_ATTRIBUTES

    def test_all_attribute_values_are_frozensets(self):
        for seq, attrs in _ATTRIBUTES.items():
            assert isinstance(attrs, frozenset), f"{seq} attrs is not frozenset"

    def test_known_sequences_have_expected_attributes(self):
        assert "FM" in _ATTRIBUTES["Tiger1"]
        assert "OCC" in _ATTRIBUTES["Tiger1"]
        assert "IPR" not in _ATTRIBUTES["FaceOcc1"]
        assert "OCC" in _ATTRIBUTES["FaceOcc1"]
        assert _ATTRIBUTES["David2"] == frozenset({"IPR", "OPR"})
        assert _ATTRIBUTES["Jumping"] == frozenset({"MB", "FM"})


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

class TestDiscovery:

    def test_empty_root_yields_zero_sequences(self):
        with tempfile.TemporaryDirectory() as td:
            ds = OTB100Dataset(td)
            assert len(ds) == 0

    def test_single_sequence_discovered(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td)
            assert len(ds) == 1

    def test_multiple_sequences_discovered_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["Jumping", "Car1", "David"]:
                _make_sequence(td, name)
            ds = OTB100Dataset(td)
            assert len(ds) == 3
            names = [ds[i].name for i in range(len(ds))]
            assert names == sorted(names)

    def test_nonexistent_root_raises(self):
        with pytest.raises(FileNotFoundError):
            OTB100Dataset("/nonexistent/__otb__")

    def test_directories_without_gt_are_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            # Valid sequence
            _make_sequence(td, "Tiger1")
            # Directory without groundtruth_rect.txt
            os.makedirs(os.path.join(td, "NakedDir", "img"), exist_ok=True)
            ds = OTB100Dataset(td)
            assert len(ds) == 1


# ---------------------------------------------------------------------------
# __getitem__ returns AttributedSequence
# ---------------------------------------------------------------------------

class TestGetItem:

    def test_returns_attributed_sequence_type(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td)
            seq = ds[0]
            assert isinstance(seq, AttributedSequence)

    def test_known_sequence_has_bundled_attributes(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td)
            seq = ds[0]
            assert seq.name == "Tiger1"
            assert "OCC" in seq.attributes
            assert "FM" in seq.attributes
            assert "IV" in seq.attributes

    def test_unknown_sequence_has_empty_attributes(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "MyCustomSeq999")
            ds = OTB100Dataset(td)
            seq = ds[0]
            assert seq.attributes == frozenset()

    def test_iteration_yields_attributed_sequences(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["Tiger1", "Jumping", "David2"]:
                _make_sequence(td, name)
            ds = OTB100Dataset(td)
            for seq in ds:
                assert isinstance(seq, AttributedSequence)


# ---------------------------------------------------------------------------
# filter_by_attribute
# ---------------------------------------------------------------------------

class TestFilterByAttribute:

    def test_single_attr_filter(self):
        with tempfile.TemporaryDirectory() as td:
            # Tiger1 has FM; David2 does NOT; Jumping has FM
            for name in ["Tiger1", "David2", "Jumping"]:
                _make_sequence(td, name)
            ds = OTB100Dataset(td)
            fm = ds.filter_by_attribute("FM")
            names = {fm[i].name for i in range(len(fm))}
            assert "Tiger1" in names
            assert "Jumping" in names
            assert "David2" not in names

    def test_multi_attr_filter_requires_all(self):
        with tempfile.TemporaryDirectory() as td:
            # Tiger1 has FM+OCC; Jumping has FM but NOT OCC
            for name in ["Tiger1", "Jumping"]:
                _make_sequence(td, name)
            ds = OTB100Dataset(td)
            both = ds.filter_by_attribute("FM", "OCC")
            assert len(both) == 1
            assert both[0].name == "Tiger1"

    def test_empty_result_when_no_match(self):
        with tempfile.TemporaryDirectory() as td:
            # FaceOcc1 has only OCC
            _make_sequence(td, "FaceOcc1")
            ds = OTB100Dataset(td)
            fm = ds.filter_by_attribute("FM")
            assert len(fm) == 0

    def test_invalid_code_raises_value_error(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td)
            with pytest.raises(ValueError, match="Unknown attribute"):
                ds.filter_by_attribute("BOGUS")

    def test_subset_contains_attributed_sequences(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["Tiger1", "David2"]:
                _make_sequence(td, name)
            ds = OTB100Dataset(td)
            subset = ds.filter_by_attribute("OCC")
            for seq in subset:
                assert isinstance(seq, AttributedSequence)
                assert "OCC" in seq.attributes

    def test_filter_returns_otb_subset_instance(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td)
            subset = ds.filter_by_attribute("FM")
            assert isinstance(subset, _OTBSubset)

    def test_subset_index_out_of_range_raises(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td)
            subset = ds.filter_by_attribute("FM")
            with pytest.raises(IndexError):
                _ = subset[999]


# ---------------------------------------------------------------------------
# filter_by_names
# ---------------------------------------------------------------------------

class TestFilterByNames:

    def test_filter_by_names_basic(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["Tiger1", "Car1", "David"]:
                _make_sequence(td, name)
            ds = OTB100Dataset(td)
            subset = ds.filter_by_names(["Tiger1", "David"])
            assert len(subset) == 2
            names = {subset[i].name for i in range(len(subset))}
            assert names == {"Tiger1", "David"}

    def test_filter_by_names_unknown_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td)
            subset = ds.filter_by_names(["Tiger1", "NonExistent"])
            assert len(subset) == 1

    def test_filter_by_names_empty_list(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td)
            subset = ds.filter_by_names([])
            assert len(subset) == 0


# ---------------------------------------------------------------------------
# attribute_distribution
# ---------------------------------------------------------------------------

class TestAttributeDistribution:

    def test_distribution_covers_all_11_attributes(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td)
            dist = ds.attribute_distribution()
            assert set(dist.keys()) == VALID_ATTRIBUTES

    def test_distribution_counts_correct(self):
        with tempfile.TemporaryDirectory() as td:
            # Tiger1: OCC ✓   FaceOcc1: OCC ✓   David2: OCC ✗
            for name in ["Tiger1", "FaceOcc1", "David2"]:
                _make_sequence(td, name)
            ds = OTB100Dataset(td)
            dist = ds.attribute_distribution()
            assert dist["OCC"] == 2

    def test_empty_dataset_all_zeros(self):
        with tempfile.TemporaryDirectory() as td:
            ds = OTB100Dataset(td)
            dist = ds.attribute_distribution()
            assert all(v == 0 for v in dist.values())

    def test_subset_distribution_all_have_filtered_attr(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["Tiger1", "FaceOcc1", "Jumping"]:
                _make_sequence(td, name)
            ds = OTB100Dataset(td)
            occ_sub = ds.filter_by_attribute("OCC")
            sub_dist = occ_sub.attribute_distribution()
            # Every sequence in the OCC subset must carry OCC
            assert sub_dist["OCC"] == len(occ_sub)


# ---------------------------------------------------------------------------
# sequences_with_attribute
# ---------------------------------------------------------------------------

class TestSequencesWithAttribute:

    def test_returns_sorted_list(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["Tiger1", "Jumping", "FaceOcc1"]:
                _make_sequence(td, name)
            ds = OTB100Dataset(td)
            names = ds.sequences_with_attribute("OCC")
            assert names == sorted(names)

    def test_only_discovered_sequences_returned(self):
        with tempfile.TemporaryDirectory() as td:
            # Tiger1 is on disk; Tiger2 is NOT
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td)
            names = ds.sequences_with_attribute("OCC")
            assert "Tiger2" not in names
            assert "Tiger1" in names

    def test_invalid_attribute_raises(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td)
            with pytest.raises(ValueError):
                ds.sequences_with_attribute("INVALID")

    def test_no_match_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "David2")   # only IPR, OPR
            ds = OTB100Dataset(td)
            names = ds.sequences_with_attribute("LR")
            assert names == []


# ---------------------------------------------------------------------------
# Custom attribute overrides
# ---------------------------------------------------------------------------

class TestCustomAttributes:

    def test_override_replaces_bundled_entry(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td, attributes={"Tiger1": frozenset({"SV"})})
            seq = ds[0]
            assert seq.attributes == frozenset({"SV"})

    def test_override_adds_new_entry(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "MySeq")
            ds = OTB100Dataset(td, attributes={"MySeq": frozenset({"FM", "OCC"})})
            seq = ds[0]
            assert "FM" in seq.attributes
            assert "OCC" in seq.attributes

    def test_bundled_entries_not_affected_by_other_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["Tiger1", "Jumping"]:
                _make_sequence(td, name)
            ds = OTB100Dataset(td, attributes={"Jumping": frozenset({"IV"})})
            tiger = next(s for s in ds if s.name == "Tiger1")
            # Tiger1 should still carry its original bundled attributes
            assert "OCC" in tiger.attributes
            assert "FM" in tiger.attributes


# ---------------------------------------------------------------------------
# AttributedSequence unit tests
# ---------------------------------------------------------------------------

class TestAttributedSequence:

    def _make(self, name="seq", attrs=frozenset({"FM", "OCC"})):
        return AttributedSequence(
            name=name,
            frame_paths=[],
            ground_truth=np.zeros((1, 4)),
            attributes=attrs,
        )

    def test_has_attribute_true(self):
        seq = self._make(attrs=frozenset({"FM", "OCC"}))
        assert seq.has_attribute("FM")
        assert seq.has_attribute("OCC")

    def test_has_attribute_false(self):
        seq = self._make(attrs=frozenset({"FM"}))
        assert not seq.has_attribute("SV")

    def test_attributes_immutable_frozenset(self):
        seq = self._make()
        assert isinstance(seq.attributes, frozenset)

    def test_repr_lists_attributes(self):
        seq = self._make(name="Tiger1", attrs=frozenset({"FM", "OCC"}))
        r = repr(seq)
        assert "Tiger1" in r
        assert "FM" in r
        assert "OCC" in r

    def test_repr_empty_attributes_shows_none(self):
        seq = self._make(attrs=frozenset())
        assert "none" in repr(seq)

    def test_default_attributes_empty(self):
        seq = AttributedSequence("x", [], np.zeros((1, 4)))
        assert seq.attributes == frozenset()


# ---------------------------------------------------------------------------
# Repr tests
# ---------------------------------------------------------------------------

class TestRepr:

    def test_dataset_repr(self):
        with tempfile.TemporaryDirectory() as td:
            ds = OTB100Dataset(td)
            r = repr(ds)
            assert "OTB100Dataset" in r
            assert "sequences=0" in r

    def test_subset_repr(self):
        with tempfile.TemporaryDirectory() as td:
            _make_sequence(td, "Tiger1")
            ds = OTB100Dataset(td)
            sub = ds.filter_by_attribute("FM")
            assert "_OTBSubset" in repr(sub)
