"""Tests for eovot.datasets.otb."""
import numpy as np
import pytest

from eovot.datasets.otb import (
    ALL_ATTRIBUTES, ATTRIBUTE_NAMES,
    IV, SV, OCC, DEF, MB, FM, IPR, OPR, OV, BC, LR,
    _OTB100_ATTRIBUTES, _OTB50_NAMES,
    AttributedSequence, OTB100Dataset, OTB50Dataset,
)


class TestAttributeConstants:
    def test_all_attributes_has_eleven(self):
        assert len(ALL_ATTRIBUTES) == 11

    def test_all_constants_in_set(self):
        for attr in (IV, SV, OCC, DEF, MB, FM, IPR, OPR, OV, BC, LR):
            assert attr in ALL_ATTRIBUTES

    def test_attribute_names_covers_all(self):
        assert set(ATTRIBUTE_NAMES.keys()) == ALL_ATTRIBUTES


class TestOTB100AttributeTable:
    def test_has_substantial_coverage(self):
        assert len(_OTB100_ATTRIBUTES) >= 90

    def test_basketball_has_iv_and_occ(self):
        attrs = _OTB100_ATTRIBUTES["Basketball"]
        assert IV in attrs
        assert OCC in attrs

    def test_all_otb50_names_present(self):
        for name in _OTB50_NAMES:
            assert name in _OTB100_ATTRIBUTES, (
                f"OTB-50 sequence {name!r} missing from OTB-100 attribute table"
            )

    def test_all_attribute_values_are_valid(self):
        for name, attrs in _OTB100_ATTRIBUTES.items():
            unknown = attrs - ALL_ATTRIBUTES
            assert not unknown, f"{name!r} has unknown attrs: {unknown}"


class TestOTB50Names:
    def test_exactly_fifty_sequences(self):
        assert len(_OTB50_NAMES) == 50

    def test_known_sequences_present(self):
        for name in ("Basketball", "Soccer", "Tiger1", "Woman", "Walking"):
            assert name in _OTB50_NAMES


class TestAttributedSequence:
    @staticmethod
    def _make(attrs=frozenset()):
        gt = np.zeros((3, 4))
        return AttributedSequence(
            name="Test", frame_paths=[], ground_truth=gt, attributes=attrs
        )

    def test_has_attribute_true(self):
        seq = self._make(attrs=frozenset({IV, OCC}))
        assert seq.has_attribute(IV)
        assert seq.has_attribute(OCC)

    def test_has_attribute_false(self):
        seq = self._make(attrs=frozenset({IV}))
        assert not seq.has_attribute(SV)

    def test_empty_attributes_default(self):
        seq = self._make()
        assert seq.attributes == frozenset()
        assert not seq.has_attribute(IV)

    def test_attributes_are_frozen(self):
        seq = self._make(attrs=frozenset({IV}))
        assert isinstance(seq.attributes, frozenset)


class TestOTB100Dataset:
    @pytest.fixture
    def root(self, tmp_path):
        seq = tmp_path / "Basketball"
        (seq / "img").mkdir(parents=True)
        (seq / "img" / "0001.jpg").write_bytes(b"\xff\xd8\xff")
        (seq / "groundtruth_rect.txt").write_text("100,200,50,60\n")
        return str(tmp_path)

    def test_returns_attributed_sequence(self, root):
        ds = OTB100Dataset(root)
        assert len(ds) == 1
        assert isinstance(ds[0], AttributedSequence)

    def test_basketball_attributes_correct(self, root):
        seq = OTB100Dataset(root)[0]
        assert seq.has_attribute(IV)
        assert seq.has_attribute(OCC)
        assert not seq.has_attribute(SV)

    def test_unknown_sequence_empty_attrs(self, tmp_path):
        seq = tmp_path / "ZZZUnknown"
        (seq / "img").mkdir(parents=True)
        (seq / "img" / "0001.jpg").write_bytes(b"\xff\xd8\xff")
        (seq / "groundtruth_rect.txt").write_text("0,0,10,10\n")
        ds = OTB100Dataset(str(tmp_path))
        assert ds[0].attributes == frozenset()

    def test_filter_by_attribute_match(self, root):
        ds = OTB100Dataset(root)
        result = ds.filter_by_attribute(IV)
        assert len(result) == 1
        assert result[0].name == "Basketball"

    def test_filter_by_attribute_no_match(self, root):
        ds = OTB100Dataset(root)
        # Basketball does not carry SV
        assert ds.filter_by_attribute(SV) == []

    def test_filter_by_invalid_attr_raises(self, root):
        ds = OTB100Dataset(root)
        with pytest.raises(ValueError, match="Unknown attribute"):
            ds.filter_by_attribute("INVALID")

    def test_attribute_summary_keys(self, root):
        summary = OTB100Dataset(root).attribute_summary()
        assert set(summary.keys()) == ALL_ATTRIBUTES

    def test_attribute_summary_iv_nonzero(self, root):
        summary = OTB100Dataset(root).attribute_summary()
        assert summary[IV] >= 1   # Basketball carries IV
        assert summary[SV] == 0   # Basketball does not carry SV


class TestOTB50Dataset:
    @pytest.fixture
    def root(self, tmp_path):
        # Basketball is OTB-50; Bird2 is OTB-100 only
        for name in ("Basketball", "Bird2"):
            seq = tmp_path / name
            (seq / "img").mkdir(parents=True)
            (seq / "img" / "0001.jpg").write_bytes(b"\xff\xd8\xff")
            (seq / "groundtruth_rect.txt").write_text("0,0,10,10\n")
        return str(tmp_path)

    def test_excludes_otb100_only_sequences(self, root):
        ds50 = OTB50Dataset(root)
        names = {ds50[i].name for i in range(len(ds50))}
        assert "Basketball" in names
        assert "Bird2" not in names

    def test_returns_attributed_sequence(self, root):
        ds50 = OTB50Dataset(root)
        assert isinstance(ds50[0], AttributedSequence)
