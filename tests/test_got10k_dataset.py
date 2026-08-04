"""Tests for eovot.datasets.got10k.GOT10kDataset and GOT10kSequenceMeta."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from eovot.datasets.got10k import GOT10kDataset, GOT10kSequenceMeta, _parse_meta_info


# ---------------------------------------------------------------------------
# Helpers — build a minimal GOT-10k directory tree in a temp folder
# ---------------------------------------------------------------------------

def _write_gt(seq_dir: Path, boxes: List[Tuple[float, float, float, float]]) -> None:
    gt = seq_dir / "groundtruth.txt"
    lines = [f"{x:.1f},{y:.1f},{w:.1f},{h:.1f}" for x, y, w, h in boxes]
    gt.write_text("\n".join(lines) + "\n")


def _write_frames(img_dir: Path, n: int) -> None:
    """Create tiny placeholder JPEG files so frame_paths is non-empty."""
    import struct, zlib

    def _tiny_jpg() -> bytes:
        # Minimal 1×1 white JPEG (valid enough that the path exists).
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e\x1f"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
            b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
            b"\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br"
            b"\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZ"
            b"cdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95"
            b"\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3"
            b"\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca"
            b"\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7"
            b"\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xff\xd9"
        )

    img_dir.mkdir(parents=True, exist_ok=True)
    data = _tiny_jpg()
    for i in range(1, n + 1):
        (img_dir / f"{i:08d}.jpg").write_bytes(data)


def _write_meta(seq_dir: Path, object_class: str, motion_class: str = "rigid") -> None:
    ini = seq_dir / "meta_info.ini"
    ini.write_text(
        textwrap.dedent(f"""\
            [DEFAULT]
            object_class_name = {object_class}
            motion_class = {motion_class}
            major_class = {object_class}
            root_class = {object_class}
            motion_adverb = quickly
        """)
    )


def _make_split(root: Path, split: str, sequences: List[dict]) -> None:
    """Build a split directory with specified sequences.

    Each entry in *sequences* is a dict with:
        name, boxes (list of 4-tuples), object_class, motion_class (optional)
    """
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for seq in sequences:
        seq_dir = split_dir / seq["name"]
        seq_dir.mkdir(parents=True, exist_ok=True)
        boxes = seq["boxes"]
        _write_gt(seq_dir, boxes)
        _write_frames(seq_dir / "img", len(boxes))
        if "object_class" in seq:
            _write_meta(seq_dir, seq["object_class"], seq.get("motion_class", "rigid"))
        names.append(seq["name"])

    # Write list.txt so list_sequences() uses it.
    (split_dir / "list.txt").write_text("\n".join(names) + "\n")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BOXES_A = [(10.0, 20.0, 50.0, 40.0), (12.0, 21.0, 50.0, 40.0), (14.0, 22.0, 50.0, 40.0)]
_BOXES_B = [(5.0, 5.0, 30.0, 30.0), (6.0, 6.0, 30.0, 30.0)]
_BOXES_C = [(100.0, 80.0, 60.0, 45.0), (102.0, 81.0, 60.0, 45.0)]


@pytest.fixture
def got10k_root(tmp_path: Path) -> Path:
    """Minimal GOT-10k tree with two val sequences and one train sequence."""
    _make_split(tmp_path, "val", [
        {"name": "GOT-10k_Val_000001", "boxes": _BOXES_A, "object_class": "dog"},
        {"name": "GOT-10k_Val_000002", "boxes": _BOXES_B, "object_class": "car"},
    ])
    _make_split(tmp_path, "train", [
        {"name": "GOT-10k_Train_000001", "boxes": _BOXES_C, "object_class": "dog",
         "motion_class": "non-rigid"},
    ])
    return tmp_path


# ---------------------------------------------------------------------------
# split validation
# ---------------------------------------------------------------------------

def test_invalid_split(got10k_root: Path) -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        GOT10kDataset(str(got10k_root), split="invalid")


def test_valid_splits(got10k_root: Path) -> None:
    for split in ("val", "train", "test"):
        # test split has no sequences in our fixture but must not raise
        ds = GOT10kDataset(str(got10k_root), split=split)
        assert isinstance(ds, GOT10kDataset)


# ---------------------------------------------------------------------------
# list_sequences / __len__
# ---------------------------------------------------------------------------

def test_list_sequences_uses_list_txt(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    names = ds.list_sequences()
    assert names == ["GOT-10k_Val_000001", "GOT-10k_Val_000002"]


def test_list_sequences_cached(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    names1 = ds.list_sequences()
    names2 = ds.list_sequences()
    assert names1 is names2  # same list object — cache hit


def test_len(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    assert len(ds) == 2


def test_max_sequences(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val", max_sequences=1)
    assert len(ds) == 1
    assert ds.list_sequences() == ["GOT-10k_Val_000001"]


def test_max_sequences_larger_than_total(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val", max_sequences=100)
    assert len(ds) == 2


def test_list_sequences_fallback_no_list_txt(tmp_path: Path) -> None:
    """When list.txt is absent, directories with img/ are discovered instead."""
    split_dir = tmp_path / "val"
    seq_dir = split_dir / "GOT-10k_Val_Fallback"
    seq_dir.mkdir(parents=True)
    _write_gt(seq_dir, _BOXES_A)
    _write_frames(seq_dir / "img", len(_BOXES_A))
    # No list.txt written deliberately.

    ds = GOT10kDataset(str(tmp_path), split="val")
    assert len(ds) == 1
    assert ds.list_sequences()[0] == "GOT-10k_Val_Fallback"


# ---------------------------------------------------------------------------
# __getitem__ / load_sequence
# ---------------------------------------------------------------------------

def test_getitem_sequence_name(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    seq = ds[0]
    assert seq.name == "GOT-10k_Val_000001"


def test_getitem_ground_truth_shape(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    seq = ds[0]
    assert seq.ground_truth.shape == (len(_BOXES_A), 4)


def test_getitem_ground_truth_values(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    seq = ds[0]
    np.testing.assert_allclose(seq.ground_truth[0], [10.0, 20.0, 50.0, 40.0])


def test_getitem_init_bbox(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    seq = ds[0]
    assert seq.init_bbox == (10.0, 20.0, 50.0, 40.0)


def test_getitem_frame_count(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    seq = ds[0]
    assert len(seq) == len(_BOXES_A)


def test_getitem_index_error(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    with pytest.raises(IndexError):
        _ = ds[99]


def test_getitem_negative_index_error(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    with pytest.raises(IndexError):
        _ = ds[-1]


def test_load_sequence_missing_gt(got10k_root: Path) -> None:
    """FileNotFoundError when groundtruth.txt is absent."""
    split_dir = Path(got10k_root) / "val" / "GOT-10k_Val_000001"
    gt_file = split_dir / "groundtruth.txt"
    gt_file.unlink()  # remove GT

    ds = GOT10kDataset(str(got10k_root), split="val")
    with pytest.raises(FileNotFoundError, match="groundtruth.txt"):
        ds.load_sequence("GOT-10k_Val_000001")


def test_load_sequence_missing_img_dir(got10k_root: Path) -> None:
    """FileNotFoundError when img/ directory is absent."""
    import shutil
    img_dir = Path(got10k_root) / "val" / "GOT-10k_Val_000001" / "img"
    shutil.rmtree(img_dir)

    ds = GOT10kDataset(str(got10k_root), split="val")
    with pytest.raises(FileNotFoundError, match="img/"):
        ds.load_sequence("GOT-10k_Val_000001")


def test_gt_whitespace_delimiter(tmp_path: Path) -> None:
    """groundtruth.txt can use spaces instead of commas."""
    split_dir = tmp_path / "val"
    seq_dir = split_dir / "SEQ1"
    seq_dir.mkdir(parents=True)
    (seq_dir / "groundtruth.txt").write_text("10 20 50 40\n12 21 50 40\n")
    _write_frames(seq_dir / "img", 2)
    (split_dir / "list.txt").write_text("SEQ1\n")

    ds = GOT10kDataset(str(tmp_path), split="val")
    seq = ds[0]
    assert seq.ground_truth.shape == (2, 4)
    np.testing.assert_allclose(seq.ground_truth[0], [10.0, 20.0, 50.0, 40.0])


# ---------------------------------------------------------------------------
# name property
# ---------------------------------------------------------------------------

def test_name_property(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    assert ds.name == "GOT-10k-val"


def test_name_train_split(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="train")
    assert ds.name == "GOT-10k-train"


# ---------------------------------------------------------------------------
# categories property
# ---------------------------------------------------------------------------

def test_categories_returns_sorted_unique(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    cats = ds.categories
    assert cats == ["car", "dog"]


def test_categories_train_split(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="train")
    assert ds.categories == ["dog"]


def test_categories_no_meta_info(tmp_path: Path) -> None:
    """Sequences without meta_info.ini are silently excluded from categories."""
    split_dir = tmp_path / "val"
    seq_dir = split_dir / "SEQ_NO_META"
    seq_dir.mkdir(parents=True)
    _write_gt(seq_dir, _BOXES_A)
    _write_frames(seq_dir / "img", len(_BOXES_A))
    (split_dir / "list.txt").write_text("SEQ_NO_META\n")

    ds = GOT10kDataset(str(tmp_path), split="val")
    # No meta → empty class → not included in categories
    assert ds.categories == []


def test_categories_respects_max_sequences(got10k_root: Path) -> None:
    """max_sequences cap restricts which metadata is read."""
    ds = GOT10kDataset(str(got10k_root), split="val", max_sequences=1)
    # Only first sequence (dog) should contribute.
    assert ds.categories == ["dog"]


# ---------------------------------------------------------------------------
# sequence_meta
# ---------------------------------------------------------------------------

def test_sequence_meta_fields(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    meta = ds.sequence_meta("GOT-10k_Val_000001")
    assert isinstance(meta, GOT10kSequenceMeta)
    assert meta.name == "GOT-10k_Val_000001"
    assert meta.object_class == "dog"
    assert meta.motion_class == "rigid"
    assert meta.motion_adverb == "quickly"


def test_sequence_meta_non_rigid(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="train")
    meta = ds.sequence_meta("GOT-10k_Train_000001")
    assert meta.motion_class == "non-rigid"


def test_sequence_meta_missing_ini(tmp_path: Path) -> None:
    """Missing meta_info.ini yields all-empty-string fields (no error)."""
    split_dir = tmp_path / "val"
    seq_dir = split_dir / "SEQ_NO_META"
    seq_dir.mkdir(parents=True)
    _write_gt(seq_dir, _BOXES_A)
    _write_frames(seq_dir / "img", len(_BOXES_A))
    (split_dir / "list.txt").write_text("SEQ_NO_META\n")

    ds = GOT10kDataset(str(tmp_path), split="val")
    meta = ds.sequence_meta("SEQ_NO_META")
    assert meta.object_class == ""
    assert meta.motion_class == ""


def test_sequence_meta_cached(got10k_root: Path) -> None:
    """sequence_meta returns the same object on repeated calls (cache hit)."""
    ds = GOT10kDataset(str(got10k_root), split="val")
    m1 = ds.sequence_meta("GOT-10k_Val_000001")
    m2 = ds.sequence_meta("GOT-10k_Val_000001")
    assert m1 is m2


# ---------------------------------------------------------------------------
# filter_by_category
# ---------------------------------------------------------------------------

def test_filter_by_category_single(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    dogs = ds.filter_by_category("dog")
    assert len(dogs) == 1
    assert dogs.list_sequences() == ["GOT-10k_Val_000001"]


def test_filter_by_category_multiple(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    both = ds.filter_by_category("dog", "car")
    assert len(both) == 2


def test_filter_by_category_no_match(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    birds = ds.filter_by_category("bird")
    assert len(birds) == 0
    assert birds.categories == []


def test_filter_by_category_returns_loadable_sequences(got10k_root: Path) -> None:
    """Filtered dataset must still load sequences without error."""
    ds = GOT10kDataset(str(got10k_root), split="val")
    dogs = ds.filter_by_category("dog")
    seq = dogs[0]
    assert seq.name == "GOT-10k_Val_000001"
    assert seq.ground_truth.shape[1] == 4


def test_filter_by_category_empty_raises(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    with pytest.raises(ValueError, match="at least one class"):
        ds.filter_by_category()


def test_filter_preserves_split(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    dogs = ds.filter_by_category("dog")
    assert dogs.split == "val"


def test_filter_shares_meta_cache(got10k_root: Path) -> None:
    """filter_by_category shares the parent's metadata cache."""
    ds = GOT10kDataset(str(got10k_root), split="val")
    # Warm cache on parent
    _ = ds.sequence_meta("GOT-10k_Val_000001")
    dogs = ds.filter_by_category("dog")
    # Cache should be shared — already populated
    assert "GOT-10k_Val_000001" in dogs._meta_cache


def test_filter_chained(got10k_root: Path) -> None:
    """Filtering an already-filtered view further narrows results."""
    ds = GOT10kDataset(str(got10k_root), split="val")
    both = ds.filter_by_category("dog", "car")
    dogs_only = both.filter_by_category("dog")
    assert len(dogs_only) == 1


# ---------------------------------------------------------------------------
# _parse_meta_info (unit test)
# ---------------------------------------------------------------------------

def test_parse_meta_info(tmp_path: Path) -> None:
    ini = tmp_path / "meta_info.ini"
    ini.write_text(
        "[DEFAULT]\n"
        "object_class_name = airplane\n"
        "motion_class = rigid\n"
        "major_class = vehicle\n"
        "root_class = object\n"
        "motion_adverb = smoothly\n"
    )
    meta = _parse_meta_info(ini, "SEQ1")
    assert meta.object_class == "airplane"
    assert meta.motion_class == "rigid"
    assert meta.major_class == "vehicle"
    assert meta.root_class == "object"
    assert meta.motion_adverb == "smoothly"
    assert meta.name == "SEQ1"


def test_parse_meta_info_missing_keys(tmp_path: Path) -> None:
    """Partial ini files produce empty strings for absent keys."""
    ini = tmp_path / "meta_info.ini"
    ini.write_text("[DEFAULT]\nobject_class_name = dog\n")
    meta = _parse_meta_info(ini, "SEQ2")
    assert meta.object_class == "dog"
    assert meta.motion_class == ""
    assert meta.motion_adverb == ""


def test_parse_meta_info_str(tmp_path: Path) -> None:
    ini = tmp_path / "meta_info.ini"
    ini.write_text("[DEFAULT]\nobject_class_name = cat\nmotion_class = non-rigid\n")
    meta = _parse_meta_info(ini, "CAT_SEQ")
    s = str(meta)
    assert "CAT_SEQ" in s
    assert "cat" in s


# ---------------------------------------------------------------------------
# GOT10kSequenceMeta dataclass
# ---------------------------------------------------------------------------

def test_sequence_meta_str(got10k_root: Path) -> None:
    ds = GOT10kDataset(str(got10k_root), split="val")
    meta = ds.sequence_meta("GOT-10k_Val_000001")
    s = str(meta)
    assert "GOT-10k_Val_000001" in s
    assert "dog" in s
