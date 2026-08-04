"""GOT-10k dataset loader for EOVOT.

GOT-10k (Generic Object Tracking 10k) is a large-scale, high-diversity
tracking benchmark with 10,000 training, 180 validation, and 180 test
sequences covering 563 object classes and 87 motion patterns.

Dataset directory layout::

    GOT-10k/
    ├── train/
    │   ├── list.txt                       # sequence names, one per line
    │   ├── GOT-10k_Train_000001/
    │   │   ├── img/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   ├── groundtruth.txt            # x,y,w,h — one box per line
    │   │   ├── absence.label              # 0/1 per frame (optional)
    │   │   └── meta_info.ini              # object class, motion patterns
    │   └── ...
    ├── val/
    └── test/

Reference:
    Huang et al., "GOT-10k: A Large High-Diversity Benchmark for Generic
    Object Tracking in the Wild." IEEE TPAMI 2021.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence as TypingSequence

import numpy as np

from .base import BaseDataset, BBox, Sequence


@dataclass
class GOT10kSequenceMeta:
    """Metadata parsed from a GOT-10k ``meta_info.ini`` file.

    All fields are optional at the file level; missing keys are stored as
    empty strings so callers can always read attributes without guarding.

    Attributes:
        name:             Sequence folder name (e.g. ``"GOT-10k_Val_000001"``).
        object_class:     Fine-grained object class (e.g. ``"dog"``).
        motion_class:     Motion rigidity — ``"rigid"`` or ``"non-rigid"``.
        major_class:      Coarser semantic grouping (e.g. ``"animal"``).
        root_class:       Broadest semantic root (e.g. ``"GOT-10k"`` root node).
        motion_adverb:    Motion manner (e.g. ``"quickly"``, ``"slowly"``).
    """

    name: str
    object_class: str = ""
    motion_class: str = ""
    major_class: str = ""
    root_class: str = ""
    motion_adverb: str = ""

    def __str__(self) -> str:
        return (
            f"GOT10kSequenceMeta[{self.name}] "
            f"class={self.object_class!r}  motion={self.motion_class!r}"
        )


def _parse_meta_info(ini_path: Path, seq_name: str) -> GOT10kSequenceMeta:
    """Parse a GOT-10k ``meta_info.ini`` file into :class:`GOT10kSequenceMeta`.

    Uses :mod:`configparser` with ``[DEFAULT]`` section; fields absent from
    the file default to empty strings so callers never see KeyError.

    Args:
        ini_path: Path to ``meta_info.ini`` inside a sequence directory.
        seq_name: Sequence name embedded in the returned dataclass.

    Returns:
        Populated :class:`GOT10kSequenceMeta` instance.
    """
    cfg = configparser.ConfigParser()
    cfg.read(str(ini_path), encoding="utf-8")
    # GOT-10k stores everything under [DEFAULT]; configparser reads it via
    # cfg.defaults() or cfg['DEFAULT'].  We use .get with a fallback.
    def _get(key: str) -> str:
        val = cfg.defaults().get(key, "").strip()
        # Also check a bare [DEFAULT] section explicitly written as such.
        if not val and cfg.has_section("DEFAULT"):  # pragma: no cover
            val = cfg.get("DEFAULT", key, fallback="").strip()
        return val

    return GOT10kSequenceMeta(
        name=seq_name,
        object_class=_get("object_class_name"),
        motion_class=_get("motion_class"),
        major_class=_get("major_class"),
        root_class=_get("root_class"),
        motion_adverb=_get("motion_adverb"),
    )


class GOT10kDataset(BaseDataset):
    """Dataset loader for GOT-10k (train / val / test splits).

    Implements the :class:`~eovot.datasets.base.BaseDataset` interface via
    ``__len__`` and ``__getitem__``, so it works directly with
    :class:`~eovot.benchmark.engine.BenchmarkEngine`.

    Frames are loaded lazily on iteration (not pre-loaded into RAM), keeping
    memory usage constant regardless of dataset size.

    Args:
        root: Path to the GOT-10k root directory.  Must contain
            ``train/``, ``val/``, or ``test/`` subdirectories.
        split: Dataset split — one of ``"train"``, ``"val"``, ``"test"``.
            Default: ``"val"``.
        max_sequences: Optional upper limit on the number of sequences
            returned.  Useful for quick smoke tests without downloading
            the full dataset.

    Example::

        dataset = GOT10kDataset("/data/GOT-10k", split="val")
        print(len(dataset))
        seq = dataset[0]
        print(seq.name, len(seq))

        # Filter to a single object category:
        dogs = dataset.filter_by_category("dog")
        print(len(dogs), "dog sequences")
    """

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root: str,
        split: str = "val",
        max_sequences: Optional[int] = None,
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS!r}, got {split!r}")
        self.root = root
        self.split = split
        self.max_sequences = max_sequences
        self._split_dir = Path(root) / split
        self._seq_names: Optional[List[str]] = None
        # Cache for sequence metadata, populated lazily.
        self._meta_cache: Dict[str, GOT10kSequenceMeta] = {}

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.list_sequences())

    def __getitem__(self, idx: int) -> Sequence:
        names = self.list_sequences()
        if idx < 0 or idx >= len(names):
            raise IndexError(f"Sequence index {idx} out of range [0, {len(names)})")
        return self.load_sequence(names[idx])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"GOT-10k-{self.split}"

    @property
    def categories(self) -> List[str]:
        """Sorted list of unique object-class names present in this split.

        Reads ``meta_info.ini`` for every sequence in
        :meth:`list_sequences` and extracts the ``object_class_name`` field.
        Sequences whose ``meta_info.ini`` is missing or whose class name is
        empty are skipped.

        Returns:
            Sorted list of unique class-name strings (e.g.
            ``["airplane", "dog", "person"]``).

        Example::

            dataset = GOT10kDataset("/data/GOT-10k", split="val")
            print(dataset.categories[:5])
        """
        seen = set()
        for seq_name in self.list_sequences():
            meta = self.sequence_meta(seq_name)
            if meta.object_class:
                seen.add(meta.object_class)
        return sorted(seen)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_sequences(self) -> List[str]:
        """Return the list of sequence names for this split.

        Reads ``list.txt`` when present; falls back to enumerating
        subdirectories that contain an ``img/`` folder.
        """
        if self._seq_names is not None:
            return self._seq_names

        list_file = self._split_dir / "list.txt"
        if list_file.exists():
            with open(list_file) as fh:
                names = [ln.strip() for ln in fh if ln.strip()]
        else:
            names = sorted(
                d.name
                for d in self._split_dir.iterdir()
                if d.is_dir() and (d / "img").is_dir()
            )

        if self.max_sequences is not None:
            names = names[: self.max_sequences]
        self._seq_names = names
        return self._seq_names

    def sequence_meta(self, seq_name: str) -> GOT10kSequenceMeta:
        """Return :class:`GOT10kSequenceMeta` for a single sequence.

        Parses ``meta_info.ini`` inside the sequence directory (if present)
        and caches the result so repeated calls are free.  When the file
        does not exist all metadata fields are returned as empty strings.

        Args:
            seq_name: Sequence folder name (e.g. ``"GOT-10k_Val_000001"``).

        Returns:
            :class:`GOT10kSequenceMeta` with fields from ``meta_info.ini``.
        """
        if seq_name in self._meta_cache:
            return self._meta_cache[seq_name]

        ini_path = self._split_dir / seq_name / "meta_info.ini"
        if ini_path.is_file():
            meta = _parse_meta_info(ini_path, seq_name)
        else:
            meta = GOT10kSequenceMeta(name=seq_name)
        self._meta_cache[seq_name] = meta
        return meta

    def filter_by_category(self, *classes: str) -> "GOT10kDataset":
        """Return a new :class:`GOT10kDataset` view restricted to *classes*.

        Iterates over :meth:`list_sequences`, reads each sequence's
        ``meta_info.ini`` to determine its object class, and keeps only
        sequences matching one of the requested class names.  The returned
        dataset shares the same ``root`` and ``split`` but has an independent
        (pre-filtered) sequence list.

        Args:
            *classes: One or more object-class names to keep
                (e.g. ``"dog"``  or ``"airplane", "car"``).

        Returns:
            A new :class:`GOT10kDataset` instance whose
            :meth:`list_sequences` returns only matching sequences.

        Raises:
            ValueError: If *classes* is empty.

        Example::

            dataset = GOT10kDataset("/data/GOT-10k", split="val")
            animals = dataset.filter_by_category("dog", "cat", "horse")
            print(len(animals), "animal sequences")
        """
        if not classes:
            raise ValueError("filter_by_category() requires at least one class name.")
        target = set(classes)
        filtered = [
            seq_name
            for seq_name in self.list_sequences()
            if self.sequence_meta(seq_name).object_class in target
        ]
        view = GOT10kDataset(root=self.root, split=self.split, max_sequences=None)
        # Bypass normal discovery by pre-loading the filtered list.
        view._seq_names = filtered
        # Share the metadata cache to avoid redundant ini parses.
        view._meta_cache = self._meta_cache
        return view

    def load_sequence(self, seq_name: str) -> Sequence:
        """Load a single GOT-10k sequence by name.

        Frames are referenced by path (lazy I/O) rather than pre-loaded,
        matching the :class:`~eovot.datasets.base.Sequence` contract.

        Args:
            seq_name: Sequence folder name (e.g. ``"GOT-10k_Val_000001"``).

        Returns:
            :class:`~eovot.datasets.base.Sequence` with frame paths and GT
            boxes aligned to the same length.  Frames are loaded lazily on
            iteration — no images are read into memory here.

        Raises:
            FileNotFoundError: If ``groundtruth.txt`` or ``img/`` are missing.
        """
        seq_dir = self._split_dir / seq_name

        gt_file = seq_dir / "groundtruth.txt"
        if not gt_file.exists():
            raise FileNotFoundError(
                f"groundtruth.txt not found for sequence '{seq_name}' at {gt_file}.\n"
                "Note: GOT-10k test-split annotations are withheld by the evaluation "
                "server — use split='val' for local evaluation."
            )
        gt_boxes = self._load_groundtruth(gt_file)

        img_dir = seq_dir / "img"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"img/ directory not found at {img_dir}")

        # Collect paths in chronological order; merge JPG and PNG.
        frame_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        frame_paths = sorted(frame_paths)
        if not frame_paths:
            raise FileNotFoundError(f"No JPEG/PNG frames found in {img_dir}")

        # Align frame count and GT length (some sequences may differ by one).
        n = min(len(frame_paths), len(gt_boxes))
        frame_paths = frame_paths[:n]
        gt_boxes = gt_boxes[:n]

        return Sequence(
            name=seq_name,
            frame_paths=[str(p) for p in frame_paths],
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_groundtruth(gt_file: Path) -> List[BBox]:
        """Parse ``groundtruth.txt`` into a list of ``(x, y, w, h)`` tuples.

        Handles both comma-separated and whitespace-delimited files,
        and skips blank lines.
        """
        boxes: List[BBox] = []
        with open(gt_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = [p for p in line.replace("\t", ",").replace(" ", ",").split(",") if p]
                if len(parts) < 4:
                    continue
                x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                boxes.append((x, y, w, h))
        return boxes
