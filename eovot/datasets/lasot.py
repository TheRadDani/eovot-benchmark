"""LaSOT dataset loader for EOVOT.

LaSOT (Large-scale Single Object Tracking) is a large-scale, long-term
tracking benchmark with 1,400 sequences across 70 object categories
(20 sequences per category), each averaging 2,500 frames.

Dataset directory layout::

    LaSOT/
    ├── airplane/
    │   ├── airplane-1/
    │   │   ├── img/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   ├── groundtruth.txt       # x,y,w,h — one box per line
    │   │   ├── full_occlusion.txt    # 0/1 per frame (optional)
    │   │   └── out_of_view.txt       # 0/1 per frame (optional)
    │   └── airplane-2/
    │       └── ...
    ├── basketball/
    │   └── ...
    └── ...

Only sequences listed in ``testing_set.txt`` (if present) are used when
``split="test"``; otherwise all discovered sequences are loaded.

Reference:
    Fan et al., "LaSOT: A High-quality Large-scale Single Object Tracking
    Benchmark." CVPR 2019. Extended version in IJCV 2021.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, BBox, Sequence

# Canonical LaSOT test-split category list (70 categories, 280 sequences).
# Used as a fallback when testing_set.txt is absent.
_LASOT_TEST_CATEGORIES = [
    "airplane", "basketball", "bear", "bicycle", "bird", "boat", "book",
    "bottle", "bus", "car", "cat", "cattle", "chameleon", "coin", "crab",
    "crocodile", "cup", "deer", "dog", "drone", "electricfan", "elephant",
    "flag", "fox", "frog", "gametarget", "gecko", "giraffe", "goldfish",
    "gorilla", "guitar", "hand", "hat", "helmet", "hippo", "horse", "kangaroo",
    "kite", "leopard", "licenseplate", "lion", "lizard", "microphone", "monkey",
    "motorcycle", "mouse", "person", "pig", "pool", "rabbit", "racing",
    "robot", "sepia", "shark", "sheep", "skateboard", "spider", "squirrel",
    "surfboard", "swing", "tank", "tiger", "toaster", "train", "truck",
    "turtle", "umbrella", "volleyball", "yoyo", "zebra",
]

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


class LaSOTDataset(BaseDataset):
    """Dataset loader for LaSOT (train / test splits).

    Args:
        root: Path to the LaSOT root directory containing one sub-folder
            per object category (e.g. ``airplane/``, ``car/``, …).
        split: ``"train"`` to load all sequences not in the test set, or
            ``"test"`` to load only test sequences.  Default: ``"test"``.
        categories: Optional list of category names to restrict loading.
            Useful for partial evaluations (e.g. ``["car", "person"]``).
        max_sequences: Optional cap on the total number of sequences returned.
        exclude_occluded: If ``True``, sequences with a full-occlusion ratio
            above *occlusion_threshold* are skipped.  Default: ``False``.
        occlusion_threshold: Fraction of occluded frames above which a
            sequence is excluded when *exclude_occluded* is ``True``.

    Example::

        # Evaluate only on 'car' and 'airplane' categories
        dataset = LaSOTDataset(
            "/data/LaSOT",
            split="test",
            categories=["car", "airplane"],
        )
        for seq in dataset:
            print(seq.name, len(seq))
    """

    SPLITS = ("train", "test")

    def __init__(
        self,
        root: str,
        split: str = "test",
        categories: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
        exclude_occluded: bool = False,
        occlusion_threshold: float = 0.5,
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS!r}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.categories = categories
        self.max_sequences = max_sequences
        self.exclude_occluded = exclude_occluded
        self.occlusion_threshold = occlusion_threshold

        self._entries: Optional[List[Tuple[str, Path]]] = None  # (seq_name, seq_dir)

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._get_entries())

    def __getitem__(self, idx: int) -> Sequence:
        entries = self._get_entries()
        if idx < 0 or idx >= len(entries):
            raise IndexError(f"Index {idx} out of range for dataset with {len(entries)} sequences")
        seq_name, seq_dir = entries[idx]
        return self._load_sequence(seq_name, seq_dir)

    @property
    def name(self) -> str:
        return f"LaSOT-{self.split}"

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def list_sequences(self) -> List[str]:
        """Return all sequence names in the selected split."""
        return [name for name, _ in self._get_entries()]

    def load_sequence(self, seq_name: str) -> Sequence:
        """Load a sequence by its full name (e.g. ``"airplane-1"``).

        Args:
            seq_name: Sequence name in the format ``"<category>-<index>"``.

        Returns:
            :class:`~eovot.datasets.base.Sequence` with lazy frame loading.

        Raises:
            KeyError: If *seq_name* is not found in the current split.
            FileNotFoundError: If the sequence directory or GT file is missing.
        """
        entries = self._get_entries()
        lookup = {name: d for name, d in entries}
        if seq_name not in lookup:
            raise KeyError(f"Sequence '{seq_name}' not found in {self.name}")
        return self._load_sequence(seq_name, lookup[seq_name])

    def category_stats(self) -> Dict[str, int]:
        """Return a mapping of category → number of sequences."""
        stats: Dict[str, int] = {}
        for name, _ in self._get_entries():
            cat = name.rsplit("-", 1)[0]
            stats[cat] = stats.get(cat, 0) + 1
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_entries(self) -> List[Tuple[str, Path]]:
        if self._entries is not None:
            return self._entries

        test_names = self._load_test_split_names()
        cat_filter = set(self.categories) if self.categories else None

        entries: List[Tuple[str, Path]] = []

        if not self.root.is_dir():
            raise FileNotFoundError(f"LaSOT root not found: {self.root}")

        for cat_dir in sorted(self.root.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat = cat_dir.name
            if cat_filter and cat not in cat_filter:
                continue

            for seq_dir in sorted(cat_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                seq_name = seq_dir.name
                if not (seq_dir / "groundtruth.txt").exists():
                    continue
                if not (seq_dir / "img").is_dir():
                    continue

                in_test = seq_name in test_names
                if self.split == "test" and not in_test:
                    continue
                if self.split == "train" and in_test:
                    continue

                if self.exclude_occluded:
                    occ_path = seq_dir / "full_occlusion.txt"
                    if occ_path.exists() and self._occlusion_ratio(occ_path) > self.occlusion_threshold:
                        continue

                entries.append((seq_name, seq_dir))

        if self.max_sequences is not None:
            entries = entries[: self.max_sequences]

        self._entries = entries
        return self._entries

    def _load_test_split_names(self) -> set:
        """Return the set of sequence names belonging to the test split.

        Reads ``testing_set.txt`` from the root if present; otherwise falls
        back to the built-in list of canonical LaSOT test categories
        (last 4 sequences per category, i.e. sequences 17–20).
        """
        testing_file = self.root / "testing_set.txt"
        if testing_file.exists():
            with open(testing_file) as fh:
                return {ln.strip() for ln in fh if ln.strip()}

        # Fallback: use the canonical test categories and sequences 17–20.
        test_names: set = set()
        for cat in _LASOT_TEST_CATEGORIES:
            for idx in range(1, 21):
                seq_name = f"{cat}-{idx}"
                candidate = self.root / cat / seq_name
                if candidate.is_dir():
                    test_names.add(seq_name)
        return test_names

    def _load_sequence(self, seq_name: str, seq_dir: Path) -> Sequence:
        gt_path = seq_dir / "groundtruth.txt"
        gt = self._parse_groundtruth(gt_path)

        img_dir = seq_dir / "img"
        frame_paths = sorted(
            str(p) for p in img_dir.iterdir()
            if p.suffix.lower() in _IMG_EXTS
        )
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {img_dir}")

        n = min(len(frame_paths), len(gt))
        return Sequence(
            name=seq_name,
            frame_paths=frame_paths[:n],
            ground_truth=np.array(gt[:n], dtype=np.float64),
        )

    @staticmethod
    def _parse_groundtruth(gt_path: Path) -> List[BBox]:
        """Parse a LaSOT groundtruth.txt into a list of (x, y, w, h) tuples."""
        boxes: List[BBox] = []
        with open(gt_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = [
                    p for p in line.replace("\t", ",").replace(" ", ",").split(",") if p
                ]
                if len(parts) < 4:
                    continue
                x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                boxes.append((x, y, w, h))
        return boxes

    @staticmethod
    def _occlusion_ratio(occ_path: Path) -> float:
        """Return the fraction of frames marked as fully occluded."""
        with open(occ_path) as fh:
            flags = [int(ln.strip()) for ln in fh if ln.strip()]
        return sum(flags) / len(flags) if flags else 0.0
