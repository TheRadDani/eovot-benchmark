"""LaSOT dataset loader for EOVOT.

LaSOT (Large-scale Single Object Tracking) is a long-term tracking benchmark
with 1,400 sequences across 70 object categories (20 sequences per category).
Sequences average ~2,500 frames, making it significantly longer than OTB or
GOT-10k and well-suited for evaluating drift resilience on edge devices.

Dataset directory layout::

    LaSOT/
    ├── airplane/
    │   ├── airplane-1/
    │   │   ├── img/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   ├── groundtruth.txt      # x,y,w,h — one box per line, comma-sep
    │   │   ├── full_occlusion.txt   # 0/1 per frame
    │   │   └── out_of_view.txt      # 0/1 per frame
    │   ├── airplane-2/
    │   └── ...
    ├── basketball/
    └── ...

Reference:
    Fan et al., "LaSOT: A High-Quality Benchmark for Large-Scale Single
    Object Tracking." CVPR 2019. Extended version: IJCV 2021.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import BaseDataset, BBox, Sequence


class LaSOTDataset(BaseDataset):
    """Dataset loader for LaSOT.

    Args:
        root: Path to the LaSOT root directory containing one subdirectory
            per object category.
        categories: Optional list of category names to include (e.g.
            ``["airplane", "bird"]``).  When *None* all discovered categories
            are used.
        max_sequences: Optional cap on the total number of sequences returned.
            Applied after category filtering; useful for quick smoke tests.

    Example::

        # Load all categories
        dataset = LaSOTDataset("/data/LaSOT")
        print(len(dataset))         # up to 1400

        # Load only two categories, at most 10 sequences
        dataset = LaSOTDataset("/data/LaSOT", categories=["airplane", "bird"],
                               max_sequences=10)
        for seq in dataset:
            print(seq.name, len(seq))
    """

    _GT_FILENAME = "groundtruth.txt"
    _IMG_DIR = "img"

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
    ) -> None:
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"LaSOT root directory not found: {root}")

        self.root = root_path
        self.max_sequences = max_sequences
        self._entries: List[tuple] = self._discover(categories)

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Sequence:
        category, seq_name, seq_dir = self._entries[idx]
        return self._load_sequence(category, seq_name, seq_dir)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self, categories: Optional[List[str]]) -> List[tuple]:
        """Walk the root directory to enumerate (category, seq_name, seq_dir) triples."""
        if categories is not None:
            cat_dirs = sorted(
                [self.root / c for c in categories if (self.root / c).is_dir()]
            )
        else:
            cat_dirs = sorted(
                [p for p in self.root.iterdir() if p.is_dir()]
            )

        entries = []
        for cat_dir in cat_dirs:
            category = cat_dir.name
            for seq_dir in sorted(cat_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                gt_file = seq_dir / self._GT_FILENAME
                img_dir = seq_dir / self._IMG_DIR
                if gt_file.is_file() and img_dir.is_dir():
                    entries.append((category, seq_dir.name, seq_dir))

        if self.max_sequences is not None:
            entries = entries[: self.max_sequences]
        return entries

    # ------------------------------------------------------------------
    # Sequence loading
    # ------------------------------------------------------------------

    def _load_sequence(self, category: str, seq_name: str, seq_dir: Path) -> Sequence:
        """Load a single LaSOT sequence.

        Args:
            category: Category name (e.g. ``"airplane"``).
            seq_name: Sequence folder name (e.g. ``"airplane-1"``).
            seq_dir: Full path to the sequence directory.

        Returns:
            :class:`~eovot.datasets.base.Sequence` with frame paths and GT
            boxes aligned to the same length.

        Raises:
            FileNotFoundError: If ``groundtruth.txt`` or ``img/`` are missing.
        """
        gt_file = seq_dir / self._GT_FILENAME
        if not gt_file.is_file():
            raise FileNotFoundError(
                f"groundtruth.txt not found for sequence '{seq_name}' at {gt_file}"
            )

        img_dir = seq_dir / self._IMG_DIR
        if not img_dir.is_dir():
            raise FileNotFoundError(f"img/ directory not found at {img_dir}")

        gt_boxes = self._load_groundtruth(gt_file)

        frame_paths = sorted(
            [
                str(p)
                for p in img_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        if not frame_paths:
            raise FileNotFoundError(f"No JPEG/PNG frames found in {img_dir}")

        # Align frame count and GT box count (may differ by one at boundaries).
        n = min(len(frame_paths), len(gt_boxes))
        frame_paths = frame_paths[:n]
        gt_array = np.array(gt_boxes[:n], dtype=np.float64)

        return Sequence(
            name=f"{category}/{seq_name}",
            frame_paths=frame_paths,
            ground_truth=gt_array,
        )

    # ------------------------------------------------------------------
    # Ground-truth parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _load_groundtruth(gt_file: Path) -> List[BBox]:
        """Parse ``groundtruth.txt`` into a list of ``(x, y, w, h)`` tuples.

        LaSOT uses comma-separated values; this parser also tolerates
        whitespace-delimited files for robustness.
        """
        boxes: List[BBox] = []
        with open(gt_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # Normalise to comma-separated, then split
                parts = [
                    p
                    for p in line.replace("\t", ",").replace(" ", ",").split(",")
                    if p
                ]
                if len(parts) < 4:
                    continue
                x, y, w, h = (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )
                boxes.append((x, y, w, h))
        return boxes

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @property
    def categories(self) -> List[str]:
        """Sorted list of unique category names present in this dataset view."""
        seen: dict = {}
        for category, _, _ in self._entries:
            seen[category] = True
        return list(seen)

    def sequences_for_category(self, category: str) -> List[str]:
        """Return all sequence names belonging to *category*."""
        return [
            seq_name
            for cat, seq_name, _ in self._entries
            if cat == category
        ]
