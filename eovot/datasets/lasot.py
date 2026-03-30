"""LaSOT dataset loader for EOVOT.

LaSOT (Large-scale Single Object Tracking) is a high-quality benchmark with
1,400 sequences across 70 object categories (20 sequences per category).
Each sequence has at least 1,000 frames, making it the longest-sequence
benchmark in common use.

Dataset directory layout::

    LaSOT/
    ├── testing_set.txt          # optional: one sequence name per line (test split)
    ├── airplane/
    │   ├── airplane-1/
    │   │   ├── img/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   ├── groundtruth.txt      # x,y,w,h — one box per line (comma-separated)
    │   │   ├── full_occlusion.txt   # 0/1 per frame (optional)
    │   │   └── out_of_view.txt      # 0/1 per frame (optional)
    │   └── airplane-2/
    │       └── ...
    ├── basketball/
    └── ...

Split handling
--------------
If a ``testing_set.txt`` file exists at the root of the dataset, its contents
are used to separate train/test sequences.  Otherwise all discovered sequences
are returned regardless of the ``split`` argument.

Reference:
    Fan et al., "LaSOT: A High-quality Benchmark for Large-scale Single Object
    Tracking." CVPR 2019.  Extended journal version: IJCV 2021.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Set

import numpy as np

from .base import BaseDataset, BBox, Sequence

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class LaSOTDataset(BaseDataset):
    """Dataset loader for LaSOT (train / test / all splits).

    Args:
        root: Path to the LaSOT root directory containing per-category
            subdirectories (e.g. ``airplane/``, ``basketball/``, …).
        split: One of ``"train"``, ``"test"``, or ``"all"``.
            Requires a ``testing_set.txt`` in *root* to distinguish train
            from test.  Falls back to ``"all"`` if the file is absent.
            Default: ``"test"``.
        max_sequences: Optional cap on the number of sequences discovered.
            Useful for quick smoke-tests. Default: ``None`` (no cap).

    Example::

        dataset = LaSOTDataset("/data/LaSOT", split="test")
        print(len(dataset))   # 280 if testing_set.txt is present
        seq = dataset[0]
        print(seq.name, len(seq))
    """

    SPLITS = ("train", "test", "all")

    def __init__(
        self,
        root: str,
        split: str = "test",
        max_sequences: Optional[int] = None,
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS!r}, got {split!r}")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"LaSOT root directory not found: {root}")
        self.root = Path(root)
        self.split = split
        self.max_sequences = max_sequences
        self._seq_dirs: List[Path] = self._discover()

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._seq_dirs)

    def __getitem__(self, idx: int) -> Sequence:
        if idx < 0 or idx >= len(self._seq_dirs):
            raise IndexError(f"Sequence index {idx} out of range [0, {len(self._seq_dirs)})")
        return self._load_sequence(self._seq_dirs[idx])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"LaSOT-{self.split}"

    @property
    def categories(self) -> List[str]:
        """Sorted list of object categories present in this split."""
        return sorted({d.parent.name for d in self._seq_dirs})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover(self) -> List[Path]:
        """Walk the root directory and collect valid sequence paths."""
        all_seq_dirs: List[Path] = []
        for category_dir in sorted(self.root.iterdir()):
            if not category_dir.is_dir() or category_dir.name.startswith("."):
                continue
            for seq_dir in sorted(category_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                gt_file = seq_dir / "groundtruth.txt"
                img_dir = seq_dir / "img"
                if gt_file.is_file() and img_dir.is_dir():
                    all_seq_dirs.append(seq_dir)

        # Apply train/test split filtering when testing_set.txt is available.
        if self.split != "all":
            test_names = self._load_testing_set()
            if test_names is not None:
                if self.split == "test":
                    all_seq_dirs = [d for d in all_seq_dirs if d.name in test_names]
                else:  # "train"
                    all_seq_dirs = [d for d in all_seq_dirs if d.name not in test_names]

        if self.max_sequences is not None:
            all_seq_dirs = all_seq_dirs[: self.max_sequences]

        return all_seq_dirs

    def _load_testing_set(self) -> Optional[Set[str]]:
        """Return the set of test-split sequence names from ``testing_set.txt``.

        Returns ``None`` if the file does not exist (caller falls back to
        returning all sequences).
        """
        testing_file = self.root / "testing_set.txt"
        if not testing_file.is_file():
            return None
        with open(testing_file) as fh:
            return {ln.strip() for ln in fh if ln.strip()}

    def _load_sequence(self, seq_dir: Path) -> Sequence:
        """Build a :class:`~eovot.datasets.base.Sequence` from *seq_dir*."""
        gt_file = seq_dir / "groundtruth.txt"
        img_dir = seq_dir / "img"

        gt_boxes = _load_groundtruth(gt_file)

        frame_paths = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in _IMG_EXTS
        )
        if not frame_paths:
            raise FileNotFoundError(f"No image frames found in {img_dir}")

        # Align frame count and GT length (LaSOT is usually exact, but guard anyway).
        n = min(len(frame_paths), len(gt_boxes))

        return Sequence(
            name=seq_dir.name,
            frame_paths=[str(p) for p in frame_paths[:n]],
            ground_truth=np.array(gt_boxes[:n], dtype=np.float64),
        )


def _load_groundtruth(gt_file: Path) -> List[BBox]:
    """Parse ``groundtruth.txt`` into a list of ``(x, y, w, h)`` tuples.

    Handles both comma-separated and whitespace-delimited files.
    Lines with fewer than 4 values are skipped.
    """
    boxes: List[BBox] = []
    with open(gt_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Normalise separators: tabs and spaces → commas, then split.
            parts = [p for p in line.replace("\t", ",").replace(" ", ",").split(",") if p]
            if len(parts) < 4:
                continue
            x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            boxes.append((x, y, w, h))
    return boxes
