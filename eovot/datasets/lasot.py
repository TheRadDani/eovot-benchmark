"""LaSOT dataset loader for EOVOT.

LaSOT (Large-scale Single Object Tracking) is a high-quality benchmark with
1,400 sequences covering 70 object categories and an average sequence length
of ~2,500 frames — making it the standard benchmark for long-term tracking
evaluation.

Reference:
    Fan et al. "LaSOT: A High-quality Large-scale Single Object Tracking
    Benchmark." CVPR 2019.  https://vision.cs.stonybrook.edu/~lasot/

Expected directory layout::

    <root>/
      <category>/              # e.g.  airplane, ball, bear, ...
        <category>-<index>/    # e.g.  airplane-1, airplane-2, ...
          img/
            00000001.jpg
            ...
          groundtruth.txt      # x,y,w,h per line (comma-separated)
          full_occlusion.txt   # 0/1 per frame (optional)
          out_of_view.txt      # 0/1 per frame (optional)

Split files (``train.txt`` / ``test.txt``) may optionally reside in *root*;
when present, only the listed sequence names are loaded for the requested
split.  The standard test split contains 280 sequences (70 categories × 4);
the train split contains 1,120 sequences (70 categories × 16).
"""

from __future__ import annotations

import os
from typing import List, Optional, Set, Tuple

import numpy as np

from .base import BaseDataset, Sequence

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_VALID_SPLITS = {"train", "test", "all"}


class LaSOTDataset(BaseDataset):
    """Loader for the LaSOT tracking benchmark.

    Args:
        root: Path to the LaSOT root directory, which contains per-category
            sub-directories and optionally ``train.txt`` / ``test.txt``.
        split: Subset to load — ``"train"``, ``"test"``, or ``"all"``
            (default).  When a ``<split>.txt`` file exists in *root* it is
            used to filter sequences; otherwise all discovered sequences are
            returned regardless of *split*.
        max_sequences: Optional cap on the number of sequences loaded.
            Useful for quick smoke-tests without a full dataset download.

    Example::

        dataset = LaSOTDataset("/data/LaSOT", split="test")
        for seq in dataset:
            print(seq.name, len(seq))

        # Index-based access
        seq = dataset[0]
        for frame in seq:
            bbox = tracker.update(frame)
    """

    _GT_FILENAME = "groundtruth.txt"
    _IMG_DIR = "img"

    def __init__(
        self,
        root: str,
        split: str = "all",
        max_sequences: Optional[int] = None,
    ) -> None:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"LaSOT root not found: {root}")
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}"
            )
        self.root = root
        self.split = split
        self._allowed: Optional[Set[str]] = self._load_split_list(split)
        self._entries: List[Tuple[str, str]] = self._discover(max_sequences)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_split_list(self, split: str) -> Optional[Set[str]]:
        """Return the set of allowed sequence names for *split*.

        Returns ``None`` when ``split == "all"`` or no split file is found,
        meaning all discovered sequences pass through the filter.
        """
        if split == "all":
            return None
        split_file = os.path.join(self.root, f"{split}.txt")
        if not os.path.isfile(split_file):
            return None
        with open(split_file) as fh:
            return {line.strip() for line in fh if line.strip()}

    def _discover(self, max_sequences: Optional[int]) -> List[Tuple[str, str]]:
        """Walk *root* and collect valid ``(sequence_name, sequence_dir)`` pairs.

        A directory is considered a valid sequence when it contains both a
        ``groundtruth.txt`` file and an ``img/`` sub-directory.
        """
        entries: List[Tuple[str, str]] = []

        for category in sorted(os.listdir(self.root)):
            cat_dir = os.path.join(self.root, category)
            if not os.path.isdir(cat_dir):
                continue  # skip split .txt files and other non-directories

            for seq_name in sorted(os.listdir(cat_dir)):
                seq_dir = os.path.join(cat_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    continue

                gt_path = os.path.join(seq_dir, self._GT_FILENAME)
                img_dir = os.path.join(seq_dir, self._IMG_DIR)
                if not (os.path.isfile(gt_path) and os.path.isdir(img_dir)):
                    continue  # not a valid sequence directory

                if self._allowed is not None and seq_name not in self._allowed:
                    continue  # filtered by split list

                entries.append((seq_name, seq_dir))
                if max_sequences is not None and len(entries) >= max_sequences:
                    return entries

        return entries

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Sequence:
        name, seq_dir = self._entries[idx]
        gt_path = os.path.join(seq_dir, self._GT_FILENAME)
        img_dir = os.path.join(seq_dir, self._IMG_DIR)

        # LaSOT ground-truth is always comma-delimited x,y,w,h
        try:
            gt = np.loadtxt(gt_path, delimiter=",")
        except ValueError:
            gt = np.loadtxt(gt_path)
        if gt.ndim == 1:
            gt = gt[np.newaxis, :]

        frame_paths = sorted(
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in _IMG_EXTS
        )
        return Sequence(name=name, frame_paths=frame_paths, ground_truth=gt)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def list_categories(self) -> List[str]:
        """Return a sorted list of unique object categories in the loaded split.

        LaSOT sequence names follow the ``<category>-<index>`` convention, so
        the category label is extracted by stripping the trailing ``-<index>``.
        """
        categories: Set[str] = set()
        for name, _ in self._entries:
            parts = name.rsplit("-", 1)
            categories.add(parts[0] if len(parts) == 2 and parts[1].isdigit() else name)
        return sorted(categories)

    def sequences_for_category(self, category: str) -> List[Sequence]:
        """Return all loaded sequences belonging to *category*.

        Args:
            category: Object category string (e.g. ``"airplane"``).

        Returns:
            List of :class:`~eovot.datasets.base.Sequence` objects whose
            names begin with ``<category>-``.
        """
        prefix = f"{category}-"
        return [
            self[i]
            for i, (name, _) in enumerate(self._entries)
            if name.startswith(prefix) or name == category
        ]
