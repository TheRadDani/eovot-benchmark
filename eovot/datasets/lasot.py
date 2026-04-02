"""LaSOT dataset loader for EOVOT.

LaSOT (Large-scale Single Object Tracking) is a benchmark with 1,400 sequences
across 70 object categories, featuring long-term tracking scenarios with full
occlusion and out-of-view events.

Dataset directory layout::

    LaSOT/
    ├── testing_set.txt            # optional: sequence names for the test split
    ├── airplane/
    │   ├── airplane-1/
    │   │   ├── img/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   ├── groundtruth.txt    # x,y,w,h per frame (comma-delimited)
    │   │   ├── full_occlusion.txt # 0/1 per frame
    │   │   └── out_of_view.txt    # 0/1 per frame
    │   └── airplane-2/
    │       └── ...
    ├── basketball/
    └── ...

The official train/test split is encoded in ``testing_set.txt`` at the dataset
root (280 test sequences).  When that file is absent all discovered sequences
are treated as a single pool and returned regardless of *split*.

Reference:
    Fan et al., "LaSOT: A High-quality Benchmark for Large-scale Single Object
    Tracking." CVPR 2019.  https://arxiv.org/abs/1809.07845
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import BaseDataset, BBox, Sequence


class LaSOTDataset(BaseDataset):
    """Dataset loader for LaSOT (train / test splits).

    Args:
        root: Path to the LaSOT root directory containing per-category
            subdirectories (e.g. ``airplane/``, ``basketball/`` …).
        split: One of ``"train"``, ``"test"``, or ``"all"``.
            The split is resolved via ``testing_set.txt`` in *root*.
            If the file is absent, *split* is ignored and all discovered
            sequences are returned.
        max_sequences: Optional upper limit on the number of sequences
            returned.  Useful for quick smoke tests.

    Example::

        dataset = LaSOTDataset("/data/LaSOT", split="test", max_sequences=20)
        for seq in dataset:
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
        self.root = root
        self.split = split
        self.max_sequences = max_sequences
        self._root = Path(root)
        self._sequences: Optional[List[Path]] = None

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._get_sequences())

    def __getitem__(self, idx: int) -> Sequence:
        seqs = self._get_sequences()
        if idx < 0 or idx >= len(seqs):
            raise IndexError(f"Sequence index {idx} out of range (0–{len(seqs) - 1})")
        return self._load_sequence(seqs[idx])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_sequences(self) -> List[Path]:
        """Discover and filter sequence directories, caching the result."""
        if self._sequences is not None:
            return self._sequences

        # Collect all <category>/<seq_name>/ directories that have a GT file.
        all_seqs: List[Path] = []
        for cat_dir in sorted(self._root.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name.startswith("."):
                continue
            for seq_dir in sorted(cat_dir.iterdir()):
                if seq_dir.is_dir() and (seq_dir / "groundtruth.txt").exists():
                    all_seqs.append(seq_dir)

        # Apply the official train/test split if the split file exists.
        if self.split != "all":
            split_file = self._root / "testing_set.txt"
            if split_file.exists():
                with open(split_file) as fh:
                    test_names = {ln.strip() for ln in fh if ln.strip()}
                if self.split == "test":
                    all_seqs = [s for s in all_seqs if s.name in test_names]
                else:  # "train"
                    all_seqs = [s for s in all_seqs if s.name not in test_names]
            # If testing_set.txt is absent we silently return all sequences.

        if self.max_sequences is not None:
            all_seqs = all_seqs[: self.max_sequences]

        self._sequences = all_seqs
        return self._sequences

    def _load_sequence(self, seq_dir: Path) -> Sequence:
        """Load a single LaSOT sequence from *seq_dir*.

        Args:
            seq_dir: Absolute path to the sequence directory (e.g.
                ``<root>/airplane/airplane-1/``).

        Returns:
            :class:`~eovot.datasets.base.Sequence` with frame paths and GT
            boxes aligned to the same length.

        Raises:
            FileNotFoundError: If ``img/`` or ``groundtruth.txt`` are absent.
        """
        gt_file = seq_dir / "groundtruth.txt"
        if not gt_file.exists():
            raise FileNotFoundError(f"groundtruth.txt not found at {gt_file}")

        img_dir = seq_dir / "img"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"img/ directory not found at {img_dir}")

        frame_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        frame_paths = sorted(frame_paths)
        if not frame_paths:
            raise FileNotFoundError(f"No JPEG/PNG frames found in {img_dir}")

        gt_boxes = self._load_groundtruth(gt_file)
        n = min(len(frame_paths), len(gt_boxes))
        frame_path_strs = [str(p) for p in frame_paths[:n]]
        gt_array = np.array(gt_boxes[:n], dtype=np.float64)

        return Sequence(name=seq_dir.name, frame_paths=frame_path_strs, ground_truth=gt_array)

    @staticmethod
    def _load_groundtruth(gt_file: Path) -> List[BBox]:
        """Parse ``groundtruth.txt`` into a list of ``(x, y, w, h)`` tuples.

        Handles comma-separated and whitespace-delimited files and skips
        blank lines.
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
