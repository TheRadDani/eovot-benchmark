"""LaSOT dataset loader for EOVOT.

LaSOT (Large-scale Single Object Tracking) is a high-quality, large-scale
benchmark with 1,400 sequences across 70 object categories (20 sequences/category).
Each sequence is annotated with per-frame bounding boxes, full-occlusion labels,
and out-of-view labels.

Dataset directory layout::

    LaSOT/
    ├── testing_set.txt             # 280 test sequence names, one per line
    ├── training_set.txt            # 1120 training sequence names, one per line
    ├── airplane/
    │   ├── airplane-1/
    │   │   ├── img/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   ├── groundtruth.txt     # x,y,w,h — comma-separated, one per line
    │   │   ├── full_occlusion.txt  # 0/1 per frame
    │   │   └── out_of_view.txt     # 0/1 per frame
    │   └── airplane-2/
    │       └── ...
    └── ...

Reference:
    Fan et al., "LaSOT: A High-quality Benchmark for Large-scale Single Object
    Tracking." CVPR 2019. https://arxiv.org/abs/1809.07845
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import BaseDataset, BBox, Sequence


class LaSOTDataset(BaseDataset):
    """Dataset loader for LaSOT (train / test splits).

    Args:
        root: Path to the LaSOT root directory containing per-class subdirectories.
        split: One of ``"train"`` or ``"test"``. Uses ``training_set.txt`` /
            ``testing_set.txt`` when present; falls back to loading all sequences.
        max_sequences: Optional upper limit on the number of sequences evaluated.
            Useful for quick smoke tests without the full 1,400-sequence dataset.

    Example::

        dataset = LaSOTDataset("/data/LaSOT", split="test")
        for seq in dataset:
            print(seq.name, len(seq))
    """

    SPLITS = ("train", "test")
    _SPLIT_FILES = {"train": "training_set.txt", "test": "testing_set.txt"}

    def __init__(
        self,
        root: str,
        split: str = "test",
        max_sequences: Optional[int] = None,
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS!r}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.max_sequences = max_sequences
        self._seq_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def list_sequences(self) -> List[str]:
        """Return sequence names for the selected split.

        Reads ``testing_set.txt`` / ``training_set.txt`` when present;
        otherwise enumerates all ``<class>/<class>-<id>/`` directories.
        """
        if self._seq_names is not None:
            return self._seq_names

        split_file = self.root / self._SPLIT_FILES[self.split]
        if split_file.exists():
            with open(split_file) as fh:
                names = [ln.strip() for ln in fh if ln.strip()]
        else:
            names = self._discover_all_sequences()

        if self.max_sequences is not None:
            names = names[: self.max_sequences]
        self._seq_names = names
        return self._seq_names

    def __len__(self) -> int:
        return len(self.list_sequences())

    def __getitem__(self, idx: int) -> Sequence:
        seq_names = self.list_sequences()
        return self.load_sequence(seq_names[idx])

    def load_sequence(self, seq_name: str) -> Sequence:
        """Load a single LaSOT sequence.

        Args:
            seq_name: Sequence identifier of the form ``"<class>-<id>"``
                (e.g. ``"airplane-1"``).

        Returns:
            :class:`~eovot.datasets.base.Sequence` with frame paths and
            ground-truth boxes aligned to the same length.

        Raises:
            FileNotFoundError: If ``groundtruth.txt`` or ``img/`` are missing.
        """
        # Sequence directories are nested: <root>/<class>/<class>-<id>/
        class_name = seq_name.rsplit("-", 1)[0]
        seq_dir = self.root / class_name / seq_name

        gt_file = seq_dir / "groundtruth.txt"
        if not gt_file.exists():
            raise FileNotFoundError(
                f"groundtruth.txt not found for sequence '{seq_name}' at {gt_file}"
            )
        gt_boxes = self._load_groundtruth(gt_file)

        img_dir = seq_dir / "img"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"img/ directory not found at {img_dir}")

        frame_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        frame_paths = sorted(frame_paths)
        if not frame_paths:
            raise FileNotFoundError(f"No JPEG/PNG frames found in {img_dir}")

        # Align frame count and GT length (some sequences differ by one frame).
        n = min(len(frame_paths), len(gt_boxes))
        frame_paths = frame_paths[:n]
        gt_boxes = gt_boxes[:n]

        return Sequence(
            name=seq_name,
            frame_paths=[str(p) for p in frame_paths],
            ground_truth=np.array(gt_boxes, dtype=np.float64),
        )

    @property
    def name(self) -> str:
        return f"LaSOT-{self.split}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_all_sequences(self) -> List[str]:
        """Walk the root directory to find all valid sequence folders."""
        names: List[str] = []
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            for seq_dir in sorted(class_dir.iterdir()):
                if seq_dir.is_dir() and (seq_dir / "groundtruth.txt").exists():
                    names.append(seq_dir.name)
        return names

    @staticmethod
    def _load_groundtruth(gt_file: Path) -> List[BBox]:
        """Parse ``groundtruth.txt`` into ``(x, y, w, h)`` tuples.

        Handles comma-separated and whitespace-delimited files; skips blank lines.
        """
        boxes: List[BBox] = []
        with open(gt_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = [
                    p for p in line.replace("\t", ",").replace(" ", ",").split(",") if p
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
