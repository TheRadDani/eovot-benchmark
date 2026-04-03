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
    │   │   └── meta_info.ini
    │   └── ...
    ├── val/
    └── test/

Reference:
    Huang et al., "GOT-10k: A Large High-Diversity Benchmark for Generic
    Object Tracking in the Wild." IEEE TPAMI 2021.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import BaseDataset, BBox, Sequence


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

    def _load_sequence(self, seq_name: str) -> Sequence:
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

    @property
    def name(self) -> str:
        return f"GOT-10k-{self.split}"

        return Sequence(name=seq_name, frame_paths=frame_paths_str, ground_truth=gt_array)

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
