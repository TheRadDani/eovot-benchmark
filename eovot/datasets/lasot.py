"""LaSOT dataset loader for EOVOT.

LaSOT (Large-Scale Single Object Tracking) is a long-term tracking benchmark
with 1,400 sequences across 70 object categories (20 sequences per category).
Sequences average ~2,500 frames, making it the most challenging persistence
test available for single-object trackers.

Dataset directory layout::

    LaSOT/
    ├── airplane/
    │   ├── airplane-1/
    │   │   ├── img/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   ├── groundtruth.txt      # x,y,w,h — one box per line
    │   │   ├── full_occlusion.txt   # 0/1 per frame
    │   │   └── out_of_view.txt      # 0/1 per frame
    │   └── airplane-2/
    │       └── ...
    ├── basketball/
    └── ...  (70 categories total)

Reference:
    Fan et al., "LaSOT: A High-Quality Benchmark for Large-Scale Single
    Object Tracking." CVPR 2019 / IJCV 2021.
    https://cis.temple.edu/lasot/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from .base import BaseDataset, BBox, Sequence


class LaSOTDataset(BaseDataset):
    """Dataset loader for LaSOT.

    Args:
        root: Path to the LaSOT root directory containing per-category
            subdirectories (e.g. ``airplane/``, ``basketball/`` …).
        categories: Optional list of category names to include.  If
            ``None`` all discovered categories are used.
        split: Informal split label used in reports; LaSOT does not ship
            a ``list.txt`` split file — pass ``"test"`` or ``"train"`` to
            match the subset you downloaded.  Default: ``"test"``.
        max_sequences: Optional cap on the number of sequences returned.
            Useful for quick smoke tests.

    Example::

        dataset = LaSOTDataset("/data/LaSOT", categories=["airplane", "bird"])
        print(len(dataset))            # number of sequences
        seq = dataset[0]
        print(seq.name, len(seq))      # e.g. "airplane-1  1000"
    """

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        split: str = "test",
        max_sequences: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.max_sequences = max_sequences
        if not self.root.is_dir():
            raise FileNotFoundError(f"LaSOT root not found: {root}")

        allowed: Optional[Set[str]] = set(categories) if categories else None
        self._entries: List[Dict] = self._discover(allowed)

        if max_sequences is not None:
            self._entries = self._entries[:max_sequences]

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Sequence:
        entry = self._entries[idx]
        return self._load(entry)

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def _discover(self, allowed: Optional[Set[str]]) -> List[Dict]:
        """Walk ``root/`` and collect all valid sequence directories."""
        entries: List[Dict] = []
        for cat_dir in sorted(self.root.iterdir()):
            if not cat_dir.is_dir():
                continue
            if allowed is not None and cat_dir.name not in allowed:
                continue
            for seq_dir in sorted(cat_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                gt_file = seq_dir / "groundtruth.txt"
                img_dir = seq_dir / "img"
                if gt_file.is_file() and img_dir.is_dir():
                    entries.append(
                        {
                            "name": seq_dir.name,
                            "category": cat_dir.name,
                            "seq_dir": seq_dir,
                            "gt_file": gt_file,
                            "img_dir": img_dir,
                        }
                    )
        return entries

    # ------------------------------------------------------------------
    # Private loading
    # ------------------------------------------------------------------

    def _load(self, entry: Dict) -> Sequence:
        """Load a single LaSOT sequence from a pre-discovered entry dict."""
        seq_dir: Path = entry["seq_dir"]
        gt_file: Path = entry["gt_file"]
        img_dir: Path = entry["img_dir"]

        gt_boxes = _load_groundtruth(gt_file)

        frame_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        frame_paths = sorted(frame_paths)
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {img_dir}")

        # Align to shortest of frames vs annotations.
        n = min(len(frame_paths), len(gt_boxes))
        frame_paths = frame_paths[:n]
        gt_boxes = gt_boxes[:n]

        gt_arr = np.array(gt_boxes, dtype=np.float64)
        return Sequence(
            name=entry["name"],
            frame_paths=[str(p) for p in frame_paths],
            ground_truth=gt_arr,
        )

    @property
    def name(self) -> str:
        return f"LaSOT-{self.split}"

    def categories(self) -> List[str]:
        """Return sorted list of unique category names present in this dataset."""
        return sorted({e["category"] for e in self._entries})


# ---------------------------------------------------------------------------
# Module-level helper (no class state needed)
# ---------------------------------------------------------------------------

def _load_groundtruth(gt_file: Path) -> List[BBox]:
    """Parse a LaSOT ``groundtruth.txt`` into ``(x, y, w, h)`` tuples.

    LaSOT uses comma-separated values, one box per line.  Blank lines are
    skipped; lines with fewer than 4 fields are ignored.
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
