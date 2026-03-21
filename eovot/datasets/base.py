"""Dataset abstractions for EOVOT.

Provides:
- :class:`Sequence` — a named sequence of frames + ground-truth boxes.
- :class:`BaseDataset` — abstract interface for all dataset loaders.
- :class:`OTBDataset` — concrete loader for OTB-style dataset layouts.

OTB directory layout expected by :class:`OTBDataset`::

    <root>/
      <sequence_name>/
        img/
          0001.jpg
          0002.jpg
          ...
        groundtruth_rect.txt   # one row per frame: x y w h

The ground-truth file may use comma or whitespace delimiters.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple

import cv2
import numpy as np

BBox = Tuple[float, float, float, float]
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class Sequence:
    """A single tracking sequence: ordered frames and per-frame ground truth."""

    def __init__(self, name: str, frame_paths: List[str], ground_truth: np.ndarray) -> None:
        if ground_truth.ndim != 2 or ground_truth.shape[1] != 4:
            raise ValueError(f"ground_truth must be shape (N, 4), got {ground_truth.shape}")
        self.name = name
        self._frame_paths = frame_paths
        self.ground_truth = ground_truth

    def __len__(self) -> int:
        return len(self._frame_paths)

    def __repr__(self) -> str:
        return f"Sequence(name={self.name!r}, frames={len(self)})"

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield BGR frames as ``(H, W, 3)`` uint8 arrays."""
        for path in self._frame_paths:
            frame = cv2.imread(path)
            if frame is None:
                raise FileNotFoundError(f"Cannot read frame: {path}")
            yield frame

    @property
    def init_bbox(self) -> BBox:
        """Ground-truth bounding box for the first frame."""
        return tuple(self.ground_truth[0])  # type: ignore[return-value]


class BaseDataset(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Sequence: ...

    def __iter__(self) -> Iterator[Sequence]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sequences={len(self)})"


class OTBDataset(BaseDataset):
    """Loader for OTB-style tracking datasets.

    Args:
        root: Path to the dataset root directory.

    Example::

        dataset = OTBDataset("/data/OTB100")
        for seq in dataset:
            print(seq.name, len(seq))
    """

    _GT_FILENAME = "groundtruth_rect.txt"
    _IMG_DIR = "img"

    def __init__(self, root: str) -> None:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")
        self.root = root
        self._entries: List[Tuple[str, str]] = self._discover()

    def _discover(self) -> List[Tuple[str, str]]:
        entries = []
        for name in sorted(os.listdir(self.root)):
            seq_dir = os.path.join(self.root, name)
            gt_path = os.path.join(seq_dir, self._GT_FILENAME)
            img_dir = os.path.join(seq_dir, self._IMG_DIR)
            if os.path.isdir(seq_dir) and os.path.isfile(gt_path) and os.path.isdir(img_dir):
                entries.append((name, seq_dir))
        return entries

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Sequence:
        name, seq_dir = self._entries[idx]
        gt_path = os.path.join(seq_dir, self._GT_FILENAME)
        img_dir = os.path.join(seq_dir, self._IMG_DIR)
        try:
            gt = np.loadtxt(gt_path, delimiter=",")
        except ValueError:
            gt = np.loadtxt(gt_path)
        if gt.ndim == 1:
            gt = gt[np.newaxis, :]
        frame_paths = sorted(
            [
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if os.path.splitext(f)[1].lower() in _IMG_EXTS
            ]
        )
        return Sequence(name=name, frame_paths=frame_paths, ground_truth=gt)
