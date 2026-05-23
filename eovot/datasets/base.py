"""Dataset abstractions for EOVOT.

Provides:
- :class:`Sequence` — a named sequence of frames + ground-truth boxes.
- :class:`BaseDataset` — abstract interface for all dataset loaders.

The full-featured OTB loader (OTB-50/100, attribute filtering) lives in
:mod:`eovot.datasets.otb`.  ``OTBDataset`` is re-exported from this module
for backward compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple

import cv2
import numpy as np

BBox = Tuple[float, float, float, float]


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


# Re-export for backward compatibility — the canonical implementation is in otb.py.
# This import is intentionally placed at the bottom to avoid a circular dependency
# during package initialisation (otb.py imports from base.py).
from .otb import OTBDataset as OTBDataset  # noqa: E402
