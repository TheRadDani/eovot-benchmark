"""VideoFileDataset — load arbitrary video files as EOVOT benchmark sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

from .base import BaseDataset, BBox


class VideoFileSequence:
    """A benchmark sequence backed by an arbitrary video file.

    Exposes the same duck-typed interface as :class:`~eovot.datasets.base.Sequence`
    (``name``, ``init_bbox``, ``ground_truth``, ``__len__``, ``__iter__``), so it
    works directly with :class:`~eovot.benchmark.engine.BenchmarkEngine` without
    any changes to the engine.

    Parameters
    ----------
    video_path:
        Path to a video file (any format supported by ``cv2.VideoCapture``).
    init_bbox:
        ``(x, y, w, h)`` bounding box giving the object location in frame 0.
    name:
        Human-readable label; defaults to the video file stem.
    ground_truth:
        Optional array of shape ``(N, 4)`` with per-frame GT boxes ``(x,y,w,h)``.
        When ``None``, a placeholder filled with *init_bbox* for every frame is
        generated on first access so BenchmarkEngine can still compute metrics.
    max_frames:
        Hard cap on the number of frames loaded (applied before ground_truth len).
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        init_bbox: BBox,
        name: Optional[str] = None,
        ground_truth: Optional[np.ndarray] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        self._video_path = Path(video_path)
        self._init_bbox: BBox = tuple(init_bbox)  # type: ignore[assignment]
        self.name: str = name if name is not None else self._video_path.stem
        self._gt: Optional[np.ndarray] = ground_truth
        self._max_frames = max_frames
        self._length: Optional[int] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def video_path(self) -> Path:
        return self._video_path

    @property
    def init_bbox(self) -> BBox:
        return self._init_bbox

    @property
    def ground_truth(self) -> np.ndarray:
        """Per-frame GT boxes ``(N, 4)``; placeholder when not supplied."""
        if self._gt is None:
            n = len(self)
            x, y, w, h = self._init_bbox
            self._gt = np.tile(
                np.array([x, y, w, h], dtype=np.float64), (max(n, 1), 1)
            )
        return self._gt

    def __len__(self) -> int:
        if self._length is not None:
            return self._length
        try:
            import cv2
            cap = cv2.VideoCapture(str(self._video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception:
            total = 0
        if self._gt is not None:
            total = min(total, len(self._gt))
        if self._max_frames is not None:
            total = min(total, self._max_frames)
        self._length = max(total, 0)
        return self._length

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield BGR frames as ``(H, W, 3)`` uint8 arrays."""
        import cv2

        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self._video_path}")
        try:
            limit = self._max_frames
            if self._gt is not None:
                gt_limit = len(self._gt)
                limit = gt_limit if limit is None else min(limit, gt_limit)

            frame_idx = 0
            while True:
                if limit is not None and frame_idx >= limit:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
                frame_idx += 1
        finally:
            cap.release()

    def __repr__(self) -> str:
        return (
            f"VideoFileSequence(name={self.name!r}, "
            f"path={str(self._video_path)!r}, "
            f"max_frames={self._max_frames!r})"
        )


class VideoFileDataset(BaseDataset):
    """A benchmark dataset composed of arbitrary video files.

    Each element is a :class:`VideoFileSequence`. Use the class-method
    constructors for the most common cases.

    Parameters
    ----------
    sequences:
        Pre-built list of :class:`VideoFileSequence` objects.
    """

    def __init__(self, sequences: List[VideoFileSequence]) -> None:
        self._sequences = list(sequences)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> VideoFileSequence:
        return self._sequences[idx]

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        entries: List[
            Union[
                Tuple[Union[str, Path], BBox],
                Tuple[Union[str, Path], BBox, np.ndarray],
            ]
        ],
        max_frames: Optional[int] = None,
    ) -> "VideoFileDataset":
        """Build a dataset from ``(video_path, init_bbox[, ground_truth])`` tuples.

        Parameters
        ----------
        entries:
            Each element is either ``(path, bbox)`` or ``(path, bbox, gt_array)``.
        max_frames:
            Frame cap applied uniformly to every sequence.
        """
        sequences = []
        for entry in entries:
            if len(entry) == 2:
                path, bbox = entry  # type: ignore[misc]
                gt = None
            else:
                path, bbox, gt = entry  # type: ignore[misc]
            sequences.append(
                VideoFileSequence(path, bbox, ground_truth=gt, max_frames=max_frames)
            )
        return cls(sequences)

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        init_bbox: BBox,
        extensions: Optional[List[str]] = None,
        max_frames: Optional[int] = None,
    ) -> "VideoFileDataset":
        """Build a dataset from all video files found in a directory.

        Parameters
        ----------
        directory:
            Folder to scan (non-recursive).
        init_bbox:
            Initial bounding box applied to every discovered video.
        extensions:
            Lower-case extensions without leading dot to match.
            Defaults to ``["mp4", "avi", "mov", "mkv"]``.
        max_frames:
            Frame cap applied uniformly to every sequence.
        """
        if extensions is None:
            extensions = ["mp4", "avi", "mov", "mkv"]
        exts = {f".{e.lstrip('.')}" for e in extensions}
        directory = Path(directory)
        video_files = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        return cls(
            [
                VideoFileSequence(p, init_bbox, max_frames=max_frames)
                for p in video_files
            ]
        )
