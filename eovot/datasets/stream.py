"""Video stream dataset source for EOVOT benchmark.

Enables benchmarking trackers on real-world video files and live webcam feeds,
bridging the gap between offline dataset evaluation and real-world edge
deployment scenarios.

Two classes are provided:

- :class:`VideoStreamSequence` — wraps any ``cv2.VideoCapture``-compatible
  source (video file path, webcam index, RTSP URL) as a single tracking
  sequence.  An optional ground-truth file enables accuracy metrics; without
  one the sequence can still measure efficiency metrics (FPS, latency, memory).

- :class:`VideoStreamDataset` — wraps one or more video sources as a
  ``BaseDataset``, making them compatible with :class:`~eovot.benchmark.engine.BenchmarkEngine`
  without any code changes.

Typical usage — benchmark KCF on a video file::

    from eovot.datasets.stream import VideoStreamDataset
    from eovot.trackers.kcf import KCFTracker
    from eovot.benchmark.engine import BenchmarkEngine

    dataset = VideoStreamDataset.from_video(
        path="highway.mp4",
        init_bbox=(120, 45, 80, 60),  # x, y, w, h on the first frame
    )
    engine = BenchmarkEngine(verbose=True)
    result = engine.run(KCFTracker(), dataset, dataset_name="highway.mp4")
    print(result)

Webcam usage::

    dataset = VideoStreamDataset.from_webcam(
        device_index=0,
        init_bbox=(200, 150, 60, 80),
        max_frames=300,
    )

Ground-truth usage (for accuracy metrics)::

    dataset = VideoStreamDataset.from_video(
        path="car_track.mp4",
        init_bbox=(50, 30, 100, 70),
        gt_path="car_track_gt.txt",   # one row per frame: x y w h
    )

Notes
-----
- Frames are yielded lazily; the entire video is never held in memory.
- ``cv2.VideoCapture`` releases are handled automatically via ``__iter__``.
- For webcam sources the total frame count is unknown until ``max_frames``
  is exhausted or the feed closes.
- When no ground truth is provided, all IoU values are ``NaN`` and accuracy
  metrics (AUC) will be ``NaN`` as well; only efficiency metrics are valid.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional, Sequence as TypingSequence, Tuple, Union

import cv2
import numpy as np

from .base import BaseDataset, BBox, Sequence

__all__ = ["VideoStreamSequence", "VideoStreamDataset"]

# A sentinel value used to fill GT rows when no ground truth is available.
_NAN_BOX = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)


# ---------------------------------------------------------------------------
# VideoStreamSequence
# ---------------------------------------------------------------------------


class VideoStreamSequence(Sequence):
    """A single tracking sequence backed by a ``cv2.VideoCapture`` source.

    Unlike :class:`~eovot.datasets.base.Sequence`, frames are read lazily
    from the video source each time the sequence is iterated, avoiding the
    need to load the entire video into memory.

    Args:
        name: Human-readable sequence identifier.
        source: A ``cv2.VideoCapture``-compatible source — a file path
            string, an integer webcam device index, or an RTSP/HTTP URL.
        init_bbox: Initial bounding box ``(x, y, w, h)`` for the first
            frame.  This is the box provided to ``tracker.initialize()``.
        ground_truth: Optional ``(N, 4)`` array of GT boxes in ``x y w h``
            format, one row per frame.  When ``None``, all GT rows are
            filled with ``NaN`` and accuracy metrics will be ``NaN``.
        max_frames: Maximum frames to yield per iteration.  ``None`` means
            unlimited (read until the source is exhausted).  Useful for
            webcam sources where the total length is unknown.
        name_hint: Used internally by :class:`VideoStreamDataset` to
            display the source identifier in logs.

    Example::

        seq = VideoStreamSequence(
            name="car",
            source="car.mp4",
            init_bbox=(50, 30, 100, 70),
        )
        for frame in seq:
            print(frame.shape)
    """

    def __init__(
        self,
        name: str,
        source: Union[str, int],
        init_bbox: BBox,
        ground_truth: Optional[np.ndarray] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        self._source = source
        self._init_bbox_raw = tuple(float(v) for v in init_bbox)
        self._max_frames = max_frames
        self._has_gt = ground_truth is not None

        # Determine frame count for __len__: for file-based sources we can
        # query the property; for webcams it's unknown until iteration.
        self._frame_count: int = self._probe_frame_count()

        # Build ground_truth array.  The base-class Sequence.__init__
        # validates shape, so we always pass a proper (N, 4) array.
        if ground_truth is not None:
            gt = np.asarray(ground_truth, dtype=np.float64)
            if gt.ndim == 1:
                gt = gt[np.newaxis, :]
        else:
            # Fill with NaN so the engine can still run; accuracy will be NaN.
            n = self._frame_count if self._frame_count > 0 else (max_frames or 1)
            gt = np.full((n, 4), np.nan, dtype=np.float64)

        # We bypass Sequence.__init__ because we store frames lazily, not as
        # paths.  Assign attributes manually.
        self.name = name
        self._frame_paths: List[str] = []   # unused but kept for compat
        self.ground_truth = gt
        self._init_bbox_value: BBox = tuple(self._init_bbox_raw)  # type: ignore

    # ------------------------------------------------------------------
    # BaseSequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of frames in the sequence (may be approximate for webcams)."""
        if self._max_frames is not None:
            return min(self._frame_count, self._max_frames) if self._frame_count > 0 else self._max_frames
        return self._frame_count

    def __repr__(self) -> str:
        return (
            f"VideoStreamSequence(name={self.name!r}, "
            f"source={self._source!r}, "
            f"frames={len(self)}, "
            f"has_gt={self._has_gt})"
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield BGR frames as ``(H, W, 3)`` uint8 arrays.

        Opens a new ``VideoCapture`` on each iteration so the sequence can be
        iterated multiple times (e.g., once per benchmark run).
        """
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {self._source!r}.  "
                "Check that the path exists, the codec is installed, or the "
                "device index is valid."
            )
        try:
            frame_idx = 0
            while True:
                if self._max_frames is not None and frame_idx >= self._max_frames:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                yield frame
                frame_idx += 1
        finally:
            cap.release()

    @property
    def init_bbox(self) -> BBox:
        """Ground-truth bounding box for the first frame (``x, y, w, h``)."""
        return self._init_bbox_value  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _probe_frame_count(self) -> int:
        """Query total frame count from the capture backend.

        Returns ``0`` for live sources where the count is unknown.
        """
        if isinstance(self._source, int):
            return 0  # webcam: unknown length
        try:
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                return 0
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # Some codecs return -1 or 0 for unknown length.
            return max(count, 0)
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# VideoStreamDataset
# ---------------------------------------------------------------------------


class VideoStreamDataset(BaseDataset):
    """A :class:`~eovot.datasets.base.BaseDataset` backed by video sources.

    Each sequence corresponds to one video file or webcam feed.  Compatible
    with :class:`~eovot.benchmark.engine.BenchmarkEngine` without modification.

    Construct via the convenience class-methods rather than directly:

    - :meth:`from_video` — single video file
    - :meth:`from_webcam` — live webcam feed
    - :meth:`from_sources` — multiple sources at once

    Args:
        sequences: List of :class:`VideoStreamSequence` objects.
    """

    def __init__(self, sequences: List[VideoStreamSequence]) -> None:
        if not sequences:
            raise ValueError("VideoStreamDataset requires at least one sequence.")
        self._sequences = sequences

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> VideoStreamSequence:
        return self._sequences[idx]

    def __repr__(self) -> str:
        return f"VideoStreamDataset(sequences={len(self)})"

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_video(
        cls,
        path: Union[str, Path],
        init_bbox: BBox,
        name: Optional[str] = None,
        gt_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
    ) -> "VideoStreamDataset":
        """Create a dataset from a single video file.

        Args:
            path: Path to a video file (MP4, AVI, MOV, etc.).
            init_bbox: Initial bounding box ``(x, y, w, h)`` for frame 0.
            name: Sequence name shown in results.  Defaults to the filename
                without extension.
            gt_path: Optional path to a ground-truth text file.  Each row
                must contain four values ``x y w h`` (whitespace or comma
                separated).  Rows beyond the video length are silently
                ignored; missing rows are padded with ``NaN``.
            max_frames: Optional frame cap.

        Returns:
            A :class:`VideoStreamDataset` containing one sequence.

        Example::

            ds = VideoStreamDataset.from_video(
                path="drone_footage.mp4",
                init_bbox=(100, 80, 60, 50),
                gt_path="drone_footage_gt.txt",
            )
        """
        path = Path(path)
        seq_name = name or path.stem
        gt = _load_gt(gt_path) if gt_path is not None else None
        seq = VideoStreamSequence(
            name=seq_name,
            source=str(path),
            init_bbox=init_bbox,
            ground_truth=gt,
            max_frames=max_frames,
        )
        return cls([seq])

    @classmethod
    def from_webcam(
        cls,
        device_index: int = 0,
        init_bbox: BBox = (0, 0, 100, 100),
        name: Optional[str] = None,
        max_frames: int = 300,
    ) -> "VideoStreamDataset":
        """Create a dataset from a live webcam feed.

        Args:
            device_index: OpenCV camera index (``0`` for the default webcam).
            init_bbox: Initial bounding box ``(x, y, w, h)`` for frame 0.
            name: Sequence name.  Defaults to ``"webcam_{device_index}"``.
            max_frames: Frame limit.  Webcam feeds are infinite, so a cap is
                required.  Default: ``300`` frames (~10 s at 30 FPS).

        Returns:
            A :class:`VideoStreamDataset` containing one sequence.

        Example::

            ds = VideoStreamDataset.from_webcam(
                device_index=0,
                init_bbox=(160, 120, 80, 80),
                max_frames=600,
            )
        """
        seq_name = name or f"webcam_{device_index}"
        seq = VideoStreamSequence(
            name=seq_name,
            source=device_index,
            init_bbox=init_bbox,
            ground_truth=None,
            max_frames=max_frames,
        )
        return cls([seq])

    @classmethod
    def from_sources(
        cls,
        sources: List[dict],
    ) -> "VideoStreamDataset":
        """Create a multi-sequence dataset from a list of source descriptors.

        Each descriptor is a dict with the following keys:

        - ``source`` (required): file path string or webcam int.
        - ``init_bbox`` (required): ``(x, y, w, h)`` tuple for frame 0.
        - ``name`` (optional): sequence name.
        - ``gt_path`` (optional): ground-truth file path.
        - ``max_frames`` (optional): frame cap.

        Args:
            sources: List of source descriptor dicts.

        Returns:
            A :class:`VideoStreamDataset` with one sequence per descriptor.

        Example::

            ds = VideoStreamDataset.from_sources([
                {"source": "video1.mp4", "init_bbox": (50, 30, 80, 60)},
                {"source": "video2.avi", "init_bbox": (10, 20, 40, 40), "name": "clip2"},
                {"source": 0, "init_bbox": (100, 100, 50, 50), "max_frames": 200},
            ])
        """
        seqs: List[VideoStreamSequence] = []
        for i, desc in enumerate(sources):
            src = desc["source"]
            bbox: BBox = tuple(desc["init_bbox"])  # type: ignore[assignment]
            seq_name = desc.get("name") or (
                Path(str(src)).stem if isinstance(src, str) else f"webcam_{src}"
            )
            gt_path = desc.get("gt_path")
            gt = _load_gt(gt_path) if gt_path is not None else None
            max_frames = desc.get("max_frames")
            seqs.append(
                VideoStreamSequence(
                    name=seq_name,
                    source=src,
                    init_bbox=bbox,
                    ground_truth=gt,
                    max_frames=max_frames,
                )
            )
        return cls(seqs)


# ---------------------------------------------------------------------------
# Ground-truth loader
# ---------------------------------------------------------------------------


def _load_gt(path: Union[str, Path]) -> np.ndarray:
    """Load a ground-truth bounding-box file.

    Supports whitespace-separated and comma-separated formats.  Each row must
    have four columns: ``x y w h``.

    Args:
        path: Path to the GT text file.

    Returns:
        ``(N, 4)`` float64 array of bounding boxes.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    try:
        gt = np.loadtxt(str(path), delimiter=",")
    except ValueError:
        gt = np.loadtxt(str(path))
    if gt.ndim == 1:
        gt = gt[np.newaxis, :]
    if gt.shape[1] != 4:
        raise ValueError(
            f"Ground-truth file must have 4 columns (x y w h), "
            f"got {gt.shape[1]} in {path}"
        )
    return gt.astype(np.float64)
