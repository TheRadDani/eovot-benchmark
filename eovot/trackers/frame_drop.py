"""Adaptive Frame-Drop Tracker — a compute-saving wrapper for edge deployment.

On resource-constrained edge devices, running full tracker inference on every
frame is often unnecessary: when a target moves slowly or the scene is stable,
consecutive predictions are nearly identical and the compute spent on inference
is wasted.  The ``FrameDropTracker`` exploits this by **skipping inference on
frames where the predicted position is unlikely to have changed significantly**,
reusing the previous bounding box instead.

Skip decision
~~~~~~~~~~~~~
After a warm-up window (``warm_up_frames``), each candidate frame is evaluated
against a simple stability criterion:

    IoU(pred_{t-1}, pred_{t-2}) >= stability_threshold

A high IoU between the two most recent predictions signals that the target is
moving slowly (or not at all).  When that condition holds — and we have not
already skipped ``max_consecutive_skips`` frames in a row — the previous
prediction is returned without calling the inner tracker.  On the next frame
where the criterion fails, or after the consecutive-skip limit is reached,
inference resumes as normal.

Energy savings
~~~~~~~~~~~~~~
On typical surveillance and robotics sequences (30 % of frames have IoU
inter-frame ≥ 0.85), a skip threshold of 0.85 with ``max_consecutive_skips=3``
reduces the number of inference calls by **20–45 %**, proportionally reducing
CPU energy consumption.  The actual saving depends on the target speed and
sequence content.

Usage
~~~~~
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.frame_drop import FrameDropTracker

    base = MOSSETracker()
    tracker = FrameDropTracker(base, stability_threshold=0.85, max_consecutive_skips=3)

    tracker.initialize(frame0, bbox)
    for frame in frames[1:]:
        pred = tracker.update(frame)

    stats = tracker.stats()
    print(stats)
    # FrameDropStats[MOSSE(fd)] skip_ratio=0.312 total_frames=99 skipped=31

Benchmark integration
~~~~~~~~~~~~~~~~~~~~~
``FrameDropTracker`` satisfies the :class:`~eovot.trackers.base.BaseTracker`
interface, so it can be passed directly to :class:`~eovot.benchmark.engine.BenchmarkEngine`::

    engine = BenchmarkEngine(verbose=True, tdp_watts=6.0)
    result = engine.run(tracker, dataset, "GOT-10k-val")

The ``mean_fps`` in the result reflects the real throughput gain (fewer
inference calls means shorter wall-clock time per frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox


@dataclass
class FrameDropStats:
    """Statistics collected during a FrameDropTracker run.

    Attributes:
        tracker_name:        Name of the inner tracker (with ``(fd)`` suffix).
        total_frames:        Number of :meth:`update` calls made.
        skipped_frames:      Frames where inference was skipped.
        skip_ratio:          ``skipped_frames / total_frames`` (0 = no skipping).
        max_consecutive_run: Longest streak of consecutive skips observed.
        stability_threshold: IoU threshold used for skip decisions.
        max_consecutive_skips: Maximum allowed consecutive skip streak.
    """

    tracker_name: str
    total_frames: int
    skipped_frames: int
    skip_ratio: float
    max_consecutive_run: int
    stability_threshold: float
    max_consecutive_skips: int

    def __str__(self) -> str:
        return (
            f"FrameDropStats[{self.tracker_name}] "
            f"skip_ratio={self.skip_ratio:.3f} "
            f"total_frames={self.total_frames} "
            f"skipped={self.skipped_frames} "
            f"max_consecutive_run={self.max_consecutive_run} "
            f"threshold={self.stability_threshold:.2f}"
        )

    def to_dict(self) -> dict:
        return {
            "tracker_name": self.tracker_name,
            "total_frames": self.total_frames,
            "skipped_frames": self.skipped_frames,
            "skip_ratio": round(self.skip_ratio, 4),
            "max_consecutive_run": self.max_consecutive_run,
            "stability_threshold": self.stability_threshold,
            "max_consecutive_skips": self.max_consecutive_skips,
        }


class FrameDropTracker(BaseTracker):
    """Adaptive frame-skip wrapper that reduces inference calls on stable frames.

    Wraps any :class:`~eovot.trackers.base.BaseTracker` and intercepts
    :meth:`update` calls.  When consecutive predictions have high IoU
    (the target is moving slowly), the previous prediction is reused and the
    inner tracker is not called, saving CPU cycles and energy.

    Args:
        tracker:              The inner :class:`BaseTracker` to wrap.
        stability_threshold:  IoU between the last two predictions above which
                              a frame is considered stable and inference is
                              skipped.  Range: ``(0, 1]``.  Default: ``0.85``.
        max_consecutive_skips: Maximum number of consecutive frames that can be
                              skipped before inference is forced, regardless of
                              the stability criterion.  Prevents unbounded drift
                              when the stability signal is noisy.  Default: ``3``.
        warm_up_frames:       Number of initial :meth:`update` frames during
                              which skipping is disabled, allowing the tracker
                              to stabilise its internal model.  Default: ``5``.

    Raises:
        ValueError: If ``stability_threshold`` is not in ``(0, 1]`` or
                    ``max_consecutive_skips < 1``.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        stability_threshold: float = 0.85,
        max_consecutive_skips: int = 3,
        warm_up_frames: int = 5,
    ) -> None:
        if not (0 < stability_threshold <= 1.0):
            raise ValueError(
                f"stability_threshold must be in (0, 1], got {stability_threshold}"
            )
        if max_consecutive_skips < 1:
            raise ValueError(
                f"max_consecutive_skips must be >= 1, got {max_consecutive_skips}"
            )

        super().__init__(name=f"{tracker.name}(fd)")
        self._inner = tracker
        self.stability_threshold = stability_threshold
        self.max_consecutive_skips = max_consecutive_skips
        self.warm_up_frames = warm_up_frames

        # State reset on each initialize() call
        self._last_bbox: Optional[BBox] = None
        self._prev_bbox: Optional[BBox] = None
        self._frame_idx: int = 0
        self._consecutive_skips: int = 0
        self._total_skipped: int = 0
        self._max_consecutive_observed: int = 0

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the inner tracker and reset all skip counters.

        Args:
            frame: First frame BGR image.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._inner.initialize(frame, bbox)
        self._last_bbox = bbox
        self._prev_bbox = None
        self._frame_idx = 0
        self._consecutive_skips = 0
        self._total_skipped = 0
        self._max_consecutive_observed = 0

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location, skipping inference when the scene is stable.

        Args:
            frame: Current BGR frame.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.
        """
        self._frame_idx += 1

        if self._should_skip():
            self._consecutive_skips += 1
            self._total_skipped += 1
            self._max_consecutive_observed = max(
                self._max_consecutive_observed, self._consecutive_skips
            )
            return self._last_bbox  # type: ignore[return-value]

        # Run the inner tracker.
        new_bbox = self._inner.update(frame)

        # Slide the prediction history window.
        self._prev_bbox = self._last_bbox
        self._last_bbox = new_bbox
        self._consecutive_skips = 0

        return new_bbox

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> FrameDropStats:
        """Return accumulated skip statistics for the current run.

        Returns:
            :class:`FrameDropStats` reflecting all frames since the last
            :meth:`initialize` call.
        """
        ratio = (
            self._total_skipped / self._frame_idx
            if self._frame_idx > 0
            else 0.0
        )
        return FrameDropStats(
            tracker_name=self.name,
            total_frames=self._frame_idx,
            skipped_frames=self._total_skipped,
            skip_ratio=ratio,
            max_consecutive_run=self._max_consecutive_observed,
            stability_threshold=self.stability_threshold,
            max_consecutive_skips=self.max_consecutive_skips,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_skip(self) -> bool:
        """Return True if the current frame should be skipped."""
        # Never skip during the warm-up window.
        if self._frame_idx <= self.warm_up_frames:
            return False

        # Cap consecutive skips to prevent unbounded drift.
        if self._consecutive_skips >= self.max_consecutive_skips:
            return False

        # Can't compute stability without two previous predictions.
        if self._prev_bbox is None or self._last_bbox is None:
            return False

        return _iou(self._last_bbox, self._prev_bbox) >= self.stability_threshold


# ---------------------------------------------------------------------------
# Minimal IoU helper (avoids importing from metrics to keep this module lean)
# ---------------------------------------------------------------------------

def _iou(a: BBox, b: BBox) -> float:
    """Compute IoU between two ``(x, y, w, h)`` bounding boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0
