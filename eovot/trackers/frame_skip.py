"""Adaptive frame-skip wrapper for any BaseTracker.

On resource-constrained edge devices, processing every video frame at full
rate is often infeasible.  ``FrameSkipTracker`` wraps any ``BaseTracker`` and
invokes the underlying tracker only on every ``skip_rate``-th frame, while
interpolating bounding boxes for the intermediate frames using a constant-
velocity linear predictor.

This design enables **systematic study of the accuracy–efficiency trade-off**
for a fixed tracker algorithm without re-implementing the tracker:

- ``skip_rate=1`` → standard evaluation, no skipping (baseline).
- ``skip_rate=2`` → tracker runs at half the frame rate (~2× compute saving).
- ``skip_rate=N`` → tracker runs at 1/N the frame rate (~N× compute saving).

The wrapper is fully transparent to ``BenchmarkEngine`` — it satisfies the
``BaseTracker`` interface and can be substituted anywhere a tracker is used.

Interpolation model
-------------------
Skipped frames receive bounding-box predictions from a **constant-velocity**
extrapolator: the velocity is estimated as the displacement between the two
most recently processed frames, then projected forward proportionally to the
number of frames since the last real update.  Box size is linearly
interpolated from the same two reference frames.  This simple model is
intentionally lightweight — the primary goal is a fast, deterministic baseline
that approximates where the tracker *would* have placed the box.

Usage::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.frame_skip import FrameSkipTracker

    base    = MOSSETracker()
    tracker = FrameSkipTracker(base, skip_rate=3)  # invoke MOSSE every 3rd frame

    engine = BenchmarkEngine(verbose=True)
    result = engine.run(tracker, dataset, dataset_name="OTB100")
    # tracker.name → "MOSSETracker[skip=3]"
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseTracker, BBox


class FrameSkipTracker(BaseTracker):
    """Wraps a BaseTracker to process only every *skip_rate*-th frame.

    Frames that are not processed by the underlying tracker receive bounding-
    box predictions extrapolated from the last two processed frames using a
    constant-velocity model.

    Args:
        tracker: Any concrete :class:`~eovot.trackers.base.BaseTracker`.
        skip_rate: Number of frames between two consecutive tracker invocations.
            ``1`` disables skipping (every frame is processed, equivalent to
            the unwrapped tracker).  Must be a positive integer.

    Raises:
        ValueError: If ``skip_rate < 1``.

    Attributes:
        skip_rate: The configured skip rate (read-only after construction).

    Example::

        from eovot.trackers.kcf import KCFTracker

        wrapped = FrameSkipTracker(KCFTracker(), skip_rate=2)
        # wrapped.name  →  "KCFTracker[skip=2]"
    """

    def __init__(self, tracker: BaseTracker, skip_rate: int = 2) -> None:
        if skip_rate < 1:
            raise ValueError(f"skip_rate must be >= 1, got {skip_rate!r}")
        super().__init__(name=f"{tracker.name}[skip={skip_rate}]")
        self._tracker = tracker
        self.skip_rate = skip_rate

        # Per-sequence state — reset on initialize().
        self._frame_index: int = 0
        self._last_bbox: Optional[BBox] = None   # bbox from the most recent process-frame
        self._prev_bbox: Optional[BBox] = None   # bbox from the process-frame before that
        self._last_processed_at: int = 0         # _frame_index value when last processed

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the underlying tracker and reset all skip state.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth ``(x, y, w, h)`` for the first frame.
        """
        self._tracker.initialize(frame, bbox)
        self._frame_index = 0
        self._last_bbox = bbox
        self._prev_bbox = bbox   # zero velocity at start
        self._last_processed_at = 0

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location for the next frame.

        Invokes the underlying tracker when ``frame_index % skip_rate == 0``;
        otherwise extrapolates from the last two processed-frame positions.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted ``(x, y, w, h)`` bounding box.
        """
        self._frame_index += 1

        if self._frame_index % self.skip_rate == 0:
            bbox = self._tracker.update(frame)
            self._prev_bbox = self._last_bbox
            self._last_bbox = bbox
            self._last_processed_at = self._frame_index
        else:
            bbox = self._extrapolate()

        return bbox

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extrapolate(self) -> BBox:
        """Project the last known position forward with constant velocity.

        Velocity is the per-frame displacement between ``_prev_bbox`` and
        ``_last_bbox``.  For a frame that is *k* steps after the last
        processed frame, the predicted position is offset by *k × velocity*.
        Box size is linearly interpolated (clamped to a minimum of 1 px).
        """
        assert self._last_bbox is not None and self._prev_bbox is not None

        frames_ahead = self._frame_index - self._last_processed_at

        lx, ly, lw, lh = self._last_bbox
        px, py, pw, ph = self._prev_bbox

        # Constant-velocity position update.
        vx = lx - px
        vy = ly - py
        ex = lx + frames_ahead * vx
        ey = ly + frames_ahead * vy

        # Linear size interpolation clamped above zero.
        alpha = min(frames_ahead / max(self.skip_rate, 1), 1.0)
        ew = max(lw + alpha * (lw - pw), 1.0)
        eh = max(lh + alpha * (lh - ph), 1.0)

        return (ex, ey, ew, eh)

    def __repr__(self) -> str:
        return (
            f"FrameSkipTracker(tracker={self._tracker!r}, "
            f"skip_rate={self.skip_rate})"
        )
