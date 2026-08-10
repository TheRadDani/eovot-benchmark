"""Adaptive frame-skipping tracker wrapper for edge deployment.

Implements the *dynamic inference* strategy described in EOVOT's roadmap:
rather than running the underlying tracker on every frame, the wrapper
monitors prediction stability and skips expensive updates on frames where
the target is moving smoothly and predictably — substituting a lightweight
constant-velocity extrapolation instead.

This enables direct benchmarking of the accuracy-latency trade-off that
arises from adaptive inference, a technique used in production edge-tracking
pipelines (drones, robotics, surveillance) to stay within compute budgets.

Design
------
Confidence score
    Computed from the normalised standard deviation of recent centre
    displacements (position jitter).  Low jitter → target moves smoothly
    → high confidence that constant-velocity prediction will be accurate
    enough for the next frame.

    confidence = 1 / (1 + position_jitter)

    where ``position_jitter = std(||Δcenter||) / mean_box_diagonal`` over
    the last ``window_size`` frames.

Skip decision
    A frame is skipped (constant-velocity prediction used) when:

    1. ``confidence >= confidence_threshold``
    2. ``consecutive_skips < max_consecutive_skips``
    3. Enough history is available (at least ``window_size`` frames)

Constant-velocity prediction
    On skipped frames the predicted box is extrapolated from the velocity
    (mean displacement) of the recent window.  Position is updated; size
    is kept constant.

Metrics exposed
    ``skip_count`` and ``skip_rate`` are stored on the tracker object
    after each sequence so they can be logged alongside the standard
    EOVOT profiling metrics to measure the accuracy-efficiency trade-off.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive import AdaptiveTracker

    base   = MOSSETracker()
    tracker = AdaptiveTracker(base, confidence_threshold=0.80, max_consecutive_skips=3)

    # Use identically to any other BaseTracker
    tracker.initialize(first_frame, init_bbox)
    for frame in remaining_frames:
        bbox = tracker.update(frame)

    print(f"Skip rate: {tracker.skip_rate:.1%}")
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox


class AdaptiveTracker(BaseTracker):
    """Wraps any :class:`BaseTracker` with adaptive frame-skipping.

    On frames where the underlying tracker's confidence is high (i.e. the
    target moves smoothly and steadily), the full tracker update is skipped
    and replaced by constant-velocity prediction.  This reduces per-frame
    computation on resource-constrained edge hardware at the cost of a
    small, controlled accuracy reduction.

    Args:
        tracker:              Underlying :class:`BaseTracker` to wrap.
        confidence_threshold: Minimum confidence in ``[0, 1]`` required
            to enable skipping.  Higher values mean skipping only happens
            when the target moves very smoothly.  Default: ``0.80``.
        max_consecutive_skips: Maximum number of consecutive frames that
            may be skipped before a full tracker update is forced.
            Prevents drift accumulation on long smooth segments.
            Default: ``3``.
        window_size:          Number of recent predictions used to estimate
            motion stability.  Must be >= 2.  Default: ``8``.

    Attributes:
        skip_count (int):  Total frames skipped since last :meth:`initialize`.
        total_updates (int): Total frames processed (including skips).
        skip_rate (float): ``skip_count / total_updates`` after a run.

    Example::

        from eovot.trackers.kcf import KCFTracker
        from eovot.trackers.adaptive import AdaptiveTracker

        adaptive = AdaptiveTracker(
            KCFTracker(),
            confidence_threshold=0.75,
            max_consecutive_skips=4,
        )
        adaptive.initialize(frame0, init_box)
        for frame in frames[1:]:
            bbox = adaptive.update(frame)
        print(f"Skipped {adaptive.skip_count}/{adaptive.total_updates} frames "
              f"({adaptive.skip_rate:.1%})")
    """

    def __init__(
        self,
        tracker: BaseTracker,
        confidence_threshold: float = 0.80,
        max_consecutive_skips: int = 3,
        window_size: int = 8,
    ) -> None:
        if not isinstance(tracker, BaseTracker):
            raise TypeError(
                f"tracker must be a BaseTracker instance, got {type(tracker).__name__}"
            )
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )
        if max_consecutive_skips < 0:
            raise ValueError(
                f"max_consecutive_skips must be >= 0, got {max_consecutive_skips}"
            )
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")

        super().__init__(
            name=f"Adaptive({tracker.name},conf={confidence_threshold:.2f},skip={max_consecutive_skips})"
        )
        self._tracker = tracker
        self.confidence_threshold = confidence_threshold
        self.max_consecutive_skips = max_consecutive_skips
        self.window_size = window_size

        # Rolling prediction history for motion-stability estimation.
        self._history: Deque[BBox] = deque(maxlen=window_size)
        self._consecutive_skips: int = 0

        # Public metrics — populated during a run.
        self.skip_count: int = 0
        self.total_updates: int = 0

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the underlying tracker and reset adaptive state.

        Args:
            frame: First frame as a ``(H, W, 3)`` BGR uint8 array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._history.clear()
        self._history.append(bbox)
        self._consecutive_skips = 0
        self.skip_count = 0
        self.total_updates = 0

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location in the current frame.

        Runs the underlying tracker if confidence is below threshold or the
        consecutive-skip limit is reached; otherwise uses constant-velocity
        prediction.

        Args:
            frame: Current frame as a ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if not self._history:
            raise RuntimeError(
                "AdaptiveTracker is not initialised. Call initialize() first."
            )

        self.total_updates += 1

        should_skip = (
            len(self._history) >= self.window_size
            and self._confidence() >= self.confidence_threshold
            and self._consecutive_skips < self.max_consecutive_skips
        )

        if should_skip:
            bbox = self._constant_velocity_prediction()
            self._consecutive_skips += 1
            self.skip_count += 1
        else:
            bbox = self._tracker.update(frame)
            self._consecutive_skips = 0

        self._history.append(bbox)
        return bbox

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def skip_rate(self) -> float:
        """Fraction of update frames that were skipped (0 to 1)."""
        if self.total_updates == 0:
            return 0.0
        return self.skip_count / self.total_updates

    @property
    def underlying_tracker(self) -> BaseTracker:
        """The wrapped :class:`BaseTracker` instance."""
        return self._tracker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _confidence(self) -> float:
        """Estimate prediction confidence from recent motion stability.

        Returns a value in ``(0, 1]``.  High values indicate the target has
        been moving smoothly in recent frames, making constant-velocity
        extrapolation reliable.

        The confidence degrades as position jitter (normalised std of
        centre displacements) increases:

            confidence = 1 / (1 + position_jitter)

        where ``position_jitter = std(||Δcenter||) / mean_box_diagonal``.
        """
        boxes = np.array(self._history, dtype=np.float64)   # (K, 4)
        centers = boxes[:, :2] + boxes[:, 2:] / 2.0         # (K, 2)
        displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)  # (K-1,)

        diagonals = np.sqrt(boxes[:, 2] ** 2 + boxes[:, 3] ** 2)
        mean_diag = float(diagonals.mean())

        if mean_diag < 1e-6 or len(displacements) == 0:
            return 0.0

        jitter = float(displacements.std()) / mean_diag
        return 1.0 / (1.0 + jitter)

    def _constant_velocity_prediction(self) -> BBox:
        """Extrapolate the next box using constant-velocity assumption.

        Computes the mean velocity (pixels/frame) from the recent history
        window and adds it to the last known position.  Box size is
        preserved (no scale change predicted).

        Returns:
            Extrapolated bounding box ``(x, y, w, h)``.
        """
        boxes = np.array(self._history, dtype=np.float64)   # (K, 4)
        centers = boxes[:, :2] + boxes[:, 2:] / 2.0         # (K, 2) — (cx, cy)
        # Mean velocity over the window
        velocity = np.diff(centers, axis=0).mean(axis=0)    # (2,) — (vx, vy)

        last = boxes[-1]
        last_cx = last[0] + last[2] / 2.0
        last_cy = last[1] + last[3] / 2.0

        new_cx = last_cx + velocity[0]
        new_cy = last_cy + velocity[1]
        w, h = float(last[2]), float(last[3])

        return (new_cx - w / 2.0, new_cy - h / 2.0, w, h)
