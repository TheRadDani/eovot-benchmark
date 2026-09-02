"""Adaptive inference budget tracker for edge-deployment optimisation.

Dynamically adjusts per-frame computation by skipping tracker updates on
"easy" frames (stable motion, minimal scale change) and running the full
wrapped tracker on "hard" frames (fast motion, significant deformation).

Unlike :class:`~eovot.trackers.frame_skip.FrameSkipTracker`, which uses a
static skip interval, this tracker decides each frame whether to update or
coast on a constant-velocity prediction, based on a rolling motion complexity
estimate.  This enables a better accuracy–speed trade-off on sequences where
difficulty varies over time (e.g. a target that slows down mid-sequence).

Motion Complexity Score (C)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each frame the tracker computes a scalar complexity estimate::

    C = w_motion × ||Δcenter|| / diagonal  +  w_scale × |area_ratio − 1|

where

- ``||Δcenter||`` — L2 displacement of the predicted box center between
  consecutive frames, normalised by the box diagonal for scale invariance.
- ``area_ratio`` — ratio of consecutive predicted box areas; deviation from
  1.0 indicates scale change.
- ``w_motion``, ``w_scale`` — configurable weights (default 1.0, 0.5).

C is then exponentially smoothed (``smoothing`` weight ``α``).

Adaptive Policy
~~~~~~~~~~~~~~~
- **Enter easy mode**: smoothed C < ``easy_threshold``.  The tracker coasts
  for subsequent frames — re-using the last predicted bbox shifted by the
  rolling mean velocity — without calling the wrapped tracker at all.
- **Exit easy mode**: smoothed C ≥ ``hard_threshold``.  The tracker calls
  the wrapped tracker this frame and returns to "hard" mode.

In hard mode the tracker always calls the wrapped tracker every frame.

Statistics
~~~~~~~~~~
After evaluation, ``skip_rate``, ``n_updates``, and ``n_coasted`` quantify
the effective compute reduction achieved for the sequence.

Typical usage::

    from eovot.trackers.adaptive_budget import AdaptiveBudgetTracker
    from eovot.trackers.mosse import MOSSETracker

    base    = MOSSETracker()
    tracker = AdaptiveBudgetTracker(base, easy_threshold=0.04, hard_threshold=0.12)
    tracker.initialize(frame0, bbox0)
    for frame in frames[1:]:
        bbox = tracker.update(frame)

    print(f"skip rate : {tracker.skip_rate:.1%}")
    print(f"updates   : {tracker.n_updates}")
    print(f"coasted   : {tracker.n_coasted}")
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox


class AdaptiveBudgetTracker(BaseTracker):
    """Wrap any BaseTracker with adaptive per-frame skip based on motion complexity.

    Args:
        tracker:         Wrapped :class:`BaseTracker` instance.  Its
            ``initialize`` and ``update`` methods are called only when the
            adaptive policy decides an update is necessary.
        easy_threshold:  Smoothed complexity below which the tracker enters
            "easy mode" (coasting).  Default ``0.05``.
        hard_threshold:  Smoothed complexity at or above which the tracker
            exits easy mode and runs a full update.  Must be ≥
            ``easy_threshold``.  Default ``0.15``.
        smoothing:       Exponential smoothing weight ``α ∈ (0, 1]`` for the
            complexity estimate.  Larger values respond faster to complexity
            spikes; smaller values give a more stable signal.  Default
            ``0.3``.
        w_motion:        Weight for the motion term in the complexity formula.
            Default ``1.0``.
        w_scale:         Weight for the scale-change term.  Default ``0.5``.
        history_len:     Length of the rolling bbox history used for motion
            estimation (not used for decisions, available for debugging).
            Default ``8``.
        name:            Optional override for the tracker name shown in
            benchmark reports.  Defaults to ``"Adaptive(<wrapped_name>)"``.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        easy_threshold: float = 0.05,
        hard_threshold: float = 0.15,
        smoothing: float = 0.3,
        w_motion: float = 1.0,
        w_scale: float = 0.5,
        history_len: int = 8,
        name: Optional[str] = None,
    ) -> None:
        if easy_threshold > hard_threshold:
            raise ValueError(
                f"easy_threshold ({easy_threshold}) must be ≤ hard_threshold ({hard_threshold})."
            )
        if not 0.0 < smoothing <= 1.0:
            raise ValueError(f"smoothing must be in (0, 1], got {smoothing}.")

        super().__init__(name or f"Adaptive({tracker.name})")
        self._tracker = tracker
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold
        self.smoothing = smoothing
        self.w_motion = w_motion
        self.w_scale = w_scale
        self.history_len = history_len

        # Runtime state — reset in initialize()
        self._prev_bbox: Optional[BBox] = None
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        self._scale_ratio: float = 1.0
        self._smooth_complexity: float = 0.0
        self._in_easy_mode: bool = False
        self._bbox_history: Deque[BBox] = deque(maxlen=history_len)

        # Public statistics
        self.n_updates: int = 0
        """Number of frames where the wrapped tracker was called."""
        self.n_coasted: int = 0
        """Number of frames where a coast prediction was used instead."""
        self.complexity_history: List[float] = []
        """Per-frame smoothed complexity values (updated on every call to update)."""

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the wrapped tracker and reset all runtime state.

        Args:
            frame: First frame (H × W × 3 BGR uint8).
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._prev_bbox = bbox
        self._velocity = (0.0, 0.0)
        self._scale_ratio = 1.0
        self._smooth_complexity = 0.0
        self._in_easy_mode = False
        self._bbox_history.clear()
        self._bbox_history.append(bbox)
        self.n_updates = 0
        self.n_coasted = 0
        self.complexity_history.clear()

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location, calling the wrapped tracker only when needed.

        The adaptive policy:
        1. In **easy mode**: coast using the constant-velocity prediction.
           If the complexity of the coast prediction exceeds ``hard_threshold``,
           exit easy mode and run the wrapped tracker instead.
        2. In **hard mode**: always run the wrapped tracker.  If complexity
           drops below ``easy_threshold``, enter easy mode for the next frame.

        Args:
            frame: Current frame (H × W × 3 BGR uint8).

        Returns:
            Predicted bounding box ``(x, y, w, h)``.
        """
        if self._in_easy_mode:
            coast_bbox = self._coast_predict()
            complexity = self._raw_complexity(coast_bbox)
            self._smooth_complexity = self._ema(complexity)

            if self._smooth_complexity >= self.hard_threshold:
                # Complexity spike — exit easy mode, run real tracker now.
                self._in_easy_mode = False
                bbox = self._tracker.update(frame)
                self.n_updates += 1
                complexity_of_real = self._raw_complexity(bbox)
                self._smooth_complexity = self._ema(complexity_of_real)
            else:
                bbox = coast_bbox
                self.n_coasted += 1
        else:
            bbox = self._tracker.update(frame)
            self.n_updates += 1
            complexity = self._raw_complexity(bbox)
            self._smooth_complexity = self._ema(complexity)
            if self._smooth_complexity < self.easy_threshold:
                self._in_easy_mode = True

        self.complexity_history.append(self._smooth_complexity)
        if self._prev_bbox is not None:
            self._update_motion_model(self._prev_bbox, bbox)
        self._prev_bbox = bbox
        self._bbox_history.append(bbox)
        return bbox

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def skip_rate(self) -> float:
        """Fraction of update calls where coasting was used instead of the tracker.

        Returns:
            Float in ``[0, 1]``.  ``0.0`` means the wrapped tracker ran on
            every frame; ``1.0`` means it never ran after initialization.
        """
        total = self.n_updates + self.n_coasted
        return self.n_coasted / total if total > 0 else 0.0

    @property
    def wrapped_tracker(self) -> BaseTracker:
        """The underlying tracker that this wrapper delegates to."""
        return self._tracker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ema(self, value: float) -> float:
        """Single-step exponential moving average update."""
        α = self.smoothing
        return α * value + (1.0 - α) * self._smooth_complexity

    def _raw_complexity(self, bbox: BBox) -> float:
        """Compute the raw (un-smoothed) motion complexity given a new bbox."""
        if self._prev_bbox is None:
            return 0.0
        px, py, pw, ph = self._prev_bbox
        cx, cy, cw, ch = bbox

        diag = math.sqrt(pw * pw + ph * ph) + 1e-6
        dx = (cx + cw / 2.0) - (px + pw / 2.0)
        dy = (cy + ch / 2.0) - (py + ph / 2.0)
        motion_norm = math.sqrt(dx * dx + dy * dy) / diag

        prev_area = pw * ph + 1e-6
        curr_area = cw * ch + 1e-6
        scale_change = abs(curr_area / prev_area - 1.0)

        return self.w_motion * motion_norm + self.w_scale * scale_change

    def _coast_predict(self) -> BBox:
        """Return a constant-velocity bbox prediction without running the tracker."""
        if self._prev_bbox is None:
            return (0.0, 0.0, 1.0, 1.0)
        px, py, pw, ph = self._prev_bbox
        vx, vy = self._velocity
        nx = px + vx
        ny = py + vy
        nw = max(pw * self._scale_ratio, 1.0)
        nh = max(ph * self._scale_ratio, 1.0)
        return (nx, ny, nw, nh)

    def _update_motion_model(self, prev_bbox: BBox, curr_bbox: BBox) -> None:
        """Update rolling velocity and scale-ratio estimates via EMA."""
        px, py, pw, ph = prev_bbox
        cx, cy, cw, ch = curr_bbox

        dvx = (cx + cw / 2.0) - (px + pw / 2.0)
        dvy = (cy + ch / 2.0) - (py + ph / 2.0)
        α = self.smoothing
        old_vx, old_vy = self._velocity
        self._velocity = (
            α * dvx + (1.0 - α) * old_vx,
            α * dvy + (1.0 - α) * old_vy,
        )

        prev_area = pw * ph + 1e-6
        curr_area = cw * ch + 1e-6
        ratio = math.sqrt(curr_area / prev_area)
        self._scale_ratio = α * ratio + (1.0 - α) * self._scale_ratio

    def __repr__(self) -> str:
        return (
            f"AdaptiveBudgetTracker("
            f"tracker={self._tracker!r}, "
            f"easy={self.easy_threshold}, hard={self.hard_threshold}, "
            f"skip_rate={self.skip_rate:.2%})"
        )
