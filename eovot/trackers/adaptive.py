"""AdaptiveBudgetTracker — AIMD-based frame-skip rate controller.

Unlike :class:`~eovot.trackers.frame_skip.FrameSkipTracker` whose skip rate
is fixed at construction time, :class:`AdaptiveBudgetTracker` adjusts the
number of frames skipped *dynamically* between each real tracker call to
keep the achieved throughput near a user-supplied ``target_fps`` budget.

The controller implements an **AIMD** (Additive Increase, Multiplicative
Decrease) rule — the same family used in TCP congestion control — applied to
the skip count *n* (frames skipped between two real tracker calls):

* **Increase phase** (behind budget): if the rolling mean latency per tracker
  call implies ``actual_fps < target_fps``, increase *n* by 1 (up to
  ``max_skip``).
* **Decrease phase** (over budget): if ``actual_fps > 1.25 × target_fps``,
  halve *n* (minimum 0) to reclaim accuracy headroom.

The ``warmup_frames`` initial frames always run the tracker at full rate so
that the latency estimator has a valid baseline before adaptation starts.

The wrapped tracker's ``name`` gets an ``_adaptive{fps:.0f}fps`` suffix so
benchmark reports distinguish this configuration from the fixed-rate wrapper.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive import AdaptiveBudgetTracker

    tracker = AdaptiveBudgetTracker(MOSSETracker(), target_fps=30.0, max_skip=5)
    # Use tracker with any BenchmarkEngine as normal
    stats = tracker.adaptation_stats
    print(f"mean skip fraction: {stats['mean_skip_fraction']:.2%}")
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox


class AdaptiveBudgetTracker(BaseTracker):
    """Wrap any tracker with an AIMD skip-rate controller.

    The controller adjusts the per-call frame-skip count *n* (frames between
    two real tracker invocations) after every real tracker call based on a
    rolling latency window:

    * If rolling-mean latency × ``target_fps > 1.0`` (behind budget) →
      ``n = min(n + 1, max_skip)``
    * If rolling-mean latency × ``target_fps < 0.80`` (≥ 20 % headroom) →
      ``n = max(n // 2, 0)``
    * Otherwise → ``n`` unchanged.

    The first ``warmup_frames`` updates always run the tracker regardless of
    the current skip count so that the latency estimator is valid before
    adaptation starts.

    Args:
        tracker:       Any :class:`BaseTracker` to wrap.
        target_fps:    Desired throughput in frames per second.  The controller
            tries to keep achieved FPS near this value.  Must be > 0.
        max_skip:      Maximum number of consecutive frames that may be skipped
            between two real tracker calls.  Hard ceiling on the skip count.
            Default 7.
        warmup_frames: Number of initial frames always processed at full rate
            for baseline latency estimation.  Default 5.
        window_size:   Number of recent real-tracker latency samples used for
            the rolling mean.  Default 8.

    Raises:
        ValueError: If *target_fps* ≤ 0 or *max_skip* < 0 or *warmup_frames* < 0.
    """

    _INCREASE_THRESHOLD: float = 1.0   # actual_fps / target_fps below this → increase skip
    _DECREASE_THRESHOLD: float = 1.25  # actual_fps / target_fps above this → decrease skip

    def __init__(
        self,
        tracker: BaseTracker,
        target_fps: float = 30.0,
        max_skip: int = 7,
        warmup_frames: int = 5,
        window_size: int = 8,
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")
        if max_skip < 0:
            raise ValueError(f"max_skip must be >= 0, got {max_skip}")
        if warmup_frames < 0:
            raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames}")

        super().__init__(name=f"{tracker.name}_adaptive{target_fps:.0f}fps")
        self._tracker = tracker
        self.target_fps = target_fps
        self.max_skip = max_skip
        self.warmup_frames = warmup_frames
        self.window_size = window_size

        # --- mutable state (reset on initialize) ---
        self._frame_idx: int = 0          # total frames seen since initialize()
        self._skip_counter: int = 0       # frames remaining in current skip run
        self._current_skip: int = 0       # current AIMD skip count n
        self._last_bbox: Optional[BBox] = None
        self._latency_window: Deque[float] = deque(maxlen=window_size)
        self._adaptation_count: int = 0
        self._total_skipped: int = 0
        self._total_frames: int = 0

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the wrapped tracker and reset all controller state.

        Args:
            frame: First frame as a ``(H, W, 3)`` BGR uint8 array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._frame_idx = 0
        self._skip_counter = 0
        self._current_skip = 0
        self._last_bbox = bbox
        self._latency_window.clear()
        self._adaptation_count = 0
        self._total_skipped = 0
        self._total_frames = 0

    def update(self, frame: np.ndarray) -> BBox:
        """Return a prediction, running the real tracker or repeating the last bbox.

        After every real tracker call the AIMD controller updates the skip
        count *n* based on the observed latency.

        Args:
            frame: Current frame as a ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._last_bbox is None:
            raise RuntimeError(
                "AdaptiveBudgetTracker has not been initialised. Call initialize() first."
            )

        self._frame_idx += 1
        self._total_frames += 1

        in_warmup = self._frame_idx <= self.warmup_frames
        is_active_frame = in_warmup or (self._skip_counter == 0)

        if is_active_frame:
            t0 = time.perf_counter()
            self._last_bbox = self._tracker.update(frame)
            latency = time.perf_counter() - t0

            self._latency_window.append(latency)
            self._skip_counter = self._current_skip  # reset countdown

            if not in_warmup and len(self._latency_window) >= 2:
                self._adapt()
        else:
            # Passive frame — return cached prediction
            self._skip_counter -= 1
            self._total_skipped += 1

        return self._last_bbox

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def underlying_tracker(self) -> BaseTracker:
        """The wrapped :class:`BaseTracker` instance."""
        return self._tracker

    @property
    def current_skip(self) -> int:
        """Current AIMD skip count (frames between two real tracker calls)."""
        return self._current_skip

    @property
    def adaptation_stats(self) -> Dict[str, float]:
        """Snapshot of adaptation statistics for the current sequence.

        Returns a dict with:

        * ``target_fps`` — the configured budget.
        * ``estimated_fps`` — FPS derived from the rolling latency mean;
          ``0.0`` if no samples yet.
        * ``current_skip`` — current skip count *n*.
        * ``mean_skip_fraction`` — fraction of frames returned from cache
          (0.0 = no skipping, 1.0 = fully skipped).
        * ``adaptation_count`` — total AIMD adjustments made so far.
        * ``total_frames`` — total frames processed (active + passive).
        * ``active_frames`` — frames where the real tracker ran.
        """
        mean_lat = (
            sum(self._latency_window) / len(self._latency_window)
            if self._latency_window
            else 0.0
        )
        est_fps = 1.0 / mean_lat if mean_lat > 0 else 0.0
        active = self._total_frames - self._total_skipped
        skip_frac = self._total_skipped / self._total_frames if self._total_frames else 0.0
        return {
            "target_fps": self.target_fps,
            "estimated_fps": est_fps,
            "current_skip": float(self._current_skip),
            "mean_skip_fraction": skip_frac,
            "adaptation_count": float(self._adaptation_count),
            "total_frames": float(self._total_frames),
            "active_frames": float(active),
        }

    # ------------------------------------------------------------------
    # AIMD controller
    # ------------------------------------------------------------------

    def _adapt(self) -> None:
        """Apply one AIMD step based on the current rolling latency mean."""
        mean_lat = sum(self._latency_window) / len(self._latency_window)
        # target_period = 1 / target_fps; ratio = mean_lat / target_period
        ratio = mean_lat * self.target_fps  # >1 → behind budget; <1 → headroom

        old_skip = self._current_skip
        if ratio > self._INCREASE_THRESHOLD:
            # Behind budget: add one more skip frame
            self._current_skip = min(self._current_skip + 1, self.max_skip)
        elif ratio < (1.0 / self._DECREASE_THRESHOLD):
            # Well under budget: halve the skip count (minimum 0)
            self._current_skip = max(self._current_skip // 2, 0)

        if self._current_skip != old_skip:
            self._adaptation_count += 1
