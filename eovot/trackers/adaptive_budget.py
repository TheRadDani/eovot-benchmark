"""AdaptiveBudgetTracker — runtime-adaptive edge deployment wrapper.

Unlike :class:`~eovot.trackers.frame_skip.FrameSkipTracker` (fixed skip rate)
and :class:`~eovot.trackers.scale.ResolutionScaledTracker` (fixed resolution),
this wrapper monitors its own per-frame latency and dynamically adjusts *both*
the skip rate and the input resolution to stay within a user-supplied latency
budget.  It implements a simple integral controller:

1. After every *control_window* frames the observed mean latency is compared to
   ``1000 / target_fps`` ms (the budget).
2. If actual latency exceeds budget by more than ``tolerance``, the controller
   either increases the skip rate (up to ``max_skip_rate``) or reduces the
   resolution scale (down to ``min_scale``).
3. If actual latency is well below budget, the controller relaxes constraints
   one step at a time (preferring accuracy over speed).

The tracker reports its adaptation history via :attr:`adaptation_log` so that
post-hoc analysis can reveal how aggressively the budget was exercised.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive_budget import AdaptiveBudgetTracker
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, num_frames=200)
    engine  = BenchmarkEngine(verbose=False)

    tracker = AdaptiveBudgetTracker(MOSSETracker(), target_fps=60.0)
    result  = engine.run(tracker, dataset, dataset_name="Syn")
    print(f"FPS achieved: {result.mean_fps:.1f}")
    print(f"Adaptations:  {len(tracker.adaptation_log)}")
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Literal, Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox

ControlAction = Literal["increase_skip", "decrease_skip", "reduce_scale", "increase_scale", "hold"]


@dataclass
class AdaptationEvent:
    """One controller decision recorded during a run."""

    frame_idx: int
    """Frame index at which the decision was made."""
    observed_latency_ms: float
    """Mean latency over the last control window (ms)."""
    budget_ms: float
    """Target latency budget (ms) = 1000 / target_fps."""
    action: ControlAction
    """What the controller decided to do."""
    new_skip_rate: int
    """Skip rate after the decision."""
    new_scale: float
    """Resolution scale after the decision."""


class AdaptiveBudgetTracker(BaseTracker):
    """Wrap any tracker to hit a target FPS budget by dynamic adaptation.

    The controller monitors the *actual* per-frame latency using a sliding
    window and adjusts two knobs:

    * **Skip rate** (1 – ``max_skip_rate``): how many frames are skipped
      between real tracker updates.  Skipped frames repeat the last prediction.
    * **Resolution scale** (``min_scale`` – 1.0): the input frame is resized
      by this factor before being passed to the underlying tracker.  The
      predicted box is rescaled back to the original resolution.

    The controller prefers relaxing (reducing skip rate / increasing scale)
    when latency is comfortably below budget, and tightens constraints
    (increasing skip rate first, then reducing scale) when the budget is
    exceeded.

    Args:
        tracker:         Any :class:`BaseTracker` to wrap.
        target_fps:      Desired throughput in frames per second.
        max_skip_rate:   Maximum allowed skip rate (≥ 1).  Default: ``4``.
        min_scale:       Minimum allowed resolution scale (> 0, ≤ 1.0).
            Default: ``0.25`` (quarter resolution).
        tolerance:       Fractional tolerance around the budget before the
            controller acts.  ``0.1`` means ±10 % headroom before adjusting.
            Default: ``0.10``.
        control_window:  Number of frames between controller evaluations.
            Default: ``10``.

    Raises:
        ValueError: If any parameter is out of range.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        target_fps: float = 30.0,
        max_skip_rate: int = 4,
        min_scale: float = 0.25,
        tolerance: float = 0.10,
        control_window: int = 10,
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")
        if max_skip_rate < 1:
            raise ValueError(f"max_skip_rate must be >= 1, got {max_skip_rate}")
        if not (0.0 < min_scale <= 1.0):
            raise ValueError(f"min_scale must be in (0, 1], got {min_scale}")
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        if control_window < 1:
            raise ValueError(f"control_window must be >= 1, got {control_window}")

        super().__init__(name=f"{tracker.name}_adaptive{int(target_fps)}fps")
        self._tracker = tracker
        self.target_fps = target_fps
        self.max_skip_rate = max_skip_rate
        self.min_scale = min_scale
        self.tolerance = tolerance
        self.control_window = control_window

        # Runtime state — reset by initialize()
        self._frame_idx: int = 0
        self._last_bbox: Optional[BBox] = None
        self._current_skip_rate: int = 1
        self._current_scale: float = 1.0
        self._latency_window: Deque[float] = deque(maxlen=control_window)
        self.adaptation_log: List[AdaptationEvent] = []

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the underlying tracker and reset adaptation state.

        Args:
            frame: First frame as a ``(H, W, 3)`` BGR uint8 array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._frame_idx = 0
        self._last_bbox = bbox
        self._current_skip_rate = 1
        self._current_scale = 1.0
        self._latency_window.clear()
        self.adaptation_log.clear()

        scaled_frame = self._scale_frame(frame)
        scaled_bbox = self._scale_bbox_down(bbox, frame.shape)
        self._tracker.initialize(scaled_frame, scaled_bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location, adapting processing to meet the FPS budget.

        On *active* frames (``frame_idx % skip_rate == 0``) the underlying
        tracker is run at possibly reduced resolution; the result is rescaled
        back to original coordinates.  On *passive* frames the last prediction
        is returned immediately.

        The controller fires every ``control_window`` active frames.

        Args:
            frame: Current frame as a ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` in *original* resolution.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._last_bbox is None:
            raise RuntimeError(
                "AdaptiveBudgetTracker has not been initialised. Call initialize() first."
            )

        self._frame_idx += 1

        if self._frame_idx % self._current_skip_rate != 0:
            # Passive frame — return last prediction without calling the tracker
            return self._last_bbox

        # Active frame — run the underlying tracker at current scale
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        scaled_frame = self._scale_frame(frame)
        raw_bbox = self._tracker.update(scaled_frame)
        elapsed_ms = (time.perf_counter() - t0) * 1_000.0

        # Rescale prediction back to original resolution
        self._last_bbox = self._scale_bbox_up(raw_bbox, h, w)
        self._latency_window.append(elapsed_ms)

        # Controller fires when the window is full
        if len(self._latency_window) == self.control_window:
            self._control_step(frame)

        return self._last_bbox

    # ------------------------------------------------------------------
    # Adaptation controller
    # ------------------------------------------------------------------

    def _control_step(self, frame: np.ndarray) -> None:
        """Evaluate the latency budget and adjust skip rate / scale if needed."""
        budget_ms = 1_000.0 / self.target_fps
        observed_ms = float(np.mean(self._latency_window))
        ratio = observed_ms / budget_ms  # > 1 means over budget

        action: ControlAction = "hold"

        if ratio > 1.0 + self.tolerance:
            # Over budget — tighten constraints
            if self._current_skip_rate < self.max_skip_rate:
                self._current_skip_rate += 1
                action = "increase_skip"
            elif self._current_scale > self.min_scale:
                # Reduce scale by 25 % steps, floored at min_scale
                self._current_scale = max(
                    self.min_scale, round(self._current_scale - 0.25, 4)
                )
                # Re-initialize the underlying tracker at the new scale
                action = "reduce_scale"
            # else already at maximum tightening — nothing more to do

        elif ratio < 1.0 - self.tolerance:
            # Under budget — relax constraints (prefer quality)
            if self._current_scale < 1.0:
                self._current_scale = min(
                    1.0, round(self._current_scale + 0.25, 4)
                )
                action = "increase_scale"
            elif self._current_skip_rate > 1:
                self._current_skip_rate -= 1
                action = "decrease_skip"

        self.adaptation_log.append(
            AdaptationEvent(
                frame_idx=self._frame_idx,
                observed_latency_ms=round(observed_ms, 3),
                budget_ms=round(budget_ms, 3),
                action=action,
                new_skip_rate=self._current_skip_rate,
                new_scale=self._current_scale,
            )
        )

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _scale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return a resized copy of *frame* at the current scale.

        If scale is 1.0 the original array is returned unchanged (zero copy).
        """
        if self._current_scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        new_w = max(1, int(w * self._current_scale))
        new_h = max(1, int(h * self._current_scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _scale_bbox_down(self, bbox: BBox, original_shape: Tuple[int, ...]) -> BBox:
        """Scale a bounding box from original resolution down to tracker resolution."""
        s = self._current_scale
        x, y, w, h = bbox
        return (x * s, y * s, w * s, h * s)

    def _scale_bbox_up(self, bbox: BBox, orig_h: int, orig_w: int) -> BBox:
        """Scale a bounding box from tracker resolution back to original resolution."""
        s = self._current_scale
        if s == 1.0:
            return bbox
        x, y, w, h = bbox
        inv = 1.0 / s
        return (x * inv, y * inv, w * inv, h * inv)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def underlying_tracker(self) -> BaseTracker:
        """The wrapped :class:`BaseTracker` instance."""
        return self._tracker

    @property
    def current_skip_rate(self) -> int:
        """Current skip rate (1 = no skipping)."""
        return self._current_skip_rate

    @property
    def current_scale(self) -> float:
        """Current resolution scale (1.0 = full resolution)."""
        return self._current_scale

    def adaptation_summary(self) -> dict:
        """Return a concise summary of all adaptation decisions made during the run.

        Returns:
            Dict with counts of each :data:`ControlAction` taken, the final
            skip rate, and the final scale.
        """
        from collections import Counter
        counts = Counter(e.action for e in self.adaptation_log)
        return {
            "total_control_steps": len(self.adaptation_log),
            "actions": dict(counts),
            "final_skip_rate": self._current_skip_rate,
            "final_scale": self._current_scale,
        }
