"""DualModeTracker — latency-aware adaptive meta-tracker.

A tracker that dynamically routes each frame to either a *fast* tracker or
an *accurate* tracker based on measured rolling-window latency:

- When the rolling-average latency of recent frames is **within** the
  latency budget, the *accurate* tracker is used.
- When latency **exceeds** the budget (device under load, thermal throttling,
  or memory pressure), the *fast* tracker takes over.

On every mode switch, the newly-activated tracker is re-initialised at the
last known bounding box so tracking remains continuous across switches.

This provides a simple but principled baseline for adaptive-inference
tracking — a policy that optimises for real-time throughput while keeping
accuracy as high as the hardware budget allows.  It is directly inspired by
the EOVOT design goal of "RL-based adaptive trackers and dynamic inference";
DualModeTracker is the deterministic open-loop baseline that future RL
policies can be compared against.

Example::

    from eovot.trackers import MOSSETracker, KCFTracker
    from eovot.trackers.dual_mode import DualModeTracker

    fast     = MOSSETracker()
    accurate = KCFTracker()
    tracker  = DualModeTracker(fast=fast, accurate=accurate, budget_ms=25.0)

    tracker.initialize(first_frame, init_bbox)
    for frame in remaining_frames:
        bbox = tracker.update(frame)
    print(tracker.stats.summary())
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

from .base import BaseTracker, BBox


@dataclass
class SwitchEvent:
    """Records a single mode-switch event during tracking."""

    frame_index: int
    from_mode: str          # "fast" or "accurate"
    to_mode: str
    rolling_latency_ms: float
    budget_ms: float

    def __str__(self) -> str:
        return (
            f"SwitchEvent(frame={self.frame_index} "
            f"{self.from_mode}→{self.to_mode} "
            f"lat={self.rolling_latency_ms:.2f}ms "
            f"budget={self.budget_ms:.1f}ms)"
        )


@dataclass
class DualModeStats:
    """Runtime statistics collected by :class:`DualModeTracker`."""

    total_frames: int = 0
    fast_frames: int = 0
    accurate_frames: int = 0
    switches: List[SwitchEvent] = field(default_factory=list)

    @property
    def fast_fraction(self) -> float:
        """Fraction of frames processed by the fast tracker."""
        if self.total_frames == 0:
            return 0.0
        return self.fast_frames / self.total_frames

    @property
    def accurate_fraction(self) -> float:
        """Fraction of frames processed by the accurate tracker."""
        if self.total_frames == 0:
            return 0.0
        return self.accurate_frames / self.total_frames

    @property
    def num_switches(self) -> int:
        """Total number of mode switches."""
        return len(self.switches)

    def summary(self) -> dict:
        """Return a JSON-serialisable summary dict."""
        return {
            "total_frames": self.total_frames,
            "fast_frames": self.fast_frames,
            "accurate_frames": self.accurate_frames,
            "fast_fraction": round(self.fast_fraction, 4),
            "accurate_fraction": round(self.accurate_fraction, 4),
            "num_switches": self.num_switches,
        }


class DualModeTracker(BaseTracker):
    """Latency-aware meta-tracker that switches between fast and accurate modes.

    At each frame the tracker inspects the rolling-average latency of the
    last *window_size* :meth:`update` calls.  When that average exceeds
    *budget_ms* it switches to *fast*; when it drops back below budget it
    returns to *accurate*.  On every switch the newly-activated tracker is
    re-initialised at the last predicted bounding box.

    Both underlying trackers are always initialised on the first frame so
    that switching has no warm-up latency.

    Args:
        fast:        Lightweight tracker to use under latency pressure
                     (e.g., MOSSE, CamShift).
        accurate:    Higher-quality tracker to use when budget has headroom
                     (e.g., KCF, CSRT).
        budget_ms:   Per-frame latency budget in milliseconds.
                     Default: ``33.3`` (≈ 30 FPS).
        window_size: Number of recent frames used to compute the rolling
                     average latency.  Larger values reduce noise but slow
                     adaptation.  Default: ``5``.
        name:        Name shown in benchmark reports.  Derived from the
                     constituent tracker names when omitted.
    """

    def __init__(
        self,
        fast: BaseTracker,
        accurate: BaseTracker,
        budget_ms: float = 33.3,
        window_size: int = 5,
        name: Optional[str] = None,
    ) -> None:
        if budget_ms <= 0:
            raise ValueError(f"budget_ms must be positive, got {budget_ms}")
        if window_size < 1:
            raise ValueError(f"window_size must be ≥ 1, got {window_size}")
        derived_name = name or f"DualMode({fast.name}+{accurate.name})"
        super().__init__(name=derived_name)

        self.fast = fast
        self.accurate = accurate
        self.budget_ms = budget_ms
        self.window_size = window_size

        self._active: BaseTracker = accurate
        self._active_name: str = "accurate"
        self._last_bbox: Optional[BBox] = None
        self._frame_index: int = 0
        self._latency_window: Deque[float] = deque(maxlen=window_size)
        self._stats: DualModeStats = DualModeStats()

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise both internal trackers on the first frame.

        Both trackers are initialised so that switching is instantaneous:
        the newly activated tracker always has a current model of the target.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self.fast.initialize(frame, bbox)
        self.accurate.initialize(frame, bbox)
        self._active = self.accurate
        self._active_name = "accurate"
        self._last_bbox = bbox
        self._frame_index = 0
        self._latency_window.clear()
        self._stats = DualModeStats()

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location, routing to the appropriate tracker.

        The routing decision is made *before* calling the underlying tracker,
        using the latency measured in previous frames.  This avoids any
        circular dependency on this frame's latency.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.
        """
        self._frame_index += 1
        self._maybe_switch(frame)

        t0 = time.perf_counter()
        bbox = self._active.update(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._latency_window.append(elapsed_ms)
        self._last_bbox = bbox

        self._stats.total_frames += 1
        if self._active_name == "fast":
            self._stats.fast_frames += 1
        else:
            self._stats.accurate_frames += 1

        return bbox

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def stats(self) -> DualModeStats:
        """Runtime statistics: frame counts, mode fractions, switch log."""
        return self._stats

    @property
    def current_mode(self) -> str:
        """The currently active tracking mode: ``"fast"`` or ``"accurate"``."""
        return self._active_name

    @property
    def rolling_latency_ms(self) -> float:
        """Rolling-average latency of the last *window_size* frames (ms).

        Returns ``0.0`` when no frames have been processed yet.
        """
        if not self._latency_window:
            return 0.0
        return float(sum(self._latency_window) / len(self._latency_window))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_switch(self, frame: np.ndarray) -> None:
        """Switch the active tracker if rolling latency crosses the budget."""
        if not self._latency_window:
            return  # insufficient history — stay with current tracker

        rolling = self.rolling_latency_ms
        should_be_fast = rolling > self.budget_ms
        currently_fast = self._active_name == "fast"

        if should_be_fast == currently_fast:
            return  # no change needed

        new_mode = "fast" if should_be_fast else "accurate"
        new_tracker = self.fast if should_be_fast else self.accurate

        if self._last_bbox is not None:
            new_tracker.initialize(frame, self._last_bbox)

        self._stats.switches.append(
            SwitchEvent(
                frame_index=self._frame_index,
                from_mode=self._active_name,
                to_mode=new_mode,
                rolling_latency_ms=rolling,
                budget_ms=self.budget_ms,
            )
        )

        self._active = new_tracker
        self._active_name = new_mode
