"""Closed-loop latency-adaptive tracker wrapper for edge deployment.

Edge devices must guarantee real-time performance under variable scene
complexity.  Static configurations (fixed resolution, fixed skip rate) cannot
handle the full workload range: a scale=0.5 that keeps the tracker under
budget on low-texture sequences may exceed budget on complex ones, while a
scale=0.25 that handles complex scenes wastes quality on easy ones.

:class:`AdaptiveBudgetTracker` closes this loop.  It monitors per-frame
latency against a configurable budget and automatically adjusts input
resolution at runtime:

- When the rolling-average latency **exceeds** the budget, the tracker
  scales down by ``scale_step`` (to a minimum of ``min_scale``).
- When the rolling-average latency drops **below** ``restore_headroom × budget``,
  the tracker scales up by ``scale_step / 2`` (to a maximum of ``1.0``).

Scale changes trigger a re-initialization of the inner tracker at the new
resolution using the most recent predicted bounding box.  This mirrors the
adaptive inference pattern used in real-time systems where inference engines
switch between model configurations without stopping the tracking loop.

Adaptation events are logged to :attr:`adaptation_events` for post-run
analysis of how frequently and aggressively the budget controller intervened.

Example::

    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.adaptive_budget import AdaptiveBudgetTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    tracker = AdaptiveBudgetTracker(KCFTracker(), budget_ms=20.0)

    dataset = SyntheticDataset(num_sequences=5)
    engine  = BenchmarkEngine(verbose=False)
    result  = engine.run(tracker, dataset, dataset_name="Synthetic")
    print(result)
    print(f"Adaptations: {len(tracker.adaptation_events)}")
    for event in tracker.adaptation_events[:3]:
        print(event)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np

from .base import BaseTracker, BBox


@dataclass
class AdaptationEvent:
    """One recorded scale-adjustment decision by :class:`AdaptiveBudgetTracker`.

    Attributes:
        frame_idx: Index of the frame at which the adaptation occurred.
        old_scale: Scale factor before the adjustment.
        new_scale: Scale factor after the adjustment.
        trigger_latency_ms: Rolling-average latency (ms) that triggered the
            adjustment.
        budget_ms: The configured latency budget (ms) at this event.
        direction: ``"down"`` when resolution was reduced to reclaim budget;
            ``"up"`` when headroom allowed a partial restoration.
    """

    frame_idx: int
    old_scale: float
    new_scale: float
    trigger_latency_ms: float
    budget_ms: float
    direction: str  # "up" | "down"

    def __str__(self) -> str:
        arrow = "↓" if self.direction == "down" else "↑"
        return (
            f"AdaptationEvent[frame={self.frame_idx}] "
            f"scale {self.old_scale:.2f} → {self.new_scale:.2f} {arrow}  "
            f"latency={self.trigger_latency_ms:.2f} ms  budget={self.budget_ms:.2f} ms"
        )


class AdaptiveBudgetTracker(BaseTracker):
    """Tracker wrapper with closed-loop latency control via dynamic resolution.

    Wraps any :class:`~.base.BaseTracker` and continuously adjusts the input
    frame resolution to maintain a per-frame latency budget.  When processing
    time consistently exceeds ``budget_ms``, the tracker lowers resolution;
    when headroom is available, it partially restores it.

    The adaptation algorithm uses a rolling-average latency estimator (size
    ``window_size``) to smooth over transient spikes before committing to a
    scale change.  This prevents oscillation from single slow frames.

    Scale changes trigger a re-initialization of the inner tracker at the new
    resolution, using the most recent prediction as the new initialization
    bounding box.  This keeps the inner tracker's internal state consistent
    with the new frame size.

    Args:
        tracker: Inner :class:`~.base.BaseTracker` to wrap.  Must implement
            ``initialize()`` and ``update()``.
        budget_ms: Target per-frame latency ceiling in milliseconds.  The
            controller aims to keep rolling-average latency at or below this
            value.
        scale_step: Linear resolution change per downward adaptation step.
            Default: ``0.15``.  Upward restorations use ``scale_step / 2``
            to avoid oscillation.
        min_scale: Minimum allowed linear scale factor.  Default: ``0.25``
            (6.25% of original pixel area — suitable for 1080p sources on
            embedded hardware).
        window_size: Number of recent frames used for the rolling latency
            average.  Larger windows are more stable but slower to react.
            Default: ``8``.
        restore_headroom: Fraction of budget below which scale is increased.
            Default: ``0.75`` (scale up when latency < 75% of budget).

    Attributes:
        adaptation_events: Ordered list of :class:`AdaptationEvent` objects
            recording each scale change, for post-run analysis.
        current_scale: Current linear scale factor (read-only property).
    """

    def __init__(
        self,
        tracker: BaseTracker,
        budget_ms: float,
        scale_step: float = 0.15,
        min_scale: float = 0.25,
        window_size: int = 8,
        restore_headroom: float = 0.75,
    ) -> None:
        if budget_ms <= 0:
            raise ValueError(f"budget_ms must be positive, got {budget_ms}")
        if not 0 < min_scale <= 1.0:
            raise ValueError(f"min_scale must be in (0, 1], got {min_scale}")
        if not 0 < scale_step < 1.0:
            raise ValueError(f"scale_step must be in (0, 1), got {scale_step}")
        if not 0 < restore_headroom < 1.0:
            raise ValueError(f"restore_headroom must be in (0, 1), got {restore_headroom}")

        super().__init__(name=f"{tracker.name}_adaptive{int(budget_ms)}ms")
        self._tracker = tracker
        self._budget_ms = budget_ms
        self._scale_step = scale_step
        self._min_scale = min_scale
        self._window_size = window_size
        self._restore_headroom = restore_headroom

        self._scale: float = 1.0
        self._latency_window: Deque[float] = deque(maxlen=window_size)
        self._last_bbox: Optional[BBox] = None
        self._frame_idx: int = 0
        self._initialized: bool = False

        self.adaptation_events: List[AdaptationEvent] = []

    @property
    def current_scale(self) -> float:
        """Current linear scale factor applied to input frames."""
        return self._scale

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialize the inner tracker at full resolution.

        Resets all adaptation state (scale, latency window, event log) and
        initializes the wrapped tracker at ``scale=1.0``.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8 array.
            bbox:  Initial bounding box ``(x, y, w, h)``.
        """
        self._scale = 1.0
        self._latency_window.clear()
        self._last_bbox = bbox
        self._frame_idx = 0
        self._initialized = True
        self.adaptation_events = []

        scaled_frame = self._scale_frame(frame, self._scale)
        scaled_bbox = self._scale_bbox(bbox, self._scale)
        self._tracker.initialize(scaled_frame, scaled_bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location with dynamic resolution adaptation.

        Scales the frame to the current resolution, times the inner tracker,
        and adjusts the scale factor for subsequent frames based on the
        rolling-average latency against the configured budget.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8 array.

        Returns:
            Predicted bounding box in **original** frame coordinates ``(x, y, w, h)``.
        """
        if not self._initialized:
            raise RuntimeError("AdaptiveBudgetTracker.initialize() must be called first.")

        self._frame_idx += 1

        scaled_frame = self._scale_frame(frame, self._scale)

        t0 = time.perf_counter()
        scaled_bbox = self._tracker.update(scaled_frame)
        elapsed_ms = (time.perf_counter() - t0) * 1_000.0

        self._latency_window.append(elapsed_ms)
        pred_bbox = self._unscale_bbox(scaled_bbox, self._scale)
        self._last_bbox = pred_bbox

        if len(self._latency_window) == self._window_size:
            self._adapt(frame, pred_bbox)

        return pred_bbox

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def adaptation_summary(self) -> Dict:
        """Return a concise summary of adaptation behaviour for this run.

        Returns:
            Dict with keys:

            * ``n_adaptations`` — total number of scale-change events.
            * ``n_down`` — number of downward scale adjustments.
            * ``n_up`` — number of upward scale restorations.
            * ``min_scale_reached`` — lowest scale factor used.
            * ``final_scale`` — scale factor at the last frame.
            * ``budget_ms`` — configured latency budget.
        """
        downs = [e for e in self.adaptation_events if e.direction == "down"]
        ups = [e for e in self.adaptation_events if e.direction == "up"]
        min_scale = min((e.new_scale for e in self.adaptation_events), default=self._scale)
        return {
            "n_adaptations": len(self.adaptation_events),
            "n_down": len(downs),
            "n_up": len(ups),
            "min_scale_reached": round(min_scale, 4),
            "final_scale": round(self._scale, 4),
            "budget_ms": self._budget_ms,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adapt(self, current_frame: np.ndarray, current_bbox: BBox) -> None:
        """Check rolling latency and adjust scale if needed."""
        avg_ms = float(np.mean(list(self._latency_window)))

        if avg_ms > self._budget_ms and self._scale > self._min_scale:
            old_scale = self._scale
            new_scale = max(self._min_scale, self._scale - self._scale_step)
            self._apply_scale_change(current_frame, current_bbox, old_scale, new_scale, avg_ms, "down")

        elif avg_ms < self._restore_headroom * self._budget_ms and self._scale < 1.0:
            old_scale = self._scale
            new_scale = min(1.0, self._scale + self._scale_step / 2.0)
            self._apply_scale_change(current_frame, current_bbox, old_scale, new_scale, avg_ms, "up")

    def _apply_scale_change(
        self,
        frame: np.ndarray,
        bbox: BBox,
        old_scale: float,
        new_scale: float,
        trigger_ms: float,
        direction: str,
    ) -> None:
        """Change scale and re-initialize the inner tracker at the new resolution."""
        self._scale = new_scale
        self._latency_window.clear()

        scaled_frame = self._scale_frame(frame, new_scale)
        scaled_bbox = self._scale_bbox(bbox, new_scale)
        self._tracker.initialize(scaled_frame, scaled_bbox)

        self.adaptation_events.append(
            AdaptationEvent(
                frame_idx=self._frame_idx,
                old_scale=old_scale,
                new_scale=new_scale,
                trigger_latency_ms=trigger_ms,
                budget_ms=self._budget_ms,
                direction=direction,
            )
        )

    @staticmethod
    def _scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
        """Resize a frame by a linear scale factor using area interpolation."""
        if scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(frame, (new_w, new_h), interpolation=interp)

    @staticmethod
    def _scale_bbox(bbox: BBox, scale: float) -> BBox:
        """Scale a bounding box by a linear factor."""
        x, y, w, h = bbox
        return (x * scale, y * scale, w * scale, h * scale)

    @staticmethod
    def _unscale_bbox(bbox: BBox, scale: float) -> BBox:
        """Map a bounding box from scaled coordinates back to original."""
        if scale == 1.0:
            return bbox
        x, y, w, h = bbox
        inv = 1.0 / scale
        return (x * inv, y * inv, w * inv, h * inv)

    def __repr__(self) -> str:
        return (
            f"AdaptiveBudgetTracker("
            f"tracker={self._tracker.name!r}, "
            f"budget_ms={self._budget_ms}, "
            f"scale=[{self._min_scale}, 1.0], "
            f"step={self._scale_step})"
        )
