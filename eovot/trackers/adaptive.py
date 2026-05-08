"""Adaptive budget-aware tracker that routes frames between two sub-trackers.

:class:`AdaptiveTracker` wraps a *primary* (accurate, potentially slow) and a
*fallback* (fast, potentially less accurate) tracker.  A
:class:`~eovot.profiling.budget.BudgetMonitor` tracks rolling frame latency
and selects which tracker to trust for each prediction.

Both sub-trackers receive every ``initialize`` and ``update`` call so that
the fallback stays warm and can take over cleanly at any point without a
cold-start penalty.

This design directly targets edge deployment scenarios (Raspberry Pi 4,
Jetson Nano) where a fixed tracker may either be too slow for the target
frame rate or leave CPU headroom unused.

Typical usage::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.adaptive import AdaptiveTracker
    from eovot.profiling.budget import ComputeBudget

    budget  = ComputeBudget(target_fps=30.0, switch_margin=0.10)
    tracker = AdaptiveTracker(
        primary=KCFTracker(),
        fallback=MOSSETracker(),
        budget=budget,
    )

    tracker.initialize(first_frame, init_bbox)
    for frame in remaining_frames:
        bbox = tracker.update(frame)

    print(tracker.routing_summary)
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from .base import BaseTracker, BBox
from ..profiling.budget import BudgetMonitor, ComputeBudget, RoutingDecision


class AdaptiveTracker(BaseTracker):
    """Budget-aware tracker that dynamically switches between two sub-trackers.

    Each frame the :class:`~eovot.profiling.budget.BudgetMonitor` consults
    the rolling p75 latency to decide whether the primary tracker is keeping
    pace with the target FPS.  When it is, the primary tracker's prediction
    is used; when it isn't, the fallback's prediction is returned instead.

    **Both** sub-trackers are always updated on every frame, ensuring:

    * No cold-start when switching modes mid-sequence.
    * Accurate profiling of the combined per-frame cost (reported in the
      benchmark engine as total ``elapsed_ms``).

    Args:
        primary:  High-accuracy tracker (e.g. KCF, CSRT).
        fallback: Fast tracker that stays within budget (e.g. MOSSE).
        budget:   :class:`~eovot.profiling.budget.ComputeBudget` specifying
                  the target FPS and switch-margin.
        name:     Optional display name.  Defaults to
                  ``"Adaptive(<primary.name>/<fallback.name>)"``.

    Example::

        budget  = ComputeBudget(target_fps=30.0)
        tracker = AdaptiveTracker(
            primary=KCFTracker(),
            fallback=MOSSETracker(),
            budget=budget,
        )
    """

    def __init__(
        self,
        primary: BaseTracker,
        fallback: BaseTracker,
        budget: ComputeBudget,
        name: Optional[str] = None,
    ) -> None:
        display_name = name or f"Adaptive({primary.name}/{fallback.name})"
        super().__init__(display_name)
        self.primary = primary
        self.fallback = fallback
        self.budget = budget
        self.monitor = BudgetMonitor(budget)
        self._decisions: List[RoutingDecision] = []

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise both sub-trackers and reset the budget monitor.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self.primary.initialize(frame, bbox)
        self.fallback.initialize(frame, bbox)
        self.monitor.reset()
        self._decisions.clear()

    def update(self, frame: np.ndarray) -> BBox:
        """Update both sub-trackers; return the selected tracker's prediction.

        The selection (primary or fallback) is determined by
        :meth:`~eovot.profiling.budget.BudgetMonitor.should_use_primary`
        *before* the update so the decision is based on historical latency
        rather than the current frame's timing.

        The total elapsed time (both tracker updates combined) is recorded
        in the monitor so future routing decisions reflect the true per-frame
        cost of running both trackers in parallel.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` from the selected tracker.
        """
        use_primary = self.monitor.should_use_primary()

        t0 = time.perf_counter()
        pred_primary = self.primary.update(frame)
        pred_fallback = self.fallback.update(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1_000.0

        decision = self.monitor.record_frame(elapsed_ms, used_primary=use_primary)
        self._decisions.append(decision)

        return pred_primary if use_primary else pred_fallback

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    @property
    def routing_summary(self) -> dict:
        """Routing statistics for the frames processed since last ``initialize``.

        Returns a dict with keys: ``target_fps``, ``budget_ms``,
        ``total_frames``, ``primary_ratio``, ``budget_violation_rate``,
        ``mean_utilization``.
        """
        return self.monitor.summary()

    @property
    def decisions(self) -> List[RoutingDecision]:
        """Per-frame routing decisions since last ``initialize`` (read-only)."""
        return list(self._decisions)
