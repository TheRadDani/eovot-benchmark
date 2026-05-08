"""Runtime compute-budget monitoring for adaptive tracker control.

Provides :class:`ComputeBudget` (hardware constraint specification) and
:class:`BudgetMonitor` (rolling-window latency tracker) used by
:class:`~eovot.trackers.adaptive.AdaptiveTracker` to decide whether to
route each frame to a fast fallback or a more accurate primary tracker.

Typical usage::

    from eovot.profiling.budget import BudgetMonitor, ComputeBudget

    budget = ComputeBudget(target_fps=30.0)
    monitor = BudgetMonitor(budget)

    for frame in video:
        use_primary = monitor.should_use_primary()
        t0 = time.perf_counter()
        bbox = primary.update(frame) if use_primary else fallback.update(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        monitor.record_frame(elapsed_ms, used_primary=use_primary)

    print(monitor.summary())
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np


@dataclass
class ComputeBudget:
    """Hardware resource constraints for real-time tracking.

    Attributes:
        target_fps:    Desired minimum frame rate.  Defines the per-frame
                       time budget: ``budget_ms = 1000 / target_fps``.
        max_memory_mb: Maximum allowed process memory in MiB.  ``None``
                       disables memory-based routing decisions.
        window:        Number of recent frames used for rolling statistics.
                       Default: ``20``.
        switch_margin: Fraction of the budget reserved as a safety margin.
                       If the rolling p75 latency exceeds
                       ``budget_ms × (1 − switch_margin)`` the monitor
                       recommends the fast fallback.  Default: ``0.10``.
    """

    target_fps: float
    max_memory_mb: Optional[float] = None
    window: int = 20
    switch_margin: float = 0.10

    def __post_init__(self) -> None:
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {self.target_fps}")
        if not 0.0 <= self.switch_margin < 1.0:
            raise ValueError(
                f"switch_margin must be in [0, 1), got {self.switch_margin}"
            )

    @property
    def budget_ms(self) -> float:
        """Per-frame time budget in milliseconds."""
        return 1_000.0 / self.target_fps


@dataclass
class RoutingDecision:
    """Record of a single frame's tracker-routing decision.

    Attributes:
        frame_index:  Zero-based index of the frame.
        used_primary: ``True`` when the primary tracker was selected.
        elapsed_ms:   Wall-clock time for the tracker update (ms).
        budget_ms:    Per-frame budget at the time of the decision (ms).
        over_budget:  ``True`` when ``elapsed_ms > budget_ms``.
    """

    frame_index: int
    used_primary: bool
    elapsed_ms: float
    budget_ms: float
    over_budget: bool

    @property
    def utilization(self) -> float:
        """Elapsed time as a fraction of the per-frame budget."""
        return self.elapsed_ms / self.budget_ms if self.budget_ms > 0 else 0.0


class BudgetMonitor:
    """Track rolling frame latency and recommend a tracker mode for each frame.

    The monitor maintains a sliding window of recent frame latencies.  It
    recommends the primary (accurate) tracker when the p75 latency is
    comfortably within the per-frame budget, and falls back to the fast
    tracker when the budget is being exceeded.

    During a *warm-up* period (fewer than ``window // 2`` recorded frames)
    the primary tracker is always recommended so the window fills with real
    measurements before making routing decisions.

    Args:
        budget: :class:`ComputeBudget` specifying target FPS and constraints.

    Example::

        budget = ComputeBudget(target_fps=30.0, window=10)
        monitor = BudgetMonitor(budget)

        for frame in frames:
            use_primary = monitor.should_use_primary()
            ...
            monitor.record_frame(elapsed_ms, used_primary=use_primary)

        print(monitor.summary())
    """

    def __init__(self, budget: ComputeBudget) -> None:
        self.budget = budget
        self._latencies: Deque[float] = deque(maxlen=budget.window)
        self._decisions: List[RoutingDecision] = []
        self._frame_idx: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record_frame(self, elapsed_ms: float, used_primary: bool) -> RoutingDecision:
        """Record the outcome of processing one frame.

        Args:
            elapsed_ms:   Wall-clock time taken by the tracker update (ms).
            used_primary: Whether the primary (accurate) tracker was used.

        Returns:
            :class:`RoutingDecision` for this frame.
        """
        self._latencies.append(elapsed_ms)
        over_budget = elapsed_ms > self.budget.budget_ms
        decision = RoutingDecision(
            frame_index=self._frame_idx,
            used_primary=used_primary,
            elapsed_ms=elapsed_ms,
            budget_ms=self.budget.budget_ms,
            over_budget=over_budget,
        )
        self._decisions.append(decision)
        self._frame_idx += 1
        return decision

    def should_use_primary(self) -> bool:
        """Recommend whether to use the primary tracker on the next frame.

        Returns:
            ``True``  — use the primary (accurate) tracker.
            ``False`` — use the fallback (fast) tracker.

        The recommendation is based on the p75 of recent frame latencies:

        * During warm-up (< ``window // 2`` frames) always returns ``True``.
        * Otherwise returns ``True`` iff
          ``p75_latency < budget_ms × (1 − switch_margin)``.
        """
        warm_up_frames = max(1, self.budget.window // 2)
        if len(self._latencies) < warm_up_frames:
            return True
        p75 = float(np.percentile(list(self._latencies), 75))
        threshold = self.budget.budget_ms * (1.0 - self.budget.switch_margin)
        return p75 < threshold

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def primary_ratio(self) -> float:
        """Fraction of frames routed to the primary tracker."""
        if not self._decisions:
            return 0.0
        return sum(1 for d in self._decisions if d.used_primary) / len(self._decisions)

    @property
    def budget_violation_rate(self) -> float:
        """Fraction of frames that exceeded the per-frame budget."""
        if not self._decisions:
            return 0.0
        return sum(1 for d in self._decisions if d.over_budget) / len(self._decisions)

    @property
    def mean_utilization(self) -> float:
        """Mean per-frame latency as a fraction of the budget."""
        if not self._decisions:
            return 0.0
        return float(np.mean([d.utilization for d in self._decisions]))

    def summary(self) -> dict:
        """Serialisable summary of routing statistics."""
        return {
            "target_fps": self.budget.target_fps,
            "budget_ms": round(self.budget.budget_ms, 3),
            "total_frames": self._frame_idx,
            "primary_ratio": round(self.primary_ratio, 4),
            "budget_violation_rate": round(self.budget_violation_rate, 4),
            "mean_utilization": round(self.mean_utilization, 4),
        }

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._latencies.clear()
        self._decisions.clear()
        self._frame_idx = 0
