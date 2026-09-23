"""Per-frame memory profiling for edge deployment analysis.

Unlike :class:`~eovot.profiling.profiler.Profiler`, which records only
peak RSS across an entire benchmark run, :class:`MemoryProfiler` snapshots
RSS after *every* frame so analysis can distinguish:

* **Spike** — a transient peak that returns to baseline (acceptable).
* **Steady growth** — monotonically increasing RSS over the run (potential leak).
* **Step increase** — a one-time allocation that then plateaus (common after
  model warm-up; acceptable once identified).

For long edge deployments even 1 KiB/frame accumulates to 1 MiB per 1000
frames — a meaningful concern on memory-limited hardware such as Raspberry Pi
or Jetson Nano.

Typical usage::

    from eovot.profiling.memory_profile import MemoryProfiler, MemoryBudgetChecker

    profiler = MemoryProfiler()
    tracker.initialize(first_frame, init_bbox)
    profiler.set_baseline()          # exclude model-load from per-frame stats

    for idx, frame in enumerate(sequence):
        tracker.update(frame)
        profiler.record_frame(idx)

    result = profiler.summary(tracker_name="KCF")
    print(result)                    # shows growth rate and leak flag

    checker = MemoryBudgetChecker(budget_mb=64.0)
    report  = checker.check(result)
    if not report.passes:
        raise RuntimeError(report.reason)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import psutil


@dataclass
class FrameMemorySnapshot:
    """Memory state immediately after one tracker update call."""

    frame_idx: int
    rss_mb: float
    """Resident Set Size in MiB at snapshot time."""
    delta_mb: float
    """RSS delta from the baseline recorded by :meth:`MemoryProfiler.set_baseline`."""


@dataclass
class MemoryProfileResult:
    """Per-sequence memory profiling summary."""

    tracker_name: str
    frame_count: int
    baseline_mb: float
    """RSS in MiB before the tracking loop started."""
    peak_mb: float
    """Absolute peak RSS in MiB observed during the run."""
    peak_delta_mb: float
    """Maximum RSS increase above baseline in MiB."""
    final_delta_mb: float
    """RSS delta at the last frame — net memory retained after the run."""
    mean_delta_mb: float
    """Mean RSS delta across all frames."""
    growth_rate_mb_per_frame: float
    """OLS linear regression slope (MiB/frame) of the delta series.
    Positive values indicate net memory growth over time."""
    is_leaking: bool
    """True when ``growth_rate_mb_per_frame`` exceeds the configured threshold."""
    snapshots: List[FrameMemorySnapshot] = field(default_factory=list)
    """Per-frame snapshots — populated only when ``include_snapshots=True``."""

    def __str__(self) -> str:
        leak_flag = " [LEAK DETECTED]" if self.is_leaking else ""
        return (
            f"MemoryProfileResult[{self.tracker_name}]{leak_flag}  "
            f"baseline={self.baseline_mb:.1f} MiB  "
            f"peak_delta=+{self.peak_delta_mb:.2f} MiB  "
            f"final_delta=+{self.final_delta_mb:.2f} MiB  "
            f"growth={self.growth_rate_mb_per_frame * 1000:.3f} KiB/frame  "
            f"frames={self.frame_count}"
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON export."""
        return {
            "tracker_name": self.tracker_name,
            "frame_count": self.frame_count,
            "baseline_mb": self.baseline_mb,
            "peak_mb": self.peak_mb,
            "peak_delta_mb": self.peak_delta_mb,
            "final_delta_mb": self.final_delta_mb,
            "mean_delta_mb": self.mean_delta_mb,
            "growth_rate_mb_per_frame": self.growth_rate_mb_per_frame,
            "is_leaking": self.is_leaking,
        }


class MemoryProfiler:
    """Track per-frame RSS memory to detect growth and leaks.

    Args:
        leak_threshold_mb_per_frame: OLS growth-rate threshold above which
            :attr:`MemoryProfileResult.is_leaking` is set to ``True``.
            Default 1e-3 MiB/frame (1 KiB/frame).

    Usage::

        profiler = MemoryProfiler()
        tracker.initialize(first_frame, init_bbox)
        profiler.set_baseline()         # after init, before loop
        for idx, frame in enumerate(frames):
            tracker.update(frame)
            profiler.record_frame(idx)
        result = profiler.summary("MOSSE")
    """

    DEFAULT_LEAK_THRESHOLD_MB_PER_FRAME: float = 1e-3

    def __init__(
        self,
        leak_threshold_mb_per_frame: float = DEFAULT_LEAK_THRESHOLD_MB_PER_FRAME,
    ) -> None:
        self._process = psutil.Process(os.getpid())
        self._baseline_mb: float = 0.0
        self._snapshots: List[FrameMemorySnapshot] = []
        self.leak_threshold_mb_per_frame = leak_threshold_mb_per_frame

    def set_baseline(self) -> float:
        """Record current RSS as the pre-tracking baseline.

        Call immediately before the tracking loop (after
        ``tracker.initialize()``) to exclude model-loading from per-frame
        analysis.

        Returns:
            Current baseline RSS in MiB.
        """
        self._baseline_mb = self._rss_mb()
        self._snapshots.clear()
        return self._baseline_mb

    def record_frame(self, frame_idx: int) -> FrameMemorySnapshot:
        """Snapshot RSS immediately after processing one frame.

        Args:
            frame_idx: Zero-based frame index (used for ordering; need not
                be contiguous).

        Returns:
            The :class:`FrameMemorySnapshot` just recorded.
        """
        rss = self._rss_mb()
        snap = FrameMemorySnapshot(
            frame_idx=frame_idx,
            rss_mb=rss,
            delta_mb=rss - self._baseline_mb,
        )
        self._snapshots.append(snap)
        return snap

    def summary(
        self,
        tracker_name: str = "unknown",
        include_snapshots: bool = False,
    ) -> MemoryProfileResult:
        """Compute the memory profile summary.

        Args:
            tracker_name:      Name tag embedded in the result.
            include_snapshots: Embed per-frame snapshots in the result for
                               downstream plotting or further analysis.

        Returns:
            :class:`MemoryProfileResult` with growth rate and leak flag.

        Raises:
            ValueError: If no frames have been recorded yet.
        """
        if not self._snapshots:
            raise ValueError("No frames recorded. Call record_frame() first.")

        deltas = np.array([s.delta_mb for s in self._snapshots])
        rss_vals = np.array([s.rss_mb for s in self._snapshots])
        frames = np.arange(len(deltas), dtype=float)

        growth_rate = _linear_slope(frames, deltas)

        return MemoryProfileResult(
            tracker_name=tracker_name,
            frame_count=len(self._snapshots),
            baseline_mb=self._baseline_mb,
            peak_mb=float(rss_vals.max()),
            peak_delta_mb=float(deltas.max()),
            final_delta_mb=float(deltas[-1]),
            mean_delta_mb=float(deltas.mean()),
            growth_rate_mb_per_frame=growth_rate,
            is_leaking=growth_rate > self.leak_threshold_mb_per_frame,
            snapshots=list(self._snapshots) if include_snapshots else [],
        )

    def reset(self) -> None:
        """Clear all snapshots and the baseline for reuse across sequences."""
        self._baseline_mb = 0.0
        self._snapshots.clear()

    def _rss_mb(self) -> float:
        return self._process.memory_info().rss / (1024 ** 2)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return the OLS slope of *y* on *x* (ordinary least squares)."""
    if len(x) < 2:
        return 0.0
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()
    if denom == 0.0:
        return 0.0
    return float((x_c * (y - y.mean())).sum() / denom)


@dataclass
class BudgetCheckReport:
    """Result of a :class:`MemoryBudgetChecker` evaluation."""

    passes: bool
    reason: str
    excess_mb: float = 0.0
    """How far the peak delta exceeded the budget (0.0 when within budget)."""

    def __str__(self) -> str:
        return self.reason


class MemoryBudgetChecker:
    """Gate a :class:`MemoryProfileResult` against a device memory budget.

    Useful for automated deployment checks in CI that should fail when a
    tracker exceeds the memory allowance of the target hardware profile.

    Args:
        budget_mb: Maximum acceptable peak memory delta in MiB.

    Example::

        checker = MemoryBudgetChecker(budget_mb=64.0)
        report  = checker.check(result)
        if not report.passes:
            raise RuntimeError(report.reason)
    """

    def __init__(self, budget_mb: float) -> None:
        if budget_mb <= 0:
            raise ValueError("budget_mb must be positive.")
        self.budget_mb = budget_mb

    def check(self, result: MemoryProfileResult) -> BudgetCheckReport:
        """Evaluate *result* against the configured budget.

        A result **fails** when either ``peak_delta_mb > budget_mb`` or
        ``is_leaking`` is True.  Budget excess is reported even when the run
        also leaks.

        Returns:
            :class:`BudgetCheckReport` with a ``passes`` flag and a
            human-readable ``reason`` string.
        """
        over_budget = result.peak_delta_mb > self.budget_mb
        excess_mb = max(0.0, result.peak_delta_mb - self.budget_mb)

        if over_budget:
            reason = (
                f"FAIL — peak delta {result.peak_delta_mb:.2f} MiB exceeds "
                f"budget {self.budget_mb:.0f} MiB "
                f"(excess {excess_mb:.2f} MiB)."
            )
            return BudgetCheckReport(passes=False, reason=reason, excess_mb=excess_mb)

        if result.is_leaking:
            reason = (
                f"WARNING — peak delta {result.peak_delta_mb:.2f} MiB fits "
                f"budget but growth rate "
                f"{result.growth_rate_mb_per_frame * 1000:.2f} KiB/frame "
                f"exceeds leak threshold."
            )
            return BudgetCheckReport(passes=False, reason=reason, excess_mb=0.0)

        reason = (
            f"OK — peak delta {result.peak_delta_mb:.2f} MiB "
            f"< budget {self.budget_mb:.0f} MiB, no leak detected."
        )
        return BudgetCheckReport(passes=True, reason=reason, excess_mb=0.0)
