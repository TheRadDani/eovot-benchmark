"""BudgetAdaptiveTracker — real-time skip-rate adaptation for latency budgets.

Edge devices have hard real-time constraints: a 33 FPS video pipeline
gives roughly 30 ms per frame, and any tracker that occasionally spikes
above that budget will cause dropped frames or downstream stalls.

Frame skipping is the lightest-weight remedy: skip *k-1* frames and
propagate the last prediction, cutting compute by approximately *k*×.
But a fixed skip rate is pessimistic — it wastes accuracy on frames that
arrive well within budget.

:class:`BudgetAdaptiveTracker` implements a feedback loop that expands
or contracts the skip rate at runtime:

- Measures per-frame latency with :func:`time.perf_counter`.
- Maintains an exponential moving average (EMA) of latency.
- Increases the skip rate when EMA > ``budget_ms``.
- Decreases the skip rate when EMA < ``budget_ms × recovery_ratio``.
- Records every adaptation decision in ``adaptation_log`` for analysis.

This enables autonomous adaptation to varying compute budgets without
any external profiling infrastructure, and produces per-run telemetry
(histogram, compliance rate, adaptation events) that plugs directly into
the EOVOT analysis pipeline.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive_budget import BudgetAdaptiveTracker
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    base    = MOSSETracker()
    tracker = BudgetAdaptiveTracker(base, budget_ms=25.0, max_skip=4)

    engine  = BenchmarkEngine(verbose=False)
    dataset = SyntheticDataset(num_sequences=5)
    result  = engine.run(tracker, dataset, dataset_name="Synthetic")

    print(tracker.to_report())
    print(f"Mean skip rate: {tracker.mean_skip_rate:.2f}x")
    print(f"Budget compliance: {tracker.budget_compliance_rate:.1%}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .base import BaseTracker, BBox


@dataclass
class AdaptationEvent:
    """One skip-rate change recorded by :class:`BudgetAdaptiveTracker`."""

    frame_idx: int
    """1-indexed frame number at which the rate changed."""
    old_skip: int
    """Skip rate before the change."""
    new_skip: int
    """Skip rate after the change."""
    ema_latency_ms: float
    """EMA latency (ms) that triggered the adaptation."""
    reason: str
    """``"over_budget"`` or ``"under_budget"``."""

    def __str__(self) -> str:
        direction = "↑" if self.new_skip > self.old_skip else "↓"
        return (
            f"frame={self.frame_idx:>5}  "
            f"skip {self.old_skip}→{self.new_skip} {direction}  "
            f"ema={self.ema_latency_ms:.2f}ms  [{self.reason}]"
        )


class BudgetAdaptiveTracker(BaseTracker):
    """Tracker wrapper that adapts frame-skip rate to a latency budget.

    Monitors per-frame processing time via an exponential moving average
    (EMA) of observed latencies and adjusts the internal skip rate:

    - If EMA latency > ``budget_ms``: increment skip rate (up to
      ``max_skip``) to reduce compute load immediately.
    - If EMA latency < ``budget_ms × recovery_ratio``: decrement skip
      rate (down to 1) to recover tracking accuracy.

    On skipped frames the last valid prediction is repeated (``"repeat"``
    propagation, same as :class:`~eovot.trackers.frame_skip.FrameSkipTracker`
    in its default mode).  The underlying tracker's ``initialize()`` always
    runs on the first call regardless of budget state.

    Args:
        tracker:        The underlying :class:`BaseTracker` to wrap.
        budget_ms:      Target per-frame latency budget in milliseconds.
                        Must be > 0.
        max_skip:       Maximum allowed skip rate (≥ 1).  A value of 4
                        means at most 3 consecutive frames are skipped.
        ema_alpha:      EMA smoothing factor in ``(0, 1]``.  Higher values
                        react faster to latency spikes.  Default 0.2.
        recovery_ratio: Fraction of ``budget_ms`` below which the skip rate
                        is decreased.  Default 0.70 (recover at 70 % of
                        the budget).
        warmup_frames:  Frames before adaptation kicks in.  The EMA is
                        seeded during this window.  Default 5.

    Raises:
        ValueError: If ``budget_ms <= 0``, ``max_skip < 1``, or
                    ``ema_alpha`` outside ``(0, 1]``.

    Attributes:
        adaptation_log: List of :class:`AdaptationEvent` objects, one per
            skip-rate change.  Cleared on :meth:`initialize`.

    Example::

        tracker = BudgetAdaptiveTracker(KCFTracker(), budget_ms=20.0)
        tracker.initialize(frame0, bbox0)
        for frame in remaining_frames:
            pred = tracker.update(frame)
        print(tracker.mean_skip_rate)
        print(tracker.budget_compliance_rate)
    """

    def __init__(
        self,
        tracker: BaseTracker,
        budget_ms: float,
        max_skip: int = 4,
        ema_alpha: float = 0.2,
        recovery_ratio: float = 0.70,
        warmup_frames: int = 5,
    ) -> None:
        if budget_ms <= 0:
            raise ValueError(f"budget_ms must be > 0, got {budget_ms}")
        if max_skip < 1:
            raise ValueError(f"max_skip must be >= 1, got {max_skip}")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if not (0.0 < recovery_ratio < 1.0):
            raise ValueError(f"recovery_ratio must be in (0, 1), got {recovery_ratio}")

        super().__init__(name=f"{tracker.name}_budget{budget_ms:.0f}ms")
        self._tracker = tracker
        self.budget_ms = budget_ms
        self.max_skip = max_skip
        self.ema_alpha = ema_alpha
        self.recovery_ratio = recovery_ratio
        self.warmup_frames = warmup_frames

        # Runtime state — reset on initialize()
        self._skip_rate: int = 1
        self._frame_idx: int = 0
        self._frames_since_tracker_ran: int = 0
        self._last_bbox: Optional[BBox] = None
        self._ema_latency: Optional[float] = None

        # Telemetry
        self.adaptation_log: List[AdaptationEvent] = []
        self._skip_rate_counts: Dict[int, int] = {}
        self._all_latencies: List[float] = []

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the underlying tracker and reset all adaptation state.

        Args:
            frame: BGR image ``(H, W, 3)``.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._skip_rate = 1
        self._frame_idx = 0
        self._frames_since_tracker_ran = 0
        self._last_bbox = bbox
        self._ema_latency = None
        self.adaptation_log.clear()
        self._skip_rate_counts.clear()
        self._all_latencies.clear()

    def update(self, frame: np.ndarray) -> BBox:
        """Predict target location, adapting skip rate from measured latency.

        On an *active* frame (every ``skip_rate``-th frame), the underlying
        tracker runs and latency is measured.  On a *passive* frame the last
        active prediction is repeated without calling the underlying tracker.

        Args:
            frame: BGR image ``(H, W, 3)``.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._last_bbox is None:
            raise RuntimeError(
                "BudgetAdaptiveTracker has not been initialised. "
                "Call initialize() first."
            )

        self._frame_idx += 1
        self._frames_since_tracker_ran += 1

        # Record which skip rate this frame was charged to
        self._skip_rate_counts[self._skip_rate] = (
            self._skip_rate_counts.get(self._skip_rate, 0) + 1
        )

        if self._frames_since_tracker_ran >= self._skip_rate:
            # Active frame: run the real tracker and measure latency
            t0 = time.perf_counter()
            self._last_bbox = self._tracker.update(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1_000.0
            self._frames_since_tracker_ran = 0
            self._all_latencies.append(elapsed_ms)

            # Update EMA
            if self._ema_latency is None:
                self._ema_latency = elapsed_ms
            else:
                self._ema_latency = (
                    self.ema_alpha * elapsed_ms
                    + (1.0 - self.ema_alpha) * self._ema_latency
                )

            # Adapt after warmup period
            if self._frame_idx > self.warmup_frames:
                self._adapt()

        return self._last_bbox

    # ------------------------------------------------------------------
    # Adaptation logic
    # ------------------------------------------------------------------

    def _adapt(self) -> None:
        """Increase or decrease skip rate based on current EMA latency."""
        if self._ema_latency is None:
            return

        old_skip = self._skip_rate

        if self._ema_latency > self.budget_ms and self._skip_rate < self.max_skip:
            self._skip_rate += 1
            self.adaptation_log.append(
                AdaptationEvent(
                    frame_idx=self._frame_idx,
                    old_skip=old_skip,
                    new_skip=self._skip_rate,
                    ema_latency_ms=self._ema_latency,
                    reason="over_budget",
                )
            )

        elif (
            self._ema_latency < self.budget_ms * self.recovery_ratio
            and self._skip_rate > 1
        ):
            self._skip_rate -= 1
            self.adaptation_log.append(
                AdaptationEvent(
                    frame_idx=self._frame_idx,
                    old_skip=old_skip,
                    new_skip=self._skip_rate,
                    ema_latency_ms=self._ema_latency,
                    reason="under_budget",
                )
            )

    # ------------------------------------------------------------------
    # Telemetry properties
    # ------------------------------------------------------------------

    @property
    def underlying_tracker(self) -> BaseTracker:
        """The wrapped :class:`BaseTracker` instance."""
        return self._tracker

    @property
    def current_skip_rate(self) -> int:
        """Current skip rate at the end of the last processed frame."""
        return self._skip_rate

    @property
    def skip_rate_histogram(self) -> Dict[int, int]:
        """Frames spent at each skip rate: ``{skip_rate: frame_count}``."""
        return dict(self._skip_rate_counts)

    @property
    def mean_skip_rate(self) -> float:
        """Frame-weighted mean skip rate over the entire run."""
        total = sum(self._skip_rate_counts.values())
        if total == 0:
            return float(self._skip_rate)
        return sum(k * v for k, v in self._skip_rate_counts.items()) / total

    @property
    def budget_compliance_rate(self) -> float:
        """Fraction of active frames whose measured latency was ≤ ``budget_ms``."""
        if not self._all_latencies:
            return 0.0
        arr = np.array(self._all_latencies)
        return float((arr <= self.budget_ms).mean())

    @property
    def ema_latency_ms(self) -> Optional[float]:
        """Current EMA latency estimate, or ``None`` before any active frame."""
        return self._ema_latency

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_report(self) -> str:
        """Human-readable summary of adaptation behaviour over the last run."""
        hist = self.skip_rate_histogram
        hist_str = "  ".join(
            f"skip={k}: {v}f" for k, v in sorted(hist.items())
        )
        ema = (
            f"{self._ema_latency:.2f} ms"
            if self._ema_latency is not None
            else "n/a"
        )
        lines = [
            f"BudgetAdaptiveTracker [{self._tracker.name}]",
            f"  budget_ms       = {self.budget_ms:.1f} ms",
            f"  max_skip        = {self.max_skip}",
            f"  final_skip_rate = {self._skip_rate}",
            f"  mean_skip_rate  = {self.mean_skip_rate:.2f}×",
            f"  ema_latency     = {ema}",
            f"  compliance_rate = {self.budget_compliance_rate:.1%}",
            f"  adaptations     = {len(self.adaptation_log)}",
            f"  frame_histogram = {hist_str}",
        ]
        if self.adaptation_log:
            lines.append("  adaptation_log:")
            for event in self.adaptation_log[:10]:
                lines.append(f"    {event}")
            if len(self.adaptation_log) > 10:
                lines.append(
                    f"    ... and {len(self.adaptation_log) - 10} more events"
                )
        return "\n".join(lines)
