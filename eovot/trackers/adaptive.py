"""Adaptive latency-budget tracker for real-time edge deployment.

:class:`AdaptiveLatencyTracker` wraps any :class:`~.base.BaseTracker` and
dynamically adjusts its frame-skip rate at runtime to maintain a target FPS
budget.  Unlike the static :class:`~eovot.trackers.frame_skip.FrameSkipTracker`,
which fixes the skip interval before evaluation begins, this controller closes
a feedback loop on measured latency:

- **Over-budget** (EMA latency > target): skip_rate is incremented by 1
  (fewer tracker calls per second, trading accuracy for throughput).
- **Under-budget with headroom** (EMA latency < target * slack): skip_rate
  is decremented by 1 (recovering accuracy when compute headroom exists).
- **Within budget**: skip_rate is held constant for the current frame.

This models real-world edge deployment conditions where thermal throttling,
OS scheduler jitter, or varying-complexity frames cause throughput to drift
even when the average frame rate looks acceptable.

Latency estimation
------------------
Per-frame execution times are smoothed with an exponential moving average
(EMA)::

    ema_t = alpha * measured_t + (1 - alpha) * ema_t_prev

Low ``alpha`` (default ``0.1``) favours stability over responsiveness;
high ``alpha`` (→ 1.0) reacts instantly to individual spikes.  An initial
warm-up period of ``warmup_frames`` active frames is used to seed the EMA
before adaptation begins, preventing over-reaction to cold-start latency.

Adaptation log
--------------
Every call to :meth:`update` appends an :class:`AdaptationEvent` to
:attr:`adaptation_log`, recording the EMA latency, current skip_rate, and
whether the rate changed.  Use :meth:`adaptation_summary` to get aggregate
statistics without iterating the log manually.

Skipped frames receive the last active prediction (``"repeat"`` mode), identical
to :class:`~eovot.trackers.frame_skip.FrameSkipTracker`.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive import AdaptiveLatencyTracker

    base    = MOSSETracker()
    tracker = AdaptiveLatencyTracker(base, target_fps=30.0)

    tracker.initialize(first_frame, init_bbox)
    for frame in remaining_frames:
        pred = tracker.update(frame)

    summary = tracker.adaptation_summary()
    print(f"Adaptation events : {summary['num_adaptations']}")
    print(f"Mean skip rate    : {summary['mean_skip_rate']:.2f}")
    print(f"Mean EMA latency  : {summary['mean_ema_latency_ms']:.2f} ms")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox

_BBox = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AdaptationEvent:
    """One frame's adaptation record from :class:`AdaptiveLatencyTracker`.

    Attributes:
        frame_index:      0-based counter across all frames since initialization.
        ema_latency_ms:   EMA-smoothed latency at this frame (ms).
        skip_rate_before: skip_rate value at the start of this frame.
        skip_rate_after:  skip_rate value after any adjustment (same if unchanged).
        was_active:       ``True`` when the underlying tracker was actually called;
                          ``False`` for passively-skipped frames.
    """

    frame_index: int
    ema_latency_ms: float
    skip_rate_before: int
    skip_rate_after: int
    was_active: bool

    @property
    def rate_changed(self) -> bool:
        """``True`` when this frame triggered an adaptation."""
        return self.skip_rate_after != self.skip_rate_before


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class AdaptiveLatencyTracker(BaseTracker):
    """Wrapper that dynamically adapts frame-skip rate to meet a FPS budget.

    Args:
        tracker:        Any :class:`~.base.BaseTracker` to wrap.  Its
                        ``initialize()`` and ``update()`` implement the
                        actual tracking algorithm.
        target_fps:     Target throughput in frames per second.  The
                        controller tries to keep ``1 / ema_latency_s``
                        at or above this value.  Must be > 0.
        min_skip:       Minimum skip rate (≥ 1).  ``1`` means every frame
                        is processed — no skipping.
        max_skip:       Maximum skip rate.  Caps adaptation to avoid
                        excessive accuracy loss.  Default ``8``.
        ema_alpha:      EMA smoothing factor in ``(0, 1]``.  Low values
                        react slowly to latency changes; high values
                        react aggressively to individual spikes.
                        Default ``0.10``.
        slack:          Fraction of budget below which the controller
                        decreases skip_rate.  ``slack=0.75`` means
                        ``skip_rate`` is reduced only when EMA latency
                        is below 75 % of the target budget.  This prevents
                        oscillation around the threshold.  Default ``0.75``.
        warmup_frames:  Number of initial active frames used to seed the
                        EMA before adaptation starts.  During warm-up,
                        skip_rate stays at ``min_skip``.  Default ``5``.

    Raises:
        ValueError: On invalid parameter combinations.

    Example::

        tracker = AdaptiveLatencyTracker(KCFTracker(), target_fps=25.0, max_skip=4)
        tracker.initialize(frame0, bbox0)
        for frame in frames[1:]:
            bbox = tracker.update(frame)
        print(tracker.adaptation_summary())
    """

    def __init__(
        self,
        tracker: BaseTracker,
        target_fps: float = 30.0,
        min_skip: int = 1,
        max_skip: int = 8,
        ema_alpha: float = 0.10,
        slack: float = 0.75,
        warmup_frames: int = 5,
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if not (1 <= min_skip <= max_skip):
            raise ValueError(
                f"Need 1 ≤ min_skip ≤ max_skip, got min_skip={min_skip}, max_skip={max_skip}"
            )
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if not (0.0 < slack < 1.0):
            raise ValueError(f"slack must be in (0, 1), got {slack}")
        if warmup_frames < 0:
            raise ValueError(f"warmup_frames must be ≥ 0, got {warmup_frames}")

        super().__init__(name=f"{tracker.name}_adaptive{target_fps:.0f}fps")
        self._tracker = tracker
        self.target_fps = target_fps
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.ema_alpha = ema_alpha
        self.slack = slack
        self.warmup_frames = warmup_frames

        # State reset by initialize()
        self._skip_rate: int = min_skip
        self._ema_latency_ms: float = 1_000.0 / target_fps   # seed with budget
        self._frame_index: int = 0
        self._active_count: int = 0                           # active frames since init
        self._last_pred: Optional[_BBox] = None
        self.adaptation_log: List[AdaptationEvent] = []

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialize the wrapped tracker and reset adaptation state.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8 array.
            bbox:  Ground-truth ``(x, y, w, h)`` for the first frame.
        """
        self._tracker.initialize(frame, bbox)
        self._last_pred = tuple(bbox)  # type: ignore[assignment]
        self._skip_rate = self.min_skip
        self._ema_latency_ms = 1_000.0 / self.target_fps
        self._frame_index = 0
        self._active_count = 0
        self.adaptation_log = []

    def update(self, frame: np.ndarray) -> BBox:
        """Return the predicted bounding box for the current frame.

        On *active* frames (index divisible by ``skip_rate``), the underlying
        tracker is called and latency is measured.  On *passive* frames the
        last active prediction is returned immediately (zero extra latency).
        After each active frame the EMA is updated and skip_rate is adjusted.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8 array.

        Returns:
            Predicted ``(x, y, w, h)`` bounding box.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._last_pred is None:
            raise RuntimeError("update() called before initialize()")

        self._frame_index += 1
        skip_rate_before = self._skip_rate
        is_active = (self._frame_index % self._skip_rate == 0)

        if is_active:
            t0 = time.perf_counter()
            pred = self._tracker.update(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1_000.0
            self._last_pred = pred
            self._active_count += 1

            # Update EMA only after warm-up frames
            if self._active_count > self.warmup_frames:
                self._ema_latency_ms = (
                    self.ema_alpha * elapsed_ms
                    + (1.0 - self.ema_alpha) * self._ema_latency_ms
                )
                self._skip_rate = self._adapt(self._ema_latency_ms)
        else:
            pred = self._last_pred  # type: ignore[assignment]

        self.adaptation_log.append(
            AdaptationEvent(
                frame_index=self._frame_index,
                ema_latency_ms=self._ema_latency_ms,
                skip_rate_before=skip_rate_before,
                skip_rate_after=self._skip_rate,
                was_active=is_active,
            )
        )
        return pred

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @property
    def num_adaptations(self) -> int:
        """Number of frames on which skip_rate was changed."""
        return sum(1 for e in self.adaptation_log if e.rate_changed)

    @property
    def mean_skip_rate(self) -> float:
        """Mean skip_rate across all logged frames (1.0 = no skipping)."""
        if not self.adaptation_log:
            return float(self._skip_rate)
        return float(np.mean([e.skip_rate_after for e in self.adaptation_log]))

    def adaptation_summary(self) -> Dict:
        """Return a plain dict summarising the adaptation trajectory.

        Returns:
            Dict with the following keys:

            * ``"target_fps"`` — configured FPS budget.
            * ``"num_frames"`` — total frames processed (active + passive).
            * ``"num_active_frames"`` — frames on which the tracker was called.
            * ``"num_adaptations"`` — frames that triggered a rate change.
            * ``"mean_skip_rate"`` — mean effective skip rate.
            * ``"final_skip_rate"`` — skip_rate at end of sequence.
            * ``"mean_ema_latency_ms"`` — mean EMA latency across active frames.
            * ``"effective_fps"`` — estimated throughput from EMA latency and
              mean skip_rate (approximation; not a wall-clock measurement).
        """
        if not self.adaptation_log:
            return {
                "target_fps": self.target_fps,
                "num_frames": 0,
                "num_active_frames": 0,
                "num_adaptations": 0,
                "mean_skip_rate": float(self._skip_rate),
                "final_skip_rate": self._skip_rate,
                "mean_ema_latency_ms": self._ema_latency_ms,
                "effective_fps": 0.0,
            }

        active_events = [e for e in self.adaptation_log if e.was_active]
        mean_ema = float(np.mean([e.ema_latency_ms for e in active_events])) if active_events else 0.0
        mean_skip = float(np.mean([e.skip_rate_after for e in self.adaptation_log]))
        # Effective FPS: tracker runs every mean_skip frames at mean_ema ms/active-frame,
        # but only the tracker call latency is measured; passive frames are near-zero.
        effective_fps = (
            (1_000.0 / mean_ema) * mean_skip if mean_ema > 0 else 0.0
        )

        return {
            "target_fps": self.target_fps,
            "num_frames": len(self.adaptation_log),
            "num_active_frames": len(active_events),
            "num_adaptations": self.num_adaptations,
            "mean_skip_rate": round(mean_skip, 3),
            "final_skip_rate": self.adaptation_log[-1].skip_rate_after,
            "mean_ema_latency_ms": round(mean_ema, 3),
            "effective_fps": round(effective_fps, 2),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adapt(self, ema_ms: float) -> int:
        """Return updated skip_rate given the current EMA latency.

        Args:
            ema_ms: Current EMA latency in milliseconds.

        Returns:
            New skip_rate clamped to ``[min_skip, max_skip]``.
        """
        budget_ms = 1_000.0 / self.target_fps
        if ema_ms > budget_ms:
            # Over-budget: increase skip_rate to reduce tracker calls
            return min(self._skip_rate + 1, self.max_skip)
        elif ema_ms < budget_ms * self.slack:
            # Comfortable headroom: recover accuracy by reducing skip_rate
            return max(self._skip_rate - 1, self.min_skip)
        return self._skip_rate
