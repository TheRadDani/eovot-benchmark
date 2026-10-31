"""Latency-adaptive frame-skip tracker for edge deployment.

:class:`LatencyAdaptiveTracker` wraps *any* :class:`~eovot.trackers.base.BaseTracker`
with a **dynamic** frame-skip policy driven by real-time latency monitoring.

Unlike :class:`~eovot.trackers.frame_skip.FrameSkipTracker` — which uses a
fixed skip rate configured at construction time — this tracker measures actual
per-frame inference latency and adapts on the fly to stay within a
user-specified latency budget (e.g. 33.3 ms for 30 FPS, 16.7 ms for 60 FPS).

Adaptation policy
-----------------
The policy is based on a **sliding window** of the last ``window_size``
per-frame latencies:

1. After each active (non-skipped) tracker update, the elapsed time is
   recorded in a circular buffer.
2. Before the next frame, the mean latency over the window is compared to
   ``latency_budget_ms × skip_threshold_ratio``.
3. If the rolling mean **exceeds** the threshold, the skip counter is
   incremented — the tracker enters a "skip run" of up to ``max_skip``
   consecutive skipped frames.
4. After a skip run, one active frame is processed to re-anchor the
   position estimate; then the policy re-evaluates.

When a frame is skipped, the last predicted bounding box is returned
unchanged (``"repeat"`` mode) or linearly extrapolated from the last two
predictions (``"linear"`` mode) — matching the behaviour of the fixed
:class:`~eovot.trackers.frame_skip.FrameSkipTracker`.

Why this matters for EOVOT
--------------------------
Edge devices experience variable load from background processes, thermal
throttling, and memory pressure.  A fixed skip rate that is safe at peak
load wastes throughput under normal conditions; a fixed rate tuned for
normal load will violate latency SLAs when the device is stressed.  The
adaptive policy automatically navigates this tradeoff, enabling
*latency-budget-compliant* tracking without sacrificing accuracy more than
necessary.

Typical usage::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive_skip import LatencyAdaptiveTracker
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=3)
    engine  = BenchmarkEngine(verbose=False)

    # 30 FPS budget — skip frames when rolling latency exceeds 26.7 ms (= 33.3 × 0.8)
    adaptive = LatencyAdaptiveTracker(MOSSETracker(), latency_budget_ms=33.3)
    result   = engine.run(adaptive, dataset, dataset_name="Synthetic-Adaptive")

    stats = adaptive.get_stats()
    print(f"Effective skip rate: {stats.skip_rate:.1%}")
    print(f"Budget violations:   {stats.budget_violations}")
    print(result)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox

SkipMode = Literal["repeat", "linear"]


@dataclass
class AdaptiveSkipStats:
    """Runtime statistics emitted by :class:`LatencyAdaptiveTracker`.

    Attributes:
        total_frames:         Total frames seen (initialize frame excluded).
        active_frames:        Frames on which the wrapped tracker actually ran.
        skipped_frames:       Frames returned from the skip policy without inference.
        skip_rate:            ``skipped / total`` — effective skip fraction.
        mean_active_latency_ms: Mean per-frame latency over all *active* frames (ms).
        latency_budget_ms:    The configured budget this tracker was tuned for.
        budget_violations:    Active frames where latency exceeded ``latency_budget_ms``.
        max_consecutive_skip: Longest run of consecutively skipped frames.
    """

    total_frames: int
    active_frames: int
    skipped_frames: int
    skip_rate: float
    mean_active_latency_ms: float
    latency_budget_ms: float
    budget_violations: int
    max_consecutive_skip: int

    def __str__(self) -> str:
        return (
            f"AdaptiveSkipStats("
            f"total={self.total_frames}, "
            f"active={self.active_frames}, "
            f"skipped={self.skipped_frames} ({self.skip_rate:.1%}), "
            f"mean_lat={self.mean_active_latency_ms:.2f} ms, "
            f"budget={self.latency_budget_ms:.1f} ms, "
            f"violations={self.budget_violations}, "
            f"max_consec_skip={self.max_consecutive_skip})"
        )


class LatencyAdaptiveTracker(BaseTracker):
    """Wrap any tracker with a dynamic latency-budget-aware frame-skip policy.

    Args:
        tracker:              The underlying tracker to wrap.
        latency_budget_ms:    Target per-frame latency ceiling in milliseconds.
                              Default: ``33.3`` (30 FPS target).
        window_size:          Number of recent active-frame latencies to average
                              when deciding whether to skip.  Larger windows are
                              more stable but slower to react.  Default: ``10``.
        skip_threshold_ratio: Fraction of ``latency_budget_ms`` at which skipping
                              begins.  ``0.8`` means skip when rolling mean >
                              26.7 ms for a 33.3 ms budget.  Default: ``0.8``.
        max_skip:             Maximum consecutive frames to skip before forcing
                              one active update (prevents infinite skip runs on
                              very slow devices).  Default: ``4``.
        mode:                 Interpolation for skipped frames.  ``"repeat"``
                              returns the last bbox; ``"linear"`` extrapolates
                              from the last velocity estimate.  Default:
                              ``"repeat"``.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        latency_budget_ms: float = 33.3,
        window_size: int = 10,
        skip_threshold_ratio: float = 0.8,
        max_skip: int = 4,
        mode: SkipMode = "repeat",
    ) -> None:
        if latency_budget_ms <= 0:
            raise ValueError(f"latency_budget_ms must be positive, got {latency_budget_ms}")
        if not 0.0 < skip_threshold_ratio <= 1.0:
            raise ValueError(f"skip_threshold_ratio must be in (0, 1], got {skip_threshold_ratio}")
        if max_skip < 1:
            raise ValueError(f"max_skip must be >= 1, got {max_skip}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        super().__init__(name=f"{tracker.name}_adaptive")
        self._tracker = tracker
        self.latency_budget_ms = latency_budget_ms
        self.window_size = window_size
        self.skip_threshold = latency_budget_ms * skip_threshold_ratio
        self.max_skip = max_skip
        self.mode = mode

        # Per-sequence state (reset on initialize)
        self._latency_window: Deque[float] = deque(maxlen=window_size)
        self._last_bbox: Optional[BBox] = None
        self._prev_bbox: Optional[BBox] = None  # for "linear" extrapolation
        self._consec_skipped: int = 0

        # Aggregate statistics across the whole run
        self._total_frames: int = 0
        self._active_frames: int = 0
        self._skipped_frames: int = 0
        self._active_latencies: list = []
        self._budget_violations: int = 0
        self._max_consec_skip: int = 0

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialize the wrapped tracker and reset per-sequence skip state."""
        self._tracker.initialize(frame, bbox)
        self._last_bbox = bbox
        self._prev_bbox = None
        self._consec_skipped = 0
        self._latency_window.clear()

    def update(self, frame: np.ndarray) -> BBox:
        """Return a predicted bbox, running the tracker or skipping the frame.

        The skip decision is made before each call based on the rolling latency
        average from the previous ``window_size`` active frames:

        - If the window is still filling (< ``window_size`` samples) or the
          rolling mean is below ``skip_threshold``, the frame is **active**:
          the tracker runs and latency is recorded.
        - Otherwise (rolling mean ≥ threshold **and** ``_consec_skipped <
          max_skip``), the frame is **skipped**: the last bbox is returned
          without calling the wrapped tracker.
        - Once ``max_skip`` consecutive frames have been skipped, the next
          frame is always active regardless of latency.
        """
        assert self._last_bbox is not None, "update() called before initialize()"

        self._total_frames += 1

        if self._should_skip():
            self._consec_skipped += 1
            self._skipped_frames += 1
            self._max_consec_skip = max(self._max_consec_skip, self._consec_skipped)
            return self._interpolated_bbox()

        # Active frame: run the wrapped tracker and record latency
        self._consec_skipped = 0
        t0 = time.perf_counter()
        bbox = self._tracker.update(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1_000.0

        self._active_frames += 1
        self._active_latencies.append(elapsed_ms)
        self._latency_window.append(elapsed_ms)

        if elapsed_ms > self.latency_budget_ms:
            self._budget_violations += 1

        self._prev_bbox = self._last_bbox
        self._last_bbox = bbox
        return bbox

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_skip(self) -> bool:
        """Return True when the rolling latency mean exceeds the skip threshold."""
        if self._consec_skipped >= self.max_skip:
            return False
        if len(self._latency_window) < self.window_size:
            return False
        rolling_mean = float(np.mean(self._latency_window))
        return rolling_mean >= self.skip_threshold

    def _interpolated_bbox(self) -> BBox:
        """Return the next predicted bbox using the configured skip mode."""
        assert self._last_bbox is not None
        if self.mode == "linear" and self._prev_bbox is not None:
            # Extrapolate: last_bbox + (last_bbox - prev_bbox)
            lx, ly, lw, lh = self._last_bbox
            px, py, pw, ph = self._prev_bbox
            return (lx + (lx - px), ly + (ly - py), lw, lh)
        return self._last_bbox

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> AdaptiveSkipStats:
        """Return aggregate runtime statistics since the last :meth:`initialize`.

        Aggregates are accumulated across all frames since the tracker was
        last initialized.  Call after benchmark evaluation for the full run
        statistics.

        Returns:
            :class:`AdaptiveSkipStats` with all counters populated.
        """
        total = self._total_frames
        active = self._active_frames
        skipped = self._skipped_frames
        skip_rate = skipped / total if total > 0 else 0.0
        mean_lat = (
            float(np.mean(self._active_latencies))
            if self._active_latencies
            else 0.0
        )
        return AdaptiveSkipStats(
            total_frames=total,
            active_frames=active,
            skipped_frames=skipped,
            skip_rate=skip_rate,
            mean_active_latency_ms=mean_lat,
            latency_budget_ms=self.latency_budget_ms,
            budget_violations=self._budget_violations,
            max_consecutive_skip=self._max_consec_skip,
        )

    def reset_stats(self) -> None:
        """Clear all accumulated statistics (useful between benchmark runs)."""
        self._total_frames = 0
        self._active_frames = 0
        self._skipped_frames = 0
        self._active_latencies.clear()
        self._budget_violations = 0
        self._max_consec_skip = 0
