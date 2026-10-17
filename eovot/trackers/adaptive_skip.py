"""Motion-adaptive frame-skip tracker for edge-optimized inference.

Extends the fixed-rate :class:`~eovot.trackers.frame_skip.FrameSkipTracker`
with a **policy layer** that dynamically adjusts the skip rate each frame based
on estimated target dynamics, enabling compute-proportional tracking:

- When the target moves slowly (or is stable) → increase skip rate to save
  compute.
- When the target moves fast → reduce skip rate to maintain accuracy.

This closes a key gap between fixed-budget frame skipping and the roadmap goal
of "RL-based adaptive trackers": the policies here serve as learned-policy
baselines that any RL agent must outperform to justify its training cost.

Two built-in policies are provided:

:class:`VelocityAdaptivePolicy`
    Adjusts skip rate inversely proportional to the estimated target velocity
    normalised by the bounding-box diagonal.  Fast motion → rate 1 (every
    frame); slow/zero motion → max_skip.

:class:`BudgetAdaptivePolicy`
    Given a target FPS budget and measured per-frame latency, computes the
    required skip rate to meet the budget.  Useful for meeting hard real-time
    constraints on edge hardware without over-provisioning compute.

Custom policies can be implemented by subclassing :class:`AdaptiveSkipPolicy`
and overriding :meth:`~AdaptiveSkipPolicy.predict_skip_rate`.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive_skip import AdaptiveSkipTracker, VelocityAdaptivePolicy
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, motion="random")
    engine  = BenchmarkEngine(verbose=False)

    policy  = VelocityAdaptivePolicy(min_skip=1, max_skip=4, low_threshold=0.02)
    tracker = AdaptiveSkipTracker(MOSSETracker(), policy=policy)

    result = engine.run(tracker, dataset, dataset_name="Synthetic")
    print(result)
    print(f"Mean skip rate  : {tracker.mean_skip_rate:.2f}x")
    print(f"Tracker call %  : {tracker.tracker_call_fraction * 100:.1f}%")
    print(f"Skip histogram  : {tracker.skip_rate_histogram()}")
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Abstract policy interface
# ---------------------------------------------------------------------------

class AdaptiveSkipPolicy(ABC):
    """Abstract base class for adaptive frame-skip policies.

    Subclass this and implement :meth:`predict_skip_rate` to create a
    custom policy.  The :class:`AdaptiveSkipTracker` calls this method
    after each tracker update to decide how many subsequent frames to skip.
    """

    @abstractmethod
    def predict_skip_rate(
        self,
        bbox_history: Deque[BBox],
        latency_history: Deque[float],
    ) -> int:
        """Decide the skip rate to apply after the current tracker update.

        Args:
            bbox_history: Rolling window of recent bounding-box predictions,
                most-recent last.  Contains at most ``history_len`` entries.
            latency_history: Rolling window of recent per-frame tracker
                latencies in **milliseconds**, most-recent last.

        Returns:
            Skip rate k ≥ 1.  k=1 means process every frame; k=2 means
            process every other frame (one update every 2 frames), etc.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state.  Called when the tracker is re-initialised."""


# ---------------------------------------------------------------------------
# Velocity-adaptive policy
# ---------------------------------------------------------------------------

class VelocityAdaptivePolicy(AdaptiveSkipPolicy):
    """Skip-rate policy driven by normalised target velocity.

    Estimates the target's velocity as the mean centre-point displacement
    over a rolling window, normalised by the bounding-box diagonal.  Maps
    that velocity to a skip rate via a piecewise-linear schedule:

    - velocity ≤ ``low_threshold``  → ``max_skip`` (target nearly stationary)
    - velocity ≥ ``high_threshold`` → ``min_skip`` (fast motion, track every frame)
    - intermediate              → linearly interpolated between the two extremes

    Normalisation by box diagonal makes the thresholds scale-invariant across
    sequences with very different target sizes.

    Args:
        min_skip: Minimum skip rate (1 = process every frame).  Applied when
            the target is moving fast.  Must be ≥ 1.
        max_skip: Maximum skip rate.  Applied when the target is nearly
            stationary.  Must be ≥ min_skip.
        low_threshold: Normalised velocity below which ``max_skip`` is used.
            Default 0.02 (2 % of box diagonal per frame).
        high_threshold: Normalised velocity above which ``min_skip`` is used.
            Default 0.15 (15 % of box diagonal per frame).
        window: Number of recent predictions used to estimate average velocity.
            Default 5.
    """

    def __init__(
        self,
        min_skip: int = 1,
        max_skip: int = 4,
        low_threshold: float = 0.02,
        high_threshold: float = 0.15,
        window: int = 5,
    ) -> None:
        if min_skip < 1:
            raise ValueError(f"min_skip must be >= 1, got {min_skip}.")
        if max_skip < min_skip:
            raise ValueError(f"max_skip must be >= min_skip, got max={max_skip} min={min_skip}.")
        if low_threshold >= high_threshold:
            raise ValueError(
                f"low_threshold must be < high_threshold, "
                f"got low={low_threshold} high={high_threshold}."
            )
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.window = window

    def predict_skip_rate(
        self,
        bbox_history: Deque[BBox],
        latency_history: Deque[float],
    ) -> int:
        if len(bbox_history) < 2:
            # Insufficient history — use min_skip (conservative)
            return self.min_skip

        bboxes = list(bbox_history)[-self.window:]
        displacements: List[float] = []

        for i in range(1, len(bboxes)):
            prev = bboxes[i - 1]
            curr = bboxes[i]

            # Centre-point displacement
            prev_cx = prev[0] + prev[2] / 2.0
            prev_cy = prev[1] + prev[3] / 2.0
            curr_cx = curr[0] + curr[2] / 2.0
            curr_cy = curr[1] + curr[3] / 2.0
            disp = np.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2)

            # Normalise by mean diagonal
            diag_prev = np.sqrt(prev[2] ** 2 + prev[3] ** 2)
            diag_curr = np.sqrt(curr[2] ** 2 + curr[3] ** 2)
            mean_diag = (diag_prev + diag_curr) / 2.0
            norm_vel = float(disp / mean_diag) if mean_diag > 0 else 0.0
            displacements.append(norm_vel)

        avg_velocity = float(np.mean(displacements)) if displacements else 0.0

        # Piecewise-linear mapping: slow → max_skip, fast → min_skip
        if avg_velocity <= self.low_threshold:
            return self.max_skip
        if avg_velocity >= self.high_threshold:
            return self.min_skip

        t = (avg_velocity - self.low_threshold) / (self.high_threshold - self.low_threshold)
        raw = self.max_skip - t * (self.max_skip - self.min_skip)
        return int(np.clip(round(raw), self.min_skip, self.max_skip))


# ---------------------------------------------------------------------------
# Budget-adaptive policy
# ---------------------------------------------------------------------------

class BudgetAdaptivePolicy(AdaptiveSkipPolicy):
    """Skip-rate policy that targets a fixed FPS budget.

    Computes the required skip rate from the measured per-frame latency so
    that the effective throughput (tracker FPS × skip rate) meets
    ``fps_budget``.  This is the most principled policy for edge deployments
    with hard real-time constraints: it automatically adapts as the host CPU
    load varies without needing any velocity estimation.

    An optional ``host_to_device_factor`` lets you project host-measured
    latencies onto a slower target device — e.g. pass ``8.0`` when the
    benchmark runs on a desktop but will deploy on a Raspberry Pi 4 that is
    ~8× slower per-core.

    Args:
        fps_budget: Target sustained FPS on the deployment device.  Must be > 0.
        min_skip: Minimum skip rate (floor). Default 1.
        max_skip: Maximum skip rate (ceiling). Default 8.
        latency_window: Number of recent latency samples used for the EMA
            estimate.  Default 10.
        host_to_device_factor: Multiplier applied to host latency before
            computing the required skip rate.  Use to project onto a slower
            deployment target.  Default 1.0 (profile on actual device).
    """

    def __init__(
        self,
        fps_budget: float,
        min_skip: int = 1,
        max_skip: int = 8,
        latency_window: int = 10,
        host_to_device_factor: float = 1.0,
    ) -> None:
        if fps_budget <= 0:
            raise ValueError(f"fps_budget must be positive, got {fps_budget}.")
        if min_skip < 1:
            raise ValueError(f"min_skip must be >= 1, got {min_skip}.")
        if max_skip < min_skip:
            raise ValueError(f"max_skip must be >= min_skip, got max={max_skip} min={min_skip}.")
        if host_to_device_factor <= 0:
            raise ValueError(f"host_to_device_factor must be positive, got {host_to_device_factor}.")
        self.fps_budget = fps_budget
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.latency_window = latency_window
        self.host_to_device_factor = host_to_device_factor

    def predict_skip_rate(
        self,
        bbox_history: Deque[BBox],
        latency_history: Deque[float],
    ) -> int:
        if not latency_history:
            return self.min_skip

        recent = list(latency_history)[-self.latency_window:]
        avg_latency_ms = float(np.mean(recent)) * self.host_to_device_factor

        # Skip rate needed so that (fps_budget) frames/s are processed
        # total while only (fps_budget / skip) frames hit the tracker.
        target_latency_ms = 1000.0 / self.fps_budget
        required_skip = avg_latency_ms / target_latency_ms

        return int(np.clip(round(required_skip), self.min_skip, self.max_skip))


# ---------------------------------------------------------------------------
# AdaptiveSkipTracker
# ---------------------------------------------------------------------------

class AdaptiveSkipTracker(BaseTracker):
    """Dynamic frame-skip wrapper driven by an :class:`AdaptiveSkipPolicy`.

    Unlike the fixed-rate :class:`~eovot.trackers.frame_skip.FrameSkipTracker`,
    this wrapper queries the policy **after each tracker update** to decide how
    many subsequent frames to skip.  Skipped frames receive a linearly
    extrapolated bounding-box prediction based on the last two tracker outputs.

    The wrapped tracker's ``name`` is preserved with an ``_adaptive`` suffix so
    that :class:`~eovot.benchmark.engine.BenchmarkEngine` reports it distinctly
    from the underlying tracker and from a fixed-rate variant.

    Args:
        tracker:     The underlying :class:`BaseTracker` to wrap.
        policy:      An :class:`AdaptiveSkipPolicy` instance that decides the
                     skip rate after each tracker update.
        history_len: Maximum length of the rolling bbox and latency history
                     windows passed to the policy.  Default 20.

    Attributes:
        mean_skip_rate (float): Effective mean skip rate across all processed
            frames since last :meth:`initialize` call.
        tracker_call_fraction (float): Fraction of frames that triggered an
            actual underlying tracker update (0 < value ≤ 1).
    """

    def __init__(
        self,
        tracker: BaseTracker,
        policy: AdaptiveSkipPolicy,
        history_len: int = 20,
    ) -> None:
        super().__init__(name=f"{tracker.name}_adaptive")
        self._tracker = tracker
        self._policy = policy
        self._history_len = history_len

        # Rolling windows passed to the policy
        self._bbox_history: Deque[BBox] = deque(maxlen=history_len)
        self._latency_history: Deque[float] = deque(maxlen=history_len)

        # Internal state
        self._frames_until_update: int = 0
        self._prev_bbox: Optional[BBox] = None
        self._prev_prev_bbox: Optional[BBox] = None
        self._last_skip_rate: int = 1
        self._step_in_skip: int = 0

        # Diagnostics
        self._total_frames: int = 0
        self._tracker_calls: int = 0
        self._skip_rates_used: List[int] = []

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the wrapped tracker and reset all adaptive-policy state.

        Args:
            frame: First frame (H, W, 3) BGR uint8 array.
            bbox:  Initial ground-truth bounding box (x, y, w, h).
        """
        self._tracker.initialize(frame, bbox)
        self._policy.reset()
        self._bbox_history.clear()
        self._latency_history.clear()
        self._bbox_history.append(bbox)
        self._prev_bbox = bbox
        self._prev_prev_bbox = None
        self._frames_until_update = 0
        self._last_skip_rate = 1
        self._step_in_skip = 0
        self._total_frames = 0
        self._tracker_calls = 0
        self._skip_rates_used = []

    def update(self, frame: np.ndarray) -> BBox:
        """Return the predicted bounding box for the current frame.

        Calls the wrapped tracker when the countdown reaches zero; otherwise
        returns a linearly extrapolated position based on the two most recent
        tracker outputs.

        Args:
            frame: Current frame (H, W, 3) BGR uint8 array.

        Returns:
            Predicted bounding box (x, y, w, h).

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._prev_bbox is None:
            raise RuntimeError(
                "AdaptiveSkipTracker is not initialised. Call initialize() first."
            )

        self._total_frames += 1

        if self._frames_until_update <= 0:
            # ------ Run the underlying tracker ------
            t0 = time.perf_counter()
            bbox = self._tracker.update(frame)
            latency_ms = (time.perf_counter() - t0) * 1_000.0

            self._tracker_calls += 1
            self._latency_history.append(latency_ms)
            self._prev_prev_bbox = self._prev_bbox
            self._prev_bbox = bbox
            self._bbox_history.append(bbox)
            self._step_in_skip = 0

            # Query policy for next skip rate
            skip_rate = max(1, self._policy.predict_skip_rate(
                self._bbox_history, self._latency_history
            ))
            self._last_skip_rate = skip_rate
            self._skip_rates_used.append(skip_rate)
            self._frames_until_update = skip_rate - 1

            return bbox

        # ------ Skipped frame: linear extrapolation ------
        self._frames_until_update -= 1
        self._step_in_skip += 1

        if self._prev_prev_bbox is not None:
            prev2 = self._prev_prev_bbox
            prev1 = self._prev_bbox
            # Per-step velocity from the last two tracker outputs
            vx = prev1[0] - prev2[0]
            vy = prev1[1] - prev2[1]
            vw = prev1[2] - prev2[2]
            vh = prev1[3] - prev2[3]
            step = self._step_in_skip
            bbox = (
                prev1[0] + vx * step,
                prev1[1] + vy * step,
                max(1.0, prev1[2] + vw * step),
                max(1.0, prev1[3] + vh * step),
            )
        else:
            bbox = self._prev_bbox

        return bbox

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def mean_skip_rate(self) -> float:
        """Mean skip rate observed across all tracker-update frames."""
        if not self._skip_rates_used:
            return 1.0
        return float(np.mean(self._skip_rates_used))

    @property
    def tracker_call_fraction(self) -> float:
        """Fraction of frames where the underlying tracker was called (0 < value ≤ 1)."""
        if self._total_frames == 0:
            return 1.0
        return self._tracker_calls / self._total_frames

    def skip_rate_histogram(self) -> dict:
        """Frequency histogram of skip-rates used during the run.

        Returns:
            Dict mapping each observed skip rate (int) to its occurrence
            count (int).  Sums to the number of tracker-update frames.
        """
        if not self._skip_rates_used:
            return {}
        rates, counts = np.unique(self._skip_rates_used, return_counts=True)
        return {int(r): int(c) for r, c in zip(rates, counts)}

    def __repr__(self) -> str:
        return (
            f"AdaptiveSkipTracker(tracker={self._tracker.name!r}, "
            f"policy={self._policy.__class__.__name__})"
        )
