"""Adaptive-compute tracker wrapper for latency-budget-aware inference.

Edge devices such as the Raspberry Pi 4 and NVIDIA Jetson Nano experience
variable per-frame latency due to thermal throttling, OS jitter, and
memory pressure.  When the underlying tracker risks breaching a latency
SLA (e.g. a 33 ms budget for 30 FPS operation), the system should fall
back to a lightweight predictor rather than drop frames or violate
real-time constraints.

:class:`AdaptiveComputeTracker` wraps any :class:`~.base.BaseTracker`
and dynamically switches between two inference modes:

* **Full mode** — the underlying tracker's ``update()`` is called.
  Used when the tracker has been consistently within budget.
* **Kalman mode** — :class:`~.kalman.KalmanBoxPredictor` propagates
  the last known state at near-zero cost (< 0.1 ms on any CPU).
  Used when the rolling mean latency plus one standard deviation would
  exceed ``latency_budget_ms``.

The switching policy is conservative: the tracker returns to full mode
only after ``recovery_frames`` consecutive frames are completed within
the budget, preventing rapid oscillation.

Example::

    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.adaptive_compute import AdaptiveComputeTracker
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    base    = KCFTracker()
    tracker = AdaptiveComputeTracker(base, latency_budget_ms=33.0)

    engine  = BenchmarkEngine(verbose=False)
    dataset = SyntheticDataset(num_sequences=3, num_frames=100)
    result  = engine.run(tracker, dataset, dataset_name="Synthetic")

    stats = tracker.compute_stats()
    print(f"Full-mode frames : {stats['full_frames']}")
    print(f"Kalman frames    : {stats['kalman_frames']}")
    print(f"Budget violations: {stats['budget_violations']}")
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np

from .base import BaseTracker, BBox
from .kalman import KalmanBoxPredictor


class AdaptiveComputeTracker(BaseTracker):
    """Budget-aware tracker that falls back to Kalman prediction under load.

    The wrapper estimates whether calling the underlying tracker's
    ``update()`` will exceed ``latency_budget_ms`` by computing a
    rolling mean and standard deviation over the last ``window_size``
    frame latencies.  If
    ``mean_latency + safety_sigma * std_latency > latency_budget_ms``
    the frame is served by the Kalman predictor instead.

    Args:
        tracker:            Any :class:`~.base.BaseTracker` to wrap.
        latency_budget_ms:  Maximum allowed per-frame latency in
            milliseconds.  Set to the reciprocal of the target FPS
            times 1000 (e.g. 33.3 ms for 30 FPS).
        window_size:        Number of recent frames used to estimate
            latency distribution.  Smaller windows react faster;
            larger windows are more stable.  Default: 20.
        safety_sigma:       Number of standard deviations added to the
            mean when estimating worst-case latency.  Higher values
            are more conservative (fewer Kalman skips, more budget
            violations possible).  Default: 1.5.
        recovery_frames:    Frames the tracker must complete within
            budget before switching back from Kalman mode to full mode.
            Default: 5.
        warmup_frames:      Frames run in full mode unconditionally at
            sequence start to prime the latency estimate.  Default: 10.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        latency_budget_ms: float = 33.3,
        window_size: int = 20,
        safety_sigma: float = 1.5,
        recovery_frames: int = 5,
        warmup_frames: int = 10,
    ) -> None:
        if latency_budget_ms <= 0:
            raise ValueError(f"latency_budget_ms must be positive, got {latency_budget_ms}")
        super().__init__(name=f"{tracker.name}_adaptive")
        self._tracker = tracker
        self.latency_budget_ms = latency_budget_ms
        self._window: Deque[float] = deque(maxlen=window_size)
        self._safety_sigma = safety_sigma
        self._recovery_frames = recovery_frames
        self._warmup_frames = warmup_frames

        # State counters
        self._frame_idx: int = 0
        self._in_kalman_mode: bool = False
        self._recovery_count: int = 0

        # Stats
        self._full_frames: int = 0
        self._kalman_frames: int = 0
        self._budget_violations: int = 0

        # Kalman predictor
        self._kf = KalmanBoxPredictor()
        self._last_bbox: BBox = (0.0, 0.0, 1.0, 1.0)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialize the underlying tracker and the Kalman predictor.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._kf.reset()
        self._kf.initialize(bbox)
        self._last_bbox = bbox

        # Reset state for this sequence
        self._frame_idx = 0
        self._in_kalman_mode = False
        self._recovery_count = 0
        self._window.clear()

    def update(self, frame: np.ndarray) -> BBox:
        """Return the predicted bounding box for the current frame.

        Runs the underlying tracker in full mode or falls back to Kalman
        prediction depending on the estimated latency versus budget.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.
        """
        self._frame_idx += 1

        use_kalman = self._should_use_kalman()

        if use_kalman:
            # Kalman-only frame: advance filter without a measurement
            pred = self._kf.predict()
            self._last_bbox = pred
            self._kalman_frames += 1
            return pred

        # Full inference path
        t0 = time.perf_counter()
        pred = self._tracker.update(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1_000.0

        # Update latency window
        self._window.append(elapsed_ms)

        # Check for budget violation
        if elapsed_ms > self.latency_budget_ms:
            self._budget_violations += 1

        # Update Kalman with the tracker measurement
        self._kf.predict()
        self._kf.correct(pred)
        self._last_bbox = pred
        self._full_frames += 1

        # Mode transition: switch to Kalman if estimate exceeds budget
        if len(self._window) >= self._warmup_frames and self._estimate_exceeds_budget():
            self._in_kalman_mode = True
            self._recovery_count = 0

        # Mode transition: recover from Kalman to full
        if self._in_kalman_mode:
            if elapsed_ms <= self.latency_budget_ms:
                self._recovery_count += 1
                if self._recovery_count >= self._recovery_frames:
                    self._in_kalman_mode = False
                    self._recovery_count = 0
            else:
                self._recovery_count = 0

        return pred

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def compute_stats(self) -> Dict[str, object]:
        """Return a summary of adaptive-compute behaviour for this sequence.

        Returns:
            Dict with keys ``full_frames``, ``kalman_frames``,
            ``budget_violations``, ``kalman_ratio`` (fraction of frames
            served by Kalman), and ``mean_full_latency_ms``.
        """
        total = self._full_frames + self._kalman_frames
        kalman_ratio = self._kalman_frames / total if total > 0 else 0.0
        mean_lat = float(np.mean(list(self._window))) if self._window else 0.0
        return {
            "full_frames": self._full_frames,
            "kalman_frames": self._kalman_frames,
            "budget_violations": self._budget_violations,
            "kalman_ratio": kalman_ratio,
            "mean_full_latency_ms": mean_lat,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_use_kalman(self) -> bool:
        """Return True if this frame should be served by Kalman prediction."""
        # Always run full mode during warm-up
        if self._frame_idx <= self._warmup_frames:
            return False
        return self._in_kalman_mode

    def _estimate_exceeds_budget(self) -> bool:
        """Return True when the latency estimate (mean + k*std) > budget."""
        arr = np.array(self._window)
        mean = arr.mean()
        std = arr.std() if len(arr) > 1 else 0.0
        return bool(mean + self._safety_sigma * std > self.latency_budget_ms)
