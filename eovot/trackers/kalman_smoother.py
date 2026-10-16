"""Kalman filter post-processor for smoothing tracker bounding-box streams.

Real-time edge deployments — robot controllers, video analytics pipelines,
autonomous navigation systems — consume the bounding-box stream produced by
a tracker and typically pass it directly to a Kalman or PID controller.
Raw tracker outputs have high frame-to-frame jitter (measured by
:class:`~eovot.metrics.temporal.TemporalConsistencyAnalyzer`), which
amplifies noise in downstream controllers and reduces deployment quality.

:class:`KalmanSmoothedTracker` wraps *any* ``BaseTracker`` with a constant-
velocity Kalman filter that:

* **Smooths** position and size predictions (reduces σ_pos and σ_scale).
* **Fills** single-frame drops by propagating the filter state when the
  underlying tracker returns a degenerate (zero-area) box.
* **Preserves** accuracy — the filter trusts good predictions by tuning
  measurement noise ``R`` to the tracker's expected error level.

State vector (8D)
-----------------
``[cx, cy, w, h, vcx, vcy, vw, vh]``

where ``(cx, cy)`` is the box centre and ``(vx, vy)`` / ``(vw, vh)`` are
the velocities.  The constant-velocity transition model propagates state
one frame at a time, and the 4D measurement ``[cx, cy, w, h]`` (from the
underlying tracker) corrects the estimate.

Typical usage::

    from eovot.trackers.kalman_smoother import KalmanSmoothedTracker
    from eovot.trackers.kcf import KCFTracker

    raw = KCFTracker()
    smoothed = KalmanSmoothedTracker(raw, process_noise=1e-3, measurement_noise=1e-1)
    smoothed.initialize(first_frame, init_bbox)

    for frame in video:
        bbox = smoothed.update(frame)   # Kalman-smoothed (x, y, w, h)

The ``smoothness_gain`` parameter controls the accuracy–smoothness tradeoff:
lower values trust the Kalman model more (smoother but lags fast motion);
higher values follow the raw tracker more closely (less smooth but more
responsive).  The default ``1.0`` is a balanced starting point.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseTracker, BBox


class KalmanSmoothedTracker(BaseTracker):
    """Constant-velocity Kalman filter wrapper for any BaseTracker.

    Wraps an existing tracker to post-process its bounding-box stream with a
    linear Kalman filter.  The wrapper is fully transparent to the benchmark
    engine: it exposes the same ``initialize`` / ``update`` interface as any
    other ``BaseTracker``.

    Args:
        tracker: Any ``BaseTracker`` instance to wrap.  Its predictions are
            used as Kalman measurements.
        process_noise: Variance of the state-transition noise (Q diagonal).
            Controls how quickly the filter adapts to acceleration.
            Larger values follow the tracker more closely; smaller values
            apply heavier smoothing.  Default: ``1e-3``.
        measurement_noise: Variance of the observation noise (R diagonal).
            Should reflect the expected pixel-level error of the underlying
            tracker.  Default: ``1e-1`` (~0.3 px RMSE, reasonable for
            classical trackers on standard resolutions).
        smoothness_gain: Multiplicative scaling applied to ``measurement_noise``.
            Values > 1 produce a smoother (less responsive) output;
            values < 1 follow the raw tracker more aggressively.
            Default: ``1.0`` (no scaling).
        name: Optional override for the tracker name used in reports.
            Defaults to ``"Kalman(<wrapped_name>)"``.

    Example::

        from eovot.trackers.kalman_smoother import KalmanSmoothedTracker
        from eovot.trackers.mosse import MOSSETracker

        smoother = KalmanSmoothedTracker(MOSSETracker(), measurement_noise=0.5)
        smoother.initialize(frame0, bbox0)
        for frame in frames[1:]:
            smooth_bbox = smoother.update(frame)
    """

    # State dimensionality
    _STATE_DIM = 8    # [cx, cy, w, h, vcx, vcy, vw, vh]
    _MEAS_DIM = 4     # [cx, cy, w, h]

    def __init__(
        self,
        tracker: BaseTracker,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-1,
        smoothness_gain: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or f"Kalman({tracker.name})")
        if process_noise <= 0:
            raise ValueError(f"process_noise must be positive, got {process_noise}")
        if measurement_noise <= 0:
            raise ValueError(f"measurement_noise must be positive, got {measurement_noise}")
        if smoothness_gain <= 0:
            raise ValueError(f"smoothness_gain must be positive, got {smoothness_gain}")

        self._tracker = tracker
        self._pn = float(process_noise)
        self._mn = float(measurement_noise) * float(smoothness_gain)

        # Kalman matrices — built lazily in initialize() so they stay typed
        self._F: np.ndarray = self._build_F()
        self._H: np.ndarray = self._build_H()
        self._Q: np.ndarray = np.eye(self._STATE_DIM) * self._pn
        self._R: np.ndarray = np.eye(self._MEAS_DIM) * self._mn

        # Filter state — initialised in initialize()
        self._x: Optional[np.ndarray] = None   # (8,) state estimate
        self._P: Optional[np.ndarray] = None   # (8, 8) covariance

    # ------------------------------------------------------------------
    # Kalman matrices
    # ------------------------------------------------------------------

    @staticmethod
    def _build_F() -> np.ndarray:
        """Constant-velocity state transition matrix (8×8)."""
        F = np.eye(8, dtype=np.float64)
        # cx += vcx, cy += vcy, w += vw, h += vh (one frame step = 1)
        F[0, 4] = 1.0  # cx += vcx
        F[1, 5] = 1.0  # cy += vcy
        F[2, 6] = 1.0  # w  += vw
        F[3, 7] = 1.0  # h  += vh
        return F

    @staticmethod
    def _build_H() -> np.ndarray:
        """Observation matrix: extracts [cx, cy, w, h] from 8D state (4×8)."""
        H = np.zeros((4, 8), dtype=np.float64)
        H[0, 0] = 1.0  # cx
        H[1, 1] = 1.0  # cy
        H[2, 2] = 1.0  # w
        H[3, 3] = 1.0  # h
        return H

    # ------------------------------------------------------------------
    # Box ↔ state conversions
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_to_meas(bbox: BBox) -> np.ndarray:
        """Convert (x, y, w, h) to measurement vector [cx, cy, w, h]."""
        x, y, w, h = bbox
        return np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=np.float64)

    @staticmethod
    def _state_to_bbox(x: np.ndarray) -> BBox:
        """Convert 8D state vector to (x, y, w, h)."""
        cx, cy, w, h = x[0], x[1], x[2], x[3]
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        return (float(cx - w / 2.0), float(cy - h / 2.0), w, h)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the underlying tracker and the Kalman filter state.

        The filter is seeded with the ground-truth bounding box; velocity
        is initialised to zero.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        meas = self._bbox_to_meas(bbox)
        self._x = np.zeros(self._STATE_DIM, dtype=np.float64)
        self._x[:4] = meas
        self._P = np.eye(self._STATE_DIM, dtype=np.float64) * 10.0

    def update(self, frame: np.ndarray) -> BBox:
        """Predict with the underlying tracker, then apply Kalman correction.

        If the underlying tracker returns a degenerate box (zero area), the
        measurement update step is skipped and the filter propagates on its
        own estimate.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            Kalman-smoothed bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._x is None or self._P is None:
            raise RuntimeError("KalmanSmoothedTracker.update() called before initialize()")

        # --- Prediction step ---
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # --- Measurement from underlying tracker ---
        raw_bbox = self._tracker.update(frame)
        raw_w, raw_h = raw_bbox[2], raw_bbox[3]
        use_measurement = raw_w > 0 and raw_h > 0

        if use_measurement:
            z = self._bbox_to_meas(raw_bbox)

            # Innovation
            y_innov = z - self._H @ x_pred
            S = self._H @ P_pred @ self._H.T + self._R
            # Kalman gain
            K = P_pred @ self._H.T @ np.linalg.inv(S)
            # Update
            self._x = x_pred + K @ y_innov
            I_KH = np.eye(self._STATE_DIM) - K @ self._H
            self._P = I_KH @ P_pred
        else:
            # Underlying tracker returned a degenerate box — propagate only
            self._x = x_pred
            self._P = P_pred

        return self._state_to_bbox(self._x)
