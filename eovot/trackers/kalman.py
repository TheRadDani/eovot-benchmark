"""Kalman Filter Tracker — motion-based prediction baseline.

Implements a constant-velocity Kalman filter for bounding box tracking using
pure NumPy.  The filter models object state as:

    x = [cx, cy, s, r, dcx, dcy, ds]

where ``(cx, cy)`` is the box centre, ``s = w*h`` is the area (scale),
``r = w/h`` is the aspect ratio (treated as constant), and
``(dcx, dcy, ds)`` are their respective velocities.

This is the same kinematic model used in SORT (Bewley et al., 2016) and
inherited by ByteTrack and many production multi-object trackers.  Within
EOVOT it serves as:

- A fast **motion-only baseline** that makes no appearance assumptions.
- A reference for comparing motion-model accuracy versus appearance-based
  trackers (MOSSE, KCF, CSRT).
- A lightweight primitive for future multi-object tracking modules.

Tracking protocol
-----------------
- On ``initialize``: convert bbox → state vector, set identity covariances.
- On ``update``: Kalman *predict* step projects state forward by one step;
  the observed bounding box from a detection or the last prediction is used
  as the *measurement*; the *update* step corrects the state.

No appearance model is used — the tracker will drift on abrupt motion
changes.  This is intentional: the benchmark should expose this weakness.

Computational cost
------------------
All operations are matrix multiplications over 7-dimensional state vectors.
Per-frame cost is O(1) and negligible (<< 1 µs) — the tracker is always
limited by frame I/O, not filter math.

References
----------
- Bewley, A. et al. "Simple online and realtime tracking." ICIP 2016.
- Kalman, R.E. "A new approach to linear filtering and prediction." 1960.

Usage::

    from eovot.trackers.kalman import KalmanFilterTracker

    tracker = KalmanFilterTracker()
    tracker.initialize(frame, bbox)   # (x, y, w, h)
    for frame in sequence:
        predicted_bbox = tracker.update(frame)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox


class KalmanFilterTracker(BaseTracker):
    """Constant-velocity Kalman filter bounding box tracker.

    State vector: ``[cx, cy, s, r, dcx, dcy, ds]`` (7-D).
    Measurement vector: ``[cx, cy, s, r]`` (4-D).

    The aspect ratio ``r`` is modelled as constant (no velocity term) to
    improve stability on most natural objects.

    Args:
        process_noise_scale: Diagonal scale for the process noise covariance
            ``Q``.  Higher values allow faster state adaptation at the cost
            of prediction stability.  Default: ``1.0``.
        measurement_noise_scale: Diagonal scale for the measurement noise
            covariance ``R``.  Higher values trust the filter prediction over
            raw observations.  Default: ``1.0``.
        uncertainty_scale: Initial off-diagonal uncertainty added to ``P``
            for the velocity components.  Default: ``10.0``.
        min_size: Minimum predicted box dimension (pixels).  Clamps the
            output to avoid degenerate boxes after drift.  Default: ``1.0``.
    """

    name = "KalmanFilter"

    # State / measurement dimensions
    _DIM_X = 7
    _DIM_Z = 4

    def __init__(
        self,
        process_noise_scale: float = 1.0,
        measurement_noise_scale: float = 1.0,
        uncertainty_scale: float = 10.0,
        min_size: float = 1.0,
    ) -> None:
        super().__init__(name=self.name)
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        self.uncertainty_scale = uncertainty_scale
        self.min_size = min_size

        # State transition matrix (constant velocity model).
        self._F = np.eye(self._DIM_X, dtype=np.float64)
        # Position components couple to velocity: x[0:3] += x[4:7]
        for i in range(3):
            self._F[i, i + 4] = 1.0

        # Measurement matrix: we observe [cx, cy, s, r] directly.
        self._H = np.zeros((self._DIM_Z, self._DIM_X), dtype=np.float64)
        self._H[:4, :4] = np.eye(4)

        # Process noise covariance Q (tuned empirically for natural video).
        q_diag = np.array([1.0, 1.0, 10.0, 10.0, 0.01, 0.01, 0.0001],
                          dtype=np.float64)
        self._Q = np.diag(q_diag * process_noise_scale)

        # Measurement noise covariance R.
        r_diag = np.array([1.0, 1.0, 10.0, 10.0], dtype=np.float64)
        self._R = np.diag(r_diag * measurement_noise_scale)

        # State vector and covariance (initialised in initialize()).
        self._x: Optional[np.ndarray] = None   # (7,)
        self._P: Optional[np.ndarray] = None   # (7, 7)
        self._initialised: bool = False
        self._last_bbox: Optional[BBox] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the Kalman filter from the first frame ground-truth box.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.  Not used by this tracker
                (no appearance model) but required by the BaseTracker interface.
            bbox: Ground-truth bounding box ``(x, y, w, h)``.
        """
        z = self._bbox_to_obs(bbox)

        # Initialise state: position from measurement, velocities at zero.
        self._x = np.zeros(self._DIM_X, dtype=np.float64)
        self._x[:4] = z

        # Initial covariance: high uncertainty on velocities.
        p_diag = np.array([10.0, 10.0, 10.0, 10.0,
                           self.uncertainty_scale,
                           self.uncertainty_scale,
                           self.uncertainty_scale],
                          dtype=np.float64)
        self._P = np.diag(p_diag)

        self._initialised = True
        self._last_bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location using the Kalman filter.

        Performs a predict step (project state forward) followed by a
        *self-measurement* update using the last predicted position as the
        observation.  This makes the tracker a pure motion predictor that
        does not rely on frame content for its update step — a deliberate
        design choice that makes the tracker fast (sub-microsecond) and
        appropriate as a motion-only baseline.

        Args:
            frame: Current BGR image.  Not used; present for API compliance.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` in frame coordinates.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if not self._initialised or self._x is None or self._P is None:
            raise RuntimeError(
                "KalmanFilterTracker.update() called before initialize()."
            )

        # Predict
        x_pred, P_pred = self._predict(self._x, self._P)

        # Use prediction as self-measurement (motion-only tracker).
        z = x_pred[:4]
        x_upd, P_upd = self._correct(x_pred, P_pred, z)

        self._x = x_upd
        self._P = P_upd

        bbox = self._state_to_bbox(self._x)
        self._last_bbox = bbox
        return bbox

    # ------------------------------------------------------------------
    # Kalman filter mechanics
    # ------------------------------------------------------------------

    def _predict(
        self, x: np.ndarray, P: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman predict step: project state and covariance forward."""
        x_pred = self._F @ x
        P_pred = self._F @ P @ self._F.T + self._Q
        return x_pred, P_pred

    def _correct(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman update step: correct prediction with measurement ``z``."""
        # Innovation and covariance.
        y = z - self._H @ x_pred
        S = self._H @ P_pred @ self._H.T + self._R
        # Kalman gain.
        K = P_pred @ self._H.T @ np.linalg.inv(S)
        # Updated state and covariance.
        x_upd = x_pred + K @ y
        I_KH = np.eye(self._DIM_X) - K @ self._H
        P_upd = I_KH @ P_pred
        return x_upd, P_upd

    # ------------------------------------------------------------------
    # Coordinate conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_to_obs(bbox: BBox) -> np.ndarray:
        """Convert ``(x, y, w, h)`` bounding box to ``[cx, cy, s, r]``."""
        x, y, w, h = bbox
        w = max(w, 1.0)
        h = max(h, 1.0)
        return np.array([
            x + w / 2.0,        # cx
            y + h / 2.0,        # cy
            w * h,              # s  (area)
            w / h,              # r  (aspect ratio)
        ], dtype=np.float64)

    def _state_to_bbox(self, x: np.ndarray) -> BBox:
        """Convert state vector ``[cx, cy, s, r, ...]`` to ``(x, y, w, h)``."""
        cx, cy, s, r = x[0], x[1], x[2], x[3]
        # Guard against degenerate state values.
        s = max(s, self.min_size ** 2)
        r = max(r, 1e-3)
        w = float(np.sqrt(s * r))
        h = float(s / max(w, self.min_size))
        w = max(w, self.min_size)
        h = max(h, self.min_size)
        return (float(cx - w / 2.0), float(cy - h / 2.0), w, h)

    # ------------------------------------------------------------------
    # Inspection properties
    # ------------------------------------------------------------------

    @property
    def state_vector(self) -> Optional[np.ndarray]:
        """Current Kalman state ``[cx, cy, s, r, dcx, dcy, ds]`` or ``None``."""
        return self._x.copy() if self._x is not None else None

    @property
    def velocity(self) -> Optional[np.ndarray]:
        """Estimated velocity ``[dcx, dcy, ds]`` or ``None`` before init."""
        return self._x[4:7].copy() if self._x is not None else None

    @property
    def covariance(self) -> Optional[np.ndarray]:
        """State covariance matrix ``(7, 7)`` or ``None`` before init."""
        return self._P.copy() if self._P is not None else None

    @property
    def position_uncertainty(self) -> Optional[float]:
        """Trace of the position block of ``P`` (sum of position variances)."""
        if self._P is None:
            return None
        return float(np.trace(self._P[:2, :2]))

    def __repr__(self) -> str:
        state = "uninitialised" if self._x is None else f"vel={self._x[4:7].round(2)}"
        return f"KalmanFilterTracker({state})"
