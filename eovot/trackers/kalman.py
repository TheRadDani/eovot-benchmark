"""Constant-velocity Kalman filter for bounding-box state estimation.

Provides a pure-NumPy implementation suitable for CPU-only edge devices
that cannot afford the overhead of a deep-learning motion model.

State vector
~~~~~~~~~~~~
The filter tracks an 8-dimensional state::

    s = [x, y, w, h, vx, vy, vw, vh]

where ``(x, y, w, h)`` is the bounding box in pixel coordinates
(top-left corner + dimensions) and ``(vx, vy, vw, vh)`` are the
corresponding velocities (pixels / frame).

Measurement vector
~~~~~~~~~~~~~~~~~~
Only the position part ``(x, y, w, h)`` is observed from tracker output.

Usage::

    from eovot.trackers.kalman import KalmanBoxPredictor

    kf = KalmanBoxPredictor()
    kf.initialize((100, 50, 80, 60))   # first GT box

    # Prediction step (no measurement available)
    pred = kf.predict()

    # Prediction + correction (tracker provided a measurement)
    pred = kf.predict()
    kf.correct((102, 51, 80, 60))

    # Innovation covariance norm — indicator of Kalman confidence
    score = kf.innovation_norm
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


class KalmanBoxPredictor:
    """Constant-velocity Kalman filter for single-target bounding-box tracking.

    The filter is intentionally minimal: no measurement noise adaptation, no
    manoeuvre detection — just a standard linear Kalman filter implemented in
    20-line NumPy arithmetic so it runs in microseconds on any hardware.

    Args:
        process_noise:     Scalar added to the diagonal of the process-noise
            matrix ``Q``.  Larger values let the filter react faster to
            sudden motion changes at the cost of noisier predictions.
        measurement_noise: Scalar added to the diagonal of the
            measurement-noise matrix ``R``.  Represents how much we trust
            the tracker's output box.  Increase when the tracker is known
            to be noisy.
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
    ) -> None:
        self._dim_x = 8  # [x, y, w, h, vx, vy, vw, vh]
        self._dim_z = 4  # [x, y, w, h]

        dt = 1.0  # one frame time-step

        # State transition matrix F  (constant-velocity model)
        self.F = np.eye(self._dim_x, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = dt

        # Measurement matrix H  (observe position part only)
        self.H = np.zeros((self._dim_z, self._dim_x), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)

        # Process-noise covariance Q
        self.Q = np.eye(self._dim_x, dtype=np.float64) * process_noise

        # Measurement-noise covariance R
        self.R = np.eye(self._dim_z, dtype=np.float64) * measurement_noise

        # State vector and covariance — uninitialised until :meth:`initialize`
        self.x: Optional[np.ndarray] = None   # (8,)
        self.P: Optional[np.ndarray] = None   # (8, 8)

        self._innovation_norm: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, bbox: BBox) -> None:
        """Set the filter state from a known bounding box (first frame).

        Velocities are initialised to zero and the covariance is set to
        a large diagonal (high uncertainty in velocities, low in position).

        Args:
            bbox: Initial bounding box ``(x, y, w, h)`` in pixel coordinates.
        """
        x0, y0, w0, h0 = bbox
        self.x = np.array([x0, y0, w0, h0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(self._dim_x, dtype=np.float64)
        # Velocities start with high uncertainty
        for i in range(4, 8):
            self.P[i, i] = 100.0

        self._innovation_norm = 0.0

    def predict(self) -> BBox:
        """Advance the filter by one time step and return the predicted box.

        Must be called before :meth:`correct` on each frame.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called yet.
        """
        if self.x is None or self.P is None:
            raise RuntimeError("KalmanBoxPredictor must be initialized before predict().")

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self._state_to_bbox()

    def correct(self, bbox: BBox) -> BBox:
        """Update the filter state with a tracker measurement.

        Computes the Kalman gain and incorporates the measurement to produce
        a corrected state estimate.  Also stores the normalised innovation
        magnitude for downstream confidence estimation.

        Args:
            bbox: Tracker-provided bounding box ``(x, y, w, h)``.

        Returns:
            Corrected (posterior) bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If :meth:`predict` has not been called this frame.
        """
        if self.x is None or self.P is None:
            raise RuntimeError("Call predict() before correct().")

        z = np.array(bbox, dtype=np.float64)

        # Innovation
        y_innov = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y_innov
        I = np.eye(self._dim_x)
        self.P = (I - K @ self.H) @ self.P

        # Normalised innovation magnitude — small = high confidence
        self._innovation_norm = float(np.sqrt(y_innov @ y_innov))

        return self._state_to_bbox()

    @property
    def innovation_norm(self) -> float:
        """L2 norm of the most recent innovation vector (pixels).

        A small value (< 5 px) indicates the tracker output closely matches
        the Kalman prediction — the tracker is running reliably and the
        prediction model is well-calibrated.  A large value suggests either
        rapid target motion or tracker drift.
        """
        return self._innovation_norm

    @property
    def is_initialized(self) -> bool:
        """True after :meth:`initialize` has been called."""
        return self.x is not None

    def reset(self) -> None:
        """Clear internal state (call before reusing on a new sequence)."""
        self.x = None
        self.P = None
        self._innovation_norm = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_to_bbox(self) -> BBox:
        """Extract ``(x, y, w, h)`` from the state vector."""
        x, y, w, h = self.x[:4]  # type: ignore[index]
        return (float(x), float(y), max(1.0, float(w)), max(1.0, float(h)))
