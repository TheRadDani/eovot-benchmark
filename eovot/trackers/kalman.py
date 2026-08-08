"""Constant-velocity Kalman filter tracker for EOVOT.

Models target motion as a 2-D constant-velocity kinematic process with
additional state components for bounding-box scale change.  The tracker
is entirely appearance-free: after initialisation from the first-frame
ground-truth box it predicts purely from past motion, making it the
canonical *kinematic baseline* for tracker comparison studies.

State vector (8-D): ``[x, y, w, h, vx, vy, vw, vh]``

  * ``(x, y)``         — top-left corner of the bounding box (pixels)
  * ``(w, h)``         — box width and height (pixels)
  * ``(vx, vy)``       — translational velocity (px / frame)
  * ``(vw, vh)``       — scale-change velocity (px / frame)

The constant-velocity transition assumes ``position[t] = position[t-1] + velocity``.

Research value
--------------
Comparing ``KalmanTracker`` against appearance-based trackers (MOSSE, KCF)
reveals how much of tracking accuracy is due to visual re-detection vs.
kinematic predictability.  Sequences where the Kalman tracker is competitive
are dominated by smooth, predictable motion; large gaps signal the need for
strong appearance modeling.

Hybrid tracking
---------------
:meth:`observe` accepts external bounding-box detections (from a re-detector,
ground-truth oracle, or another tracker) and fuses them with the kinematic
prediction using the Kalman update equations.  This enables using
``KalmanTracker`` as a smooth interpolator between sparse detections — a
common pattern in multi-stage tracking pipelines.

References
----------
Welch, G., & Bishop, G. (1995). An introduction to the Kalman filter.
UNC-Chapel Hill, TR 95-041.

Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016).
Simple online and realtime tracking (SORT). IEEE ICIP 2016.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox


class KalmanTracker(BaseTracker):
    """Constant-velocity Kalman filter tracker — pure motion-model baseline.

    Uses an 8-dimensional state vector ``[x, y, w, h, vx, vy, vw, vh]``
    and a linear constant-velocity motion model to predict the target's
    bounding box in each frame without reading any pixel content.

    Args:
        name:              Human-readable identifier used in benchmark reports.
                           Default: ``"Kalman"``.
        process_noise:     Scalar process noise variance :math:`\\sigma^2_q`.
                           Controls how quickly state uncertainty grows during
                           prediction — higher values allow faster adaptation
                           to non-constant-velocity motion at the cost of
                           noisier predictions.  Default: ``1e-2``.
        measurement_noise: Scalar measurement noise variance :math:`\\sigma^2_r`
                           applied when :meth:`observe` is called.  Lower values
                           make the filter trust external observations more.
                           Default: ``1e-1``.
        initial_vel_var:   Initial variance assigned to the velocity state
                           components.  A large value means the filter is
                           initially uncertain about the target's velocity and
                           will adapt quickly when observations are provided.
                           Default: ``10.0``.

    Example::

        from eovot.trackers.kalman import KalmanTracker

        tracker = KalmanTracker(process_noise=1e-2)
        tracker.initialize(first_frame, (100, 80, 60, 40))

        # Pure kinematic prediction (no appearance)
        for frame in sequence[1:]:
            bbox = tracker.update(frame)

        # Hybrid mode: correct with an external detector
        tracker.update(frame)
        tracker.observe((detected_x, detected_y, detected_w, detected_h))
    """

    def __init__(
        self,
        name: str = "Kalman",
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
        initial_vel_var: float = 10.0,
    ) -> None:
        super().__init__(name)
        self._q = float(process_noise)
        self._r = float(measurement_noise)
        self._v0 = float(initial_vel_var)

        # State transition matrix F: x_{k} = F @ x_{k-1}
        # [x, y, w, h] ← [x, y, w, h] + dt * [vx, vy, vw, vh]  (dt = 1 frame)
        self._F: np.ndarray = np.eye(8, dtype=np.float64)
        for i in range(4):
            self._F[i, i + 4] = 1.0

        # Observation matrix H: z_k = H @ x_k  (observe position+size only)
        self._H: np.ndarray = np.zeros((4, 8), dtype=np.float64)
        np.fill_diagonal(self._H, 1.0)

        # Process noise covariance Q (isotropic)
        self._Q: np.ndarray = np.eye(8, dtype=np.float64) * self._q

        # Measurement noise covariance R (isotropic)
        self._R: np.ndarray = np.eye(4, dtype=np.float64) * self._r

        # State mean and covariance — populated in initialize()
        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the Kalman filter from the first-frame ground-truth box.

        Args:
            frame: BGR image (not read — the kinematic model has no appearance).
            bbox:  Initial bounding box ``(x, y, w, h)`` in pixel coordinates.
        """
        x, y, w, h = (float(v) for v in bbox)

        # Mean: position from GT bbox; velocities start at zero
        self._x = np.array([x, y, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Covariance: low uncertainty on position (we just observed it),
        # high uncertainty on velocity (completely unknown initially)
        self._P = np.diag(
            [self._r, self._r, self._r, self._r,
             self._v0, self._v0, self._v0, self._v0]
        ).astype(np.float64)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the next bounding box using the constant-velocity model.

        This is a pure **prediction** step — no pixel content is read.
        The ``frame`` parameter is accepted for ``BaseTracker`` API
        compatibility but is not used.

        Args:
            frame: BGR image (ignored by the kinematic model).

        Returns:
            Predicted bounding box ``(x, y, w, h)``.  Width and height are
            clamped to a minimum of 1 pixel to prevent degenerate boxes.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._x is None or self._P is None:
            raise RuntimeError(
                "KalmanTracker not initialised — call initialize() before update()."
            )

        # Predict step: propagate state and covariance forward one frame
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

        return self._state_to_bbox()

    def observe(self, bbox: BBox) -> None:
        """Fuse an external bounding-box measurement into the Kalman state.

        Enables *hybrid tracking*: call after :meth:`update` whenever an
        external detector supplies a position fix.  The Kalman update
        equations blend the kinematic prediction with the observation,
        weighted by their respective noise covariances.

        Typical pattern::

            pred = tracker.update(frame)
            if detector_has_detection:
                tracker.observe(detector_bbox)
            # tracker.state is now the fused estimate

        Args:
            bbox: External bounding-box observation ``(x, y, w, h)``.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._x is None or self._P is None:
            raise RuntimeError(
                "KalmanTracker not initialised — call initialize() before observe()."
            )

        z = np.array([float(v) for v in bbox], dtype=np.float64)

        # Innovation (residual between observation and prediction)
        y_innov = z - self._H @ self._x

        # Innovation covariance
        S = self._H @ self._P @ self._H.T + self._R

        # Kalman gain
        K = self._P @ self._H.T @ np.linalg.inv(S)

        # State update
        self._x = self._x + K @ y_innov

        # Covariance update (Joseph / symmetric form for numerical stability)
        I_KH = np.eye(8, dtype=np.float64) - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

    # ------------------------------------------------------------------
    # Diagnostic properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> Optional[np.ndarray]:
        """Current state vector ``[x, y, w, h, vx, vy, vw, vh]``, or ``None``."""
        return self._x.copy() if self._x is not None else None

    @property
    def covariance(self) -> Optional[np.ndarray]:
        """Current state covariance matrix ``(8, 8)``, or ``None``."""
        return self._P.copy() if self._P is not None else None

    @property
    def velocity(self) -> Optional[np.ndarray]:
        """Current velocity estimate ``[vx, vy, vw, vh]``, or ``None``."""
        return self._x[4:].copy() if self._x is not None else None

    @property
    def uncertainty(self) -> Optional[float]:
        """Scalar uncertainty: trace of the state covariance matrix."""
        return float(np.trace(self._P)) if self._P is not None else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _state_to_bbox(self) -> BBox:
        """Extract and validate a bounding box from the current state."""
        x = float(self._x[0])
        y = float(self._x[1])
        w = max(float(self._x[2]), 1.0)
        h = max(float(self._x[3]), 1.0)
        return (x, y, w, h)
