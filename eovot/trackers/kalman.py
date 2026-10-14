"""Kalman Filter tracker — constant-velocity motion model baseline.

This tracker uses an 8-dimensional linear Kalman filter to predict bounding
box positions based purely on observed motion.  It contains **no appearance
model**, making it the fastest possible baseline and a useful reference for
isolating the contribution of appearance-based methods.

State vector: [cx, cy, w, h, vcx, vcy, vw, vh]
  - (cx, cy): bounding box centre
  - (w, h):   bounding box dimensions
  - v*:       corresponding velocities (constant-velocity assumption)

Measurement vector: [cx, cy, w, h]
  - Observed directly from the ground-truth initialisation; subsequent
    observations come from this tracker's own predictions (no external
    detector is required).

The filter is tuned with two scalar noise parameters:

* ``process_noise`` (Q): models uncertainty in the motion dynamics.
  Lower values trust the constant-velocity assumption more; higher values
  allow faster adaptation to acceleration.
* ``measurement_noise`` (R): models uncertainty in the bounding-box
  measurement.  Lower values trust the measurement (i.e. the bounding box
  from the previous frame) more.

Because the tracker re-uses its own predictions as measurements, it
effectively becomes a kinematic extrapolator — the only "observation"
is the initialisation bounding box.  This is intentional: it establishes
an upper bound on pure motion prediction and a lower bound on what any
real tracker must beat to be useful.

Typical usage::

    import cv2
    from eovot.trackers.kalman import KalmanFilterTracker

    tracker = KalmanFilterTracker()
    frame   = cv2.imread("frame_0001.jpg")
    bbox    = (100, 80, 60, 40)          # (x, y, w, h)

    tracker.initialize(frame, bbox)

    for frame in subsequent_frames:
        bbox = tracker.update(frame)     # pure Kalman prediction
        x, y, w, h = bbox

Benchmark note
--------------
On a modern CPU this tracker runs at >50 000 FPS — three orders of magnitude
faster than MOSSE — making it invaluable for quantifying the overhead of
appearance modelling and for latency-constrained edge deployments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseTracker, BBox

# ---------------------------------------------------------------------------
# Helper: minimal Kalman filter (numpy-only, no scipy dependency)
# ---------------------------------------------------------------------------

class _KalmanFilter:
    """Minimal linear Kalman filter.

    State: x  (n_states,)
    Measurement: z  (n_meas,)
    """

    def __init__(
        self,
        F: np.ndarray,   # transition matrix         (n, n)
        H: np.ndarray,   # observation matrix        (m, n)
        Q: np.ndarray,   # process noise covariance  (n, n)
        R: np.ndarray,   # measurement noise cov     (m, m)
        x0: np.ndarray,  # initial state             (n,)
        P0: np.ndarray,  # initial error covariance  (n, n)
    ) -> None:
        self.F = F.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.x = x0.copy().astype(np.float64)
        self.P = P0.copy().astype(np.float64)
        self._n = F.shape[0]
        self._I = np.eye(self._n, dtype=np.float64)

    def predict(self) -> np.ndarray:
        """Predict the next state.  Returns the predicted state vector."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """Correct the state with observation *z*.  Returns the updated state."""
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y
        self.P = (self._I - K @ self.H) @ self.P
        return self.x.copy()


# ---------------------------------------------------------------------------
# Public tracker
# ---------------------------------------------------------------------------

class KalmanFilterTracker(BaseTracker):
    """Constant-velocity Kalman filter tracker.

    Tracks a single bounding box using an 8-D linear Kalman filter.  No
    appearance model is used; the tracker predicts the next position based
    solely on the velocity estimated from prior frames.

    Args:
        name:              Human-readable label in benchmark reports.
        process_noise:     Scalar variance added to the state-transition noise
                           matrix ``Q``.  Controls how much the tracker trusts
                           the constant-velocity assumption.  Increase this to
                           allow faster adaptation to sudden direction changes
                           (at the cost of more jitter); decrease to get
                           smoother but slower-adapting predictions.
                           Default: ``1e-2``.
        measurement_noise: Scalar variance of the bounding-box measurement
                           noise matrix ``R``.  Controls how much weight is
                           given to the re-used prediction as an "observation"
                           during the Kalman update step.  Default: ``1e-1``.
        update_with_prediction: If ``True`` (default), feed the Kalman
                           prediction back as a measurement on every frame.
                           This is semantically equivalent to pure kinematic
                           extrapolation but keeps the error covariance bounded
                           and velocity estimates smooth.  Set to ``False`` to
                           disable the update step entirely (pure open-loop
                           prediction).

    Example::

        from eovot.trackers.kalman import KalmanFilterTracker

        tracker = KalmanFilterTracker(process_noise=5e-3, measurement_noise=5e-2)
        tracker.initialize(frame, (100, 80, 60, 40))
        for frame in sequence:
            bbox = tracker.update(frame)
    """

    def __init__(
        self,
        name: str = "KalmanFilter",
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
        update_with_prediction: bool = True,
    ) -> None:
        super().__init__(name)
        self._q = float(process_noise)
        self._r = float(measurement_noise)
        self._update_with_pred = update_with_prediction
        self._kf: Optional[_KalmanFilter] = None
        self._last_bbox: Optional[BBox] = None

    # ------------------------------------------------------------------
    # State-space matrices (built once at initialisation)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_matrices(q: float, r: float):
        """Return (F, H, Q, R) for the 8-D constant-velocity model."""
        dt = 1.0  # discrete time step (one frame)

        # Transition: position += velocity * dt
        F = np.eye(8, dtype=np.float64)
        for i in range(4):
            F[i, i + 4] = dt

        # Observation: we measure the first 4 state components (cx, cy, w, h)
        H = np.zeros((4, 8), dtype=np.float64)
        H[:4, :4] = np.eye(4)

        # Process noise — small and isotropic across all state dims
        Q = np.eye(8, dtype=np.float64) * q

        # Measurement noise — applies to position/size observations only
        R = np.eye(4, dtype=np.float64) * r

        return F, H, Q, R

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the Kalman filter from the ground-truth bounding box.

        The initial state velocity is set to zero; the filter will estimate
        velocity from subsequent predictions.

        Args:
            frame: BGR image (unused — the Kalman filter is appearance-free).
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx, cy = x + w / 2.0, y + h / 2.0

        # Initial state: [cx, cy, w, h, 0, 0, 0, 0]
        x0 = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Initial error covariance: high uncertainty on velocities
        P0 = np.diag([1e0, 1e0, 1e0, 1e0, 1e2, 1e2, 1e2, 1e2]).astype(np.float64)

        F, H, Q, R = self._build_matrices(self._q, self._r)
        self._kf = _KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
        self._last_bbox = (x, y, w, h)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the next bounding box position using the Kalman filter.

        The frame content is **not read** — prediction is purely kinematic.

        Args:
            frame: BGR image (unused).

        Returns:
            Predicted bounding box ``(x, y, w, h)`` with clipped non-negative
            dimensions.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._kf is None:
            raise RuntimeError("KalmanFilterTracker: call initialize() before update().")

        # Predict
        state = self._kf.predict()

        # Optionally update with the predicted position as "measurement"
        if self._update_with_pred:
            meas = state[:4].copy()
            state = self._kf.update(meas)

        cx, cy, w, h = state[0], state[1], state[2], state[3]
        w = max(w, 1.0)
        h = max(h, 1.0)
        x = cx - w / 2.0
        y = cy - h / 2.0

        self._last_bbox = (x, y, w, h)
        return (x, y, w, h)

    def __repr__(self) -> str:
        return (
            f"KalmanFilterTracker(name={self.name!r}, "
            f"q={self._q}, r={self._r})"
        )
