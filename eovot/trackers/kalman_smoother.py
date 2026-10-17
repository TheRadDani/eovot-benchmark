"""Kalman filter post-processor for visual object tracking predictions.

Addresses the gap between *measuring* temporal jitter (via
:class:`~eovot.metrics.temporal.TemporalConsistencyAnalyzer`) and *fixing*
it.  Wraps any :class:`~eovot.trackers.base.BaseTracker` with a
constant-velocity Kalman filter that:

1. **Smooths** bounding-box predictions, reducing position and scale jitter.
2. **Detects** tracker drift via the Mahalanobis innovation distance.
3. **Coasts** through brief occlusions using the motion model alone.

State and measurement model
---------------------------
The filter tracks 8 state variables::

    x = [cx, cy, w, h, vx, vy, vw, vh]ᵀ

where ``(cx, cy)`` is the box centre, ``(w, h)`` is the box size, and
``(vx, vy, vw, vh)`` are their respective velocities.  The constant-velocity
transition is::

    x_{t+1} = F · x_t + noise,   F = I₄ ⊕ I₄  (block-diagonal with dt=1)

Each tracker prediction provides a 4-D measurement ``z = [cx, cy, w, h]ᵀ``.

Innovation gating
-----------------
The normalised innovation ``γ = νᵀ S⁻¹ ν`` (Mahalanobis distance squared)
follows a χ²(4) distribution under the Gaussian assumption.  When ``γ``
exceeds ``gate_threshold`` the measurement is considered an outlier and the
tracker enters **coast** mode — subsequent frames use the motion prediction
only.  Consecutive coast frames accumulate in :attr:`coasted_frames`; once
``max_coast_frames`` is exceeded, the tracker enters **lost** state.

Confidence score
----------------
Each prediction carries a :attr:`confidence` in ``(0, 1]``::

    confidence = exp(-γ / (2 · gate_threshold))

High confidence (≈ 1.0) means the tracker and filter agree closely.
Low confidence signals drift, occlusion, or target loss.

Example
-------
::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kalman_smoother import KalmanSmootherTracker

    ds      = SyntheticDataset(num_sequences=1, num_frames=200)
    seq     = ds[0]

    raw     = MOSSETracker()
    smooth  = KalmanSmootherTracker(raw, process_noise=1.0, measurement_noise=4.0)

    frames  = list(seq)
    smooth.initialize(frames[0], seq.init_bbox)

    for frame in frames[1:]:
        bbox = smooth.update(frame)
        print(f"conf={smooth.confidence:.3f}  state={smooth.state}  bbox={bbox}")
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseTracker, BBox


class KalmanSmootherTracker(BaseTracker):
    """Constant-velocity Kalman filter wrapper around any :class:`BaseTracker`.

    Args:
        tracker: The underlying tracker that generates raw predictions.
        process_noise: Scalar scaling of the process noise covariance ``Q``.
            Higher values allow the filter to trust tracker jumps more,
            reducing smoothing but increasing responsiveness.  Default: ``1.0``.
        measurement_noise: Scalar scaling of the measurement noise covariance
            ``R``.  Higher values trust the tracker less and rely more on the
            motion model.  Default: ``4.0`` (moderate smoothing).
        gate_threshold: Mahalanobis distance squared threshold for innovation
            gating.  Measurements with ``γ > gate_threshold`` are rejected.
            The χ²(4) 99th percentile is ~13.3; default ``16.0`` is slightly
            more permissive to avoid over-rejection.
        max_coast_frames: Maximum consecutive frames allowed in coast mode
            before the tracker is declared **lost**.  Default: ``10``.
        name: Human-readable name for benchmark reports.  If ``None``,
            defaults to ``"Kalman(<inner_name>)"``.

    Attributes:
        confidence: Per-frame confidence score in ``(0, 1]``.  Updated after
            every :meth:`update` call.  ``1.0`` before the first call.
        state: One of ``"initializing"``, ``"tracking"``, ``"coast"``,
            ``"lost"``.
        coasted_frames: Number of consecutive frames in coast/lost mode.
        innovations: History of Mahalanobis distances (γ) for diagnostics.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        process_noise: float = 1.0,
        measurement_noise: float = 4.0,
        gate_threshold: float = 16.0,
        max_coast_frames: int = 10,
        name: Optional[str] = None,
    ) -> None:
        _name = name if name is not None else f"Kalman({tracker.name})"
        super().__init__(name=_name)

        if process_noise <= 0:
            raise ValueError(f"process_noise must be positive, got {process_noise}")
        if measurement_noise <= 0:
            raise ValueError(f"measurement_noise must be positive, got {measurement_noise}")
        if gate_threshold <= 0:
            raise ValueError(f"gate_threshold must be positive, got {gate_threshold}")
        if max_coast_frames < 1:
            raise ValueError(f"max_coast_frames must be >= 1, got {max_coast_frames}")

        self._tracker = tracker
        self._process_noise = float(process_noise)
        self._measurement_noise = float(measurement_noise)
        self._gate_threshold = float(gate_threshold)
        self._max_coast_frames = int(max_coast_frames)

        # Public state
        self.confidence: float = 1.0
        self.state: str = "initializing"
        self.coasted_frames: int = 0
        self.innovations: list = []

        # Filter matrices — initialized in initialize()
        self._x: Optional[np.ndarray] = None   # state (8,)
        self._P: Optional[np.ndarray] = None   # covariance (8, 8)
        self._F: np.ndarray = self._make_F()
        self._H: np.ndarray = self._make_H()
        self._Q: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialize the inner tracker and set up Kalman state.

        Args:
            frame: First frame BGR image.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)

        x, y, w, h = (float(v) for v in bbox)
        cx, cy = x + w / 2.0, y + h / 2.0

        # Initial state: position known, velocity = 0
        self._x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Initial covariance: high uncertainty on velocity
        self._P = np.diag([
            w * w,      # cx variance ~ box width
            h * h,      # cy variance ~ box height
            w * w,      # w variance
            h * h,      # h variance
            100.0,      # vx large initial uncertainty
            100.0,      # vy
            100.0,      # vw
            100.0,      # vh
        ])

        # Scale noise matrices based on typical target size
        scale = max(w * h, 1.0)
        self._Q = self._make_Q(self._process_noise, scale)
        self._R = np.diag([
            self._measurement_noise * w,
            self._measurement_noise * h,
            self._measurement_noise * w,
            self._measurement_noise * h,
        ])

        self.confidence = 1.0
        self.state = "tracking"
        self.coasted_frames = 0
        self.innovations = []

    def update(self, frame: np.ndarray) -> BBox:
        """Predict, fuse with tracker measurement, and return smoothed bbox.

        Args:
            frame: Current frame BGR image.

        Returns:
            Smoothed bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._x is None:
            raise RuntimeError(
                "KalmanSmootherTracker is not initialized. Call initialize() first."
            )

        # 1. Kalman prediction step
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # 2. Get raw tracker prediction — guard against tracker crashes and
        #    degenerate bboxes (e.g. target drifted fully off-screen).
        tracker_valid = False
        z = None
        try:
            raw_bbox = self._tracker.update(frame)
            rx, ry, rw, rh = (float(v) for v in raw_bbox)
            if rw > 0 and rh > 0:
                rcx, rcy = rx + rw / 2.0, ry + rh / 2.0
                z = np.array([rcx, rcy, rw, rh], dtype=np.float64)
                tracker_valid = True
        except BaseException:
            pass  # inner tracker crashed (e.g. cv2.error) — treat as coast frame

        # 3. Compute innovation and gate (only when tracker produced a valid bbox)
        if tracker_valid and z is not None:
            z_pred = self._H @ x_pred            # predicted measurement (4,)
            nu = z - z_pred                       # innovation (4,)
            S = self._H @ P_pred @ self._H.T + self._R  # innovation covariance (4,4)
            try:
                S_inv = np.linalg.inv(S)
                gamma = float(nu @ S_inv @ nu)    # Mahalanobis² distance
            except np.linalg.LinAlgError:
                gamma = self._gate_threshold + 1.0
            gamma = max(0.0, gamma)
        else:
            # No valid tracker measurement — force coast
            nu = np.zeros(4, dtype=np.float64)
            S_inv = np.eye(4, dtype=np.float64)
            gamma = self._gate_threshold + 1.0

        self.innovations.append(gamma)

        # 4. Confidence score
        self.confidence = float(np.exp(-gamma / (2.0 * self._gate_threshold)))

        if tracker_valid and gamma <= self._gate_threshold and self.state != "lost":
            # Accepted measurement: full Kalman update
            K = P_pred @ self._H.T @ S_inv       # gain (8, 4)
            self._x = x_pred + K @ nu
            self._P = (np.eye(8) - K @ self._H) @ P_pred
            self.state = "tracking"
            self.coasted_frames = 0
        else:
            # Rejected or invalid measurement: coast on motion prediction
            self._x = x_pred
            self._P = P_pred
            self.coasted_frames += 1
            if self.coasted_frames > self._max_coast_frames:
                self.state = "lost"
            else:
                self.state = "coast"

        # 5. Convert filtered state to (x, y, w, h)
        cx_f, cy_f, w_f, h_f = (
            self._x[0], self._x[1], self._x[2], self._x[3]
        )
        w_f = max(w_f, 1.0)
        h_f = max(h_f, 1.0)
        return (cx_f - w_f / 2.0, cy_f - h_f / 2.0, w_f, h_f)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_F() -> np.ndarray:
        """Build the 8×8 constant-velocity state transition matrix."""
        F = np.eye(8, dtype=np.float64)
        # Position += velocity * dt (dt=1 frame)
        F[0, 4] = 1.0  # cx += vx
        F[1, 5] = 1.0  # cy += vy
        F[2, 6] = 1.0  # w  += vw
        F[3, 7] = 1.0  # h  += vh
        return F

    @staticmethod
    def _make_H() -> np.ndarray:
        """Build the 4×8 measurement matrix (observe position/size only)."""
        H = np.zeros((4, 8), dtype=np.float64)
        H[0, 0] = 1.0  # measure cx
        H[1, 1] = 1.0  # measure cy
        H[2, 2] = 1.0  # measure w
        H[3, 3] = 1.0  # measure h
        return H

    @staticmethod
    def _make_Q(process_noise: float, scale: float) -> np.ndarray:
        """Build the 8×8 process noise covariance matrix.

        Position noise is scaled by ``scale`` (target area) to keep the
        filter adaptive to different target sizes.  Velocity noise is lower
        to enforce smooth motion.
        """
        pos_var = process_noise * np.sqrt(scale)
        vel_var = process_noise * np.sqrt(scale) * 0.1
        return np.diag([
            pos_var, pos_var, pos_var, pos_var,
            vel_var, vel_var, vel_var, vel_var,
        ])

    def mean_innovation(self) -> float:
        """Return the mean Mahalanobis distance over all processed frames."""
        if not self.innovations:
            return 0.0
        return float(np.mean(self.innovations))

    def __repr__(self) -> str:
        return (
            f"KalmanSmootherTracker("
            f"tracker={self._tracker.name!r}, "
            f"pn={self._process_noise}, "
            f"mn={self._measurement_noise}, "
            f"gate={self._gate_threshold})"
        )
