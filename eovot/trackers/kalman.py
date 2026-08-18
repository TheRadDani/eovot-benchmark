"""Kalman Filter Tracker for EOVOT.

Implements a constant-velocity Kalman filter as a standalone tracking baseline.
The filter operates in bounding-box space with a linear motion model, predicting
the next position even when the appearance model is uncertain.

State vector  : [x, y, w, h, vx, vy, vw, vh]  (position + velocity)
Measurement   : [x, y, w, h] via normalized cross-correlation template matching

Roles in EOVOT:

1. **Pure motion baseline** — interpretable tracker that does not rely on
   learned features or gradient descent, only physics-inspired kinematics.
2. **Motion model component** — drop-in replacement for the Kalman state
   estimator in SORT-family multi-object trackers.  Each track in SORT uses
   exactly this formulation.
3. **Occlusion robustness** — when template correlation is below
   ``ncc_threshold``, the tracker relies solely on the Kalman prediction
   rather than an unreliable measurement update, gracefully bridging gaps.

Reference:
    Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction
    Problems." Journal of Basic Engineering, 82(1), 35–45.

    Bewley, A. et al. (2016). "Simple Online and Realtime Tracking." ICIP.
    (SORT applies this exact constant-velocity formulation to each track.)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class KalmanTracker(BaseTracker):
    """Constant-velocity Kalman filter tracker with NCC appearance matching.

    The tracker maintains an 8-dimensional state vector
    ``[x, y, w, h, vx, vy, vw, vh]`` with a first-order constant-velocity
    motion model.  A normalised cross-correlation (NCC) appearance check
    decides whether the appearance measurement is reliable before each
    Kalman update step, making the tracker robust to partial occlusions.

    Args:
        process_noise: Diagonal scale for the process-noise covariance
            ``Q``.  Larger values trust the measurement more than the motion
            prediction.  Default: ``1e-2``.
        measurement_noise: Diagonal scale for the measurement-noise
            covariance ``R``.  Larger values trust the motion model more.
            Default: ``1e-1``.
        initial_covariance: Initial uncertainty in the state estimate.
            Default: ``1.0``.
        learning_rate: EMA weight for the appearance template update.
            ``0.0`` → fixed template; ``1.0`` → always replace.
            Default: ``0.05``.
        ncc_threshold: Minimum normalised cross-correlation (NCC) score
            to accept an appearance measurement.  Below this the tracker
            skips the Kalman update and predicts purely from kinematics.
            Range: ``[-1, 1]``.  Default: ``0.3``.
        search_pad: Padding factor around the predicted bounding box when
            extracting the appearance patch.  Default: ``1.5``.

    Example::

        from eovot.trackers.kalman import KalmanTracker

        tracker = KalmanTracker()
        tracker.initialize(first_frame, bbox)
        for frame in subsequent_frames:
            pred_bbox = tracker.update(frame)
    """

    def __init__(
        self,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
        initial_covariance: float = 1.0,
        learning_rate: float = 0.05,
        ncc_threshold: float = 0.3,
        search_pad: float = 1.5,
    ) -> None:
        super().__init__(name="Kalman")
        self._q_scale = process_noise
        self._r_scale = measurement_noise
        self._p0_scale = initial_covariance
        self._lr = learning_rate
        self._ncc_threshold = ncc_threshold
        self._search_pad = search_pad

        # Kalman matrices (initialised in initialize())
        self._x: Optional[np.ndarray] = None   # state (8,)
        self._P: Optional[np.ndarray] = None   # error covariance (8, 8)
        self._F: Optional[np.ndarray] = None   # transition (8, 8)
        self._H: Optional[np.ndarray] = None   # measurement (4, 8)
        self._Q: Optional[np.ndarray] = None   # process noise (8, 8)
        self._R: Optional[np.ndarray] = None   # measurement noise (4, 4)

        self._template: Optional[np.ndarray] = None   # grayscale appearance patch
        self._target_sz: Optional[Tuple[int, int]] = None  # (w, h)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the Kalman filter on the first frame.

        Args:
            frame: BGR image ``(H, W, 3)`` or grayscale ``(H, W)``.
            bbox:  Initial bounding box ``(x, y, w, h)`` in pixels.
        """
        x, y, w, h = (float(v) for v in bbox)
        tw, th = max(1, int(round(w))), max(1, int(round(h)))
        self._target_sz = (tw, th)

        # State: [x, y, w, h, vx, vy, vw, vh]; velocities initialised to 0.
        self._x = np.array([x, y, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Constant-velocity transition matrix (dt = 1 frame).
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = 1.0
        self._F = F

        # Measurement matrix: observe [x, y, w, h] from state.
        H = np.zeros((4, 8), dtype=np.float64)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        self._H = H

        self._Q = np.eye(8, dtype=np.float64) * self._q_scale
        self._R = np.eye(4, dtype=np.float64) * self._r_scale

        # Higher initial uncertainty for velocity components.
        self._P = np.eye(8, dtype=np.float64) * self._p0_scale
        self._P[4:, 4:] *= 10.0

        gray = self._to_gray(frame)
        self._template = self._extract_patch(gray, x + w / 2.0, y + h / 2.0, tw, th)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict target location in a new frame.

        Performs the Kalman predict step, evaluates NCC to decide whether the
        appearance measurement is reliable, conditionally runs the update step,
        and returns the current state estimate as a bounding box.

        Args:
            frame: BGR image ``(H, W, 3)`` or grayscale ``(H, W)``.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._x is None:
            raise RuntimeError("KalmanTracker not initialised. Call initialize() first.")

        # --- Predict ---
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

        px, py, pw, ph = self._x[:4]
        pw = max(1.0, pw)
        ph = max(1.0, ph)
        self._x[2], self._x[3] = pw, ph

        tw, th = self._target_sz  # type: ignore[misc]
        gray = self._to_gray(frame)

        # --- Appearance check ---
        cx_pred, cy_pred = px + pw / 2.0, py + ph / 2.0
        candidate = self._extract_patch(gray, cx_pred, cy_pred, tw, th)
        ncc = self._ncc(self._template, candidate)

        # --- Conditional update ---
        if ncc >= self._ncc_threshold:
            z = np.array([px, py, pw, ph], dtype=np.float64)
            # Refined measurement: locate best NCC position within search area.
            meas_box = self._refine_measurement(gray, cx_pred, cy_pred, pw, ph, tw, th)
            if meas_box is not None:
                z = np.array(meas_box, dtype=np.float64)

            S = self._H @ self._P @ self._H.T + self._R
            K = self._P @ self._H.T @ np.linalg.inv(S)
            y_innov = z - self._H @ self._x
            self._x = self._x + K @ y_innov
            self._P = (np.eye(8, dtype=np.float64) - K @ self._H) @ self._P

            # Template update with slow learning rate.
            if self._lr > 0.0:
                cx_upd = self._x[0] + self._x[2] / 2.0
                cy_upd = self._x[1] + self._x[3] / 2.0
                new_patch = self._extract_patch(gray, cx_upd, cy_upd, tw, th)
                self._template = (
                    (1.0 - self._lr) * self._template + self._lr * new_patch
                )

        fx, fy, fw, fh = self._x[0], self._x[1], max(1.0, self._x[2]), max(1.0, self._x[3])
        return (float(fx), float(fy), float(fw), float(fh))

    def reset(self) -> None:
        """Clear internal state so the tracker can be re-initialised."""
        self._x = None
        self._P = None
        self._template = None
        self._target_sz = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert frame to float32 grayscale in [0, 1]."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return gray.astype(np.float32) / 255.0

    def _extract_patch(
        self,
        gray: np.ndarray,
        cx: float,
        cy: float,
        tw: int,
        th: int,
    ) -> np.ndarray:
        """Extract a fixed-size grayscale patch centred at ``(cx, cy)``."""
        H, W = gray.shape[:2]
        x1 = int(round(cx - tw / 2.0))
        y1 = int(round(cy - th / 2.0))
        x2, y2 = x1 + tw, y1 + th

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - W)
        pad_b = max(0, y2 - H)

        if pad_l or pad_t or pad_r or pad_b:
            gray = np.pad(gray, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
            x1 += pad_l
            y1 += pad_t
            x2 += pad_l
            y2 += pad_t

        patch = gray[y1:y2, x1:x2]
        if patch.shape != (th, tw):
            patch = cv2.resize(patch, (tw, th))
        return patch.astype(np.float32)

    @staticmethod
    def _ncc(template: np.ndarray, candidate: np.ndarray) -> float:
        """Normalised cross-correlation in [-1, 1] between two equal-size patches."""
        t = template - template.mean()
        c = candidate - candidate.mean()
        denom = (np.linalg.norm(t) * np.linalg.norm(c))
        if denom < 1e-8:
            return 0.0
        return float(np.sum(t * c) / denom)

    def _refine_measurement(
        self,
        gray: np.ndarray,
        cx: float,
        cy: float,
        pw: float,
        ph: float,
        tw: int,
        th: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Run template matching in a search window around the predicted centre.

        Extracts a padded search region and slides the template using
        normalised cross-correlation to find the best matching position.

        Returns:
            ``(x, y, w, h)`` box for the best match, or ``None`` if the
            search window is too small for the template.
        """
        H, W = gray.shape[:2]
        sw = int(round(max(pw, tw) * self._search_pad))
        sh = int(round(max(ph, th) * self._search_pad))

        if sw < tw or sh < th:
            return None

        x1 = int(round(cx - sw / 2.0))
        y1 = int(round(cy - sh / 2.0))
        x2, y2 = x1 + sw, y1 + sh

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - W)
        pad_b = max(0, y2 - H)
        search = gray
        if pad_l or pad_t or pad_r or pad_b:
            search = np.pad(gray, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
            x1 += pad_l
            y1 += pad_t
            x2 += pad_l
            y2 += pad_t

        region = search[y1:y2, x1:x2]
        if region.shape[0] < th or region.shape[1] < tw:
            return None

        # Slide template using NCC (via FFT-based matchTemplate with CCOEFF_NORMED).
        tmpl = (self._template * 255.0).astype(np.float32)
        reg8 = (region * 255.0).astype(np.float32)
        result = cv2.matchTemplate(reg8, tmpl, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        mx, my = max_loc

        # Map back to frame coordinates.
        match_x = (x1 + mx) - pad_l
        match_y = (y1 + my) - pad_t
        return (float(match_x), float(match_y), float(tw), float(th))
