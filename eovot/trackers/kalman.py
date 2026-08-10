"""Kalman filter tracker with NCC template-matching measurement for EOVOT.

Combines a constant-velocity Kalman filter motion model with normalised
cross-correlation (NCC) template matching for visual measurement.  The
result is a two-stage Bayesian tracker: the Kalman prediction defines
where to look, NCC decides whether the target is actually there.

Motion model
------------
State vector ``x = [cx, cy, w, h, vx, vy]`` (6-D):

    cx, cy  — bounding-box centre in pixels
    w, h    — box width and height
    vx, vy  — estimated horizontal and vertical velocity (px/frame)

Width and height are modelled as constant (zero velocity in the w/h
dimensions), which is accurate for most tracking sequences and keeps the
model dimension small without sacrificing practical accuracy.

Kalman equations (standard form)
---------------------------------
Predict:
    x̂⁻ = F x̂
    P⁻  = F P Fᵀ + Q

Update (when NCC measurement is accepted):
    K   = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹
    x̂   = x̂⁻ + K (z − H x̂⁻)
    P   = (I − K H) P⁻

Template design
---------------
The appearance template includes both the target AND a surrounding
context border (``context_factor × target_size`` on each side).
Including context is critical: a plain-colour target would produce a
uniform template where NCC cannot distinguish the target location.
The target-background boundary inside the context window provides the
discriminative structure NCC relies on.

Template size:
    tmpl_w = target_w × (1 + 2 × context_factor)
    tmpl_h = target_h × (1 + 2 × context_factor)

Because the template is centred on the target, the matched top-left
position in the search region converts back to image coordinates as:

    cx_image = sreg_x1 + max_loc_x + tmpl_w / 2

Template matching measurement
------------------------------
On each frame the tracker crops a search region centred on the Kalman
prediction (``search_factor × template_size``), normalises it, and
runs ``cv2.matchTemplate`` with ``TM_CCOEFF_NORMED``.  The peak NCC
score and location give both an acceptance signal and the [cx, cy, w, h]
measurement fed to the Kalman update step.

When NCC < ``ncc_threshold`` the measurement is treated as missing and
the state is propagated by prediction only (covariance grows).

Edge deployment profile
-----------------------
Pure NumPy and OpenCV (no contrib module, no ONNX weights).  A typical
run costs one ``cv2.matchTemplate`` call plus six small matrix
multiplications per frame — expected throughput 200–500 FPS on a modern
CPU core.

Example::

    from eovot.trackers.kalman import KalmanFilterTracker

    tracker = KalmanFilterTracker()
    tracker.initialize(first_frame, (x, y, w, h))
    for frame in sequence:
        bbox = tracker.update(frame)
        print(bbox)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class KalmanFilterTracker(BaseTracker):
    """Constant-velocity Kalman filter tracker with NCC template matching.

    Args:
        process_noise: Scalar that seeds the diagonal process-noise
            covariance Q.  Higher values make the filter trust its motion
            model less and react faster to measurement corrections.
            Default: ``1.0``.
        measurement_noise: Scalar that seeds the diagonal measurement-noise
            covariance R.  Higher values make the filter trust visual
            measurements less (relies more on motion prediction).
            Default: ``10.0``.
        context_factor: Fraction of the target size added as context border
            around the target when building the appearance template.
            A value of ``1.0`` means the template is 3× the target width
            (target + 1× context on each side).  Context provides
            discriminative structure even for plain-colour targets.
            Default: ``1.0``.
        search_factor: The NCC search window is
            ``search_factor × template_w`` × ``search_factor × template_h``
            centred on the Kalman prediction.  Default: ``1.5``.
        ncc_threshold: Minimum NCC score (in ``[−1, 1]``) to accept a
            template-matching measurement.  Frames where NCC falls below
            this value perform Kalman prediction only (no update).
            Default: ``0.3``.
        learning_rate: EMA weight for online template updates.  ``0.0``
            means a completely static template; ``1.0`` replaces the
            template with the current patch every frame.  Default: ``0.05``.
        name: Human-readable identifier for benchmark reports.

    Example::

        from eovot.trackers.kalman import KalmanFilterTracker

        tracker = KalmanFilterTracker(ncc_threshold=0.4, search_factor=2.0)
        tracker.initialize(frame0, (100, 80, 60, 50))
        for frame in frames[1:]:
            x, y, w, h = tracker.update(frame)
    """

    # State dimension and index constants
    _N_STATE = 6
    _N_OBS = 4
    # State vector layout: [cx, cy, w, h, vx, vy]
    _I_CX, _I_CY, _I_W, _I_H, _I_VX, _I_VY = 0, 1, 2, 3, 4, 5

    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
        context_factor: float = 1.0,
        search_factor: float = 1.5,
        ncc_threshold: float = 0.3,
        learning_rate: float = 0.05,
        name: str = "KalmanFilter",
    ) -> None:
        super().__init__(name=name)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.context_factor = context_factor
        self.search_factor = search_factor
        self.ncc_threshold = ncc_threshold
        self.learning_rate = learning_rate

        # State transition matrix F (constant-velocity model)
        self._trans_mat = np.eye(self._N_STATE, dtype=np.float64)
        self._trans_mat[self._I_CX, self._I_VX] = 1.0
        self._trans_mat[self._I_CY, self._I_VY] = 1.0

        # Observation matrix: observe [cx, cy, w, h] from the 6-D state
        self._obs_mat = np.zeros((self._N_OBS, self._N_STATE), dtype=np.float64)
        for i in range(self._N_OBS):
            self._obs_mat[i, i] = 1.0

        # Process noise covariance Q (diagonal)
        self._Q = np.diag([
            process_noise,        # cx
            process_noise,        # cy
            process_noise * 0.1,  # w  (size changes slowly)
            process_noise * 0.1,  # h
            process_noise * 4.0,  # vx (velocity has higher uncertainty)
            process_noise * 4.0,  # vy
        ])

        # Measurement noise covariance R (diagonal)
        self._R = np.eye(self._N_OBS, dtype=np.float64) * measurement_noise

        # Internal state — set in initialize()
        self._x: Optional[np.ndarray] = None         # state vector (6,)
        self._P: Optional[np.ndarray] = None         # state covariance (6, 6)
        self._template: Optional[np.ndarray] = None  # float32 grayscale template
        self._target_sz: Optional[Tuple[int, int]] = None   # (w, h) of target
        self._template_sz: Optional[Tuple[int, int]] = None # (w, h) of template incl. context

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the tracker on the first frame.

        Args:
            frame: BGR or grayscale image ``(H, W[, C])``.
            bbox:  Ground-truth box ``(x, y, w, h)``.

        Raises:
            ValueError: If the bounding box has non-positive area.
        """
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0 or h <= 0:
            raise ValueError(
                f"KalmanFilterTracker: bounding box must have positive area, got {bbox}."
            )

        cx, cy = x + w / 2.0, y + h / 2.0

        # Initial state: position from bbox, velocity assumed zero
        self._x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)

        # Initial covariance: high uncertainty on velocity
        self._P = np.diag([
            self.measurement_noise,
            self.measurement_noise,
            self.measurement_noise * 0.1,
            self.measurement_noise * 0.1,
            self.measurement_noise * 100.0,
            self.measurement_noise * 100.0,
        ])

        self._target_sz = (max(1, int(round(w))), max(1, int(round(h))))

        # Template: target + context border for discriminative structure.
        # A plain-colour target with no context produces a uniform template
        # where NCC cannot distinguish location; context fixes this.
        tmpl_w = max(1, int(round(w * (1.0 + 2.0 * self.context_factor))))
        tmpl_h = max(1, int(round(h * (1.0 + 2.0 * self.context_factor))))
        self._template_sz = (tmpl_w, tmpl_h)

        gray = self._to_gray(frame)
        self._template = self._crop_patch(gray, cx, cy, tmpl_w, tmpl_h).astype(np.float32)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict and update the target location.

        Args:
            frame: BGR or grayscale image.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._x is None or self._P is None or self._template is None:
            raise RuntimeError(
                "KalmanFilterTracker: initialize() must be called before update()."
            )

        # 1. Kalman prediction step
        x_pred = self._trans_mat @ self._x
        P_pred = self._trans_mat @ self._P @ self._trans_mat.T + self._Q

        gray = self._to_gray(frame)
        pred_cx = float(x_pred[self._I_CX])
        pred_cy = float(x_pred[self._I_CY])
        pred_w = max(1.0, float(x_pred[self._I_W]))
        pred_h = max(1.0, float(x_pred[self._I_H]))

        # 2. Template-matching measurement
        ncc_score, meas_cx, meas_cy = self._match_template(
            gray, pred_cx, pred_cy, pred_w, pred_h
        )

        if ncc_score >= self.ncc_threshold:
            # Measurement accepted: form z = [cx, cy, w, h]
            z = np.array([meas_cx, meas_cy, pred_w, pred_h], dtype=np.float64)

            # Kalman update step
            H = self._obs_mat
            innov = z - H @ x_pred                        # innovation
            S = H @ P_pred @ H.T + self._R                # innovation covariance
            K = P_pred @ H.T @ np.linalg.inv(S)           # Kalman gain
            self._x = x_pred + K @ innov
            self._P = (np.eye(self._N_STATE) - K @ H) @ P_pred

            # Online template update (EMA) — only on accepted measurements
            out_cx = float(self._x[self._I_CX])
            out_cy = float(self._x[self._I_CY])
            tw, th = self._template_sz
            new_patch = self._crop_patch(gray, out_cx, out_cy, tw, th).astype(np.float32)
            if new_patch.shape == self._template.shape:
                lr = self.learning_rate
                self._template = (1.0 - lr) * self._template + lr * new_patch
        else:
            # Measurement rejected: propagate prediction without update
            self._x = x_pred
            self._P = P_pred

        # Convert [cx, cy, w, h, vx, vy] state back to (x, y, w, h) bbox
        cx_out = float(self._x[self._I_CX])
        cy_out = float(self._x[self._I_CY])
        w_out = max(1.0, float(self._x[self._I_W]))
        h_out = max(1.0, float(self._x[self._I_H]))
        return (cx_out - w_out / 2.0, cy_out - h_out / 2.0, w_out, h_out)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_template(
        self,
        gray: np.ndarray,
        pred_cx: float,
        pred_cy: float,
        target_w: float,
        target_h: float,
    ) -> Tuple[float, float, float]:
        """Search for the template in a region around the Kalman prediction.

        Args:
            gray:     Single-channel uint8 grayscale image.
            pred_cx:  Kalman-predicted target centre x.
            pred_cy:  Kalman-predicted target centre y.
            target_w: Current estimated target width (from Kalman state).
            target_h: Current estimated target height (from Kalman state).

        Returns:
            ``(ncc_score, matched_cx, matched_cy)`` — NCC in ``[−1, 1]``
            and the matched target-centre coordinates in image pixels.
        """
        tw, th = self._template_sz

        # Scale template to current target size (handles gentle scale drift)
        scale_x = target_w / max(1.0, self._target_sz[0])
        scale_y = target_h / max(1.0, self._target_sz[1])
        scaled_tw = max(1, int(round(tw * scale_x)))
        scaled_th = max(1, int(round(th * scale_y)))

        tmpl = self._template
        if tmpl.shape != (scaled_th, scaled_tw):
            tmpl = cv2.resize(
                tmpl, (scaled_tw, scaled_th), interpolation=cv2.INTER_LINEAR
            )

        # Search region: search_factor × template size centred on Kalman prediction
        sw = max(scaled_tw + 2, int(round(scaled_tw * self.search_factor)))
        sh = max(scaled_th + 2, int(round(scaled_th * self.search_factor)))

        # Crop (with edge-replication padding) search region from current frame
        sreg = self._crop_patch(gray, pred_cx, pred_cy, sw, sh).astype(np.float32)

        if sreg.shape[0] < tmpl.shape[0] or sreg.shape[1] < tmpl.shape[1]:
            return (-1.0, pred_cx, pred_cy)

        result = cv2.matchTemplate(sreg, tmpl, cv2.TM_CCOEFF_NORMED)
        _, ncc_score, _, max_loc = cv2.minMaxLoc(result)

        # Convert match location back to image centre coordinates.
        # sreg_x1 / sreg_y1 are the nominal image-coordinate left/top edges
        # of the search region (may be negative for targets near the border).
        # max_loc is (column, row) in the search region.
        # Target centre = search-region left-edge + template left-edge + template half-width
        sreg_x1 = pred_cx - sw / 2.0
        sreg_y1 = pred_cy - sh / 2.0
        matched_cx = sreg_x1 + max_loc[0] + scaled_tw / 2.0
        matched_cy = sreg_y1 + max_loc[1] + scaled_th / 2.0

        return (float(ncc_score), matched_cx, matched_cy)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert BGR / BGRA image to single-channel uint8 grayscale."""
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _crop_patch(
        gray: np.ndarray,
        cx: float,
        cy: float,
        w: int,
        h: int,
    ) -> np.ndarray:
        """Crop a ``(h, w)`` patch centred at ``(cx, cy)`` with edge-replication padding."""
        ih, iw = gray.shape[:2]
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2, y2 = x1 + w, y1 + h

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - iw)
        pad_b = max(0, y2 - ih)

        if pad_l or pad_t or pad_r or pad_b:
            gray = np.pad(gray, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
            x1 += pad_l
            y1 += pad_t
            x2 += pad_l
            y2 += pad_t

        patch = gray[y1:y2, x1:x2]
        if patch.shape != (h, w):
            patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
        return patch
