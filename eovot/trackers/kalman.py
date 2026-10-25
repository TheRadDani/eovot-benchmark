"""Kalman Filter Tracker with template-matching measurement.

A constant-velocity Kalman filter that uses normalised cross-correlation
(NCC) template matching for visual measurement and falls back to pure
motion prediction when NCC confidence is below threshold.

This tracker fills an important role in EOVOT: it separates appearance
cues from motion priors.  On sequences where the target moves predictably
the KF prior compensates for weak appearance measurements; on sequences
with sudden motion changes the template match provides a corrective update.

**Scientific purpose**

Used as a baseline to answer: *"How much does pure motion prediction
contribute versus appearance matching?"*  The ``use_prediction_only`` flag
lets researchers degrade the tracker to a pure Kalman predictor (no
appearance), allowing a clean ablation study.

State vector (6-D)::

    [cx, cy, w, h, vx, vy]

Measurement vector (4-D)::

    [cx, cy, w, h]

Motion model: constant velocity (F adds velocity to position each step).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class KalmanFilterTracker(BaseTracker):
    """Kalman-filtered visual tracker using NCC template matching.

    On each frame the tracker:

    1. Runs a **KF predict** step using the constant-velocity motion model.
    2. Searches for the appearance template in a neighbourhood centred on
       the predicted position using ``cv2.TM_CCOEFF_NORMED``.
    3. If NCC score ≥ ``ncc_threshold``, performs a **KF update** with the
       matched position as measurement; otherwise uses the pure KF prediction.

    Args:
        name:               Tracker identifier for benchmark reports.
        process_noise:      Process-noise scalar ``q`` (diagonal of Q).
                            Larger values let the KF trust measurements more.
        measurement_noise:  Measurement-noise scalar ``r`` (scales R by bbox
                            size).  Larger values trust the motion model more.
        search_scale:       Multiplier for the search-window size relative to
                            the predicted bounding-box dimensions.  ``2.5``
                            means the window is 2.5× the predicted box.
        ncc_threshold:      Minimum NCC score to accept a template match.
                            Range [-1, 1]; values < 0.3 indicate a weak match.
        use_prediction_only: When ``True``, the template-matching step is
                            skipped and the tracker runs as a pure motion
                            predictor.  Useful for ablation studies.

    Example::

        from eovot.trackers.kalman import KalmanFilterTracker
        from eovot.datasets.synthetic import SyntheticDataset

        tracker = KalmanFilterTracker()
        dataset = SyntheticDataset(num_sequences=1, num_frames=50)
        seq = dataset[0]
        frames = list(seq)
        tracker.initialize(frames[0], seq.init_bbox)
        for frame in frames[1:]:
            bbox = tracker.update(frame)
    """

    def __init__(
        self,
        name: str = "KalmanFilter",
        process_noise: float = 1e-3,
        measurement_noise: float = 0.5,
        search_scale: float = 2.5,
        ncc_threshold: float = 0.3,
        use_prediction_only: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.search_scale = search_scale
        self.ncc_threshold = ncc_threshold
        self.use_prediction_only = use_prediction_only

        # KF matrices — allocated in initialize()
        self._x: Optional[np.ndarray] = None   # state (6,)
        self._P: Optional[np.ndarray] = None   # covariance (6,6)
        self._F: Optional[np.ndarray] = None   # transition
        self._H: Optional[np.ndarray] = None   # observation
        self._Q: Optional[np.ndarray] = None   # process noise
        self._R: Optional[np.ndarray] = None   # measurement noise

        # Appearance template (grayscale)
        self._template: Optional[np.ndarray] = None
        self._tmpl_wh: Optional[Tuple[int, int]] = None  # (w, h) of initial template

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Set up KF state and extract the appearance template.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth box ``(x, y, w, h)``.
        """
        x, y, w, h = (float(v) for v in bbox)
        cx, cy = x + w / 2.0, y + h / 2.0

        # ---- KF state -----------------------------------------------
        self._x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)

        self._P = np.eye(6, dtype=np.float64) * 100.0

        # Constant-velocity transition: position += velocity each step.
        self._F = np.array(
            [
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Observation: we directly measure [cx, cy, w, h].
        self._H = np.zeros((4, 6), dtype=np.float64)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._H[3, 3] = 1.0

        q = self.process_noise
        self._Q = np.diag([q, q, q * 0.5, q * 0.5, q * 10.0, q * 10.0])

        r = self.measurement_noise
        self._R = np.diag([r * w, r * h, r * w, r * h])

        # ---- Appearance template ------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H_img, W_img = gray.shape
        xi = max(0, int(round(x)))
        yi = max(0, int(round(y)))
        wi = max(1, min(int(round(w)), W_img - xi))
        hi = max(1, min(int(round(h)), H_img - yi))
        self._template = gray[yi : yi + hi, xi : xi + wi].astype(np.float32)
        self._tmpl_wh = (wi, hi)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location in the next frame.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If :meth:`initialize` was not called first.
        """
        if self._x is None:
            raise RuntimeError("initialize() must be called before update().")

        # ---- KF predict ---------------------------------------------
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        cx_p, cy_p, w_p, h_p = float(x_pred[0]), float(x_pred[1]), float(x_pred[2]), float(x_pred[3])
        w_p = max(1.0, w_p)
        h_p = max(1.0, h_p)

        # ---- Template match (optional) -------------------------------
        measurement = np.array([cx_p, cy_p, w_p, h_p])
        accepted = False

        if not self.use_prediction_only and self._template is not None:
            accepted, measurement = self._template_match(frame, cx_p, cy_p, w_p, h_p)

        # ---- KF update -----------------------------------------------
        if accepted:
            z = measurement
            y_innov = z - self._H @ x_pred
            S = self._H @ P_pred @ self._H.T + self._R
            try:
                K = P_pred @ self._H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = np.zeros((6, 4))
            self._x = x_pred + K @ y_innov
            self._P = (np.eye(6) - K @ self._H) @ P_pred
        else:
            self._x = x_pred
            self._P = P_pred

        cx, cy, w_out, h_out = float(self._x[0]), float(self._x[1]), float(self._x[2]), float(self._x[3])
        w_out = max(1.0, w_out)
        h_out = max(1.0, h_out)
        return (cx - w_out / 2.0, cy - h_out / 2.0, w_out, h_out)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _template_match(
        self, frame: np.ndarray, cx_p: float, cy_p: float, w_p: float, h_p: float
    ) -> Tuple[bool, np.ndarray]:
        """Search for the template in a window centred on the KF prediction.

        Returns:
            ``(accepted, measurement)`` where *accepted* is ``True`` when
            NCC ≥ threshold and *measurement* is ``[cx, cy, w, h]``.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H_img, W_img = gray.shape

        # Search window
        sw = int(w_p * self.search_scale)
        sh = int(h_p * self.search_scale)
        sx = max(0, int(cx_p - sw / 2.0))
        sy = max(0, int(cy_p - sh / 2.0))
        sx2 = min(W_img, sx + sw)
        sy2 = min(H_img, sy + sh)

        # Resize template to current predicted box size (handles scale drift).
        tw = max(1, min(int(round(w_p)), sx2 - sx - 1))
        th = max(1, min(int(round(h_p)), sy2 - sy - 1))

        if tw < 3 or th < 3 or sx2 - sx <= tw or sy2 - sy <= th:
            return False, np.array([cx_p, cy_p, w_p, h_p])

        tmpl = cv2.resize(self._template, (tw, th), interpolation=cv2.INTER_LINEAR)
        search = gray[sy:sy2, sx:sx2].astype(np.float32)

        result = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
        if result.size == 0:
            return False, np.array([cx_p, cy_p, w_p, h_p])

        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < self.ncc_threshold:
            return False, np.array([cx_p, cy_p, w_p, h_p])

        match_cx = float(sx + max_loc[0] + tw / 2.0)
        match_cy = float(sy + max_loc[1] + th / 2.0)
        return True, np.array([match_cx, match_cy, float(tw), float(th)])
