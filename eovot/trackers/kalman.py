"""Kalman Filter tracker — motion-smoothed NCC template matching.

Combines a 4-state constant-velocity Kalman filter with OpenCV normalised
cross-correlation (NCC) template matching.  On each frame the tracker:

1. **Predicts** the target centre using the constant-velocity motion model.
2. **Measures** the target position by running NCC template matching inside
   a search window centred on the Kalman prediction.
3. **Updates** the Kalman state using the best-match position as an
   observation, or propagates the prediction alone when the template
   correlation is below ``min_correlation`` (e.g., during occlusion).
4. Returns the smoothed centre position with the fixed template size.

State vector: ``[cx, cy, vx, vy]`` — centre position and velocity.
Target size is held constant at initialisation (a common VOT simplification).

The Kalman filter smooths noisy template-match responses and suppresses
sudden trajectory jumps, making this tracker more stable than raw template
matching on jittery or low-frame-rate streams.

Reference
---------
Kalman, R. E. (1960). A new approach to linear filtering and prediction
problems. *Journal of Basic Engineering*, 82(1), 35–45.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class KalmanFilterTracker(BaseTracker):
    """Template-matching tracker with Kalman filter motion smoothing.

    Args:
        process_noise: Diagonal scale for the process noise matrix Q.
            Lower values trust the constant-velocity model more strongly;
            higher values allow the filter to react faster to acceleration.
            Default: ``0.01``.
        measurement_noise: Diagonal scale for the measurement noise matrix R.
            Lower values trust the NCC template match more; higher values
            trust the Kalman prediction more (useful in noisy backgrounds).
            Default: ``10.0``.
        search_window_scale: Search window width/height as a multiple of the
            target size used during template matching.  Must be > 1.
            Default: ``2.5``.
        min_correlation: NCC threshold below which the template match is
            discarded and the Kalman state propagated without a measurement
            update.  Increase to reduce false-update risk in cluttered scenes;
            decrease to keep the filter updated on low-contrast targets.
            Range: ``[-1, 1]``.  Default: ``0.3``.

    Example::

        from eovot.trackers.kalman import KalmanFilterTracker
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        tracker = KalmanFilterTracker()
        dataset = SyntheticDataset(num_sequences=5, num_frames=100)
        result  = BenchmarkEngine(verbose=False).run(tracker, dataset, "Syn")
        print(result)
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 10.0,
        search_window_scale: float = 2.5,
        min_correlation: float = 0.3,
    ) -> None:
        super().__init__(name="KalmanFilter")
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.search_window_scale = search_window_scale
        self.min_correlation = min_correlation

        # Kalman state: [cx, cy, vx, vy]
        self._state: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._template: Optional[np.ndarray] = None
        self._target_size: Optional[Tuple[int, int]] = None  # (w, h)

        # Kalman system matrices — constant after initialisation
        self._F: Optional[np.ndarray] = None   # state transition (4x4)
        self._H: Optional[np.ndarray] = None   # observation (2x4)
        self._Q: Optional[np.ndarray] = None   # process noise (4x4)
        self._R: Optional[np.ndarray] = None   # measurement noise (2x2)
        self._I4: np.ndarray = np.eye(4, dtype=np.float64)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the Kalman filter on the first frame.

        Args:
            frame: First frame as a ``(H, W, 3)`` BGR array.
            bbox:  Initial bounding box ``(x, y, w, h)`` in pixel coordinates.
        """
        x, y, w, h = (float(v) for v in bbox)
        cx, cy = x + w / 2.0, y + h / 2.0

        tw = max(2, int(round(w)))
        th = max(2, int(round(h)))
        self._target_size = (tw, th)

        # Initial state: stationary at the given position
        self._state = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        # High initial uncertainty so first few measurements dominate
        self._covariance = np.eye(4, dtype=np.float64) * 100.0

        # Constant-velocity state transition (dt = 1 frame)
        self._F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Observation matrix: we observe [cx, cy] only
        self._H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        self._Q = np.eye(4, dtype=np.float64) * self.process_noise
        self._R = np.eye(2, dtype=np.float64) * self.measurement_noise

        gray = self._to_gray(frame)
        self._template = self._extract_patch(gray, cx, cy, tw, th)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict and update the target state, returning the new bounding box.

        Args:
            frame: Current frame ``(H, W, 3)`` BGR array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._state is None:
            raise RuntimeError(
                "KalmanFilterTracker not initialised. Call initialize() first."
            )

        # --- Kalman predict ---
        x_pred = self._F @ self._state
        P_pred = self._F @ self._covariance @ self._F.T + self._Q

        # --- Template match observation ---
        tw, th = self._target_size
        sw = max(tw + 2, int(tw * self.search_window_scale))
        sh = max(th + 2, int(th * self.search_window_scale))
        pred_cx, pred_cy = float(x_pred[0]), float(x_pred[1])

        gray = self._to_gray(frame)
        search_patch = self._extract_patch(gray, pred_cx, pred_cy, sw, sh)
        best_cx, best_cy, correlation = self._template_match(
            search_patch, self._template, pred_cx, pred_cy, sw, sh, tw, th
        )

        # --- Kalman update (only when match is reliable) ---
        if correlation >= self.min_correlation:
            z = np.array([best_cx, best_cy], dtype=np.float64)
            S = self._H @ P_pred @ self._H.T + self._R
            K = P_pred @ self._H.T @ np.linalg.inv(S)
            self._state = x_pred + K @ (z - self._H @ x_pred)
            self._covariance = (self._I4 - K @ self._H) @ P_pred
            # Slow EMA template update to adapt to gradual appearance change
            obs_patch = self._extract_patch(
                gray, float(self._state[0]), float(self._state[1]), tw, th
            )
            if obs_patch.shape == self._template.shape:
                self._template = 0.9 * self._template + 0.1 * obs_patch
        else:
            # No reliable observation — propagate motion model only
            self._state = x_pred
            self._covariance = P_pred

        cx, cy = float(self._state[0]), float(self._state[1])
        return (cx - tw / 2.0, cy - th / 2.0, float(tw), float(th))

    def reset(self) -> None:
        """Reset all internal state so the tracker can be re-initialised."""
        self._state = None
        self._covariance = None
        self._template = None
        self._target_size = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert a BGR or grayscale frame to float32 grayscale."""
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return frame.astype(np.float32)

    @staticmethod
    def _extract_patch(
        gray: np.ndarray, cx: float, cy: float, w: int, h: int
    ) -> np.ndarray:
        """Extract a ``w x h`` patch centred at ``(cx, cy)`` with edge-padding."""
        fh, fw = gray.shape[:2]
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2, y2 = x1 + w, y1 + h

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - fw)
        pad_b = max(0, y2 - fh)
        if pad_l or pad_t or pad_r or pad_b:
            gray = np.pad(
                gray, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge"
            )
            x1 += pad_l
            y1 += pad_t
            x2 += pad_l
            y2 += pad_t

        patch = gray[y1:y2, x1:x2]
        if patch.shape[:2] != (h, w):
            patch = cv2.resize(patch, (w, h))
        return patch.astype(np.float32)

    @staticmethod
    def _template_match(
        search_patch: np.ndarray,
        template: np.ndarray,
        pred_cx: float,
        pred_cy: float,
        sw: int,
        sh: int,
        tw: int,
        th: int,
    ) -> Tuple[float, float, float]:
        """Run NCC template matching; return ``(best_cx, best_cy, correlation)``.

        The returned coordinates are in the global frame coordinate system.
        Returns ``(pred_cx, pred_cy, 0.0)`` when the search patch is too small
        for a valid match.
        """
        if search_patch.shape[0] < th or search_patch.shape[1] < tw:
            return pred_cx, pred_cy, 0.0

        result = cv2.matchTemplate(
            search_patch, template, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # max_loc is the top-left of the best match in the search patch.
        # The centre of that match in the search patch coordinate system is
        # (max_loc[0] + tw/2, max_loc[1] + th/2).  Subtracting the search
        # patch centre (sw/2, sh/2) gives the offset from pred_cx, pred_cy.
        offset_x = max_loc[0] + tw / 2.0 - sw / 2.0
        offset_y = max_loc[1] + th / 2.0 - sh / 2.0
        return float(pred_cx + offset_x), float(pred_cy + offset_y), float(max_val)
