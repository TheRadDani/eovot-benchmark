"""Kalman filter + colour-histogram hybrid tracker.

Fuses two complementary signals for robust edge deployment:

* **Motion prior** — a constant-velocity Kalman filter predicts where the
  target *should* be next frame without any appearance information.
* **Appearance model** — a normalised HSV colour histogram models what the
  target *looks like*; a Bhattacharyya-distance search within a gated window
  around the Kalman prediction finds the best-matching patch in the frame.

Fusion rule
-----------
The Kalman prediction gates the colour search to a bounded local region,
preventing full-frame search and limiting the "snap to distractor" failure
common in plain mean-shift / CamShift trackers.  If no strong colour match is
found (Bhattacharyya distance ≥ ``match_threshold``), the Kalman prediction
is returned unchanged so the tracker never jumps to a random location.

This design mirrors the classical track-then-detect paradigm used in many
lightweight edge-deployment pipelines, making it a useful reference point
when comparing correlation-filter and deep-learning trackers in EOVOT.

Example::

    from eovot.trackers.kalman_color import KalmanColorTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    tracker = KalmanColorTracker(search_factor=2.5, n_bins=16)
    dataset = SyntheticDataset(num_sequences=5, num_frames=150)
    engine  = BenchmarkEngine(verbose=False)
    result  = engine.run(tracker, dataset, dataset_name="Synthetic")
    print(result)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class KalmanColorTracker(BaseTracker):
    """Hybrid Kalman-filter + HSV colour-histogram tracker.

    The tracker maintains a 4-state constant-velocity Kalman filter
    ``[cx, cy, vx, vy]`` alongside an HSV colour histogram of the initial
    target appearance.  Each frame, the Kalman filter first predicts the new
    target centre; a colour-histogram search within a window around that
    prediction refines the box.  When colour evidence is weak the Kalman
    prediction is used as-is, providing graceful degradation under occlusion
    and illumination change.

    Args:
        name: Human-readable identifier.  Default: ``"KalmanColor"``.
        n_bins: Number of histogram bins per HSV channel (Hue and Saturation).
            More bins give a finer appearance model but are less tolerant of
            lighting changes.  Default: ``16``.
        search_factor: Multiplier on the target size that defines the
            colour-search window width/height around the Kalman prediction.
            ``2.0`` searches within a region twice the target dimensions.
            Default: ``2.0``.
        match_threshold: Bhattacharyya distance above which the colour search
            result is discarded and the Kalman prediction is used instead.
            Range ``[0, 1]``; lower values are stricter.  Default: ``0.5``.
        process_noise: Kalman process-noise covariance magnitude.
            Larger values make the filter adapt faster to sudden accelerations.
            Default: ``1e-2``.
        measurement_noise: Kalman measurement-noise covariance magnitude.
            Default: ``1e-1``.

    Example::

        tracker = KalmanColorTracker(n_bins=32, search_factor=3.0)
    """

    def __init__(
        self,
        name: str = "KalmanColor",
        n_bins: int = 16,
        search_factor: float = 2.0,
        match_threshold: float = 0.5,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
    ) -> None:
        super().__init__(name=name)
        self.n_bins = n_bins
        self.search_factor = search_factor
        self.match_threshold = match_threshold
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self._kf: cv2.KalmanFilter
        self._template_hist: np.ndarray
        self._bbox: BBox = (0.0, 0.0, 1.0, 1.0)
        self._initialized = False

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the Kalman filter and build the colour histogram template.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        x, y, w, h = self._bbox

        # 4-state [cx, cy, vx, vy], 2-measurement [cx, cy].
        self._kf = cv2.KalmanFilter(4, 2)
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)

        cx, cy = x + w / 2.0, y + h / 2.0
        self._kf.statePost = np.array(
            [[cx], [cy], [0.0], [0.0]], dtype=np.float32
        )

        self._template_hist = self._compute_histogram(frame, self._bbox)
        self._initialized = True

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location using Kalman + colour search.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.
        """
        pred_state = self._kf.predict()
        pred_cx = float(pred_state[0, 0])
        pred_cy = float(pred_state[1, 0])

        _, _, w, h = self._bbox

        # Colour search within a window gated by the Kalman prediction.
        sx, sy, sw, sh = self._search_window(pred_cx, pred_cy, w, h, frame.shape)
        best_bbox, best_dist = self._colour_search(frame, sx, sy, sw, sh, w, h)

        if best_dist < self.match_threshold:
            bx, by, bw, bh = best_bbox
            new_cx = bx + bw / 2.0
            new_cy = by + bh / 2.0
            self._kf.correct(
                np.array([[new_cx], [new_cy]], dtype=np.float32)
            )
            self._bbox = best_bbox
        else:
            # Weak colour match — keep Kalman prediction, preserve target size.
            new_bbox: BBox = (pred_cx - w / 2.0, pred_cy - h / 2.0, w, h)
            self._kf.correct(
                np.array([[pred_cx], [pred_cy]], dtype=np.float32)
            )
            self._bbox = new_bbox

        return self._bbox

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_histogram(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Extract a normalised 2D HSV histogram (Hue × Saturation) from *bbox*."""
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        fH, fW = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fW, x + max(1, w)), min(fH, y + max(1, h))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((self.n_bins, self.n_bins), dtype=np.float32)
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [self.n_bins, self.n_bins],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist  # type: ignore[return-value]

    def _search_window(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        frame_shape: Tuple[int, ...],
    ) -> Tuple[float, float, float, float]:
        """Return a search window ``(x, y, w, h)`` centred on *(cx, cy)*."""
        fH, fW = frame_shape[:2]
        sw = min(w * self.search_factor, float(fW))
        sh = min(h * self.search_factor, float(fH))
        sx = min(max(0.0, cx - sw / 2.0), fW - sw)
        sy = min(max(0.0, cy - sh / 2.0), fH - sh)
        return sx, sy, sw, sh

    def _colour_search(
        self,
        frame: np.ndarray,
        sx: float,
        sy: float,
        sw: float,
        sh: float,
        target_w: float,
        target_h: float,
    ) -> Tuple[BBox, float]:
        """Slide a target-sized window over the search region; return the best match.

        Stepping is ``target_size / 4`` for a good speed–accuracy trade-off.

        Returns:
            ``(best_bbox, best_bhattacharyya_distance)`` where distance is in
            ``[0, 1]``; lower means a better colour match.
        """
        fH, fW = frame.shape[:2]
        tw = max(1, int(target_w))
        th = max(1, int(target_h))
        isx, isy = int(sx), int(sy)

        best_bbox: BBox = (sx, sy, target_w, target_h)
        best_dist = 1.0

        step_x = max(1, tw // 4)
        step_y = max(1, th // 4)

        x_end = min(isx + int(sw) - tw + 1, fW - tw + 1)
        y_end = min(isy + int(sh) - th + 1, fH - th + 1)

        if x_end <= isx or y_end <= isy:
            return best_bbox, best_dist

        for xi in range(isx, x_end, step_x):
            for yi in range(isy, y_end, step_y):
                candidate: BBox = (float(xi), float(yi), target_w, target_h)
                hist = self._compute_histogram(frame, candidate)
                dist = float(
                    cv2.compareHist(self._template_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                )
                if dist < best_dist:
                    best_dist = dist
                    best_bbox = candidate

        return best_bbox, best_dist
