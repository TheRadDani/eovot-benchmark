"""CamShift — Continuously Adaptive Mean Shift tracker.

Reference
---------
Bradski, G. R. (1998). Computer vision face tracking for use in a
perceptual user interface. Intel Technology Journal, Q2.

Design notes
------------
* Pure OpenCV core API (``cv2.calcBackProject`` + ``cv2.CamShift``) — no
  ``contrib`` module and no extra model files, unlike CSRT/MedianFlow
  (need the legacy/contrib tracking module) or DaSiamRPN/NanoTrack
  (need downloaded ONNX weights).
* Tracks via the target's hue-channel colour distribution rather than a
  learned correlation filter (MOSSE/KCF/CSRT) or online classifier (MIL),
  giving EOVOT a baseline from a genuinely different tracking family —
  useful for comparing how colour-histogram trackers trade accuracy for
  near-zero per-frame compute on edge hardware.
* Extremely lightweight: a histogram back-projection and a few mean-shift
  iterations per frame, no FFTs or learned weights, so memory and CPU
  cost are essentially constant regardless of sequence length.
* Degrades on grayscale or low-saturation/low-value targets (the hue
  channel is undefined or noisy there) — this is a known, documented
  limitation of colour-histogram tracking, not a bug. The tracker falls
  back to the last valid bounding box when the colour-probability window
  collapses (e.g. target colour blends into the background).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import BaseTracker, BBox


class CamShiftTracker(BaseTracker):
    """CamShift colour-histogram tracker.

    Builds a hue-channel histogram of the target on the first frame, then
    on every subsequent frame back-projects that histogram across the
    whole image and runs mean-shift (via ``cv2.CamShift``) to relocate and
    resize the tracking window.

    Args:
        name:            Identifier used in benchmark reports. Default ``"CamShift"``.
        hist_bins:       Number of bins in the hue histogram. Default 16.
        sat_min/sat_max: Saturation range (0-255) used to mask out
                          near-grayscale pixels, where hue is unreliable.
        val_min/val_max: Value (brightness) range (0-255) used to mask out
                          very dark or very bright (washed-out) pixels.
        max_iter:        Maximum mean-shift iterations per frame.
        epsilon:         Mean-shift convergence threshold (pixels).

    Example::

        from eovot.trackers.camshift import CamShiftTracker

        tracker = CamShiftTracker()
        tracker.initialize(frame, (x, y, w, h))
        for frame in sequence:
            bbox = tracker.update(frame)
    """

    def __init__(
        self,
        name: str = "CamShift",
        hist_bins: int = 16,
        sat_min: int = 60,
        sat_max: int = 255,
        val_min: int = 32,
        val_max: int = 255,
        max_iter: int = 10,
        epsilon: float = 1.0,
    ) -> None:
        super().__init__(name=name)
        self.hist_bins = hist_bins
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self._term_crit = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max_iter,
            epsilon,
        )

        self._roi_hist: Optional[np.ndarray] = None
        self._track_window: Optional[tuple] = None  # (x, y, w, h) ints
        self._last_bbox: Optional[BBox] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Build the target's hue histogram from the first frame.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth box ``(x, y, w, h)``.

        Raises:
            ValueError: If the bounding box yields an empty patch.
        """
        x, y, w, h = (int(round(v)) for v in bbox)
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bounding box {bbox}: width and height must be positive.")

        H, W = frame.shape[:2]
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        self._track_window = (x, y, w, h)
        self._last_bbox = (float(x), float(y), float(w), float(h))

        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv_roi,
            (0, self.sat_min, self.val_min),
            (180, self.sat_max, self.val_max),
        )
        self._roi_hist = cv2.calcHist([hsv_roi], [0], mask, [self.hist_bins], [0, 180])
        cv2.normalize(self._roi_hist, self._roi_hist, 0, 255, cv2.NORM_MINMAX)

    def update(self, frame: np.ndarray) -> BBox:
        """Relocate the target via histogram back-projection + mean-shift.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` as floats. Falls back
            to the last valid box if the tracking window collapses
            (zero area), e.g. when the target's colour blends into the
            background.
        """
        if self._roi_hist is None or self._track_window is None:
            raise RuntimeError("initialize() must be called before update().")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0], self._roi_hist, [0, 180], 1)

        _, new_window = cv2.CamShift(back_proj, self._track_window, self._term_crit)
        x, y, w, h = new_window

        if w <= 0 or h <= 0:
            # Colour-probability window collapsed — keep last valid box and
            # window so subsequent frames can still recover the target.
            return self._last_bbox  # type: ignore[return-value]

        self._track_window = (x, y, w, h)
        bbox: BBox = (float(x), float(y), float(w), float(h))
        self._last_bbox = bbox
        return bbox
