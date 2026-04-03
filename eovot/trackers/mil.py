"""MIL — Multiple Instance Learning tracker (OpenCV built-in).

Reference
---------
Babenko, B., Yang, M. H., & Belongie, S. (2011).
Robust Object Tracking with Online Multiple Instance Learning.
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI),
33(8), 1619–1632.

Design notes
------------
* Wraps ``cv2.TrackerMIL_create()`` — ships with opencv-python ≥ 4.x,
  no extra downloads or contrib packages required.
* Uses a bag-of-instances sampling strategy: treats a region of positive
  patches as a "positive bag" and trains an online AdaBoost classifier,
  making it more robust to ambiguous positives than standard boosting.
* Slower than MOSSE (~500 FPS) and KCF (~150–350 FPS) but typically more
  accurate on target appearance change and mild occlusion.
* Falls back to the last valid bounding box when the internal tracker
  reports a failure (target fully occluded or out of frame).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import BaseTracker, BBox


class MILTracker(BaseTracker):
    """Multiple Instance Learning object tracker (OpenCV built-in).

    MIL improves on classical boosting-based trackers by learning from a
    *bag* of candidate patches rather than a single positive example per
    frame, which reduces drift caused by noisy positive samples.

    This tracker fills the gap between the fast-but-simple correlation
    filters (MOSSE, KCF) and deep-learning-based trackers.  It runs at
    ~30–80 FPS on a modern CPU core, making it viable on mid-range edge
    devices (e.g. Jetson Nano, Raspberry Pi 4 with native code).

    Args:
        name:            Human-readable identifier used in benchmark reports.
        feature_count:   Number of Haar-like features sampled from the pool.
                         Higher values increase accuracy at the cost of speed.
                         ``None`` uses the OpenCV default (250).

    Example::

        tracker = MILTracker()
        tracker.initialize(first_frame, init_bbox)
        for frame in sequence:
            pred_bbox = tracker.update(frame)
    """

    # OpenCV's MIL implementation uses fixed-size Haar feature matrices.
    # Values below this threshold trigger an internal assertion error.
    _MIN_FEATURE_COUNT: int = 250

    def __init__(
        self,
        name: str = "MIL",
        feature_count: Optional[int] = None,
    ) -> None:
        super().__init__(name)
        if feature_count is not None and feature_count < self._MIN_FEATURE_COUNT:
            raise ValueError(
                f"feature_count must be >= {self._MIN_FEATURE_COUNT} "
                f"(OpenCV internal constraint). Got {feature_count}."
            )
        self._feature_count = feature_count
        self._tracker: Optional[cv2.TrackerMIL] = None
        self._last_bbox: BBox = (0.0, 0.0, 1.0, 1.0)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise MIL tracker on the first frame of a sequence.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        if self._feature_count is not None:
            params = cv2.TrackerMIL_Params()
            params.featureSetNumFeatures = self._feature_count
            self._tracker = cv2.TrackerMIL_create(params)
        else:
            self._tracker = cv2.TrackerMIL_create()

        x, y, w, h = (max(0, int(v)) for v in bbox)
        self._last_bbox = (float(x), float(y), float(w), float(h))
        self._tracker.init(frame, (x, y, w, h))

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location in the current frame.

        Falls back to the last valid bounding box when the internal
        tracker reports a failure (``ok = False``).

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.
        """
        if self._tracker is None:
            return self._last_bbox

        ok, bbox = self._tracker.update(frame)
        if ok:
            self._last_bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        return self._last_bbox
