"""MedianFlow tracker wrapper for EOVOT.

MedianFlow (Kalal et al., ICPR 2010) is a classical forward-backward tracking
algorithm built on sparse Lucas-Kanade optical flow:

1. Forward pass: track a grid of points from frame t → t+1 via LK optical flow.
2. Backward pass: track the same points from frame t+1 → t.
3. Forward-backward error (FBE): Euclidean distance between original and
   back-tracked positions — a reliable self-consistency score.
4. Median displacement: the median (x, y) shift and scale change of the
   point subset with the lowest FBE becomes the predicted box motion.

**Failure detection**: MedianFlow explicitly signals tracking failure when the
median FBE exceeds a threshold (default 10 px) or too few points survive the
flow.  This makes it unusually well-suited to detecting occlusion or rapid
motion, properties that matter for edge deployment scenarios where silent
drift is unacceptable.

Performance characteristics:
- CPU throughput: ~80–200 FPS (single core), dominated by optical flow.
- Memory: O(grid_size²) — constant, well within edge device limits.
- Accuracy: lower than CSRT on long, cluttered sequences but excellent
  on short, fast-motion clips with clear backgrounds.

This wrapper exposes an ``ok`` flag via the underlying OpenCV tracker and
returns the last valid box if tracking fails, consistent with EOVOT's
convention for graceful degradation.

Dependency:  standard ``opencv-python`` (contrib not required).

Reference:
    Kalal et al., "Forward-Backward Error: Automatic Detection of Tracking
    Failures." ICPR 2010.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import BaseTracker, BBox

# MedianFlow is in the main OpenCV tracking module (not contrib) since 4.x.
try:
    _create_median_flow = cv2.legacy.TrackerMedianFlow_create  # OpenCV 4.5+
except AttributeError:
    try:
        _create_median_flow = cv2.TrackerMedianFlow_create  # OpenCV < 4.5
    except AttributeError:
        _create_median_flow = None  # type: ignore[assignment]


class MedianFlowTracker(BaseTracker):
    """EOVOT wrapper around OpenCV's MedianFlow tracker.

    Provides explicit failure detection: if optical flow self-consistency
    drops below the OpenCV threshold, the tracker signals failure and the
    last valid bounding box is returned.

    Args:
        name: Identifier used in benchmark reports. Default: ``"MedianFlow"``.

    Raises:
        ImportError: If neither ``cv2.legacy.TrackerMedianFlow_create`` nor
            ``cv2.TrackerMedianFlow_create`` is available in the installed
            OpenCV build.

    Example::

        from eovot.trackers.median_flow import MedianFlowTracker

        tracker = MedianFlowTracker()
        tracker.initialize(frame, (x, y, w, h))
        for frame in sequence:
            bbox = tracker.update(frame)
    """

    def __init__(self, name: str = "MedianFlow") -> None:
        if _create_median_flow is None:
            raise ImportError(
                "OpenCV MedianFlow tracker is not available in this build.\n"
                "Try: pip install opencv-python>=4.5 or opencv-contrib-python"
            )
        super().__init__(name=name)
        self._tracker: Optional[cv2.Tracker] = None
        self._last_bbox: Optional[BBox] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise MedianFlow on the first frame.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth box ``(x, y, w, h)`` in pixels.
        """
        self._tracker = _create_median_flow()
        self._last_bbox = bbox
        roi = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        self._tracker.init(frame, roi)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target bounding box in the current frame.

        Returns the median-flow estimate when tracking is confident, or the
        last valid bounding box when the forward-backward error exceeds the
        tracker's internal threshold (failure detection).

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` as floats.
        """
        if self._tracker is None:
            raise RuntimeError("initialize() must be called before update().")

        ok, roi = self._tracker.update(frame)
        if ok:
            bbox: BBox = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
            self._last_bbox = bbox
        else:
            # Tracking failure detected — return last valid box.
            bbox = self._last_bbox if self._last_bbox is not None else (0.0, 0.0, 1.0, 1.0)

        return bbox

    @property
    def has_failure_detection(self) -> bool:
        """Always ``True``; MedianFlow reports tracking failures explicitly."""
        return True
