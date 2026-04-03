"""CSRT tracker wrapper for EOVOT.

Channel and Spatial Reliability Tracking (CSRT) is one of the highest-accuracy
classical trackers in OpenCV's ``contrib`` module.  It extends DSST (Danelljan
et al., BMVC 2014) with:

- Per-channel spatial reliability maps that suppress background contributions.
- HOG + colour-names feature extraction for a rich but compact descriptor.
- Template update with a forgetting factor to handle slow appearance changes.

Compared to KCF, CSRT consistently achieves higher IoU on standard benchmarks
(OTB-100, VOT) at the cost of lower throughput (~30–60 FPS on a laptop CPU
versus 150–350 FPS for KCF).  For edge deployments that tolerate ~30 ms/frame
latency, CSRT offers a strong accuracy–efficiency operating point before adding
deep-learning components.

Dependency::

    pip install opencv-contrib-python   # contains cv2.TrackerCSRT_create

Reference:
    Lukežič et al., "Discriminative Correlation Filter Tracker with Channel
    and Spatial Reliability." CVPR 2017 / IJCV 2018.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import BaseTracker, BBox

# Guard against builds that do not include the contrib module.
try:
    _create_csrt = cv2.TrackerCSRT_create
except AttributeError:
    _create_csrt = None  # type: ignore[assignment]


class CSRTTracker(BaseTracker):
    """EOVOT wrapper around OpenCV's CSRT tracker.

    Uses the default CSRT parameters, which are well-tuned for general-purpose
    single-object tracking.  Custom parameters can be set via the OpenCV
    ``TrackerCSRT_Params`` structure and passed through ``tracker_params``.

    Args:
        name: Identifier used in benchmark reports. Default: ``"CSRT"``.
        tracker_params: Optional ``cv2.TrackerCSRT_Params`` instance for
            fine-grained control (e.g. reduced HOG cell size for smaller
            targets on edge hardware).

    Raises:
        ImportError: If OpenCV was installed without the contrib module
            (i.e. ``cv2.TrackerCSRT_create`` is missing).

    Example::

        from eovot.trackers.csrt import CSRTTracker

        tracker = CSRTTracker()
        tracker.initialize(frame, (x, y, w, h))
        for frame in sequence:
            bbox = tracker.update(frame)
    """

    def __init__(
        self,
        name: str = "CSRT",
        tracker_params: Optional[object] = None,
    ) -> None:
        if _create_csrt is None:
            raise ImportError(
                "cv2.TrackerCSRT_create is not available.\n"
                "Install the contrib build: pip install opencv-contrib-python"
            )
        super().__init__(name=name)
        self._params = tracker_params
        self._tracker: Optional[cv2.Tracker] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise CSRT on the first frame.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth box ``(x, y, w, h)`` in pixels.
        """
        if self._params is not None:
            self._tracker = cv2.TrackerCSRT_create(self._params)
        else:
            self._tracker = _create_csrt()

        roi = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        self._tracker.init(frame, roi)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target bounding box in the current frame.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` as floats.
            Returns the last known box clamped to image boundaries if
            tracking fails (OpenCV signals failure via ``ok=False``).
        """
        if self._tracker is None:
            raise RuntimeError("initialize() must be called before update().")

        ok, roi = self._tracker.update(frame)
        x, y, w, h = float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3])

        if not ok:
            # Tracking failed — clamp to image boundaries and return.
            H, W = frame.shape[:2]
            x = max(0.0, min(x, W - 1.0))
            y = max(0.0, min(y, H - 1.0))
            w = max(1.0, min(w, W - x))
            h = max(1.0, min(h, H - y))

        return (x, y, w, h)
