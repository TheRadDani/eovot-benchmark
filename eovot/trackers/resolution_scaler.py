"""Resolution-scaled tracking wrapper for EOVOT.

:class:`ResolutionScaledTracker` wraps any :class:`~eovot.trackers.base.BaseTracker`
and downscales frames before inference, then rescales predicted bounding boxes
back to the original coordinate space.

Why this matters for edge deployment
-------------------------------------
Processing fewer pixels is the cheapest path to faster tracking.  Halving the
frame dimensions reduces pixel count by 4×, which typically yields a 2–4×
speedup for correlation-filter trackers with acceptable IoU degradation.

This wrapper enables *systematic resolution scaling experiments*:

.. code-block:: python

    from eovot.trackers.resolution_scaler import ResolutionScaledTracker
    from eovot.trackers.mosse import MOSSETracker
    from eovot.benchmark.engine import BenchmarkEngine

    engine = BenchmarkEngine()
    for scale in [1.0, 0.75, 0.5, 0.25]:
        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=scale)
        result  = engine.run(tracker, dataset, dataset_name="OTB100")
        print(result)   # compare mIoU and FPS across scale factors

The experiment reveals each tracker's accuracy–efficiency Pareto front as a
function of input resolution — a critical data point for edge deployment
decisions.

Implementation notes
--------------------
* Scaling uses ``cv2.INTER_LINEAR`` for downscaling and ``cv2.INTER_LINEAR``
  for upscaling, matching the default used by most tracking papers when
  discussing input resolution.
* When ``scale_factor == 1.0`` the wrapper is a zero-overhead pass-through.
* Bounding boxes are always returned in the *original* frame coordinate space
  so the wrapper is transparent to the benchmark engine and metric computation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class ResolutionScaledTracker(BaseTracker):
    """Wrap a tracker to run on downscaled frames.

    Args:
        tracker: Any :class:`~eovot.trackers.base.BaseTracker` instance.
        scale_factor: Fraction of the original resolution to use.  Must be
            in ``(0.0, 1.0]``.  For example ``0.5`` halves each dimension
            (quarter of the original pixel count).  ``1.0`` is a no-op.
        name: Human-readable name for benchmark reports.  Defaults to
            ``"<inner_name>@<scale_pct>%"`` (e.g. ``"MOSSE@50%"``).

    Raises:
        ValueError: If *scale_factor* is outside ``(0.0, 1.0]``.

    Example::

        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.5)
        tracker.initialize(frame, bbox)
        pred = tracker.update(next_frame)
    """

    def __init__(
        self,
        tracker: BaseTracker,
        scale_factor: float = 0.5,
        name: Optional[str] = None,
    ) -> None:
        if not (0.0 < scale_factor <= 1.0):
            raise ValueError(
                f"scale_factor must be in (0.0, 1.0], got {scale_factor}"
            )
        pct = int(round(scale_factor * 100))
        display_name = name or f"{tracker.name}@{pct}%"
        super().__init__(display_name)
        self._inner = tracker
        self.scale_factor = scale_factor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Downscale *frame* and return the scaled frame plus the actual scale factors.

        Returns:
            ``(scaled_frame, sx, sy)`` where ``sx`` and ``sy`` are the
            width and height scale factors applied (may differ slightly from
            ``scale_factor`` due to integer pixel rounding).
        """
        if self.scale_factor == 1.0:
            return frame, 1.0, 1.0

        h, w = frame.shape[:2]
        new_w = max(1, int(round(w * self.scale_factor)))
        new_h = max(1, int(round(h * self.scale_factor)))
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return scaled, new_w / w, new_h / h

    def _scale_bbox_down(self, bbox: BBox, sx: float, sy: float) -> BBox:
        """Scale *bbox* from original to scaled coordinate space."""
        x, y, w, h = bbox
        return (x * sx, y * sy, w * sx, h * sy)

    def _scale_bbox_up(self, bbox: BBox, sx: float, sy: float) -> BBox:
        """Scale *bbox* from scaled back to original coordinate space."""
        x, y, w, h = bbox
        return (x / sx, y / sy, w / sx, h / sy)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialize inner tracker on the (optionally scaled) first frame.

        Args:
            frame: First video frame in original resolution.
            bbox: Ground-truth bounding box ``(x, y, w, h)`` in original pixel space.
        """
        scaled_frame, sx, sy = self._scale_frame(frame)
        self._sx = sx
        self._sy = sy
        scaled_bbox = self._scale_bbox_down(bbox, sx, sy)
        self._inner.initialize(scaled_frame, scaled_bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Run the inner tracker on a scaled frame and return prediction in original space.

        Args:
            frame: Next video frame in original resolution.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` in original pixel coordinates.
        """
        scaled_frame, sx, sy = self._scale_frame(frame)
        scaled_pred = self._inner.update(scaled_frame)
        return self._scale_bbox_up(scaled_pred, sx, sy)
