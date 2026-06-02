"""Resolution-scaling tracker wrapper for edge-deployment analysis.

ResolutionScalerTracker wraps any BaseTracker and downscales input frames
to a fraction of their original resolution before inference, then scales
bounding-box predictions back up to the original coordinate space.

Edge deployment motivation
--------------------------
Edge SoCs (Raspberry Pi, Cortex-A CPUs, mobile chipsets) are limited by
both compute throughput and memory bandwidth.  Frame resolution directly
controls both: halving each spatial dimension (scale_factor=0.5) reduces
pixel count by 4×, cutting both compute work and DRAM bandwidth roughly
proportionally.

This creates a spatial analogue to frame-skipping:

    scale_factor 1.00 → full resolution  (baseline accuracy, baseline FPS)
    scale_factor 0.75 → 56 % of pixels   (mild accuracy drop, ~1.6× FPS)
    scale_factor 0.50 → 25 % of pixels   (moderate accuracy drop, ~3× FPS)
    scale_factor 0.25 →  6 % of pixels   (significant accuracy drop, ~8× FPS)

Used together with ``scripts/run_resolution_sweep.py``, this module produces
a resolution-vs-accuracy Pareto curve for any tracker—enabling researchers
to select the Pareto-optimal operating point for their deployment target
without access to the physical device.

Coordinate-space conventions
-----------------------------
All bounding boxes follow the EOVOT ``(x, y, w, h)`` convention where
``(x, y)`` is the top-left pixel.  Scaling multiplies ``(x, y, w, h)``
by the scale factor on the way in, and divides by it on the way out, so
predictions are always returned in the original-resolution coordinate space
regardless of the scale factor used during inference.

Example::

    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.resolution_scaler import ResolutionScalerTracker

    # Run KCF on half-resolution frames
    tracker = ResolutionScalerTracker(KCFTracker(), scale_factor=0.5)

    tracker.initialize(full_res_frame, init_bbox)
    for frame in sequence:
        pred_bbox = tracker.update(frame)  # returned in full-res coordinates
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class ResolutionScalerTracker(BaseTracker):
    """Wraps a BaseTracker to operate at a reduced input resolution.

    All frames are downscaled by ``scale_factor`` before being passed to the
    inner tracker, and all bounding-box predictions are scaled back up to
    the original resolution.

    Args:
        tracker: Underlying tracker instance (any BaseTracker subclass).
        scale_factor: Spatial scale applied to frames before inference.
            Must be in the range ``(0.0, 1.0]``.  ``1.0`` is a no-op.
        interpolation: OpenCV interpolation flag used for downscaling.
            ``cv2.INTER_LINEAR`` (default) is fast and adequate for most
            tracking use cases.  Use ``cv2.INTER_AREA`` for slightly
            better quality at aggressive downscaling (scale ≤ 0.5).
    """

    def __init__(
        self,
        tracker: BaseTracker,
        scale_factor: float = 0.5,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> None:
        if not (0.0 < scale_factor <= 1.0):
            raise ValueError(
                f"scale_factor must be in (0.0, 1.0], got {scale_factor}."
            )
        super().__init__(
            name=f"ResScale({tracker.name},sf={scale_factor:.2f})"
        )
        self._tracker = tracker
        self._scale = scale_factor
        self._interp = interpolation
        self._orig_shape: Optional[Tuple[int, int]] = None  # (H, W)
        self._last_bbox: Optional[BBox] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Downscale the first frame, scale the bbox, and delegate to tracker.

        Args:
            frame: BGR image at original resolution, shape ``(H, W, 3)`` uint8.
            bbox:  Ground-truth bounding box ``(x, y, w, h)`` in original-
                   resolution pixel coordinates.
        """
        self._orig_shape = (frame.shape[0], frame.shape[1])
        self._last_bbox = bbox
        scaled_frame = self._scale_frame(frame)
        scaled_bbox = self._scale_bbox_down(bbox)
        self._tracker.initialize(scaled_frame, scaled_bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Downscale the frame, get a prediction, and scale the result back up.

        If the inner tracker raises (e.g. due to a target exiting the
        reduced-resolution frame), the last valid bounding box is returned
        so the benchmark run is not interrupted.

        Args:
            frame: BGR image at original resolution, shape ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` in original-resolution
            pixel coordinates.
        """
        scaled_frame = self._scale_frame(frame)
        try:
            scaled_pred = self._tracker.update(scaled_frame)
            bbox = self._scale_bbox_up(scaled_pred)
        except (cv2.error, Exception):
            # Fallback: repeat last known box when inner tracker cannot update
            bbox = self._last_bbox if self._last_bbox is not None else (0.0, 0.0, 1.0, 1.0)
        self._last_bbox = bbox
        return bbox

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def scale_factor(self) -> float:
        """The spatial scale factor applied to input frames."""
        return self._scale

    @property
    def pixel_reduction_factor(self) -> float:
        """Fraction of the original pixel count used during inference.

        Equal to ``scale_factor ** 2``.  A factor of 0.25 means the tracker
        sees only 25 % as many pixels as the full-resolution version.
        """
        return self._scale ** 2

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to (H * scale, W * scale) using self._interp."""
        if self._scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        new_w = max(1, int(round(w * self._scale)))
        new_h = max(1, int(round(h * self._scale)))
        return cv2.resize(frame, (new_w, new_h), interpolation=self._interp)

    def _scale_bbox_down(self, bbox: BBox) -> BBox:
        """Scale a bbox from original to downscaled coordinate space."""
        if self._scale == 1.0:
            return bbox
        x, y, w, h = bbox
        s = self._scale
        return (x * s, y * s, w * s, h * s)

    def _scale_bbox_up(self, bbox: BBox) -> BBox:
        """Scale a bbox from downscaled back to original coordinate space."""
        if self._scale == 1.0:
            return bbox
        x, y, w, h = bbox
        inv = 1.0 / self._scale
        return (x * inv, y * inv, w * inv, h * inv)
