"""Sequence-level challenge attribute analysis for EOVOT.

Computes per-sequence difficulty attributes from ground-truth bounding box
trajectories, following the OTB / VOT convention for challenge categorization.

Attributes enable *challenge-conditioned* evaluation: instead of a single
aggregate mIoU across all sequences, researchers can ask "how does tracker X
perform on fast-motion sequences specifically?" — a critical question for
edge deployment where failure modes vary by scene type.

Supported attributes
--------------------
- **fast_motion** — frame-to-frame displacement exceeds a pixel threshold.
- **scale_variation** — bounding box area changes by more than a relative ratio.
- **low_resolution** — target occupies fewer pixels than a minimum area.
- **aspect_ratio_change** — width/height ratio shifts significantly over time.
- **out_of_view** — target centre or box approaches / crosses frame boundaries.
- **partial_occlusion** — inferred from predicted IoU drops with subsequent
  recovery (requires tracker predictions; optional).

Usage::

    from eovot.datasets.attributes import SequenceAttributeAnalyzer
    import numpy as np

    gt = np.array([(10, 10, 40, 40), (30, 10, 40, 40), ...])  # (N, 4) xywh
    analyzer = SequenceAttributeAnalyzer()
    attrs = analyzer.analyze(gt, frame_size=(320, 240))
    print(attrs.active_attributes())   # e.g. ['fast_motion']
    print(attrs.fast_motion)           # True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SequenceAttributes:
    """Challenge attributes detected in a tracking sequence.

    Each boolean flag is ``True`` when the corresponding challenge is present
    for at least one frame in the sequence.  Quantitative descriptors give
    more detail for downstream analysis.
    """

    fast_motion: bool = False
    """True when any frame-to-frame displacement exceeds the pixel threshold."""

    scale_variation: bool = False
    """True when relative bounding box area change exceeds the ratio threshold."""

    low_resolution: bool = False
    """True when the minimum target area drops below the area threshold (px²)."""

    aspect_ratio_change: bool = False
    """True when relative width/height ratio change exceeds the tolerance."""

    out_of_view: bool = False
    """True when the target approaches or crosses the frame boundary."""

    partial_occlusion: bool = False
    """True when IoU drops and later recovers (requires predicted_boxes)."""

    # Quantitative descriptors
    mean_displacement_px: float = 0.0
    """Mean frame-to-frame centre displacement across the sequence (pixels)."""

    max_displacement_px: float = 0.0
    """Maximum single-frame displacement observed."""

    max_scale_ratio: float = 1.0
    """Ratio of largest to smallest bounding box area in the sequence."""

    min_bbox_area_px2: float = 0.0
    """Minimum bounding box area observed (pixels²)."""

    out_of_view_frame_count: int = 0
    """Number of frames where the target is near or outside the frame boundary."""

    def active_attributes(self) -> List[str]:
        """Return names of all attributes that are ``True``."""
        names = [
            "fast_motion",
            "scale_variation",
            "low_resolution",
            "aspect_ratio_change",
            "out_of_view",
            "partial_occlusion",
        ]
        return [n for n in names if getattr(self, n)]

    def attribute_vector(self) -> Dict[str, bool]:
        """Return a dict mapping each attribute name to its boolean value."""
        return {
            "fast_motion": self.fast_motion,
            "scale_variation": self.scale_variation,
            "low_resolution": self.low_resolution,
            "aspect_ratio_change": self.aspect_ratio_change,
            "out_of_view": self.out_of_view,
            "partial_occlusion": self.partial_occlusion,
        }

    def to_dict(self) -> Dict:
        """Serialise to a plain dict (JSON-compatible)."""
        return {
            **self.attribute_vector(),
            "mean_displacement_px": round(self.mean_displacement_px, 3),
            "max_displacement_px": round(self.max_displacement_px, 3),
            "max_scale_ratio": round(self.max_scale_ratio, 4),
            "min_bbox_area_px2": round(self.min_bbox_area_px2, 1),
            "out_of_view_frame_count": self.out_of_view_frame_count,
        }

    def __repr__(self) -> str:
        active = self.active_attributes()
        tag = ", ".join(active) if active else "none"
        return f"SequenceAttributes([{tag}])"


class SequenceAttributeAnalyzer:
    """Compute challenge attributes from a ground-truth bounding box trajectory.

    All detection thresholds have sensible defaults matching the OTB / VOT
    literature and can be overridden for domain-specific benchmarks.

    Args:
        fast_motion_px: Per-frame centre displacement threshold (pixels).
            Default: ``20.0``.
        scale_var_ratio: Relative frame-to-frame area change threshold.
            Default: ``0.25`` (25 % change triggers the attribute).
        low_res_area_px2: Bounding box area threshold (pixels²).
            Default: ``1000.0`` (~32 × 32 px target).
        aspect_ratio_tol: Relative aspect ratio change threshold.
            Default: ``0.25``.
        oov_margin: Fraction of the frame dimension defining the
            out-of-view border zone.  Default: ``0.05`` (5 % of W or H).
        occlusion_low_iou: IoU threshold below which a drop is detected.
            Default: ``0.3``.
        occlusion_recovery_iou: IoU above which recovery is confirmed.
            Default: ``0.5``.
    """

    def __init__(
        self,
        fast_motion_px: float = 20.0,
        scale_var_ratio: float = 0.25,
        low_res_area_px2: float = 1000.0,
        aspect_ratio_tol: float = 0.25,
        oov_margin: float = 0.05,
        occlusion_low_iou: float = 0.3,
        occlusion_recovery_iou: float = 0.5,
    ) -> None:
        self.fast_motion_px = fast_motion_px
        self.scale_var_ratio = scale_var_ratio
        self.low_res_area_px2 = low_res_area_px2
        self.aspect_ratio_tol = aspect_ratio_tol
        self.oov_margin = oov_margin
        self.occlusion_low_iou = occlusion_low_iou
        self.occlusion_recovery_iou = occlusion_recovery_iou

    def analyze(
        self,
        boxes: np.ndarray,
        frame_size: Optional[Tuple[int, int]] = None,
        predicted_boxes: Optional[np.ndarray] = None,
    ) -> SequenceAttributes:
        """Analyse a ground-truth trajectory for challenge attributes.

        Args:
            boxes: Ground-truth boxes, shape ``(N, 4)`` in ``(x, y, w, h)``
                pixel coordinates.
            frame_size: ``(width, height)`` of the video frames, used for
                out-of-view detection.  Pass ``None`` to skip that check.
            predicted_boxes: Optional ``(N, 4)`` tracker predictions used
                to infer partial occlusion via IoU drops.

        Returns:
            :class:`SequenceAttributes` with all detected challenges.

        Raises:
            ValueError: If ``boxes`` is not a 2-D array with 4 columns.
        """
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                f"boxes must have shape (N, 4), got {boxes.shape}"
            )

        attr = SequenceAttributes()
        n = len(boxes)

        if n == 0:
            return attr

        # Centre coordinates for displacement computation.
        cx = boxes[:, 0] + boxes[:, 2] / 2.0
        cy = boxes[:, 1] + boxes[:, 3] / 2.0

        # ---- Fast motion ------------------------------------------------
        if n >= 2:
            displacements = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
            attr.mean_displacement_px = float(displacements.mean())
            attr.max_displacement_px = float(displacements.max())
            attr.fast_motion = bool(np.any(displacements > self.fast_motion_px))

        # ---- Scale variation --------------------------------------------
        areas = boxes[:, 2] * boxes[:, 3]
        areas = np.maximum(areas, 1.0)  # guard against degenerate boxes
        attr.min_bbox_area_px2 = float(areas.min())
        attr.max_scale_ratio = float(areas.max() / areas.min())

        if n >= 2:
            relative_changes = np.abs(np.diff(areas)) / areas[:-1]
            attr.scale_variation = bool(np.any(relative_changes > self.scale_var_ratio))

        # ---- Low resolution --------------------------------------------
        attr.low_resolution = bool(attr.min_bbox_area_px2 < self.low_res_area_px2)

        # ---- Aspect ratio change ----------------------------------------
        if n >= 2:
            widths = np.maximum(boxes[:, 2], 1.0)
            heights = np.maximum(boxes[:, 3], 1.0)
            ars = widths / heights
            ar_changes = np.abs(np.diff(ars)) / np.maximum(ars[:-1], 1e-9)
            attr.aspect_ratio_change = bool(np.any(ar_changes > self.aspect_ratio_tol))

        # ---- Out of view ------------------------------------------------
        if frame_size is not None:
            W, H = float(frame_size[0]), float(frame_size[1])
            margin_x = W * self.oov_margin
            margin_y = H * self.oov_margin
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = x1 + boxes[:, 2]
            y2 = y1 + boxes[:, 3]
            oov_mask = (
                (cx < margin_x)
                | (cx > W - margin_x)
                | (cy < margin_y)
                | (cy > H - margin_y)
                | (x2 < 0)
                | (y2 < 0)
                | (x1 > W)
                | (y1 > H)
            )
            attr.out_of_view_frame_count = int(oov_mask.sum())
            attr.out_of_view = bool(attr.out_of_view_frame_count > 0)

        # ---- Partial occlusion (from predictions) -----------------------
        if predicted_boxes is not None and len(predicted_boxes) >= 2:
            pred = np.asarray(predicted_boxes, dtype=np.float64)
            ious = self._batch_iou(boxes, pred)
            attr.partial_occlusion = self._detect_occlusion(ious)

        return attr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _batch_iou(self, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Vectorised IoU between aligned (x, y, w, h) box pairs."""
        n = min(len(gt), len(pred))
        gt, pred = gt[:n], pred[:n]

        gx1, gy1 = gt[:, 0], gt[:, 1]
        gx2 = gx1 + gt[:, 2]
        gy2 = gy1 + gt[:, 3]
        px1, py1 = pred[:, 0], pred[:, 1]
        px2 = px1 + pred[:, 2]
        py2 = py1 + pred[:, 3]

        ix1 = np.maximum(gx1, px1)
        iy1 = np.maximum(gy1, py1)
        ix2 = np.minimum(gx2, px2)
        iy2 = np.minimum(gy2, py2)

        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = gt[:, 2] * gt[:, 3] + pred[:, 2] * pred[:, 3] - inter
        return inter / np.maximum(union, 1e-9)

    def _detect_occlusion(self, ious: np.ndarray) -> bool:
        """True when IoU drops below low threshold and later recovers above high."""
        low = ious < self.occlusion_low_iou
        high = ious > self.occlusion_recovery_iou

        if not np.any(low):
            return False

        first_drop = int(np.argmax(low))
        # Recovery must happen after the first drop.
        return bool(np.any(high[first_drop:]))
