"""Attribute detection for tracking sequences.

Attributes are per-frame binary flags that characterise tracking
difficulty.  They are derived entirely from ground-truth trajectories,
so they require no additional annotation files and remain consistent
across datasets.

Attributes implemented
----------------------
- **FastMotion** (FM): normalised centre displacement exceeds a threshold.
- **ScaleVariation** (SV): relative change in bounding-box area is large.
- **AspectRatioChange** (ARC): bounding-box aspect ratio changes rapidly.
- **LowResolution** (LR): bounding-box area falls below a minimum size.
- **OutOfView** (OV): bounding-box centre exits the frame boundaries.

Reference thresholds match those used in OTB-100 [Wu et al. 2015] and
are exposed as constructor parameters so users can adjust them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np


class SequenceAttribute(str, Enum):
    """Named tracking-challenge attributes."""

    FAST_MOTION = "FastMotion"
    SCALE_VARIATION = "ScaleVariation"
    ASPECT_RATIO_CHANGE = "AspectRatioChange"
    LOW_RESOLUTION = "LowResolution"
    OUT_OF_VIEW = "OutOfView"


@dataclass
class AttributeProfile:
    """Per-frame attribute presence flags for one sequence.

    Attributes:
        name:            Human-readable attribute name.
        per_frame_flags: Boolean array of shape ``(T,)`` — ``True`` on
                         frames where the attribute is active.
    """

    name: str
    per_frame_flags: np.ndarray

    @property
    def prevalence(self) -> float:
        """Fraction of frames where the attribute is active."""
        return float(self.per_frame_flags.mean()) if len(self.per_frame_flags) else 0.0

    @property
    def num_frames(self) -> int:
        """Number of frames where the attribute is active."""
        return int(self.per_frame_flags.sum())


class AttributeDetector:
    """Derive per-frame attribute flags from ground-truth bounding boxes.

    All thresholds have sensible defaults matching the OTB-100 protocol.
    Constructing with custom values allows domain-specific tuning.

    Args:
        motion_threshold:   Normalised displacement (as a fraction of the
                            bounding-box diagonal) above which a frame is
                            labelled *FastMotion*.  Default: ``0.20``
                            (≥ 20 % of the diagonal per frame).
        scale_ratio_threshold: Fractional change in bounding-box area
                            (|ΔA| / A_prev) above which *ScaleVariation*
                            is flagged.  Default: ``0.25``.
        ar_threshold:       Fractional change in aspect ratio (w/h) above
                            which *AspectRatioChange* is flagged.
                            Default: ``0.25``.
        min_bbox_area:      Bounding-box area (pixels²) below which a
                            frame is considered *LowResolution*.
                            Default: ``400`` (20 × 20 px object).
        frame_size:         Optional ``(width, height)`` of the video
                            frame.  Required to detect *OutOfView*.
    """

    def __init__(
        self,
        motion_threshold: float = 0.20,
        scale_ratio_threshold: float = 0.25,
        ar_threshold: float = 0.25,
        min_bbox_area: int = 400,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.motion_threshold = motion_threshold
        self.scale_ratio_threshold = scale_ratio_threshold
        self.ar_threshold = ar_threshold
        self.min_bbox_area = min_bbox_area
        self.frame_size = frame_size

    def detect(self, gt_boxes: np.ndarray) -> Dict[SequenceAttribute, AttributeProfile]:
        """Compute per-frame attribute flags from ground-truth boxes.

        Args:
            gt_boxes: Array of shape ``(T, 4)`` in ``(x, y, w, h)`` format.

        Returns:
            Dict mapping each :class:`SequenceAttribute` to an
            :class:`AttributeProfile` with per-frame boolean flags.
        """
        gt = np.asarray(gt_boxes, dtype=np.float64)
        T = len(gt)
        profiles: Dict[SequenceAttribute, AttributeProfile] = {}

        profiles[SequenceAttribute.FAST_MOTION] = self._fast_motion(gt, T)
        profiles[SequenceAttribute.SCALE_VARIATION] = self._scale_variation(gt, T)
        profiles[SequenceAttribute.ASPECT_RATIO_CHANGE] = self._aspect_ratio_change(gt, T)
        profiles[SequenceAttribute.LOW_RESOLUTION] = self._low_resolution(gt)
        profiles[SequenceAttribute.OUT_OF_VIEW] = self._out_of_view(gt)

        return profiles

    # ------------------------------------------------------------------
    # Private helpers — one method per attribute
    # ------------------------------------------------------------------

    def _fast_motion(self, gt: np.ndarray, T: int) -> AttributeProfile:
        if T < 2:
            flags = np.zeros(T, dtype=bool)
        else:
            cx = gt[:, 0] + gt[:, 2] / 2.0
            cy = gt[:, 1] + gt[:, 3] / 2.0
            disp = np.hypot(np.diff(cx), np.diff(cy))           # (T-1,)
            diag = np.hypot(gt[:-1, 2], gt[:-1, 3]) + 1e-6      # (T-1,)
            norm_disp = disp / diag
            flags = np.concatenate([[False], norm_disp > self.motion_threshold])
        return AttributeProfile(
            name=SequenceAttribute.FAST_MOTION.value,
            per_frame_flags=flags,
        )

    def _scale_variation(self, gt: np.ndarray, T: int) -> AttributeProfile:
        if T < 2:
            flags = np.zeros(T, dtype=bool)
        else:
            areas = gt[:, 2] * gt[:, 3]
            ratio = np.abs(np.diff(areas)) / (areas[:-1] + 1e-6)
            flags = np.concatenate([[False], ratio > self.scale_ratio_threshold])
        return AttributeProfile(
            name=SequenceAttribute.SCALE_VARIATION.value,
            per_frame_flags=flags,
        )

    def _aspect_ratio_change(self, gt: np.ndarray, T: int) -> AttributeProfile:
        if T < 2:
            flags = np.zeros(T, dtype=bool)
        else:
            ar = gt[:, 2] / (gt[:, 3] + 1e-6)
            change = np.abs(np.diff(ar)) / (ar[:-1] + 1e-6)
            flags = np.concatenate([[False], change > self.ar_threshold])
        return AttributeProfile(
            name=SequenceAttribute.ASPECT_RATIO_CHANGE.value,
            per_frame_flags=flags,
        )

    def _low_resolution(self, gt: np.ndarray) -> AttributeProfile:
        areas = gt[:, 2] * gt[:, 3]
        flags = areas < self.min_bbox_area
        return AttributeProfile(
            name=SequenceAttribute.LOW_RESOLUTION.value,
            per_frame_flags=flags,
        )

    def _out_of_view(self, gt: np.ndarray) -> AttributeProfile:
        if self.frame_size is None:
            # Cannot determine OOV without frame dimensions; return all-False.
            return AttributeProfile(
                name=SequenceAttribute.OUT_OF_VIEW.value,
                per_frame_flags=np.zeros(len(gt), dtype=bool),
            )
        W, H = self.frame_size
        cx = gt[:, 0] + gt[:, 2] / 2.0
        cy = gt[:, 1] + gt[:, 3] / 2.0
        flags = (cx < 0) | (cx > W) | (cy < 0) | (cy > H)
        return AttributeProfile(
            name=SequenceAttribute.OUT_OF_VIEW.value,
            per_frame_flags=flags,
        )
