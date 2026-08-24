"""Shared low-level utilities used across the EOVOT stack.

Modules in this package are dependency-light (NumPy only) and free of
downstream imports from :mod:`eovot.benchmark`, :mod:`eovot.trackers`,
:mod:`eovot.metrics`, or :mod:`eovot.reporting` — so any of those packages
may import from here without creating cycles.
"""

from .bbox_utils import (
    clip_bbox,
    bbox_area,
    xywh_to_xyxy,
    xyxy_to_xywh,
    bbox_center,
    batch_iou,
    batch_giou,
    is_valid_bbox,
)

__all__ = [
    "clip_bbox",
    "bbox_area",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "bbox_center",
    "batch_iou",
    "batch_giou",
    "is_valid_bbox",
]
