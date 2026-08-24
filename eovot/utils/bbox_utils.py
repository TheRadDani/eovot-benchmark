"""Vectorised bounding-box primitives shared across EOVOT.

Before this module existed, bbox arithmetic was scattered across the
codebase: :mod:`eovot.metrics.accuracy` inlined its own IoU maths,
individual trackers re-implemented clipping and centre computation, and
the reporting layer had no way to sanitise boxes read back from JSON.
That duplication had already caused subtle divergences — the top-level
``iou`` helper skipped zero-area boxes while the batched
``MetricsEngine.batch_iou`` counted them as zero-IoU — and made it hard
to add new geometric metrics (GIoU, DIoU) without touching many files.

This module centralises the primitives:

* Format conversion — :func:`xywh_to_xyxy`, :func:`xyxy_to_xywh`
* Geometry — :func:`bbox_area`, :func:`bbox_center`, :func:`clip_bbox`,
  :func:`is_valid_bbox`
* Overlap metrics — :func:`batch_iou`, :func:`batch_giou`

Every function accepts either a single ``(4,)`` bbox or a batched
``(N, 4)`` array and returns the natural NumPy shape, so downstream
callers can use the same helpers whether they hold a single prediction
or an entire sequence.

The module is intentionally dependency-light (NumPy only) so it can be
imported from any package layer, including :mod:`eovot.trackers`, without
creating cycles.

Example::

    from eovot.utils import batch_giou, clip_bbox, xywh_to_xyxy

    ious = batch_giou(preds, gts)                      # (N,) float64
    clipped = clip_bbox(pred, frame_shape=(480, 640))  # keeps box on-screen
    xyxy = xywh_to_xyxy(pred)                          # for cv2.rectangle
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

# ---------------------------------------------------------------------- #
# Type aliases                                                             #
# ---------------------------------------------------------------------- #

BBox = Tuple[float, float, float, float]
"""Single bounding box ``(x, y, w, h)`` in pixel coordinates."""

BBoxLike = Union[BBox, np.ndarray]
"""Either a length-4 tuple/list or a NumPy array of shape ``(4,)`` or ``(N, 4)``."""


# ---------------------------------------------------------------------- #
# Format conversion                                                        #
# ---------------------------------------------------------------------- #


def xywh_to_xyxy(bbox: BBoxLike) -> np.ndarray:
    """Convert ``(x, y, w, h)`` boxes to ``(x1, y1, x2, y2)``.

    Args:
        bbox: A single bbox as a length-4 sequence or a ``(N, 4)`` array of
            boxes in the ``(x, y, w, h)`` convention.

    Returns:
        NumPy array with the same leading dimensions as ``bbox`` — shape
        ``(4,)`` for a single input, ``(N, 4)`` for a batch — containing
        ``(x1, y1, x2, y2)`` where ``(x1, y1)`` is the top-left corner and
        ``(x2, y2) = (x + w, y + h)`` is the bottom-right corner.
    """
    arr = np.asarray(bbox, dtype=np.float64)
    if arr.ndim == 1:
        x, y, w, h = arr
        return np.array([x, y, x + w, y + h], dtype=np.float64)
    out = arr.copy()
    out[..., 2] = arr[..., 0] + arr[..., 2]
    out[..., 3] = arr[..., 1] + arr[..., 3]
    return out


def xyxy_to_xywh(bbox: BBoxLike) -> np.ndarray:
    """Convert ``(x1, y1, x2, y2)`` boxes to ``(x, y, w, h)``.

    Args:
        bbox: A single bbox as a length-4 sequence or a ``(N, 4)`` array of
            boxes in the ``(x1, y1, x2, y2)`` convention.

    Returns:
        NumPy array matching the shape of ``bbox`` in the ``(x, y, w, h)``
        convention where ``w = x2 - x1`` and ``h = y2 - y1``.
    """
    arr = np.asarray(bbox, dtype=np.float64)
    if arr.ndim == 1:
        x1, y1, x2, y2 = arr
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)
    out = arr.copy()
    out[..., 2] = arr[..., 2] - arr[..., 0]
    out[..., 3] = arr[..., 3] - arr[..., 1]
    return out


# ---------------------------------------------------------------------- #
# Geometry                                                                 #
# ---------------------------------------------------------------------- #


def bbox_area(bbox: BBoxLike) -> Union[float, np.ndarray]:
    """Compute the area of a bounding box (or batch of boxes).

    Args:
        bbox: ``(x, y, w, h)`` box or ``(N, 4)`` batch.

    Returns:
        Float for a single box, ``(N,)`` array for a batch.
        Negative widths/heights are clamped to zero, so the returned area
        is always ``>= 0``.
    """
    arr = np.asarray(bbox, dtype=np.float64)
    w = np.maximum(arr[..., 2], 0.0)
    h = np.maximum(arr[..., 3], 0.0)
    area = w * h
    if arr.ndim == 1:
        return float(area)
    return area


def bbox_center(bbox: BBoxLike) -> np.ndarray:
    """Compute the (x, y) centre of a bounding box (or batch of boxes).

    Args:
        bbox: ``(x, y, w, h)`` box or ``(N, 4)`` batch.

    Returns:
        Shape ``(2,)`` for a single box or ``(N, 2)`` for a batch, where
        each entry is ``(cx, cy) = (x + w/2, y + h/2)``.
    """
    arr = np.asarray(bbox, dtype=np.float64)
    if arr.ndim == 1:
        return np.array(
            [arr[0] + arr[2] / 2.0, arr[1] + arr[3] / 2.0], dtype=np.float64
        )
    return np.stack(
        [arr[..., 0] + arr[..., 2] / 2.0, arr[..., 1] + arr[..., 3] / 2.0],
        axis=-1,
    )


def clip_bbox(
    bbox: BBoxLike, frame_shape: Tuple[int, int]
) -> np.ndarray:
    """Clip a bbox so it lies fully within ``frame_shape``.

    The width and height are shrunk (not shifted) so any portion of the
    box that extends past the frame edge is trimmed off.  The top-left
    corner is clipped to ``(0, 0)`` and the bottom-right to
    ``(frame_width, frame_height)``.

    Args:
        bbox:        ``(x, y, w, h)`` box or ``(N, 4)`` batch.
        frame_shape: ``(height, width)`` — same convention as
            ``numpy.ndarray.shape[:2]`` on an image, so passing
            ``frame.shape[:2]`` directly is safe.

    Returns:
        NumPy array with the same shape as ``bbox``.  Boxes that fall
        entirely outside the frame come back as ``(x_clipped, y_clipped,
        0, 0)`` — callers can detect that with :func:`is_valid_bbox`.
    """
    h_img, w_img = frame_shape[:2]
    arr = np.asarray(bbox, dtype=np.float64)

    xyxy = xywh_to_xyxy(arr)
    if xyxy.ndim == 1:
        x1, y1, x2, y2 = xyxy
        x1 = float(np.clip(x1, 0.0, w_img))
        y1 = float(np.clip(y1, 0.0, h_img))
        x2 = float(np.clip(x2, 0.0, w_img))
        y2 = float(np.clip(y2, 0.0, h_img))
        x2 = max(x2, x1)
        y2 = max(y2, y1)
        return xyxy_to_xywh(np.array([x1, y1, x2, y2], dtype=np.float64))

    xyxy[..., 0] = np.clip(xyxy[..., 0], 0.0, w_img)
    xyxy[..., 1] = np.clip(xyxy[..., 1], 0.0, h_img)
    xyxy[..., 2] = np.clip(xyxy[..., 2], 0.0, w_img)
    xyxy[..., 3] = np.clip(xyxy[..., 3], 0.0, h_img)
    xyxy[..., 2] = np.maximum(xyxy[..., 2], xyxy[..., 0])
    xyxy[..., 3] = np.maximum(xyxy[..., 3], xyxy[..., 1])
    return xyxy_to_xywh(xyxy)


def is_valid_bbox(bbox: BBoxLike) -> Union[bool, np.ndarray]:
    """Return whether each bbox has strictly positive width and height.

    Useful for filtering degenerate predictions before scoring — e.g. when
    a tracker returns ``(x, y, 0, 0)`` after losing the target and downstream
    IoU maths would otherwise produce a spurious zero.

    Args:
        bbox: ``(x, y, w, h)`` box or ``(N, 4)`` batch.

    Returns:
        ``bool`` for a single input, ``(N,)`` boolean array for a batch.
    """
    arr = np.asarray(bbox, dtype=np.float64)
    valid = (arr[..., 2] > 0) & (arr[..., 3] > 0)
    if arr.ndim == 1:
        return bool(valid)
    return valid


# ---------------------------------------------------------------------- #
# Overlap metrics                                                          #
# ---------------------------------------------------------------------- #


def batch_iou(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Vectorised per-frame Intersection-over-Union.

    Kept in :mod:`eovot.utils` so trackers and post-processing code can
    use IoU without importing from :mod:`eovot.metrics` (which pulls in a
    much larger surface).  :class:`~eovot.metrics.accuracy.MetricsEngine`
    delegates to this helper.

    Args:
        preds: ``(N, 4)`` array of predicted boxes ``(x, y, w, h)``.
        gts:   ``(N, 4)`` array of ground-truth boxes ``(x, y, w, h)``.

    Returns:
        ``(N,)`` float64 array of IoU values in ``[0, 1]``.  If either
        input is shorter than the other, the excess rows are ignored.
        Rows with any non-positive width/height contribute IoU ``0.0``.
    """
    n = min(len(preds), len(gts))
    if n == 0:
        return np.empty(0, dtype=np.float64)
    p = np.asarray(preds[:n], dtype=np.float64)
    g = np.asarray(gts[:n], dtype=np.float64)

    ix1 = np.maximum(p[:, 0], g[:, 0])
    iy1 = np.maximum(p[:, 1], g[:, 1])
    ix2 = np.minimum(p[:, 0] + p[:, 2], g[:, 0] + g[:, 2])
    iy2 = np.minimum(p[:, 1] + p[:, 3], g[:, 1] + g[:, 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = p[:, 2] * p[:, 3] + g[:, 2] * g[:, 3] - inter

    valid = (
        (p[:, 2] > 0) & (p[:, 3] > 0)
        & (g[:, 2] > 0) & (g[:, 3] > 0)
        & (union > 0)
    )
    return np.where(valid, inter / union, 0.0)


def batch_giou(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Generalized IoU per frame — IoU that stays informative when boxes disjoint.

    GIoU (Rezatofighi et al., CVPR 2019) augments IoU with a term that
    penalises the empty gap between two non-overlapping boxes, using the
    area of the smallest enclosing axis-aligned box.  Where standard IoU
    plateaus at zero for every disjoint prediction, GIoU keeps decreasing
    smoothly toward ``-1`` as the two boxes drift apart — which is exactly
    the signal a tracker needs when it has already lost the target and
    would otherwise look "equally bad" everywhere off-screen.

    Formula per row::

        GIoU = IoU − (area(C) − area(A ∪ B)) / area(C)

    where ``C`` is the axis-aligned convex hull (smallest enclosing box)
    of ``A`` (prediction) and ``B`` (ground truth).

    Args:
        preds: ``(N, 4)`` array of predicted boxes ``(x, y, w, h)``.
        gts:   ``(N, 4)`` array of ground-truth boxes ``(x, y, w, h)``.

    Returns:
        ``(N,)`` float64 array of GIoU values in ``[-1, 1]``.  If either
        input is shorter than the other, the excess rows are ignored.
        Rows with a degenerate ground-truth or predicted box (width/height
        ``<= 0``) contribute GIoU ``0.0`` — matching :func:`batch_iou`'s
        convention so mean-GIoU aggregates stay comparable.
    """
    n = min(len(preds), len(gts))
    if n == 0:
        return np.empty(0, dtype=np.float64)
    p = np.asarray(preds[:n], dtype=np.float64)
    g = np.asarray(gts[:n], dtype=np.float64)

    ix1 = np.maximum(p[:, 0], g[:, 0])
    iy1 = np.maximum(p[:, 1], g[:, 1])
    ix2 = np.minimum(p[:, 0] + p[:, 2], g[:, 0] + g[:, 2])
    iy2 = np.minimum(p[:, 1] + p[:, 3], g[:, 1] + g[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    area_p = p[:, 2] * p[:, 3]
    area_g = g[:, 2] * g[:, 3]
    union = area_p + area_g - inter

    cx1 = np.minimum(p[:, 0], g[:, 0])
    cy1 = np.minimum(p[:, 1], g[:, 1])
    cx2 = np.maximum(p[:, 0] + p[:, 2], g[:, 0] + g[:, 2])
    cy2 = np.maximum(p[:, 1] + p[:, 3], g[:, 1] + g[:, 3])
    hull = np.maximum(0.0, cx2 - cx1) * np.maximum(0.0, cy2 - cy1)

    valid = (
        (p[:, 2] > 0) & (p[:, 3] > 0)
        & (g[:, 2] > 0) & (g[:, 3] > 0)
        & (union > 0) & (hull > 0)
    )
    iou = np.where(valid, inter / union, 0.0)
    hull_penalty = np.where(valid, (hull - union) / hull, 0.0)
    return iou - hull_penalty
