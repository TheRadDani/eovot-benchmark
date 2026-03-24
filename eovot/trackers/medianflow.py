"""MedianFlow tracker for EOVOT.

MedianFlow is a deterministic, optical-flow-based tracker with explicit
reliability estimation via forward-backward (FB) error.  It runs entirely
on CPU using OpenCV's Lucas-Kanade sparse optical flow — no deep learning,
no GPU, no pre-trained weights.

Algorithm (Kalal et al., 2010):
    1. Sample a sparse grid of feature points inside the current bbox.
    2. Track points *forward* from frame t to t+1 using LK optical flow.
    3. Track the same points *backward* from t+1 to t.
    4. Discard points whose round-trip displacement exceeds *fb_threshold*.
    5. Estimate translation as the **median** (x, y) displacement of the
       surviving points.
    6. Estimate scale change as the median ratio of inter-point distances
       between t+1 and t (NCC-based reliability can gate this step).
    7. Apply translation + scale to produce the predicted bbox.

Complexity: O(P · W · H) per frame where P is the number of tracked points
(typically 100–400) and W×H is the patch window size used by LK.

Expected throughput: ~150–600 FPS on a modern CPU (pure OpenCV C++
backend, no Python overhead in the core loop).

Reference:
    Kalal, Z., Mikolajczyk, K., & Matas, J. (2010).
    "Forward-Backward Error: Automatic Detection of Tracking Failures."
    ICPR 2010.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class MedianFlowTracker(BaseTracker):
    """Lightweight CPU tracker based on forward-backward LK optical flow.

    Args:
        grid_size: Number of feature points sampled along each axis of the
            bounding box.  Total points = ``grid_size ** 2``.  Default: 10
            (100 points).
        fb_threshold: Maximum allowed round-trip displacement (pixels) for a
            point to be considered reliable.  Default: 1.0.
        lk_window: Side length of the LK search window in pixels.
            Default: 21.
        lk_levels: Number of pyramid levels for LK.  Default: 3.

    Example::

        tracker = MedianFlowTracker()
        tracker.initialize(first_frame, (x, y, w, h))
        for frame in subsequent_frames:
            bbox = tracker.update(frame)
    """

    def __init__(
        self,
        grid_size: int = 10,
        fb_threshold: float = 1.0,
        lk_window: int = 21,
        lk_levels: int = 3,
    ) -> None:
        super().__init__(name="MedianFlow")
        self.grid_size = grid_size
        self.fb_threshold = fb_threshold
        self._lk_params = dict(
            winSize=(lk_window, lk_window),
            maxLevel=lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
        )
        # State
        self._prev_gray: Optional[np.ndarray] = None
        self._bbox: Optional[BBox] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise on the first frame.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox: Ground-truth box ``(x, y, w, h)``.
        """
        self._prev_gray = _to_gray(frame)
        self._bbox = tuple(map(float, bbox))  # type: ignore[arg-type]

    def update(self, frame: np.ndarray) -> BBox:
        """Predict target location in the next frame.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.
            If tracking fails (too few reliable points), the last known
            bbox is returned unchanged.
        """
        if self._prev_gray is None or self._bbox is None:
            raise RuntimeError("MedianFlowTracker.update() called before initialize().")

        curr_gray = _to_gray(frame)
        new_bbox = self._track(self._prev_gray, curr_gray, self._bbox)
        self._prev_gray = curr_gray
        self._bbox = new_bbox
        return new_bbox

    # ------------------------------------------------------------------
    # Private tracking logic
    # ------------------------------------------------------------------

    def _track(
        self, prev: np.ndarray, curr: np.ndarray, bbox: BBox
    ) -> BBox:
        """Core MedianFlow step: forward-backward filtering + median shift."""
        pts = _sample_grid(bbox, self.grid_size)
        if len(pts) < 4:
            return bbox

        pts0 = pts.astype(np.float32).reshape(-1, 1, 2)

        # Forward pass: prev → curr
        pts1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts0, None, **self._lk_params)
        # Backward pass: curr → prev
        pts0_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(curr, prev, pts1, None, **self._lk_params)

        if pts1 is None or pts0_back is None:
            return bbox

        # Forward-backward error
        fb_err = np.linalg.norm(pts0_back - pts0, axis=2).reshape(-1)
        valid = (st_fwd.reshape(-1) == 1) & (st_bwd.reshape(-1) == 1) & (fb_err < self.fb_threshold)

        if valid.sum() < 4:
            return bbox

        pts0_v = pts0.reshape(-1, 2)[valid]
        pts1_v = pts1.reshape(-1, 2)[valid]

        # Median translation
        dx = float(np.median(pts1_v[:, 0] - pts0_v[:, 0]))
        dy = float(np.median(pts1_v[:, 1] - pts0_v[:, 1]))

        # Median scale change (pairwise distance ratio)
        scale = _median_scale(pts0_v, pts1_v)

        x, y, w, h = bbox
        cx = x + w / 2 + dx
        cy = y + h / 2 + dy
        w2 = w * scale
        h2 = h * scale

        return (cx - w2 / 2, cy - h2 / 2, w2, h2)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to grayscale uint8."""
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _sample_grid(bbox: BBox, n: int) -> np.ndarray:
    """Sample an n×n grid of points uniformly inside *bbox*.

    Returns:
        ``(n*n, 2)`` float32 array of (x, y) coordinates.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or n < 1:
        return np.empty((0, 2), dtype=np.float32)
    xs = np.linspace(x + 0.1 * w, x + 0.9 * w, n)
    ys = np.linspace(y + 0.1 * h, y + 0.9 * h, n)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def _median_scale(pts0: np.ndarray, pts1: np.ndarray) -> float:
    """Estimate scale via median pairwise distance ratio.

    For all pairs (i, j) with i < j compute
    ``|pts1_i − pts1_j| / |pts0_i − pts0_j|`` and return the median.
    Falls back to 1.0 if fewer than 2 points are supplied.
    """
    n = len(pts0)
    if n < 2:
        return 1.0

    ratios: list = []
    for i in range(n):
        for j in range(i + 1, n):
            d0 = float(np.linalg.norm(pts0[i] - pts0[j]))
            d1 = float(np.linalg.norm(pts1[i] - pts1[j]))
            if d0 > 1e-6:
                ratios.append(d1 / d0)

    return float(np.median(ratios)) if ratios else 1.0
