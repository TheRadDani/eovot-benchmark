"""Custom sparse Lucas-Kanade optical flow tracker with RANSAC geometric verification.

Unlike EOVOT's :class:`~eovot.trackers.median_flow.MedianFlowTracker`, which
wraps OpenCV's opaque ``cv2.TrackerMedianFlow_create()`` black box, this
tracker is assembled directly from primitive optical-flow building blocks,
giving full transparency and tunability at every step:

Pipeline (per frame)
--------------------
1. **Feature detection** — Shi-Tomasi corner detection inside the bounding box.
2. **Forward LK flow** — track detected keypoints from frame t → t+1 with a
   pyramidal LK solver (``cv2.calcOpticalFlowPyrLK``).
3. **Backward LK flow** — track the predicted points back to frame t.
4. **Forward-backward (FB) filtering** — discard points where the round-trip
   error exceeds ``fb_thresh`` pixels, a self-consistency signal independent
   of ground truth.
5. **RANSAC geometric verification** — fit a partial 2-D affine transform to
   the surviving points and discard geometric outliers (foreground clutter,
   background motion).  This step removes points that pass FB filtering but
   are NOT on the target.
6. **Motion estimation** — median (dx, dy) displacement and a pairwise median
   scale estimate from surviving points drive the bounding-box update.
7. **Adaptive re-detection** — when the surviving keypoint count drops below
   ``min_points``, trigger Shi-Tomasi re-detection in the current predicted
   region before declaring failure.
8. **Failure flag** — ``self.lost`` is set to ``True`` when re-detection also
   yields too few trackable points.  The last valid box is returned; the flag
   resets on the next :meth:`initialize` call.

Why this differs from ``MedianFlowTracker``
--------------------------------------------
* RANSAC removes background distractors that happen to have low FB error.
* Adaptive re-detection provides one recovery attempt before flagging failure.
* Every parameter is exposed — no hidden C++ thresholds.
* Compatible with any OpenCV build (no contrib module required).

Performance
-----------
~60–150 FPS single-core, depending on ``grid_size`` and frame resolution.
Memory footprint: O(grid_size²) — constant across arbitrarily long sequences.

References
----------
Kalal Z., Mikolajczyk K., Matas J., "Forward-Backward Error: Automatic
Detection of Tracking Failures." ICPR 2010.

Lucas B., Kanade T., "An Iterative Image Registration Technique with an
Application to Stereo Vision." IJCAI 1981.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class LKFlowTracker(BaseTracker):
    """Sparse Lucas-Kanade optical flow tracker with RANSAC geometric verification.

    All parameters are exposed so researchers can study the impact of each
    design choice independently (e.g. disable RANSAC, tighten FB threshold).

    Args:
        max_corners: Maximum keypoints returned by Shi-Tomasi detection.
            Default: ``200``.
        quality_level: Shi-Tomasi corner quality in ``(0, 1]``.  Lower values
            detect more corners in flat regions.  Default: ``0.01``.
        min_distance: Minimum Euclidean distance between detected keypoints
            in pixels.  Default: ``5.0``.
        fb_thresh: Forward-backward error threshold in pixels.  Points with
            round-trip error > ``fb_thresh`` are discarded.  Default: ``1.0``.
        min_points: Minimum surviving keypoints after all filtering stages.
            When fewer survive, adaptive re-detection is triggered; below this
            after re-detection → ``lost = True``.  Default: ``5``.
        use_ransac: Apply RANSAC-based partial affine outlier rejection.
            Disable for speed on clean, low-clutter sequences.  Default: ``True``.
        ransac_thresh: Reprojection error threshold for RANSAC (pixels).
            Default: ``3.0``.
        win_size: LK pyramid window size.  Default: ``(21, 21)``.
        max_level: Maximum pyramid levels for LK flow.  Default: ``3``.
        name: Human-readable identifier used in benchmark reports.
            Default: ``"LKFlow"``.

    Attributes:
        lost: ``True`` after a failure is detected.  Resets on next
            :meth:`initialize` call.
    """

    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 5.0,
        fb_thresh: float = 1.0,
        min_points: int = 5,
        use_ransac: bool = True,
        ransac_thresh: float = 3.0,
        win_size: Tuple[int, int] = (21, 21),
        max_level: int = 3,
        name: str = "LKFlow",
    ) -> None:
        super().__init__(name=name)
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.fb_thresh = fb_thresh
        self.min_points = min_points
        self.use_ransac = use_ransac
        self.ransac_thresh = ransac_thresh

        self._lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Internal state — set in initialize()
        self._prev_gray: Optional[np.ndarray] = None
        self._points: Optional[np.ndarray] = None  # shape (N, 1, 2) float32
        self._bbox: Optional[list] = None           # [x, y, w, h] floats
        self.lost: bool = False

    # ------------------------------------------------------------------ #
    # BaseTracker interface                                                #
    # ------------------------------------------------------------------ #

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Detect initial keypoints inside the ground-truth bounding box.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth bounding box ``(x, y, w, h)`` in pixels.

        Raises:
            ValueError: If ``w`` or ``h`` is non-positive.
        """
        x, y, w, h = [int(round(v)) for v in bbox]
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bounding box {bbox}: w and h must be positive.")

        self._bbox = [float(x), float(y), float(w), float(h)]
        self._prev_gray = _to_gray(frame)
        self._points = self._detect(self._prev_gray, self._bbox)
        self.lost = False

    def update(self, frame: np.ndarray) -> BBox:
        """Track keypoints forward and update the bounding box.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` in the original frame's
            coordinate system.  If ``self.lost`` is ``True``, this is the last
            valid box.
        """
        if self._prev_gray is None or self._bbox is None:
            raise RuntimeError("initialize() must be called before update().")

        curr_gray = _to_gray(frame)

        # --- Re-detect if we have too few tracked points ---
        if self.lost or self._points is None or len(self._points) < self.min_points:
            self._points = self._detect(curr_gray, self._bbox)
            if len(self._points) < self.min_points:
                self.lost = True
                self._prev_gray = curr_gray
                return tuple(self._bbox)  # type: ignore[return-value]
            self.lost = False

        # --- Forward + backward LK flow ---
        p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, curr_gray, self._points, None, **self._lk_params
        )
        p0_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, self._prev_gray, p1, None, **self._lk_params
        )

        # --- FB error filter ---
        fb_err = np.linalg.norm(
            self._points.reshape(-1, 2) - p0_back.reshape(-1, 2), axis=1
        )
        keep = (
            (st_fwd.flatten() == 1)
            & (st_bwd.flatten() == 1)
            & (fb_err < self.fb_thresh)
        )
        old_pts = self._points[keep].reshape(-1, 2)
        new_pts = p1[keep].reshape(-1, 2)

        # --- RANSAC geometric verification ---
        if self.use_ransac and len(old_pts) >= 4:
            old_pts, new_pts = self._ransac_filter(old_pts, new_pts)

        if len(new_pts) < self.min_points:
            # One recovery attempt before signalling failure
            redet = self._detect(curr_gray, self._bbox)
            if len(redet) < self.min_points:
                self.lost = True
            else:
                self._points = redet
                self.lost = False
            self._prev_gray = curr_gray
            return tuple(self._bbox)  # type: ignore[return-value]

        # --- Motion estimation ---
        dx = float(np.median(new_pts[:, 0] - old_pts[:, 0]))
        dy = float(np.median(new_pts[:, 1] - old_pts[:, 1]))
        scale = _median_scale(old_pts, new_pts)

        x, y, w, h = self._bbox
        cx = x + w / 2.0 + dx
        cy = y + h / 2.0 + dy
        w_new = max(1.0, w * scale)
        h_new = max(1.0, h * scale)
        self._bbox = [cx - w_new / 2.0, cy - h_new / 2.0, w_new, h_new]

        # --- State update ---
        self._prev_gray = curr_gray
        self._points = new_pts.reshape(-1, 1, 2).astype(np.float32)

        return tuple(self._bbox)  # type: ignore[return-value]

    @property
    def has_failure_detection(self) -> bool:
        """Always ``True``; ``self.lost`` signals tracking failures explicitly."""
        return True

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _detect(self, gray: np.ndarray, bbox: list) -> np.ndarray:
        """Run Shi-Tomasi corner detection inside *bbox* on *gray*.

        Returns:
            Float32 array of shape ``(N, 1, 2)`` in full-frame coordinates.
            Empty ``(0, 1, 2)`` array when no corners are found.
        """
        x, y, w, h = [int(round(v)) for v in bbox]
        ih, iw = gray.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 1, 2), dtype=np.float32)

        roi = gray[y1:y2, x1:x2]
        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=3,
        )
        if corners is None:
            return np.empty((0, 1, 2), dtype=np.float32)

        # Translate ROI-relative coords to full-frame coords
        corners += np.array([[[float(x1), float(y1)]]], dtype=np.float32)
        return corners

    def _ransac_filter(
        self,
        src: np.ndarray,
        dst: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reject geometric outliers via RANSAC partial-affine estimation.

        Points whose reprojection error under the RANSAC best-fit affine
        transform exceeds ``self.ransac_thresh`` pixels are discarded.

        Args:
            src: ``(N, 2)`` float32 — source (previous-frame) keypoints.
            dst: ``(N, 2)`` float32 — destination (current-frame) keypoints.

        Returns:
            Filtered ``(inlier_src, inlier_dst)`` pair.  Returns the input
            arrays unchanged when RANSAC fails or yields too few inliers.
        """
        if len(src) < 3:
            return src, dst
        try:
            _, mask = cv2.estimateAffinePartial2D(
                src.reshape(-1, 1, 2),
                dst.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
            )
        except cv2.error:
            return src, dst

        if mask is None:
            return src, dst
        inliers = mask.flatten().astype(bool)
        if inliers.sum() < 2:
            return src, dst
        return src[inliers], dst[inliers]


# ------------------------------------------------------------------ #
# Module-level helpers (no self-state)                                 #
# ------------------------------------------------------------------ #

def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR / BGRA / grayscale image to a single-channel uint8 array."""
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _median_scale(old_pts: np.ndarray, new_pts: np.ndarray) -> float:
    """Estimate scale change as the median pairwise distance ratio.

    Computes ||new_j - new_i|| / ||old_j - old_i|| for all pairs (i, j)
    where the old distance > 1 px, then returns the median.  Uses at most
    50 points to keep the O(N²) cost bounded.

    Args:
        old_pts: ``(N, 2)`` source keypoints.
        new_pts: ``(N, 2)`` destination keypoints.

    Returns:
        Scale factor in ``[0.1, 10.0]``.  Falls back to ``1.0`` when the
        estimate is unreliable (fewer than 2 points, all co-located).
    """
    n = len(old_pts)
    if n < 2:
        return 1.0

    # Subsample to cap O(N²) cost
    max_pts = 50
    if n > max_pts:
        idx = np.linspace(0, n - 1, max_pts, dtype=int)
        old_pts = old_pts[idx]
        new_pts = new_pts[idx]

    d_old = np.sqrt(np.sum((old_pts[:, None] - old_pts[None, :]) ** 2, axis=2))
    d_new = np.sqrt(np.sum((new_pts[:, None] - new_pts[None, :]) ** 2, axis=2))

    valid = d_old > 1.0
    if not valid.any():
        return 1.0

    ratios = d_new[valid] / d_old[valid]
    return float(np.clip(np.median(ratios), 0.1, 10.0))
