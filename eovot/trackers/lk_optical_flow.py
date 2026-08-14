"""Lucas-Kanade Sparse Optical Flow tracker.

Reference:
    Lucas, B. D., & Kanade, T. (1981). An iterative image registration
    technique with an application to stereo vision. IJCAI-81, pp. 674–679.

    Bouguet, J.-Y. (2001). Pyramidal implementation of the affine Lucas
    Kanade feature tracker description of the algorithm. Intel Corporation.

Design notes
------------
* Uses Shi-Tomasi corner detection (``cv2.goodFeaturesToTrack``) to seed
  keypoints inside the initial bounding box.
* Tracks keypoints across frames with pyramidal LK optical flow
  (``cv2.calcOpticalFlowPyrLK``), which handles translational and mild
  scale/rotation changes efficiently on CPU.
* Estimates the new bounding box via ``cv2.estimateAffinePartial2D``
  (rotation + uniform scale + translation) applied to the matched keypoint
  pairs, giving implicit scale adaptation without explicit scale search.
* Automatically re-detects keypoints when fewer than ``min_points`` survive
  forward–backward consistency filtering, preventing silent tracker drift
  after occlusions or fast motion.
* No GPU or deep-learning framework required; expected throughput is
  200–600 FPS on a single modern CPU core, making it a strong edge baseline
  that complements the correlation-filter family (MOSSE, KCF).

Algorithmic family contrast
---------------------------
Unlike MOSSE/KCF which solve a ridge regression problem in the Fourier
domain on a dense image patch, LK Optical Flow operates on *sparse*
keypoints.  This makes it:

* More robust to appearance change (few keypoints survive even large
  illumination shifts).
* More sensitive to texture-poor targets (featureless regions yield few
  corners).
* Naturally scale-adaptive through the affine estimate.

Both families are useful in an edge benchmark — they stress-test
complementary failure modes.

Example::

    import cv2
    from eovot.trackers.lk_optical_flow import LKOpticalFlowTracker

    tracker = LKOpticalFlowTracker()
    cap = cv2.VideoCapture("video.mp4")
    ret, frame = cap.read()
    tracker.initialize(frame, (100, 80, 60, 60))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bbox = tracker.update(frame)
        print(bbox)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox

# LK termination criteria: stop after 30 iterations or when the corner
# moves less than 0.03 pixels — matches OpenCV's recommended defaults.
_LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)


class LKOpticalFlowTracker(BaseTracker):
    """Keypoint-based tracker using sparse Lucas-Kanade optical flow.

    Tracks a single object by maintaining a set of Shi-Tomasi keypoints
    inside the target region and propagating them across frames with
    pyramidal LK optical flow.  Scale and mild rotation changes are
    recovered from the affine transform between matched keypoint pairs.

    Args:
        max_corners: Maximum number of Shi-Tomasi corners to detect inside
            the initial bounding box.  More corners improve robustness on
            complex textures; fewer reduce per-frame cost.  Default: ``200``.
        quality_level: Minimum accepted quality (eigenvalue ratio) for
            corner detection — same as ``cv2.goodFeaturesToTrack``.
            Default: ``0.01``.
        min_distance: Minimum Euclidean distance (pixels) between detected
            corners; prevents clustering near a single strong edge.
            Default: ``3``.
        block_size: Neighbourhood size for corner quality computation.
            Default: ``7``.
        pyr_levels: Number of pyramid levels for LK; more levels allow
            larger inter-frame motion to be handled.  Default: ``3``.
        win_size: LK search window radius at the finest pyramid level.
            Default: ``(15, 15)``.
        min_points: If fewer than this many points survive after forward–
            backward consistency filtering, the tracker re-detects corners
            in the last known bounding box.  Default: ``5``.
        fb_error_threshold: Maximum forward–backward reprojection error
            (pixels) for a point to be considered reliably tracked.
            Default: ``1.0``.
        redetect_on_failure: Whether to re-detect keypoints when the
            tracked point count falls below ``min_points``.  Disabling this
            gives a "purer" optical flow measurement at the cost of
            potential tracker death after total occlusion.  Default: ``True``.
    """

    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 3.0,
        block_size: int = 7,
        pyr_levels: int = 3,
        win_size: Tuple[int, int] = (15, 15),
        min_points: int = 5,
        fb_error_threshold: float = 1.0,
        redetect_on_failure: bool = True,
    ) -> None:
        super().__init__(name="LKOpticalFlow")
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.pyr_levels = pyr_levels
        self.win_size = win_size
        self.min_points = min_points
        self.fb_error_threshold = fb_error_threshold
        self.redetect_on_failure = redetect_on_failure

        self._prev_gray: Optional[np.ndarray] = None
        self._points: Optional[np.ndarray] = None   # (N, 1, 2) float32
        self._bbox: Optional[BBox] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the tracker on the first frame.

        Converts the frame to grayscale, detects Shi-Tomasi corners
        inside *bbox*, and stores them as the initial keypoint set.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 array.
            bbox:  Initial bounding box ``(x, y, w, h)`` in pixel coordinates.

        Raises:
            ValueError: If *bbox* has non-positive width or height.
        """
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0 or h <= 0:
            raise ValueError(f"LKOpticalFlowTracker: invalid bbox {bbox}")

        self._bbox = (x, y, w, h)
        self._prev_gray = self._to_gray(frame)
        self._points = self._detect_corners(self._prev_gray, self._bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location in the next frame.

        Runs pyramidal LK optical flow on all stored keypoints, filters
        them with a forward–backward consistency check, estimates the
        affine transform between old and new points, and applies that
        transform to the current bounding box.

        Args:
            frame: Current BGR image as a ``(H, W, 3)`` uint8 array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._prev_gray is None or self._bbox is None:
            raise RuntimeError(
                "LKOpticalFlowTracker not initialised. Call initialize() first."
            )

        curr_gray = self._to_gray(frame)

        if self._points is not None and len(self._points) >= self.min_points:
            good_old, good_new = self._track_points(self._prev_gray, curr_gray, self._points)
            if len(good_old) >= self.min_points:
                new_bbox = self._estimate_bbox(good_old, good_new, self._bbox)
                self._bbox = new_bbox
                self._points = good_new.reshape(-1, 1, 2).astype(np.float32)
            elif self.redetect_on_failure:
                # Too few reliable points — re-detect inside last known bbox.
                self._points = self._detect_corners(curr_gray, self._bbox)
        elif self.redetect_on_failure:
            self._points = self._detect_corners(curr_gray, self._bbox)

        self._prev_gray = curr_gray
        return self._bbox

    def reset(self) -> None:
        """Clear all internal state so the tracker can be re-initialised."""
        self._prev_gray = None
        self._points = None
        self._bbox = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _detect_corners(self, gray: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        """Detect Shi-Tomasi corners restricted to the bounding box region.

        Uses an ROI mask so that only corners inside the target box are
        returned, even when the background is highly textured.

        Args:
            gray: Grayscale frame.
            bbox: Region of interest ``(x, y, w, h)``.

        Returns:
            ``(N, 1, 2)`` float32 array of corner positions in *full-frame*
            coordinates, or ``None`` when the bbox is out-of-bounds.
        """
        x, y, w, h = bbox
        fh, fw = gray.shape
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(fw, int(x + w))
        y2 = min(fh, int(y + h))
        if x2 <= x1 or y2 <= y1:
            return None

        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
            blockSize=self.block_size,
        )
        return pts  # None when no corners found

    def _track_points(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward–backward filtered Lucas-Kanade optical flow.

        Computes the forward flow (prev → curr) and backward flow
        (curr → prev).  A point is kept only if its forward–backward
        reprojection error is below ``fb_error_threshold``, filtering
        out unreliable estimates caused by occlusion or fast motion.

        Args:
            prev_gray: Grayscale frame at time *t-1*.
            curr_gray: Grayscale frame at time *t*.
            points:    ``(N, 1, 2)`` float32 keypoint positions in *prev*.

        Returns:
            ``(good_old, good_new)`` — matched pairs as ``(M, 2)`` float32
            arrays, where *M ≤ N* is the number of points that passed the
            consistency filter.
        """
        lk_kw = dict(
            winSize=self.win_size,
            maxLevel=self.pyr_levels,
            criteria=_LK_CRITERIA,
        )
        # Forward pass: prev → curr
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None, **lk_kw)
        # Backward pass: curr → prev
        p0r, st0r, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1, None, **lk_kw)

        if p1 is None or st1 is None or p0r is None or st0r is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        fb_error = np.abs(points - p0r).reshape(-1, 2).max(axis=1)
        mask = (st1.ravel() == 1) & (st0r.ravel() == 1) & (fb_error < self.fb_error_threshold)

        good_old = points.reshape(-1, 2)[mask]
        good_new = p1.reshape(-1, 2)[mask]
        return good_old, good_new

    @staticmethod
    def _estimate_bbox(
        old_pts: np.ndarray,
        new_pts: np.ndarray,
        old_bbox: BBox,
    ) -> BBox:
        """Estimate the new bounding box from matched keypoint pairs.

        When ≥ 3 points are available, fits a partial-affine transform
        (rotation + uniform scale + translation) and applies it to the
        four corners of *old_bbox*.  Falls back to median-translation-only
        when fewer than 3 points survive (sufficient for a 2-DOF estimate).

        Args:
            old_pts: ``(N, 2)`` float32 — source keypoint positions.
            new_pts: ``(N, 2)`` float32 — destination keypoint positions.
            old_bbox: Current ``(x, y, w, h)`` box.

        Returns:
            New ``(x, y, w, h)`` bounding box in pixel coordinates.
        """
        x, y, w, h = old_bbox

        if len(old_pts) >= 3:
            M, _ = cv2.estimateAffinePartial2D(
                old_pts.reshape(-1, 1, 2),
                new_pts.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            if M is not None:
                corners = np.array(
                    [[x, y], [x + w, y], [x, y + h], [x + w, y + h]],
                    dtype=np.float32,
                ).reshape(1, -1, 2)
                new_corners = cv2.transform(corners, M).reshape(-1, 2)
                nx1, ny1 = new_corners.min(axis=0)
                nx2, ny2 = new_corners.max(axis=0)
                nw, nh = max(1.0, float(nx2 - nx1)), max(1.0, float(ny2 - ny1))
                return (float(nx1), float(ny1), nw, nh)

        # Fallback: median translation only (no scale recovery).
        delta = np.median(new_pts - old_pts, axis=0)
        return (x + float(delta[0]), y + float(delta[1]), w, h)
