"""Sparse Lucas-Kanade optical-flow tracker for EOVOT.

Tracks a target bounding box by propagating a sparse set of keypoints
through Lucas-Kanade (LK) optical flow between consecutive frames.
A forward-backward error check filters unreliable flow vectors; translation
and scale are estimated from the surviving keypoints using the MedianFlow
convention (median displacement and median pairwise-distance ratio).

Why this tracker belongs in EOVOT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Correlation-filter trackers (KCF, MOSSE) build an appearance model and
search a response map every frame — useful but expensive.  Optical-flow
propagation is *O(K)* in the number of keypoints *K*, making it one of
the lightest viable tracking methods available on edge hardware.  Unlike
``FrameSkipTracker`` (which skips updates entirely on non-key frames),
``OpticalFlowTracker`` computes a *flow-based prediction every frame*,
producing smooth trajectories without the position jumps that frame skip
introduces.

Algorithm (one :meth:`update` call)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Track all stored keypoints from the previous grayscale frame to the
   current frame using ``cv2.calcOpticalFlowPyrLK`` (forward pass).
2. Track the newly found points back to the previous frame (backward pass)
   and discard pairs whose round-trip error exceeds *fb_threshold* pixels.
3. Estimate bounding-box **translation** as the median displacement of the
   surviving pairs.
4. Estimate bounding-box **scale** as the median of all pairwise-distance
   ratios between the previous and current keypoint sets (MedianFlow rule).
   Scale changes are clamped to ``[0.5, 2.0]`` per frame to resist outliers.
5. Re-detect Shi-Tomasi corners inside the updated bounding box when the
   surviving keypoint count falls below *min_corners*, and also every
   *redetect_interval* frames to resist long-term keypoint drift.

Limitations
~~~~~~~~~~~
* Pure motion propagation — no appearance model.  Tracking degrades when
  the target undergoes large appearance change or is fully occluded.
* Scale estimation becomes noisy with very few surviving keypoints.
* Not suitable as a standalone long-sequence tracker; pair with a
  re-detection module for production use.

Example::

    from eovot.trackers.optical_flow import OpticalFlowTracker

    tracker = OpticalFlowTracker(name="OptFlow")
    tracker.initialize(frame0, bbox)
    for frame in subsequent_frames:
        pred = tracker.update(frame)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class OpticalFlowTracker(BaseTracker):
    """Sparse Lucas-Kanade optical-flow bounding-box tracker.

    Args:
        name: Human-readable identifier used in benchmark reports.
            Default: ``"OpticalFlow"``.
        max_corners: Maximum number of Shi-Tomasi corners to detect inside
            the target bounding box on initialization or re-detection.
            Default: ``200``.
        min_corners: Minimum surviving corner count before triggering a
            re-detection in the current bbox. Default: ``10``.
        quality_level: Shi-Tomasi corner quality level in ``(0, 1)``.
            Lower values detect weaker corners in low-texture regions.
            Default: ``0.01``.
        min_distance: Minimum pixel separation between detected corners.
            Default: ``7``.
        fb_threshold: Forward-backward error threshold in pixels.  Flow
            vectors whose round-trip error exceeds this value are discarded.
            Default: ``1.0``.
        lk_window: Side length of the Lucas-Kanade search window in pixels.
            Default: ``21``.
        max_level: Number of pyramid levels for the LK flow computation.
            Default: ``3``.
        redetect_interval: Re-detect corners every *N* frames to resist
            long-term keypoint drift.  Set to ``0`` to disable periodic
            re-detection (only re-detect when count drops below
            *min_corners*).  Default: ``20``.

    Example::

        import cv2
        from eovot.trackers.optical_flow import OpticalFlowTracker

        tracker = OpticalFlowTracker(name="OptFlow", max_corners=100)
        cap = cv2.VideoCapture("video.mp4")

        ret, frame = cap.read()
        tracker.initialize(frame, (100, 80, 60, 50))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            bbox = tracker.update(frame)  # (x, y, w, h)
    """

    def __init__(
        self,
        name: str = "OpticalFlow",
        max_corners: int = 200,
        min_corners: int = 10,
        quality_level: float = 0.01,
        min_distance: int = 7,
        fb_threshold: float = 1.0,
        lk_window: int = 21,
        max_level: int = 3,
        redetect_interval: int = 20,
    ) -> None:
        super().__init__(name=name)

        if max_corners < 1:
            raise ValueError(f"max_corners must be >= 1, got {max_corners}.")
        if min_corners < 1:
            raise ValueError(f"min_corners must be >= 1, got {min_corners}.")
        if not (0.0 < quality_level < 1.0):
            raise ValueError(
                f"quality_level must be in (0, 1), got {quality_level}."
            )
        if fb_threshold <= 0:
            raise ValueError(
                f"fb_threshold must be positive, got {fb_threshold}."
            )

        self.max_corners = max_corners
        self.min_corners = min_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.fb_threshold = fb_threshold
        self.lk_window = lk_window
        self.max_level = max_level
        self.redetect_interval = redetect_interval

        # Runtime state — populated by initialize()
        self._prev_gray: Optional[np.ndarray] = None
        self._points: Optional[np.ndarray] = None  # (N, 1, 2) float32
        self._bbox: Optional[BBox] = None
        self._frame_count: int = 0
        self._lk_params: dict = {}

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the tracker on the first frame.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._prev_gray = _to_gray(frame)
        self._bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        self._frame_count = 0
        self._lk_params = {
            "winSize": (self.lk_window, self.lk_window),
            "maxLevel": self.max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }
        self._points = self._detect_corners(self._prev_gray, self._bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target bounding box in *frame*.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.
        """
        if self._prev_gray is None or self._bbox is None:
            raise RuntimeError(
                "OpticalFlowTracker.initialize() must be called before update()."
            )

        curr_gray = _to_gray(frame)
        self._frame_count += 1

        has_points = (
            self._points is not None and len(self._points) >= self.min_corners
        )

        if has_points:
            good_prev, good_curr = self._flow_and_filter(
                self._prev_gray, curr_gray, self._points
            )
            if len(good_prev) >= self.min_corners:
                self._bbox = _update_bbox(self._bbox, good_prev, good_curr)
                self._points = good_curr[:, None, :].astype(np.float32)
            else:
                # Too few reliable points — re-detect in updated bbox region
                self._points = self._detect_corners(curr_gray, self._bbox)
        else:
            self._points = self._detect_corners(curr_gray, self._bbox)

        # Periodic re-detection to resist long-term keypoint drift
        if (
            self.redetect_interval > 0
            and self._frame_count % self.redetect_interval == 0
        ):
            self._points = self._detect_corners(curr_gray, self._bbox)

        self._prev_gray = curr_gray
        return self._bbox  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_corners(
        self, gray: np.ndarray, bbox: BBox
    ) -> np.ndarray:
        """Detect Shi-Tomasi corners inside *bbox*.

        Args:
            gray: Grayscale frame, ``(H, W)`` uint8.
            bbox: Region to search, ``(x, y, w, h)``.

        Returns:
            ``(N, 1, 2)`` float32 array of corner locations, or an empty
            ``(0, 1, 2)`` array when no corners are found.
        """
        H, W = gray.shape[:2]
        x, y, w, h = (int(v) for v in bbox)
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        mask = np.zeros_like(gray)
        mask[y: y + h, x: x + w] = 255

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=float(self.min_distance),
            blockSize=7,
            mask=mask,
        )
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return pts.astype(np.float32)

    def _flow_and_filter(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        pts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Track *pts* forward then backward, discarding unreliable vectors.

        Args:
            prev_gray: Grayscale frame at time *t*.
            curr_gray: Grayscale frame at time *t + 1*.
            pts: ``(N, 1, 2)`` float32 corner array from the previous frame.

        Returns:
            ``(good_prev, good_curr)`` — two ``(M, 2)`` float32 arrays of
            matched point coordinates after filtering.  ``M`` may be 0.
        """
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty(
                (0, 2), dtype=np.float32
            )

        # Forward flow: prev → curr
        new_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None, **self._lk_params
        )
        # Backward flow: curr → prev
        back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, new_pts, None, **self._lk_params
        )

        # Forward-backward error (Euclidean distance of the round-trip error)
        diff = pts - back_pts                          # (N, 1, 2)
        fb_err = np.sqrt(np.sum(diff ** 2, axis=2))   # (N, 1)
        fb_err = fb_err[:, 0]                          # (N,)

        good_mask = (
            (status_fwd[:, 0] == 1)
            & (status_bwd[:, 0] == 1)
            & (fb_err < self.fb_threshold)
        )

        good_prev = pts[good_mask, 0, :]         # (M, 2)
        good_curr = new_pts[good_mask, 0, :]     # (M, 2)
        return good_prev, good_curr


# ---------------------------------------------------------------------------
# Module-level helpers (stateless, reusable by subclasses or external code)
# ---------------------------------------------------------------------------

def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR or single-channel frame to uint8 grayscale."""
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _update_bbox(
    bbox: BBox,
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
) -> BBox:
    """Estimate a new bounding box from matched keypoint displacements.

    Translation is estimated as the **median displacement** of surviving
    point pairs.  Scale is estimated as the **median of all pairwise-distance
    ratios** between the current and previous point sets (MedianFlow
    convention), clamped to ``[0.5, 2.0]`` per frame to resist outliers.

    Args:
        bbox:      Current bounding box ``(x, y, w, h)``.
        prev_pts:  ``(N, 2)`` float32 keypoint coordinates in the previous frame.
        curr_pts:  ``(N, 2)`` float32 keypoint coordinates in the current frame.

    Returns:
        Updated bounding box ``(x, y, w, h)``.
    """
    # Translation: median displacement
    dx = float(np.median(curr_pts[:, 0] - prev_pts[:, 0]))
    dy = float(np.median(curr_pts[:, 1] - prev_pts[:, 1]))

    # Scale: median pairwise-distance ratio
    scale = _estimate_scale(prev_pts, curr_pts)

    x, y, w, h = bbox
    cx = x + w / 2.0 + dx
    cy = y + h / 2.0 + dy
    w_new = w * scale
    h_new = h * scale
    return (cx - w_new / 2.0, cy - h_new / 2.0, w_new, h_new)


def _estimate_scale(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    min_dist: float = 1.0,
    clamp: Tuple[float, float] = (0.5, 2.0),
) -> float:
    """Estimate the isotropic scale change between two point sets.

    Computes the median of upper-triangle pairwise-distance ratios
    (``||p_i - p_j|| / ||q_i - q_j||`` for all pairs ``i < j``), excluding
    pairs whose previous-frame distance is below *min_dist* (degenerate).

    Args:
        prev_pts: ``(N, 2)`` points in the previous frame.
        curr_pts: ``(N, 2)`` corresponding points in the current frame.
        min_dist: Minimum distance threshold (pixels) for inclusion.
        clamp:    ``(min, max)`` clamp range for the returned scale.

    Returns:
        Estimated scale factor, clamped to *clamp*.  Returns ``1.0`` when
        fewer than two points are available or all pairs are degenerate.
    """
    n = len(prev_pts)
    if n < 2:
        return 1.0

    # Vectorised pairwise distances — upper triangle only
    i_idx, j_idx = np.triu_indices(n, k=1)
    if len(i_idx) == 0:
        return 1.0

    d_prev = np.sqrt(
        np.sum((prev_pts[i_idx] - prev_pts[j_idx]) ** 2, axis=1)
    )
    d_curr = np.sqrt(
        np.sum((curr_pts[i_idx] - curr_pts[j_idx]) ** 2, axis=1)
    )

    valid = d_prev > min_dist
    if not valid.any():
        return 1.0

    ratios = d_curr[valid] / d_prev[valid]
    scale = float(np.median(ratios))
    return float(np.clip(scale, clamp[0], clamp[1]))
