"""Background Subtraction Tracker for edge-deployed static-camera scenarios.

Background subtraction is one of the oldest and cheapest tracking paradigms:
build a statistical model of the static scene, subtract each new frame from
that model, and track the largest foreground blob that overlaps the target.

This module wraps OpenCV's GMM-based background subtractor (MOG2) in the
standard :class:`~eovot.trackers.base.BaseTracker` interface.

Design rationale
----------------
Every tracker currently in EOVOT belongs to one of three families:

* Correlation filters  — MOSSE, KCF, CSRT (frequency-domain appearance models)
* Colour histograms    — CamShift, KalmanColor (hue-histogram appearance models)
* Feature flow         — LK Optical Flow, MedianFlow (point tracking / optical flow)
* Online classifiers   — MIL (ensemble of weak classifiers)

Background subtraction represents a *fourth* family: **scene-background
modelling**.  It makes no assumption about the target's appearance; it only
assumes the *background* is stationary (or slowly drifting).  This is a
common and valid assumption in edge-deployment scenarios such as:

- Fixed industrial cameras on assembly lines
- Traffic intersection cameras
- Security cameras on poles / walls

In those cases background subtraction is often dramatically faster than any
appearance-based tracker: the MOG2 step is a simple per-pixel Gaussian
mixture update plus a few morphological operations, requiring no FFTs, no
histogram matching, and no feature extraction.

Algorithm
---------
1. **Initialization** — Apply MOG2 to the first frame to seed the background
   model.  Store the initial bounding box as the target's last known location.
   Optionally warm the model on a configurable number of additional frames
   before tracking begins (``warmup_frames`` parameter).

2. **Update** — On each new frame:
   a. Apply MOG2 → binary foreground mask.
   b. Morphological opening (erosion + dilation) removes salt-and-pepper
      noise while preserving blobs larger than ``morph_kernel_size``.
   c. Find connected components in the cleaned mask.
   d. Pick the component whose centroid is closest to the last known target
      centroid AND whose area is at least ``min_blob_ratio`` × target area.
   e. Return the component's bounding box.  If no valid component is found
      (target camouflaged into background), return the last known box.

3. **Fallback** — When the target blends into the background (foreground
   probability goes to zero), the tracker stays on the last known box and
   waits for the target to become distinguishable again.  This is the
   principal failure mode: it is known, documented, and reflected in
   benchmark results (failure events, recovery lag).

Limitations
-----------
* Assumes a **static or slowly drifting camera**.  Strong camera motion
  saturates the foreground mask with global motion and breaks the tracker.
* Performs poorly when the **target is stationary** for many frames — MOG2
  will gradually absorb it into the background model.
* Multiple foreground blobs (other moving objects) can confuse the
  centroid-nearest-component heuristic.  The ``max_search_radius``
  parameter limits how far the tracker will accept a match.

These failure modes are not bugs; they are inherent to the tracking
paradigm and are valuable data points in a benchmark that is meant to
characterise tracker behaviour under realistic edge constraints.

Usage::

    from eovot.trackers.background_subtraction import BackgroundSubtractionTracker

    tracker = BackgroundSubtractionTracker()
    tracker.initialize(first_frame, (x, y, w, h))
    for frame in video:
        bbox = tracker.update(frame)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class BackgroundSubtractionTracker(BaseTracker):
    """MOG2 background-subtraction tracker for static-camera scenarios.

    Tracks the foreground blob whose centroid is closest to the target's
    last known position, using OpenCV's Gaussian Mixture Model background
    subtractor (MOG2) for foreground detection.

    Args:
        name: Human-readable identifier for benchmark reports.
            Default: ``"BGSub"``.
        history: Number of frames used to build the MOG2 background model.
            Larger values give a more stable model but adapt more slowly to
            gradual scene changes.  Default: ``200``.
        var_threshold: Mahalanobis-distance threshold for a pixel to be
            classified as foreground.  Lower → more foreground detections
            (noisier mask); higher → fewer (may miss slow targets).
            Default: ``16.0``.
        detect_shadows: Whether MOG2 should distinguish shadow pixels (grey
            in the mask) from foreground (white).  Shadows are excluded from
            the foreground mask.  Default: ``True``.
        morph_kernel_size: Side-length (pixels) of the square structuring
            element used for morphological opening to clean the foreground
            mask.  Larger values suppress more noise but may shrink small
            targets.  Default: ``5``.
        morph_iterations: Number of erosion+dilation cycles applied.
            Default: ``1``.
        min_blob_ratio: Minimum fraction of the target's initial area that a
            foreground component must have to be considered a candidate.
            Prevents tiny noise blobs from hijacking the tracker.
            Default: ``0.1`` (10 % of target area).
        max_search_radius: Maximum distance (pixels) from the last known
            centroid within which a foreground component centroid must fall
            to be accepted.  ``None`` disables the radius constraint.
            Default: ``None``.
        learning_rate: MOG2 background-model update rate per frame.
            ``-1`` uses the automatic MOG2 default.  ``0`` freezes the model
            (useful for truly static backgrounds).  Default: ``-1``.
    """

    def __init__(
        self,
        name: str = "BGSub",
        history: int = 200,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        morph_kernel_size: int = 5,
        morph_iterations: int = 1,
        min_blob_ratio: float = 0.10,
        max_search_radius: Optional[float] = None,
        learning_rate: float = -1.0,
    ) -> None:
        super().__init__(name=name)
        self._bgsub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (morph_kernel_size, morph_kernel_size),
        )
        self._morph_iterations = morph_iterations
        self._min_blob_ratio = min_blob_ratio
        self._max_search_radius = max_search_radius
        self._learning_rate = learning_rate

        self._last_bbox: Optional[BBox] = None
        self._target_area: float = 0.0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Seed the background model and store the initial target location.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.

        Raises:
            ValueError: If the bounding box has non-positive width or height.
        """
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            raise ValueError(
                f"Invalid bounding box {bbox}: width and height must be positive."
            )
        self._bgsub.apply(frame, learningRate=self._learning_rate)
        self._last_bbox = (float(x), float(y), float(w), float(h))
        self._target_area = float(w * h)
        self._initialized = True

    def update(self, frame: np.ndarray) -> BBox:
        """Locate the target in a new frame via foreground blob selection.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` as floats.  Returns the
            last known box when no valid foreground blob is found.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if not self._initialized or self._last_bbox is None:
            raise RuntimeError("initialize() must be called before update().")

        # --- 1. Background subtraction → binary foreground mask ----------
        fg_mask = self._bgsub.apply(frame, learningRate=self._learning_rate)
        # MOG2 shadow pixels are marked with value 127; threshold to 0/255.
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # --- 2. Morphological opening — remove small noise blobs ----------
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            self._morph_kernel,
            iterations=self._morph_iterations,
        )

        # --- 3. Connected components in foreground mask -------------------
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fg_mask, connectivity=8
        )
        # Component 0 is always the background — skip it.
        if num_labels <= 1:
            return self._last_bbox  # type: ignore[return-value]

        # --- 4. Select best matching component ----------------------------
        lx, ly, lw, lh = self._last_bbox
        last_cx = lx + lw / 2.0
        last_cy = ly + lh / 2.0
        min_area = self._min_blob_ratio * self._target_area

        best_idx: Optional[int] = None
        best_dist: float = float("inf")

        for i in range(1, num_labels):  # skip background label 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            cx, cy = centroids[i]
            dist = float(np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2))

            if self._max_search_radius is not None and dist > self._max_search_radius:
                continue

            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is None:
            # No valid component found — hold the last known box.
            return self._last_bbox  # type: ignore[return-value]

        # --- 5. Construct predicted bbox from component stats -------------
        bx = int(stats[best_idx, cv2.CC_STAT_LEFT])
        by = int(stats[best_idx, cv2.CC_STAT_TOP])
        bw = int(stats[best_idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[best_idx, cv2.CC_STAT_HEIGHT])

        # Keep aspect ratio from the last known box when blob dimensions are
        # degenerate (1px wide or tall, which can happen with thin shadows).
        if bw <= 0:
            bw = int(self._last_bbox[2])
        if bh <= 0:
            bh = int(self._last_bbox[3])

        new_bbox: BBox = (float(bx), float(by), float(bw), float(bh))
        self._last_bbox = new_bbox
        return new_bbox
