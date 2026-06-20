"""Adaptive-resolution meta-tracker for edge-aware inference.

:class:`AdaptiveResolutionTracker` wraps any EOVOT-compatible tracker and
dynamically downscales input frames based on an estimated tracking
confidence.  When confidence is high the tracker processes a smaller
(cheaper) frame; when confidence drops it falls back to full resolution.

Motivation
----------
On resource-constrained edge devices (Raspberry Pi 4, Jetson Nano) the
primary inference cost is proportional to the number of pixels processed.
For well-tracked, smoothly-moving targets this full-resolution processing
is wasteful: a scaled-down frame carries enough information for accurate
localisation.  For difficult frames (occlusion, fast motion, appearance
change) full resolution is critical for recovery.

This module provides a principled, tunable solution by decoupling the
*scale policy* from the underlying *tracking algorithm*, so any tracker
in the EOVOT registry benefits automatically.

Confidence signal
-----------------
The confidence proxy is the IoU between the current and previous
predictions, weighted by target-area stability::

    conf_t = clip(IoU(pred_{t-1}, pred_t) * size_factor, 0, 1)

This requires **no ground truth at runtime** and works for any tracker
that produces bounding-box predictions.

Scale schedule
--------------
Three levels driven by two thresholds on the smoothed confidence::

    conf >= conf_high_thresh  ->  scale_high  (smallest frame, fastest)
    conf_low_thresh < conf < conf_high_thresh  ->  scale_medium
    conf <= conf_low_thresh   ->  scale_low   (full resolution, most accurate)

A *hysteresis* guard prevents rapid oscillation between levels.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive import AdaptiveResolutionTracker

    base    = MOSSETracker()
    tracker = AdaptiveResolutionTracker(base, scale_low=1.0,
                                        scale_medium=0.75, scale_high=0.5)
    tracker.initialize(frame, bbox)
    for frame in sequence:
        pred = tracker.update(frame)
        print(tracker.current_scale)  # observe scale adaptation
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class AdaptiveResolutionTracker(BaseTracker):
    """Meta-tracker that adapts input resolution to estimated tracking confidence.

    Wraps any :class:`~eovot.trackers.base.BaseTracker` and downscales frames
    when tracking confidence is high, recovering full resolution automatically
    when the tracker encounters a difficult frame.  This reduces per-frame
    compute on edge devices without sacrificing accuracy on easy frames.

    Args:
        base_tracker:      The underlying :class:`~eovot.trackers.base.BaseTracker`
                           to wrap.  Must implement ``initialize`` and ``update``.
        scale_low:         Scale factor applied when confidence is *low* (hard
                           frames, full or near-full resolution).
                           Must be in ``(0, 1]``.  Default: ``1.0``.
        scale_medium:      Scale applied at medium confidence.
                           Must satisfy ``scale_high <= scale_medium <= scale_low``.
                           Default: ``0.75``.
        scale_high:        Scale applied when confidence is *high* (easy frames,
                           smallest / cheapest frame).
                           Must be in ``(0, scale_medium]``.  Default: ``0.5``.
        conf_high_thresh:  Smoothed confidence value above which the *high*
                           (small-frame) scale is activated.  Default: ``0.65``.
        conf_low_thresh:   Smoothed confidence below which the *low*
                           (full-resolution) scale is activated.  Default: ``0.35``.
        hysteresis_frames: Number of consecutive frames a confidence level
                           must persist before the scale changes.  Prevents
                           rapid oscillation.  Default: ``3``.
        history_window:    Rolling window size for smoothing the raw confidence
                           signal.  Default: ``5``.

    Raises:
        ValueError: If scale or threshold parameters violate their constraints.

    Note:
        All predictions are returned in the **original frame's coordinate
        space** regardless of the active scale factor.
    """

    def __init__(
        self,
        base_tracker: BaseTracker,
        scale_low: float = 1.0,
        scale_medium: float = 0.75,
        scale_high: float = 0.5,
        conf_high_thresh: float = 0.65,
        conf_low_thresh: float = 0.35,
        hysteresis_frames: int = 3,
        history_window: int = 5,
    ) -> None:
        super().__init__(name=f"Adaptive({base_tracker.name})")

        if not (0.0 < scale_high <= scale_medium <= scale_low <= 1.0):
            raise ValueError(
                "Scales must satisfy 0 < scale_high <= scale_medium <= scale_low <= 1. "
                f"Got scale_high={scale_high}, scale_medium={scale_medium}, "
                f"scale_low={scale_low}"
            )
        if not (0.0 <= conf_low_thresh < conf_high_thresh <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0 <= conf_low_thresh < conf_high_thresh <= 1. "
                f"Got conf_low_thresh={conf_low_thresh}, "
                f"conf_high_thresh={conf_high_thresh}"
            )
        if hysteresis_frames < 1:
            raise ValueError(
                f"hysteresis_frames must be >= 1, got {hysteresis_frames}"
            )
        if history_window < 1:
            raise ValueError(
                f"history_window must be >= 1, got {history_window}"
            )

        self._base = base_tracker
        self._scale_low = scale_low
        self._scale_medium = scale_medium
        self._scale_high = scale_high
        self._conf_high = conf_high_thresh
        self._conf_low = conf_low_thresh
        self._hysteresis = hysteresis_frames
        self._history_window = history_window

        # Runtime state (reset on each initialize() call)
        self._current_scale: float = scale_low
        self._prev_bbox: Optional[BBox] = None
        self._target_area: float = 1.0
        self._conf_history: List[float] = []
        self._frames_at_level: int = 0
        self._current_level: int = 0  # 0=low, 1=medium, 2=high

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the wrapped tracker on the first frame.

        Resets all adaptive state so the tracker starts fresh on each
        new sequence.

        Args:
            frame: Full-resolution BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._prev_bbox = bbox
        x, y, w, h = bbox
        self._target_area = max(float(w * h), 1.0)
        self._conf_history.clear()
        self._frames_at_level = 0
        self._current_level = 0
        self._current_scale = self._scale_low

        scaled_frame, scaled_bbox = self._scale_input(frame, bbox, self._current_scale)
        self._base.initialize(scaled_frame, scaled_bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict target location with adaptive-resolution inference.

        Downscales *frame* according to the current confidence level,
        runs the base tracker, maps the prediction back to the original
        coordinate space, and updates the confidence estimate for the
        next frame.

        Args:
            frame: Full-resolution BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` in original image space.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._prev_bbox is None:
            raise RuntimeError(
                "AdaptiveResolutionTracker.update() called before initialize()."
            )
        h_orig, w_orig = frame.shape[:2]
        scale = self._current_scale

        scaled_frame = self._resize_frame(frame, scale)
        raw_pred = self._base.update(scaled_frame)

        pred = self._unscale_bbox(raw_pred, scale)
        pred = self._clamp_bbox(pred, w_orig, h_orig)

        conf = self._estimate_confidence(pred)
        self._update_scale(conf)
        self._prev_bbox = pred

        return pred

    # ------------------------------------------------------------------
    # Public introspection helpers
    # ------------------------------------------------------------------

    @property
    def current_scale(self) -> float:
        """Active scale factor that will be applied to the *next* frame."""
        return self._current_scale

    @property
    def confidence_history(self) -> List[float]:
        """Copy of the rolling confidence signal (up to ``history_window`` values)."""
        return list(self._conf_history)

    @property
    def base_tracker(self) -> BaseTracker:
        """The underlying wrapped tracker instance."""
        return self._base

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scale_input(
        self, frame: np.ndarray, bbox: BBox, scale: float
    ) -> Tuple[np.ndarray, BBox]:
        """Return ``(scaled_frame, scaled_bbox)`` for initialisation."""
        if scale == 1.0:
            return frame, bbox
        scaled_frame = self._resize_frame(frame, scale)
        x, y, w, h = bbox
        return scaled_frame, (x * scale, y * scale, w * scale, h * scale)

    @staticmethod
    def _resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
        """Return a bilinearly-resized copy of *frame* (no-op when scale == 1)."""
        if scale == 1.0:
            return frame
        fh, fw = frame.shape[:2]
        new_w = max(1, int(round(fw * scale)))
        new_h = max(1, int(round(fh * scale)))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _unscale_bbox(bbox: BBox, scale: float) -> BBox:
        """Map *bbox* from scaled space back to original image coordinates."""
        if scale == 1.0:
            return bbox
        inv = 1.0 / scale
        x, y, w, h = bbox
        return (x * inv, y * inv, w * inv, h * inv)

    @staticmethod
    def _clamp_bbox(bbox: BBox, frame_w: int, frame_h: int) -> BBox:
        """Clamp *bbox* so it lies within ``[0, frame_w) x [0, frame_h)``."""
        x, y, w, h = bbox
        x = float(max(0.0, min(x, frame_w - 1)))
        y = float(max(0.0, min(y, frame_h - 1)))
        w = float(max(1.0, min(w, frame_w - x)))
        h = float(max(1.0, min(h, frame_h - y)))
        return (x, y, w, h)

    def _estimate_confidence(self, current_bbox: BBox) -> float:
        """Estimate tracking confidence in ``[0, 1]`` from prediction smoothness.

        Computes the IoU between the current and previous predictions,
        weighted by a size-stability factor.  A smoothly-moving, stable
        target yields values near 1; a target that jumps or changes size
        abruptly yields values near 0.

        Args:
            current_bbox: Prediction for the current frame in original coords.

        Returns:
            Smoothed confidence estimate in ``[0, 1]``.
        """
        if self._prev_bbox is None:
            return 1.0

        px, py, pw, ph = self._prev_bbox
        cx, cy, cw, ch = current_bbox

        # IoU between consecutive predictions
        ix1 = max(px, cx)
        iy1 = max(py, cy)
        ix2 = min(px + pw, cx + cw)
        iy2 = min(py + ph, cy + ch)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = pw * ph + cw * ch - inter
        iou_consec = float(inter / union) if union > 0 else 0.0

        # Down-weight confidence for targets that changed size drastically
        size_factor = float(np.clip(cw * ch / self._target_area, 0.5, 2.0))
        raw_conf = float(np.clip(iou_consec * size_factor, 0.0, 1.0))

        self._conf_history.append(raw_conf)
        if len(self._conf_history) > self._history_window:
            self._conf_history.pop(0)

        return float(np.mean(self._conf_history))

    def _update_scale(self, conf: float) -> None:
        """Update the active scale level based on the smoothed confidence.

        Applies hysteresis: the level only changes after
        ``hysteresis_frames`` consecutive frames at the new desired level.

        Args:
            conf: Smoothed confidence value in ``[0, 1]``.
        """
        # Map confidence to desired level
        if conf >= self._conf_high:
            desired = 2  # high confidence -> small (fast) scale
        elif conf <= self._conf_low:
            desired = 0  # low confidence -> full resolution
        else:
            desired = 1  # medium

        if desired == self._current_level:
            self._frames_at_level += 1
        else:
            self._frames_at_level = 1
            self._current_level = desired

        # Apply hysteresis guard
        if self._frames_at_level >= self._hysteresis:
            if self._current_level == 0:
                self._current_scale = self._scale_low
            elif self._current_level == 1:
                self._current_scale = self._scale_medium
            else:
                self._current_scale = self._scale_high
