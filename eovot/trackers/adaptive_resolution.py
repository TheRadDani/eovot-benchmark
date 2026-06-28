"""Adaptive Dynamic Resolution Tracker (ADRT) for compute-budget-aware tracking.

Wraps any :class:`~eovot.trackers.base.BaseTracker` with a lightweight
feedback controller that adjusts the input frame resolution up or down at
runtime to stay within a user-specified throughput target.

Design rationale
----------------
Edge devices have fixed compute envelopes. A tracker that runs at 60 FPS on a
laptop may only achieve 5 FPS on a Raspberry Pi. Rather than accepting this
degradation passively, ADRT trades accuracy smoothly for speed by scaling the
frame before the inner tracker sees it, then projecting the predicted box back
to the original resolution.

The controller is a simple proportional regulator:

    scale ← clamp(scale × (current_fps / target_fps) ^ control_gain,
                  min_scale, max_scale)

- When behind target: scale shrinks → cheaper inference → FPS rises.
- When ahead of target: scale grows → richer features → better accuracy.
- Exponential smoothing on the latency estimate prevents single-frame spikes
  from overreacting.

This makes ADRT the first compute-adaptive tracker in EOVOT, directly serving
the project's edge-deployment thesis.

Example::

    from eovot.trackers.adaptive_resolution import AdaptiveResolutionTracker
    from eovot.trackers.kcf import KCFTracker

    inner = KCFTracker()
    tracker = AdaptiveResolutionTracker(inner, target_fps=25.0)

    # Then use tracker.initialize / tracker.update exactly like any other tracker.
    tracker.initialize(frame0, init_bbox)
    for frame in remaining_frames:
        bbox = tracker.update(frame)
        print(f"scale={tracker.current_scale:.2f}  fps={tracker.current_fps:.1f}")
"""

from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from .base import BaseTracker, BBox


class AdaptiveResolutionTracker(BaseTracker):
    """Compute-budget-aware tracker that dynamically adjusts input resolution.

    Wraps any :class:`BaseTracker` and scales each frame before it is processed,
    then maps the predicted box back to original-resolution coordinates.  A
    proportional controller updates the scale factor every frame using an
    exponentially-smoothed FPS estimate.

    Args:
        tracker: The underlying tracker instance (must implement
            :class:`BaseTracker`). The caller retains ownership; ADRT calls
            ``initialize`` / ``update`` and nothing else.
        target_fps: Desired throughput in frames per second.  The controller
            will increase resolution when actual FPS exceeds this target and
            decrease resolution when FPS falls short.  Default: ``30.0``.
        min_scale: Minimum allowed scale factor (> 0).  Prevents the frame
            from being shrunk beyond usefulness.  Default: ``0.25``.
        max_scale: Maximum allowed scale factor (≤ 1.0 recommended; values
            > 1.0 upsample the input, wasting compute without extra information).
            Default: ``1.0``.
        control_gain: Exponent controlling aggressiveness of scale adjustments.
            ``1.0`` means fully proportional correction each frame; ``0.5``
            applies the square-root of the ratio for smoother adaptation.
            Default: ``0.5``.
        smoothing: EMA coefficient for the FPS estimate.  Higher values make
            the estimate smoother but slower to react.  Default: ``0.85``.
        initial_scale: Starting scale factor before any timing data is
            available.  Defaults to the midpoint of ``[min_scale, max_scale]``.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        target_fps: float = 30.0,
        min_scale: float = 0.25,
        max_scale: float = 1.0,
        control_gain: float = 0.5,
        smoothing: float = 0.85,
        initial_scale: Optional[float] = None,
    ) -> None:
        if not isinstance(tracker, BaseTracker):
            raise TypeError(f"tracker must be a BaseTracker, got {type(tracker)}")
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if not (0 < min_scale <= max_scale):
            raise ValueError(
                f"Need 0 < min_scale <= max_scale, got {min_scale}, {max_scale}"
            )
        if not (0.0 < control_gain <= 2.0):
            raise ValueError(f"control_gain must be in (0, 2], got {control_gain}")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")

        super().__init__(name=f"ADRT[{tracker.name}]")

        self._inner = tracker
        self.target_fps = target_fps
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.control_gain = control_gain
        self.smoothing = smoothing

        _default_scale = (min_scale + max_scale) / 2.0
        self._scale: float = (
            float(np.clip(initial_scale, min_scale, max_scale))
            if initial_scale is not None
            else _default_scale
        )

        self._smoothed_latency_s: Optional[float] = None
        self._scale_history: List[float] = []
        self._latency_history: List[float] = []

    # ------------------------------------------------------------------
    # Public status properties
    # ------------------------------------------------------------------

    @property
    def current_scale(self) -> float:
        """Current frame scale factor, in ``[min_scale, max_scale]``."""
        return self._scale

    @property
    def current_fps(self) -> float:
        """EMA-smoothed FPS estimate; 0.0 before the first update call."""
        if self._smoothed_latency_s is None or self._smoothed_latency_s <= 0:
            return 0.0
        return 1.0 / self._smoothed_latency_s

    @property
    def scale_history(self) -> List[float]:
        """Scale factor at each ``update`` call (read-only copy)."""
        return list(self._scale_history)

    @property
    def latency_history_ms(self) -> List[float]:
        """Raw per-frame latency in milliseconds for each ``update`` call."""
        return [t * 1000.0 for t in self._latency_history]

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the inner tracker on a scaled-down version of the first frame.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox: Ground-truth bounding box ``(x, y, w, h)`` in original coords.
        """
        self._smoothed_latency_s = None
        self._scale_history.clear()
        self._latency_history.clear()

        scaled_frame = self._resize(frame, self._scale)
        scaled_bbox = self._scale_bbox_down(bbox, self._scale)
        self._inner.initialize(scaled_frame, scaled_bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target in *frame* using the current scale setting.

        Measures wall-clock latency of the inner ``update`` call and adjusts
        the scale factor for the next frame via the proportional controller.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` in **original** frame
            coordinates.
        """
        scaled_frame = self._resize(frame, self._scale)

        t0 = time.perf_counter()
        scaled_pred = self._inner.update(scaled_frame)
        t1 = time.perf_counter()

        raw_latency = t1 - t0
        self._latency_history.append(raw_latency)
        self._update_latency(raw_latency)

        # Project prediction back to original coordinates
        pred_bbox = self._scale_bbox_up(scaled_pred, self._scale)

        # Adapt scale for the next frame
        self._adapt_scale()
        self._scale_history.append(self._scale)

        return pred_bbox

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize(frame: np.ndarray, scale: float) -> np.ndarray:
        """Resize *frame* by *scale* using bilinear interpolation.

        If *scale* is 1.0 (within floating-point tolerance) returns the frame
        unchanged to avoid unnecessary copies.
        """
        if abs(scale - 1.0) < 1e-6:
            return frame
        h, w = frame.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _scale_bbox_down(bbox: BBox, scale: float) -> BBox:
        """Convert original-resolution bbox to scaled-down coordinates."""
        x, y, w, h = bbox
        return (x * scale, y * scale, w * scale, h * scale)

    @staticmethod
    def _scale_bbox_up(bbox: BBox, scale: float) -> BBox:
        """Project a scaled bbox back to original-resolution coordinates."""
        x, y, w, h = bbox
        inv = 1.0 / scale if scale > 0 else 1.0
        return (x * inv, y * inv, w * inv, h * inv)

    def _update_latency(self, raw: float) -> None:
        """Exponential-moving-average update for the latency estimate."""
        if self._smoothed_latency_s is None:
            self._smoothed_latency_s = raw
        else:
            self._smoothed_latency_s = (
                self.smoothing * self._smoothed_latency_s
                + (1.0 - self.smoothing) * raw
            )

    def _adapt_scale(self) -> None:
        """Proportional controller: adjust scale based on current FPS vs target.

        Uses a power law to keep the step bounded:

            scale_new = scale × (fps_current / target_fps) ^ control_gain

        The ratio ``< 1`` shrinks scale when behind target; ``> 1`` grows it
        when ahead.  The result is clamped to ``[min_scale, max_scale]``.
        """
        fps = self.current_fps
        if fps <= 0:
            return
        ratio = fps / self.target_fps
        new_scale = self._scale * (ratio ** self.control_gain)
        self._scale = float(np.clip(new_scale, self.min_scale, self.max_scale))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary dict of adaptation statistics for this run.

        Returns:
            Dict with keys: ``mean_scale``, ``min_scale_used``,
            ``max_scale_used``, ``mean_fps``, ``n_frames``.
        """
        if not self._scale_history:
            return {}
        scales = np.array(self._scale_history)
        lats = np.array(self._latency_history)
        fps_vals = 1.0 / lats[lats > 0] if (lats > 0).any() else np.array([0.0])
        return {
            "mean_scale": float(scales.mean()),
            "min_scale_used": float(scales.min()),
            "max_scale_used": float(scales.max()),
            "mean_fps": float(fps_vals.mean()),
            "n_frames": len(self._scale_history),
        }

    def __repr__(self) -> str:
        return (
            f"AdaptiveResolutionTracker("
            f"inner={self._inner.name!r}, "
            f"target_fps={self.target_fps}, "
            f"scale={self._scale:.2f}, "
            f"range=[{self.min_scale}, {self.max_scale}])"
        )
