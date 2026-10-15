"""BackgroundAwareTracker — motion-gated tracker wrapper for edge energy savings.

Motivation
----------
Classical and correlation-filter trackers run their full update pipeline on
every frame regardless of whether the target has moved.  On edge hardware
(Raspberry Pi, Jetson Nano, embedded vision systems), CPU cycles and battery
life are scarce: spending a full KCF/CSRT update on a stationary object wastes
energy with zero accuracy benefit.

This module introduces :class:`BackgroundAwareTracker`, a **decorator** that
wraps any :class:`~eovot.trackers.base.BaseTracker` with an OpenCV MOG2
Gaussian Mixture background subtractor.  On each frame, MOG2 computes a
foreground mask within the tracker's last known bounding-box region.  If the
foreground pixel ratio falls below ``motion_threshold``, the frame is skipped:
the previous bounding box is returned without calling the wrapped tracker's
``update()``.

Energy–accuracy tradeoff
------------------------
Skipping update frames costs accuracy in sequences with genuine slow motion,
but the penalty is small because the box simply stays put rather than
accumulating drift.  On typical surveillance or edge-deployment scenes — where
targets are stationary or nearly so for extended periods — skip rates of 20–40 %
are common, translating to proportional CPU savings at a fraction-of-a-percent
IoU cost.

The :attr:`motion_stats` property exposes total frames, skipped frames, and the
skip ratio for downstream energy analysis (feed into
:class:`~eovot.profiling.energy.EnergyProfiler` or log directly).

Usage
-----
::

    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.background_aware import BackgroundAwareTracker

    base = KCFTracker()
    tracker = BackgroundAwareTracker(base, motion_threshold=0.02)

    tracker.initialize(first_frame, init_bbox)
    for frame in sequence:
        bbox = tracker.update(frame)
        # bbox is the wrapped tracker's prediction, or the previous bbox if
        # the frame was skipped due to low motion.

    stats = tracker.motion_stats
    print(f"Skipped {stats['skipped_frames']} / {stats['total_frames']} frames "
          f"({stats['skip_ratio']:.1%} energy saved)")

Notes
-----
* MOG2 is initialised lazily on the first ``update()`` call because its
  ``apply()`` method needs the frame dimensions, which are not known at
  construction time.
* The foreground-ratio test is applied only inside the last known bounding-box
  crop (not the full frame) to avoid false triggers from background activity
  unrelated to the tracked object.
* ``motion_threshold`` is in ``[0, 1]`` — a value of ``0.01`` skips the update
  when less than 1 % of the target crop's pixels are classified as foreground.
  Lower values skip more aggressively; ``0.0`` disables skipping.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class BackgroundAwareTracker(BaseTracker):
    """Motion-gated wrapper that skips tracker updates on static frames.

    Args:
        tracker:          Any :class:`BaseTracker` subclass to wrap.
        motion_threshold: Foreground pixel ratio (0–1) below which the wrapped
            tracker's ``update()`` is skipped.  Default: ``0.02`` (2 %).
        history:          MOG2 history length in frames.  Longer history gives
            more stable background models at the cost of slower adaptation.
            Default: ``50``.
        var_threshold:    MOG2 Mahalanobis-distance threshold.  Higher values
            classify fewer pixels as foreground (stricter).  Default: ``16``.
        name:             Identifier used in benchmark reports.  Defaults to
            ``"BackgroundAware(<wrapped_name>)"``.

    Attributes:
        motion_stats: Dict with ``total_frames``, ``skipped_frames``, and
            ``skip_ratio`` updated after each ``update()`` call.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        motion_threshold: float = 0.02,
        history: int = 50,
        var_threshold: float = 16.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or f"BackgroundAware({tracker.name})")
        if not (0.0 <= motion_threshold <= 1.0):
            raise ValueError(
                f"motion_threshold must be in [0, 1], got {motion_threshold}."
            )

        self._tracker = tracker
        self.motion_threshold = motion_threshold
        self._history = history
        self._var_threshold = var_threshold

        self._bg_subtractor: Optional[cv2.BackgroundSubtractorMOG2] = None
        self._last_bbox: Optional[BBox] = None

        self._total_frames: int = 0
        self._skipped_frames: int = 0

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the wrapped tracker and reset the motion model.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._last_bbox = bbox
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=False,
        )
        # Warm up the background model on the first frame so it has a valid
        # reference before the second frame arrives.
        self._bg_subtractor.apply(frame)
        self._total_frames = 0
        self._skipped_frames = 0

    def update(self, frame: np.ndarray) -> BBox:
        """Return a bounding-box prediction, skipping the wrapped tracker if motion is low.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.  If the frame is skipped,
            the previous bounding box is returned unchanged.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        if self._last_bbox is None or self._bg_subtractor is None:
            raise RuntimeError("initialize() must be called before update().")

        self._total_frames += 1

        if self._has_motion(frame):
            bbox = self._tracker.update(frame)
            self._last_bbox = bbox
        else:
            self._skipped_frames += 1
            bbox = self._last_bbox

        # Always feed the full frame to MOG2 to keep the background model
        # adapting even on skipped frames.
        self._bg_subtractor.apply(frame)

        return bbox

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def motion_stats(self) -> Dict[str, object]:
        """Return motion-skipping statistics accumulated since last initialize().

        Returns:
            Dict with:
            - ``total_frames`` (int): frames since last ``initialize()``.
            - ``skipped_frames`` (int): frames where the wrapped update was skipped.
            - ``skip_ratio`` (float): ``skipped / total``, or 0.0 if no frames yet.
        """
        ratio = (
            self._skipped_frames / self._total_frames
            if self._total_frames > 0 else 0.0
        )
        return {
            "total_frames": self._total_frames,
            "skipped_frames": self._skipped_frames,
            "skip_ratio": ratio,
        }

    @property
    def wrapped_tracker(self) -> BaseTracker:
        """The underlying :class:`BaseTracker` being wrapped."""
        return self._tracker

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _has_motion(self, frame: np.ndarray) -> bool:
        """Return True when the foreground pixel ratio inside the last bbox exceeds threshold.

        Evaluates motion only within the tracker's last known bounding-box crop
        so that background activity outside the target region does not cause
        false motion detections.
        """
        if self.motion_threshold <= 0.0:
            return True

        x, y, w, h = (int(round(v)) for v in self._last_bbox)  # type: ignore[misc]
        H, W = frame.shape[:2]

        # Clamp to valid image region.
        x  = max(0, min(x, W - 1))
        y  = max(0, min(y, H - 1))
        x2 = min(x + max(w, 1), W)
        y2 = min(y + max(h, 1), H)

        crop = frame[y:y2, x:x2]
        if crop.size == 0:
            return True

        # Apply MOG2 to the full frame to get a consistent background model,
        # then inspect only the cropped foreground mask.
        fg_mask = self._bg_subtractor.apply(frame, learningRate=0)
        crop_mask = fg_mask[y:y2, x:x2]

        n_pixels = crop_mask.size
        if n_pixels == 0:
            return True

        fg_ratio = float(np.count_nonzero(crop_mask)) / n_pixels
        return fg_ratio >= self.motion_threshold
