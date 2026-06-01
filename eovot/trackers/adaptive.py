"""Confidence-gated adaptive tracker wrapper for edge-optimised deployment.

Reduces tracker compute by skipping inner-tracker update calls on frames
where optical-flow motion is low **and** the tracker has been reliably
following the target.  On video with slow-moving or temporarily stationary
targets this can eliminate 40–80 % of update calls with negligible accuracy
loss — a significant win for battery-constrained or thermally-limited edge
devices.

The wrapper exposes the same :class:`~eovot.trackers.base.BaseTracker`
interface, so it is fully transparent to :class:`~eovot.benchmark.engine.BenchmarkEngine`.

Design
------
Two independent gates must both pass before a frame is skipped:

1. **Motion gate** — dense Farnebäck optical flow is computed in the
   bounding-box region.  If the mean magnitude falls below
   ``motion_threshold`` pixels/frame the gate passes.

2. **Confidence gate** — an exponential moving average of a proxy
   confidence score (based on relative bbox displacement) must exceed
   ``confidence_threshold``.  This prevents skipping during fast-motion
   or after a near-failure recovery.

When both gates pass **and** the consecutive-skip streak has not reached
``max_skip_streak``, the wrapper propagates the previous bbox (optionally
shifted by a smoothed linear-velocity estimate).  Otherwise the inner
tracker's ``update()`` is invoked normally.

References
----------
* Farnebäck, G. (2003). Two-Frame Motion Estimation Based on Polynomial
  Expansion. SCIA 2003, LNCS 2749, pp. 363–370.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import BaseTracker, BBox


class AdaptiveTracker(BaseTracker):
    """Confidence-gated wrapper that skips tracker updates on easy frames.

    Args:
        tracker: Inner :class:`BaseTracker` to wrap.  Any tracker
            implementing the EOVOT interface is supported.
        motion_threshold: Mean optical-flow magnitude (pixels/frame)
            inside the bounding box below which the motion gate passes.
            Lower values make the tracker skip more aggressively.
            Default: ``2.0``.
        confidence_threshold: Minimum EMA confidence score (0–1) required
            for the confidence gate to pass.  Default: ``0.6``.
        confidence_window: Number of frames used for the EMA smoothing of
            the confidence signal.  Default: ``10``.
        max_skip_streak: Maximum number of consecutive frames that can be
            skipped before forcing a full update.  Prevents unbounded
            drift.  Default: ``5``.
        use_velocity: When ``True``, shifts the propagated bbox by the
            current linear-velocity estimate.  Improves propagation
            quality for smoothly-moving targets.  Default: ``True``.

    Attributes:
        skip_ratio: Fraction of update calls that were skipped (0–1).
            Populated after tracking; zero when no frames have been seen.
        frames_skipped: Raw count of skipped frames.
        frames_updated: Raw count of frames where the inner tracker ran.

    Example::

        from eovot.trackers.mosse import MOSSETracker
        from eovot.trackers.adaptive import AdaptiveTracker

        inner   = MOSSETracker(learning_rate=0.125)
        tracker = AdaptiveTracker(inner, motion_threshold=2.5, max_skip_streak=4)

        tracker.initialize(first_frame, init_bbox)
        for frame in subsequent_frames:
            bbox = tracker.update(frame)

        print(f"Skip ratio: {tracker.skip_ratio:.1%}")
        print(f"Frames updated: {tracker.frames_updated}/{tracker.frames_updated + tracker.frames_skipped}")
    """

    def __init__(
        self,
        tracker: BaseTracker,
        motion_threshold: float = 2.0,
        confidence_threshold: float = 0.6,
        confidence_window: int = 10,
        max_skip_streak: int = 5,
        use_velocity: bool = True,
    ) -> None:
        super().__init__(name=f"Adaptive({tracker.name})")
        self._inner = tracker
        self.motion_threshold = motion_threshold
        self.confidence_threshold = confidence_threshold
        self.confidence_window = confidence_window
        self.max_skip_streak = max_skip_streak
        self.use_velocity = use_velocity

        # Runtime state — reset on each initialize() call
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_bbox: Optional[BBox] = None
        self._velocity: np.ndarray = np.zeros(2, dtype=np.float64)  # (vx, vy)
        self._confidence_ema: float = 1.0
        self._skip_streak: int = 0

        # Accumulated statistics
        self._frames_total: int = 0
        self._frames_skipped: int = 0

    # ------------------------------------------------------------------ #
    # BaseTracker interface                                                #
    # ------------------------------------------------------------------ #

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the wrapped tracker and reset internal state.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._inner.initialize(frame, bbox)
        self._prev_gray = _to_gray(frame)
        self._prev_bbox = bbox
        self._velocity = np.zeros(2, dtype=np.float64)
        self._confidence_ema = 1.0
        self._skip_streak = 0
        self._frames_total = 0
        self._frames_skipped = 0

    def update(self, frame: np.ndarray) -> BBox:
        """Predict target location, skipping the inner tracker when safe.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._prev_bbox is None:
            raise RuntimeError("AdaptiveTracker not initialised — call initialize() first.")

        gray = _to_gray(frame)
        self._frames_total += 1

        motion = self._estimate_motion(gray)
        should_skip = (
            motion < self.motion_threshold
            and self._confidence_ema >= self.confidence_threshold
            and self._skip_streak < self.max_skip_streak
        )

        if should_skip:
            self._frames_skipped += 1
            self._skip_streak += 1
            pred = self._propagate()
        else:
            pred = self._inner.update(frame)
            self._skip_streak = 0
            self._update_velocity(pred)
            self._update_confidence(pred)

        self._prev_gray = gray
        self._prev_bbox = pred
        return pred

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    @property
    def skip_ratio(self) -> float:
        """Fraction of frames where the inner tracker was not called."""
        return self._frames_skipped / self._frames_total if self._frames_total else 0.0

    @property
    def frames_skipped(self) -> int:
        """Total number of skipped frames."""
        return self._frames_skipped

    @property
    def frames_updated(self) -> int:
        """Total number of frames where the inner tracker ran."""
        return self._frames_total - self._frames_skipped

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _estimate_motion(self, gray: np.ndarray) -> float:
        """Dense optical-flow magnitude sampled inside the current bbox."""
        if self._prev_gray is None or self._prev_bbox is None:
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Restrict to the bbox region to focus on target motion
        x, y, w, h = (int(v) for v in self._prev_bbox)
        ih, iw = gray.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x2 > x1 and y2 > y1:
            flow = flow[y1:y2, x1:x2]

        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(mag.mean())

    def _propagate(self) -> BBox:
        """Return the previous bbox, optionally shifted by velocity."""
        x, y, w, h = self._prev_bbox
        if self.use_velocity:
            x += self._velocity[0]
            y += self._velocity[1]
        return (float(x), float(y), float(w), float(h))

    def _update_velocity(self, new_bbox: BBox) -> None:
        """Update velocity via exponential smoothing from consecutive displacement."""
        px, py, _, _ = self._prev_bbox
        nx, ny, _, _ = new_bbox
        alpha = 0.5
        self._velocity[0] = alpha * (nx - px) + (1.0 - alpha) * self._velocity[0]
        self._velocity[1] = alpha * (ny - py) + (1.0 - alpha) * self._velocity[1]

    def _update_confidence(self, new_bbox: BBox) -> None:
        """Update EMA confidence from displacement relative to bbox diagonal."""
        px, py, pw, ph = self._prev_bbox
        nx, ny, _, _ = new_bbox
        diag = max(1.0, float(np.sqrt(pw ** 2 + ph ** 2)))
        displacement = float(np.sqrt((nx - px) ** 2 + (ny - py) ** 2))
        # Exponential decay: full confidence when stationary, near-zero after large jump
        conf = float(np.exp(-displacement / (diag * 0.5)))
        alpha = 2.0 / (self.confidence_window + 1)
        self._confidence_ema = alpha * conf + (1.0 - alpha) * self._confidence_ema


# ------------------------------------------------------------------ #
# Module-level helper                                                 #
# ------------------------------------------------------------------ #

def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR / BGRA / grayscale image to single-channel uint8."""
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
