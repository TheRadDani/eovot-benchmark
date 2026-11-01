"""Motion-adaptive frame scheduler for edge-efficient tracking.

The existing :class:`~eovot.trackers.frame_skip.FrameSkipTracker` reduces
compute by processing only every *N*-th frame with a *fixed* stride.  This
is sub-optimal: during slow phases (target barely moving) many frames are
wasted; during fast phases (rapid motion, occlusion recovery) a fixed skip
may miss the target entirely.

:class:`MotionAdaptiveTracker` replaces the fixed stride with a
*velocity-based* decision rule:

1. After each full tracker update the magnitude of the predicted center
   displacement is measured and fed into an exponential moving average (EMA)
   to maintain a smoothed estimate of target speed.
2. On the *next* frame the EMA is compared with ``motion_threshold``
   (expressed as a fraction of the current bounding-box diagonal).
   - **Below threshold** → the target is barely moving; skip the frame
     and advance the prediction by linear extrapolation using the last
     observed velocity.
   - **At or above threshold** → the target is moving appreciably; run
     the full tracker update.

This design guarantees that compute is spent where it matters (fast motion)
and saved where it does not (slow or stationary targets), directly improving
the accuracy-per-FPS trade-off for edge deployment.

The wrapper is **tracker-agnostic**: it works with any
:class:`~eovot.trackers.base.BaseTracker` subclass, including all
classicals (MOSSE, KCF, CSRT, CamShift, LKOpticalFlow) and DL trackers.

Example::

    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.motion_adaptive import MotionAdaptiveTracker

    base = KCFTracker()
    tracker = MotionAdaptiveTracker(base, motion_threshold=0.04, ema_alpha=0.3)

    # Use like any BaseTracker
    tracker.initialize(first_frame, init_bbox)
    for frame in subsequent_frames:
        bbox = tracker.update(frame)

    print(f"Skip ratio: {tracker.skip_ratio:.1%}")   # e.g. "Skip ratio: 38.2%"

Benchmarking::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    ds = SyntheticDataset(num_sequences=10, num_frames=200, motion="linear")
    engine = BenchmarkEngine(verbose=True)

    base_result = engine.run(KCFTracker(), ds, dataset_name="Synthetic")
    adaptive_result = engine.run(
        MotionAdaptiveTracker(KCFTracker(), motion_threshold=0.05),
        ds, dataset_name="Synthetic",
    )
    # Compare FPS gain vs IoU cost
    print(base_result.mean_fps, adaptive_result.mean_fps)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox


class MotionAdaptiveTracker(BaseTracker):
    """Tracker wrapper that skips frames when target motion is negligible.

    Args:
        tracker: Any :class:`~eovot.trackers.base.BaseTracker` instance to
            wrap.  The underlying tracker handles initialization and the
            full update path; this class adds the adaptive skip layer.
        motion_threshold: Fraction of the current bounding-box diagonal
            below which a frame is skipped.  E.g. ``0.05`` means: skip
            the frame when the EMA-smoothed center displacement per frame
            is less than 5 % of the box diagonal.
            - Lower values → more skipping → higher FPS, potentially lower IoU.
            - Higher values → less skipping → closer to the base tracker.
            Default: ``0.05``.
        ema_alpha: EMA smoothing factor for the motion magnitude estimate.
            Higher values make the scheduler react faster to motion changes
            (less smoothing); lower values are more conservative.
            Range: ``(0, 1]``.  Default: ``0.3``.
        max_skip_streak: Maximum number of consecutive frames that may be
            skipped before the tracker is *forced* to run a full update.
            This prevents unbounded drift when the target is stationary for
            a long time but then suddenly moves.  Default: ``5``.

    Attributes:
        skip_ratio (float): Fraction of ``update()`` calls that were
            handled by interpolation rather than the underlying tracker.
            Updated after each call.
        total_updates (int): Total number of ``update()`` calls since
            initialization.
        skipped_updates (int): Number of those calls that were skipped.

    Example::

        tracker = MotionAdaptiveTracker(MOSSETracker(), motion_threshold=0.04)
        tracker.initialize(frame0, bbox0)
        bbox1 = tracker.update(frame1)
        print(tracker.skip_ratio)  # 0.0 — first frame always processed
    """

    def __init__(
        self,
        tracker: BaseTracker,
        motion_threshold: float = 0.05,
        ema_alpha: float = 0.3,
        max_skip_streak: int = 5,
    ) -> None:
        if not (0 < ema_alpha <= 1.0):
            raise ValueError(
                f"ema_alpha must be in (0, 1], got {ema_alpha}."
            )
        if motion_threshold < 0:
            raise ValueError(
                f"motion_threshold must be non-negative, got {motion_threshold}."
            )
        if max_skip_streak < 1:
            raise ValueError(
                f"max_skip_streak must be >= 1, got {max_skip_streak}."
            )
        super().__init__(name=f"MotionAdaptive({tracker.name})")
        self._tracker = tracker
        self.motion_threshold = motion_threshold
        self.ema_alpha = ema_alpha
        self.max_skip_streak = max_skip_streak

        # State reset by initialize()
        self._last_bbox: Optional[BBox] = None
        self._last_center: Optional[Tuple[float, float]] = None
        self._last_velocity: Tuple[float, float] = (0.0, 0.0)
        self._motion_ema: float = float("inf")  # large → force full update first frame
        self._skip_streak: int = 0

        self.total_updates: int = 0
        self.skipped_updates: int = 0

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Delegate initialization to the wrapped tracker and reset state.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._last_bbox = bbox
        x, y, w, h = bbox
        self._last_center = (x + w / 2.0, y + h / 2.0)
        self._last_velocity = (0.0, 0.0)
        self._motion_ema = float("inf")
        self._skip_streak = 0
        self.total_updates = 0
        self.skipped_updates = 0

    def update(self, frame: np.ndarray) -> BBox:
        """Return the predicted bounding box for *frame*.

        Runs the full tracker update when the EMA motion exceeds the
        threshold or when ``max_skip_streak`` consecutive frames have been
        skipped.  Otherwise extrapolates linearly from the last prediction.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.
        """
        self.total_updates += 1

        # Threshold in pixels, derived from the current box diagonal
        threshold_px = self._threshold_pixels()

        skip = (
            self._motion_ema < threshold_px
            and self._skip_streak < self.max_skip_streak
            and self._last_bbox is not None
        )

        if skip:
            self.skipped_updates += 1
            self._skip_streak += 1
            return self._interpolate()

        # Full tracker update
        bbox = self._tracker.update(frame)
        self._update_motion_state(bbox)
        self._last_bbox = bbox
        self._skip_streak = 0
        return bbox

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def skip_ratio(self) -> float:
        """Fraction of update calls handled by interpolation.

        Returns:
            Value in ``[0, 1]``.  ``0.0`` before any ``update()`` call.
        """
        if self.total_updates == 0:
            return 0.0
        return self.skipped_updates / self.total_updates

    @property
    def wrapped_tracker(self) -> BaseTracker:
        """The underlying tracker being wrapped."""
        return self._tracker

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _threshold_pixels(self) -> float:
        """Convert normalised motion_threshold to pixels using current box."""
        if self._last_bbox is None:
            return 0.0
        _, _, w, h = self._last_bbox
        diag = math.sqrt(w * w + h * h)
        return self.motion_threshold * max(diag, 1.0)

    def _interpolate(self) -> BBox:
        """Return a linearly extrapolated bounding box."""
        if self._last_bbox is None:
            raise RuntimeError("MotionAdaptiveTracker.interpolate called before initialize")
        x, y, w, h = self._last_bbox
        dx, dy = self._last_velocity
        new_bbox: BBox = (x + dx, y + dy, w, h)
        # Advance state for the next potential interpolation
        self._last_bbox = new_bbox
        cx, cy = self._last_center  # type: ignore[misc]
        self._last_center = (cx + dx, cy + dy)
        return new_bbox

    def _update_motion_state(self, bbox: BBox) -> None:
        """Measure displacement from last prediction and update EMA."""
        x, y, w, h = bbox
        new_cx = x + w / 2.0
        new_cy = y + h / 2.0

        if self._last_center is not None:
            old_cx, old_cy = self._last_center
            dx = new_cx - old_cx
            dy = new_cy - old_cy
            displacement = math.sqrt(dx * dx + dy * dy)
            self._last_velocity = (dx, dy)

            if self._motion_ema == float("inf"):
                self._motion_ema = displacement
            else:
                self._motion_ema = (
                    self.ema_alpha * displacement
                    + (1.0 - self.ema_alpha) * self._motion_ema
                )
        else:
            self._last_velocity = (0.0, 0.0)
            self._motion_ema = 0.0

        self._last_center = (new_cx, new_cy)
