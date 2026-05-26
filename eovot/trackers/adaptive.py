"""Adaptive frame-skip tracker wrapper for edge deployment.

Wraps any :class:`~eovot.trackers.base.BaseTracker` and skips tracker
updates on a configurable fraction of frames, propagating the bounding box
using simple extrapolation.  This enables systematic ablation studies of the
accuracy-vs-throughput trade-off that is central to edge VOT deployment.

Background
~~~~~~~~~~
On resource-constrained hardware (Raspberry Pi, Jetson Nano) even lightweight
trackers like MOSSE and KCF may fall below real-time frame rates (≥25 FPS)
when combined with image capture and post-processing.  A principled response
is *frame skipping*: run the tracker once every N frames and extrapolate the
target location in between.  The EOVOT benchmarking goal is to quantify the
accuracy penalty incurred at each skip rate so that practitioners can select
the Pareto-optimal operating point for their deployment.

Two extrapolation strategies are provided:

``"constant"``
    The last confirmed bounding box is repeated unchanged for all skipped
    frames.  Fast, robust to scale changes, but accumulates position error
    for moving targets.

``"linear"``
    The centre and size of the bounding box are extrapolated along the
    velocity vector estimated from the two most recent tracker updates.
    Reduces centre-distance error for objects with approximately constant
    velocity.  Bounding box coordinates are clamped to the image boundary
    when the frame shape is known.

Usage
~~~~~
::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive import FrameSkipTracker

    base = MOSSETracker()
    tracker = FrameSkipTracker(base, skip_rate=3, extrapolation="linear")

    # Drop-in replacement inside BenchmarkEngine:
    from eovot.benchmark.engine import BenchmarkEngine
    engine = BenchmarkEngine()
    result = engine.run(tracker, dataset, "OTB100")

    # Inspect skip statistics after the run:
    print(f"Tracked {tracker.frames_tracked} / {tracker.total_frames} frames "
          f"(skipped {tracker.skip_ratio*100:.1f}%)")

Ablation study pattern
~~~~~~~~~~~~~~~~~~~~~~~
::

    skip_rates = [0, 1, 2, 3, 5, 9]
    for rate in skip_rates:
        t = FrameSkipTracker(MOSSETracker(), skip_rate=rate)
        result = engine.run(t, dataset, "synthetic")
        print(f"skip={rate}  mIoU={result.mean_iou:.3f}  FPS={result.mean_fps:.1f}")
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox


class FrameSkipTracker(BaseTracker):
    """A drop-in :class:`BaseTracker` that skips inner tracker updates periodically.

    Args:
        tracker:       Any :class:`BaseTracker` implementation to wrap.
        skip_rate:     Number of frames to skip between tracker updates.
                       ``0`` disables skipping (every frame is tracked).
                       ``1`` runs the tracker every second frame.
                       ``N`` runs the tracker every ``(N + 1)``-th frame.
        extrapolation: How to fill in bounding boxes on skipped frames.
                       ``"constant"`` repeats the last tracked bbox.
                       ``"linear"`` extrapolates position and size using
                       the velocity of the two most recent tracked frames.
        name:          Optional override for the tracker name shown in reports.
                       Defaults to ``"<inner_name>(skip=N)"``.

    Attributes:
        frames_tracked: Number of frames on which the inner tracker was called.
        frames_skipped: Number of frames filled by extrapolation.
        total_frames:   Total frames seen since :meth:`initialize`.
        skip_ratio:     ``frames_skipped / total_frames`` (0.0 when no frames
                        have been processed yet).
    """

    EXTRAPOLATION_MODES = ("constant", "linear")

    def __init__(
        self,
        tracker: BaseTracker,
        skip_rate: int = 1,
        extrapolation: str = "constant",
        name: Optional[str] = None,
    ) -> None:
        if skip_rate < 0:
            raise ValueError(f"skip_rate must be >= 0, got {skip_rate}.")
        if extrapolation not in self.EXTRAPOLATION_MODES:
            raise ValueError(
                f"extrapolation must be one of {self.EXTRAPOLATION_MODES}, "
                f"got {extrapolation!r}."
            )
        derived_name = name or f"{tracker.name}(skip={skip_rate})"
        super().__init__(derived_name)
        self._inner = tracker
        self.skip_rate = skip_rate
        self.extrapolation = extrapolation

        # State reset on each initialize() call.
        self._last_bbox: Optional[BBox] = None
        self._prev_bbox: Optional[BBox] = None      # two frames back, for velocity
        self._frame_counter: int = 0                # frames since initialize
        self._next_track_at: int = 0                # frame index for next inner update
        self._frame_shape: Optional[Tuple[int, int]] = None  # (H, W) for clamping

        self.frames_tracked: int = 0
        self.frames_skipped: int = 0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        return self.frames_tracked + self.frames_skipped

    @property
    def skip_ratio(self) -> float:
        tot = self.total_frames
        return self.frames_skipped / tot if tot > 0 else 0.0

    @property
    def inner_tracker(self) -> BaseTracker:
        """The wrapped tracker instance."""
        return self._inner

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise both the wrapper and the inner tracker.

        Args:
            frame: BGR uint8 image, shape ``(H, W, 3)``.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._inner.initialize(frame, bbox)
        self._last_bbox = bbox
        self._prev_bbox = None
        self._frame_counter = 0
        self._next_track_at = self.skip_rate + 1  # first skip N frames, then track
        self._frame_shape = (frame.shape[0], frame.shape[1])
        self.frames_tracked = 0
        self.frames_skipped = 0

    def update(self, frame: np.ndarray) -> BBox:
        """Return the predicted bounding box for *frame*.

        Calls the inner tracker on scheduled frames; extrapolates on the rest.

        Args:
            frame: BGR uint8 image, shape ``(H, W, 3)``.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._last_bbox is None:
            raise RuntimeError("FrameSkipTracker.update() called before initialize().")

        self._frame_counter += 1
        self._frame_shape = (frame.shape[0], frame.shape[1])

        if self.skip_rate == 0 or self._frame_counter >= self._next_track_at:
            # Run the inner tracker on this frame.
            bbox = self._inner.update(frame)
            self._prev_bbox = self._last_bbox
            self._last_bbox = bbox
            self.frames_tracked += 1
            # Schedule next tracker call.
            self._next_track_at = self._frame_counter + self.skip_rate + 1
        else:
            # Extrapolate the bounding box without running the inner tracker.
            bbox = self._extrapolate()
            self.frames_skipped += 1

        return bbox

    # ------------------------------------------------------------------
    # Extrapolation
    # ------------------------------------------------------------------

    def _extrapolate(self) -> BBox:
        """Predict the bounding box for a skipped frame."""
        if self.extrapolation == "linear" and self._prev_bbox is not None:
            return self._linear_extrapolate()
        return self._last_bbox  # type: ignore[return-value]

    def _linear_extrapolate(self) -> BBox:
        """Extrapolate centre and size linearly from the two most-recent bboxes.

        Velocity is estimated as the per-component difference between the last
        two *tracked* frames.  The extrapolated bbox is clamped to the image
        boundary if ``_frame_shape`` is available.
        """
        ax, ay, aw, ah = self._last_bbox  # type: ignore[misc]
        bx, by, bw, bh = self._prev_bbox  # type: ignore[misc]

        # Velocity = (last - prev) per component; apply once per skipped frame.
        vx = ax - bx
        vy = ay - by
        vw = aw - bw
        vh = ah - bh

        nx = ax + vx
        ny = ay + vy
        nw = max(1.0, aw + vw)
        nh = max(1.0, ah + vh)

        if self._frame_shape is not None:
            H, W = self._frame_shape
            nx = float(np.clip(nx, 0, W - 1))
            ny = float(np.clip(ny, 0, H - 1))
            nw = float(np.clip(nw, 1, W - nx))
            nh = float(np.clip(nh, 1, H - ny))

        return (nx, ny, nw, nh)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FrameSkipTracker("
            f"inner={self._inner!r}, "
            f"skip_rate={self.skip_rate}, "
            f"extrapolation={self.extrapolation!r})"
        )
