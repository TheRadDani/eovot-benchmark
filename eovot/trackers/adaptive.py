"""Adaptive tracker wrappers for EOVOT.

Currently provides :class:`FrameSkipTracker`, which wraps any
:class:`~eovot.trackers.base.BaseTracker` and processes only every
``(skip_rate + 1)``-th frame, extrapolating bounding boxes for skipped
frames.  This enables the **accuracy-throughput trade-off ablation study**:
by varying ``skip_rate`` from 0 to N you obtain a Pareto curve on the
``(mIoU, FPS)`` plane showing where a tracker sits under each throughput
constraint.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive import FrameSkipTracker

    base = MOSSETracker()
    fast = FrameSkipTracker(base, skip_rate=2, extrapolation="linear")

    fast.initialize(first_frame, init_bbox)
    for frame in video:
        bbox = fast.update(frame)   # runs base tracker every 3rd frame
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox


class FrameSkipTracker(BaseTracker):
    """Wrapper that skips ``skip_rate`` frames between tracker updates.

    The underlying tracker is called only on processed frames; on
    *skipped* frames the bounding box is extrapolated from the last
    known state.  This trades accuracy for throughput in a controllable
    and measurable way, enabling systematic ablation studies.

    Args:
        tracker:       The base tracker to wrap.  Must implement the
                       :class:`~eovot.trackers.base.BaseTracker` interface.
        skip_rate:     Number of frames to skip between real tracker
                       updates.  0 → no skipping (identical to the
                       wrapped tracker); 1 → every other frame; 2 → every
                       third frame; etc.
        extrapolation: Prediction mode for skipped frames.

                       ``"last"`` — repeat the last known bounding box.
                       Safe for slow-moving targets; zero overshoot.

                       ``"linear"`` — extrapolate using the displacement
                       between the last two real updates as a velocity
                       estimate.  Better for linear motion.

    Raises:
        ValueError: If ``skip_rate < 0`` or ``extrapolation`` is unknown.

    Properties:
        skip_ratio:            Fraction of frames that are skipped
                               (0 when ``skip_rate=0``, 0.5 when 1, ...).
        throughput_multiplier: Theoretical FPS gain relative to
                               ``skip_rate=0`` (i.e. ``skip_rate + 1``).
    """

    EXTRAPOLATION_MODES: Tuple[str, ...] = ("last", "linear")

    def __init__(
        self,
        tracker: BaseTracker,
        skip_rate: int = 1,
        extrapolation: str = "last",
    ) -> None:
        if skip_rate < 0:
            raise ValueError(
                f"skip_rate must be >= 0, got {skip_rate}."
            )
        if extrapolation not in self.EXTRAPOLATION_MODES:
            raise ValueError(
                f"Unknown extrapolation mode '{extrapolation}'. "
                f"Valid: {self.EXTRAPOLATION_MODES}"
            )
        super().__init__(
            name=f"{tracker.name}[skip={skip_rate},{extrapolation}]"
        )
        self._tracker = tracker
        self.skip_rate = skip_rate
        self.extrapolation = extrapolation

        self._frame_count: int = 0
        self._last_bbox: Optional[BBox] = None
        self._prev_bbox: Optional[BBox] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Delegate initialisation to the wrapped tracker.

        Args:
            frame: First video frame (BGR or grayscale).
            bbox:  Initial ground-truth box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._last_bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        self._prev_bbox = None
        self._frame_count = 0

    def update(self, frame: np.ndarray) -> BBox:
        """Return a bounding box for *frame*, running the tracker only on
        processed frames and extrapolating on skipped frames.

        Args:
            frame: The current video frame.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._last_bbox is None:
            raise RuntimeError(
                "FrameSkipTracker not initialised. Call initialize() first."
            )

        self._frame_count += 1

        # Determine whether to run the real tracker on this frame.
        run_tracker = (
            self.skip_rate == 0
            or self._frame_count % (self.skip_rate + 1) == 0
        )

        if run_tracker:
            bbox = self._tracker.update(frame)
            self._prev_bbox = self._last_bbox
            self._last_bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
            return self._last_bbox

        return self._extrapolate()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def skip_ratio(self) -> float:
        """Fraction of frames that are skipped (0 → no skipping)."""
        return self.skip_rate / (self.skip_rate + 1)

    @property
    def throughput_multiplier(self) -> float:
        """Theoretical FPS speedup relative to ``skip_rate=0``."""
        return float(self.skip_rate + 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extrapolate(self) -> BBox:
        """Return an extrapolated box for a skipped frame."""
        if self.extrapolation == "linear" and self._prev_bbox is not None:
            lx, ly, lw, lh = self._last_bbox  # type: ignore[misc]
            px, py, pw, ph = self._prev_bbox
            vx = lx - px
            vy = ly - py
            # Smoothed size: average of last two to reduce jitter.
            return (lx + vx, ly + vy, (lw + pw) / 2.0, (lh + ph) / 2.0)

        return self._last_bbox  # type: ignore[return-value]
