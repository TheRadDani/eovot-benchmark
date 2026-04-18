"""Frame-skip wrapper for edge-aware tracker evaluation.

On constrained hardware (Raspberry Pi, Jetson Nano, microcontrollers) it
is often impossible to run a full tracker on every camera frame while
maintaining real-time throughput.  A practical strategy is to invoke the
tracker every *k* frames and propagate the last known bounding box on
the remaining frames.

:class:`FrameSkipWrapper` implements this strategy as a transparent
:class:`~eovot.trackers.base.BaseTracker` wrapper, so any existing tracker
can be evaluated under different temporal sub-sampling rates without
modifying its implementation.

Edge deployment tradeoff
------------------------
- ``skip_rate=1``  — original tracker, no frame skipping (baseline)
- ``skip_rate=2``  — tracker called every other frame (~2× throughput gain)
- ``skip_rate=4``  — tracker called every 4th frame (~4× throughput gain)

The accuracy–efficiency tradeoff across skip rates is quantified by
:class:`~eovot.metrics.efficiency.AETCurve`.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.frame_skip import FrameSkipWrapper

    base = MOSSETracker()
    tracker = FrameSkipWrapper(base, skip_rate=3)
    # tracker.name == "MOSSE-skip3"
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseTracker, BBox


class FrameSkipWrapper(BaseTracker):
    """Wraps any :class:`~eovot.trackers.base.BaseTracker` with temporal sub-sampling.

    The underlying tracker's ``update()`` is called only on every *k*-th frame
    (where *k = skip_rate*).  On the remaining ``skip_rate - 1`` frames the
    last predicted bounding box is returned unchanged.

    This is the simplest viable edge-deployment strategy and serves as a
    lower bound on accuracy for a given throughput budget.

    Args:
        tracker:   Any :class:`~eovot.trackers.base.BaseTracker` instance.
        skip_rate: Number of frames between tracker updates.
                   ``1`` means every frame is processed (no skipping).
                   Must be a positive integer.

    Raises:
        ValueError: If *skip_rate* < 1.
    """

    def __init__(self, tracker: BaseTracker, skip_rate: int = 2) -> None:
        if skip_rate < 1:
            raise ValueError(f"skip_rate must be >= 1, got {skip_rate}")
        super().__init__(name=f"{tracker.name}-skip{skip_rate}")
        self._tracker = tracker
        self.skip_rate = skip_rate
        self._last_bbox: Optional[BBox] = None
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the underlying tracker and reset the frame counter."""
        self._tracker.initialize(frame, bbox)
        self._last_bbox = bbox
        self._frame_count = 0

    def update(self, frame: np.ndarray) -> BBox:
        """Return the tracker prediction, calling ``update`` only every *k* frames.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Bounding box ``(x, y, w, h)``.  The same box is returned for
            ``skip_rate - 1`` consecutive skipped frames.
        """
        self._frame_count += 1
        if self._frame_count % self.skip_rate == 0:
            self._last_bbox = self._tracker.update(frame)
        # _last_bbox is guaranteed non-None after initialize() is called.
        return self._last_bbox  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def base_tracker(self) -> BaseTracker:
        """The wrapped tracker instance."""
        return self._tracker

    def __repr__(self) -> str:
        return (
            f"FrameSkipWrapper("
            f"tracker={self._tracker!r}, "
            f"skip_rate={self.skip_rate})"
        )
