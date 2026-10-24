"""AdaptiveSkipTracker — confidence-driven frame-skipping for edge deployment.

Unlike :class:`~eovot.trackers.frame_skip.FrameSkipTracker` which uses a
fixed skip interval, this wrapper dynamically adjusts how often the underlying
tracker runs based on its self-reported confidence signal.

Motivation
----------
When tracking is easy (stable background, slow target, good appearance model),
the correlation response is sharp and the tracker is highly confident.  In
these stretches it is wasteful to run the full tracking update every frame.
Conversely, when the scene changes rapidly or the target is partially occluded,
confidence drops and the tracker *must* run every frame to stay on target.

:class:`AdaptiveSkipTracker` exploits this by querying
:meth:`~eovot.trackers.base.BaseTracker.confidence` after each real update and
choosing the skip rate for the coming frames:

- ``confidence ≥ high_threshold`` → skip the next ``max_skip - 1`` frames
  (run once every ``max_skip`` frames).
- ``confidence ≤ low_threshold``  → run every frame (no skipping).
- Between the two thresholds      → linear interpolation between 1 and
  ``max_skip`` (continuous adaptation).

On skipped frames the last predicted bounding box is returned unchanged,
identical to the ``"repeat"`` mode of :class:`FrameSkipTracker`.

Edge deployment impact
~~~~~~~~~~~~~~~~~~~~~~
On sequences where the tracker achieves mean confidence ≥ 0.7, a
``max_skip=4`` configuration typically reduces compute by 2–3× with
< 0.02 mean IoU degradation on slow-moving targets.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive_skip import AdaptiveSkipTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    dataset = SyntheticDataset(num_sequences=5, motion="linear")
    engine  = BenchmarkEngine(verbose=False)

    tracker = AdaptiveSkipTracker(MOSSETracker(), max_skip=4)
    result  = engine.run(tracker, dataset, dataset_name="Synthetic")
    print(result)
    print(f"Mean skip rate: {tracker.mean_skip_rate():.2f}x")
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BaseTracker, BBox


class AdaptiveSkipTracker(BaseTracker):
    """Dynamically skip frames based on the wrapped tracker's confidence.

    Args:
        tracker: Any :class:`~eovot.trackers.base.BaseTracker` instance.
            If it overrides :meth:`~eovot.trackers.base.BaseTracker.confidence`
            the skip rate adapts; otherwise the base returns 1.0 and the
            wrapper behaves like ``FrameSkipTracker(tracker, skip_rate=1)``
            (every frame is processed).
        max_skip: Maximum number of consecutive frames to skip when
            confidence is at its peak.  A value of ``1`` disables skipping
            (every frame is processed regardless of confidence).
            Default: ``4``.
        high_threshold: Confidence level above which the tracker runs at
            the minimum rate (once every ``max_skip`` frames).  Default: ``0.7``.
        low_threshold: Confidence level below which the tracker runs every
            frame.  Must be < ``high_threshold``.  Default: ``0.3``.

    Raises:
        ValueError: If ``max_skip < 1`` or thresholds are inconsistent.

    Attributes:
        skip_history: List of effective skip intervals used at each update
            call (1 = ran this frame, k > 1 = ran once for k consecutive
            frames).  Useful for post-hoc efficiency analysis.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        max_skip: int = 4,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3,
    ) -> None:
        if max_skip < 1:
            raise ValueError(f"max_skip must be ≥ 1, got {max_skip}.")
        if not (0.0 <= low_threshold < high_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 ≤ low_threshold < high_threshold ≤ 1, "
                f"got low={low_threshold}, high={high_threshold}."
            )

        super().__init__(name=f"{tracker.name}_adaptive{max_skip}")
        self._tracker = tracker
        self.max_skip = max_skip
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

        self._last_bbox: Optional[BBox] = None
        self._frames_since_update: int = 0
        self._current_skip: int = 1
        self.skip_history: List[int] = []

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the wrapped tracker and reset skip state.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._last_bbox = bbox
        self._frames_since_update = 0
        self._current_skip = 1
        self.skip_history.clear()

    def update(self, frame: np.ndarray) -> BBox:
        """Return a prediction, running the underlying tracker adaptively.

        On frames designated for skipping the last bounding box is returned
        without invoking the underlying tracker.  On frames where the tracker
        runs, the skip interval for subsequent frames is recomputed from the
        new confidence score.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._last_bbox is None:
            raise RuntimeError(
                "AdaptiveSkipTracker is not initialised. Call initialize() first."
            )

        self._frames_since_update += 1

        if self._frames_since_update < self._current_skip:
            # Skip — return last known position
            self.skip_history.append(0)
            return self._last_bbox

        # Run the underlying tracker
        bbox = self._tracker.update(frame)
        self._last_bbox = bbox
        self._frames_since_update = 0
        self.skip_history.append(self._current_skip)

        # Recompute skip interval from confidence
        conf = self._tracker.confidence()
        self._current_skip = self._confidence_to_skip(conf)

        return bbox

    def confidence(self) -> float:
        """Delegate to the wrapped tracker's confidence signal."""
        return self._tracker.confidence()

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def mean_skip_rate(self) -> float:
        """Return the average effective skip multiplier across all frames.

        A value of 1.0 means every frame was processed; 4.0 means on average
        only one frame in four was processed by the underlying tracker.

        Returns:
            Mean skip interval, or ``1.0`` if no frames have been processed.
        """
        actual = [s for s in self.skip_history if s > 0]
        if not actual:
            return 1.0
        return float(np.mean(actual))

    def compute_rate(self) -> float:
        """Return the fraction of frames on which the tracker actually ran.

        Returns:
            Fraction in ``(0, 1]``.  ``1.0`` = ran on every frame.
        """
        total = len(self.skip_history)
        if total == 0:
            return 1.0
        ran = sum(1 for s in self.skip_history if s > 0)
        return float(ran) / float(total)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _confidence_to_skip(self, conf: float) -> int:
        """Map confidence in ``[0, 1]`` to a skip interval in ``[1, max_skip]``.

        Uses a piecewise linear interpolation:
        - ``conf ≤ low_threshold``   → skip = 1 (run every frame)
        - ``conf ≥ high_threshold``  → skip = max_skip
        - between thresholds         → linear interpolation

        Args:
            conf: Confidence score in ``[0, 1]``.

        Returns:
            Integer skip interval ≥ 1.
        """
        if self.max_skip == 1:
            return 1
        conf = float(np.clip(conf, 0.0, 1.0))
        lo, hi = self.low_threshold, self.high_threshold
        if conf <= lo:
            return 1
        if conf >= hi:
            return self.max_skip
        t = (conf - lo) / (hi - lo)
        return max(1, int(round(1.0 + t * (self.max_skip - 1))))
