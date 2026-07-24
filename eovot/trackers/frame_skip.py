"""FrameSkipTracker — adaptive frame-skipping wrapper for any BaseTracker.

A key edge-deployment optimisation is to skip frames when the target is
moving slowly or the scene is stable, running the full tracker only every
*k*-th frame and propagating the last prediction for the frames in between.
This reduces average compute cost by up to *k*× at the cost of some accuracy
degradation on fast-moving sequences.

:class:`FrameSkipTracker` wraps *any* :class:`~eovot.trackers.base.BaseTracker`
and implements two skip strategies:

``"repeat"`` (default)
    Skipped frames return the last computed bounding box unchanged.  Simple,
    zero-latency overhead, but introduces position lag on fast targets.

``"linear"``
    Skipped frames receive a linearly extrapolated position based on the
    velocity vector estimated from the last two tracker updates.  More accurate
    than repeat at moderate skip rates; degrades on non-linear motion.

The wrapped tracker's ``name`` is preserved with a ``_skip{k}`` suffix so
that :class:`~eovot.benchmark.engine.BenchmarkEngine` reports it distinctly.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.frame_skip import FrameSkipTracker
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=3)
    engine  = BenchmarkEngine(verbose=False)

    # Run at full rate (baseline)
    baseline = engine.run(MOSSETracker(), dataset, dataset_name="Syn")

    # Run at 3× reduced compute — process every 3rd frame
    skipped  = engine.run(
        FrameSkipTracker(MOSSETracker(), skip_rate=3),
        dataset,
        dataset_name="Syn",
    )
    print(f"FPS gain: {skipped.mean_fps / baseline.mean_fps:.1f}×")
    print(f"IoU cost: {baseline.mean_iou - skipped.mean_iou:.4f}")
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox

SkipMode = Literal["repeat", "linear"]


class FrameSkipTracker(BaseTracker):
    """Wrap a tracker to process only every *skip_rate*-th frame.

    The wrapped tracker's ``initialize()`` and ``update()`` are called at the
    normal rate as far as the benchmark engine is concerned; internally, the
    heavy ``update()`` is forwarded to the underlying tracker only on
    *active frames* (those whose 1-indexed count is divisible by
    ``skip_rate``).  All other frames are *passive* and receive a propagated
    prediction without calling the underlying tracker.

    Args:
        tracker:   Any :class:`BaseTracker` to wrap.
        skip_rate: Process 1 in every ``skip_rate`` frames (≥ 1).
            ``skip_rate=1`` is a no-op (all frames are active).
            ``skip_rate=2`` halves compute; ``skip_rate=4`` quarters it.
        mode:      Skip strategy.  ``"repeat"`` returns the last active
            prediction unchanged; ``"linear"`` extrapolates position from
            the velocity of the last two active updates.

    Raises:
        ValueError: If *skip_rate* < 1.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        skip_rate: int = 2,
        mode: SkipMode = "repeat",
    ) -> None:
        if skip_rate < 1:
            raise ValueError(f"skip_rate must be >= 1, got {skip_rate}")
        super().__init__(name=f"{tracker.name}_skip{skip_rate}")
        self._tracker = tracker
        self.skip_rate = skip_rate
        self.mode: SkipMode = mode

        self._frame_idx: int = 0
        self._last_bbox: Optional[BBox] = None
        self._prev_bbox: Optional[BBox] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the underlying tracker and reset the frame counter.

        Args:
            frame: First frame as a ``(H, W, 3)`` BGR uint8 array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._frame_idx = 0
        self._last_bbox = bbox
        self._prev_bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        """Return a bounding-box prediction, possibly from a cached frame.

        The underlying tracker's ``update()`` is called only on *active*
        frames.  Passive frames return the last active prediction
        (``"repeat"`` mode) or a linearly extrapolated position
        (``"linear"`` mode).

        Args:
            frame: Current frame as a ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._last_bbox is None:
            raise RuntimeError(
                "FrameSkipTracker has not been initialised. Call initialize() first."
            )

        self._frame_idx += 1

        if self._frame_idx % self.skip_rate == 0:
            # Active frame: run the real tracker
            prev = self._last_bbox
            self._last_bbox = self._tracker.update(frame)
            self._prev_bbox = prev
        else:
            # Passive frame: propagate without running the tracker
            if self.mode == "linear":
                self._last_bbox = self._extrapolate()
            # else "repeat": self._last_bbox unchanged

        return self._last_bbox

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def underlying_tracker(self) -> BaseTracker:
        """The wrapped :class:`BaseTracker` instance."""
        return self._tracker

    @property
    def active_frame_count(self) -> int:
        """Number of frames on which the underlying tracker actually ran."""
        return self._frame_idx // self.skip_rate

    @property
    def skipped_frame_count(self) -> int:
        """Number of frames that were propagated without a tracker update."""
        return self._frame_idx - self.active_frame_count

    @property
    def effective_skip_fraction(self) -> float:
        """Fraction of frames that were skipped (0.0 = no skipping, 1.0 = all).

        For a stationary tracker (``skip_rate=1``) this is exactly 0.0.
        """
        if self._frame_idx == 0:
            return 0.0
        return self.skipped_frame_count / self._frame_idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extrapolate(self) -> BBox:
        """Linearly extrapolate from the velocity of the last two active frames.

        Velocity is the displacement of the bounding-box *centre* between
        the last two active predictions.  The centre is shifted by this
        velocity; box dimensions remain constant.

        Returns:
            Extrapolated ``(x, y, w, h)`` bounding box.
        """
        if self._prev_bbox is None or self._last_bbox is None:
            return self._last_bbox or (0.0, 0.0, 0.0, 0.0)

        px, py, pw, ph = self._prev_bbox
        lx, ly, lw, lh = self._last_bbox

        # Velocity of the centre between the last two active frames
        vcx = (lx + lw / 2.0) - (px + pw / 2.0)
        vcy = (ly + lh / 2.0) - (py + ph / 2.0)

        # Extrapolate by one additional frame
        new_cx = (lx + lw / 2.0) + vcx
        new_cy = (ly + lh / 2.0) + vcy

        return (new_cx - lw / 2.0, new_cy - lh / 2.0, lw, lh)
