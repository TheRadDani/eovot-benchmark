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

    def active_fps(self, total_fps: float) -> float:
        """Convert benchmark-reported FPS to inner-tracker call rate.

        The benchmark engine measures FPS over **all** frames (active and
        passive).  When ``skip_rate > 1``, passive frames are essentially
        free (a simple bbox lookup / linear extrapolation) so the reported
        FPS is dominated by passive-frame throughput and *overestimates* how
        fast the inner tracker runs.

        This method converts the profiler's ``mean_fps`` back to the rate
        at which the wrapped tracker is actually invoked:

            active_fps = total_fps / skip_rate

        Use ``active_fps`` to compare a frame-skip run against a baseline
        full-rate run in terms of inner-tracker compute load.

        Args:
            total_fps: FPS as reported by :class:`~eovot.profiling.profiler.Profiler`
                (all frames, active + passive).

        Returns:
            Effective call rate of the underlying tracker (frames per second
            at which full tracking updates occur).

        Example::

            result = engine.run(FrameSkipTracker(MOSSETracker(), skip_rate=3), ...)
            inner_rate = fst.active_fps(result.mean_fps)
            # inner_rate ≈ result.mean_fps / 3
        """
        if self.skip_rate <= 0:
            return 0.0
        return total_fps / self.skip_rate

    def theoretical_speedup(
        self, inner_latency_ms: float, passive_overhead_ms: float = 0.0
    ) -> float:
        """Estimate the wall-clock speedup factor from frame skipping.

        Models the per-sequence latency as::

            T_active  = active_frames  × inner_latency_ms
            T_passive = passive_frames × passive_overhead_ms
            T_total   = T_active + T_passive

        Speedup relative to full-rate (all frames active) is::

            speedup = (N × inner_latency_ms) / T_total
                    = skip_rate / (1 + (skip_rate - 1) × passive_overhead_ms
                                         / inner_latency_ms)

        When ``passive_overhead_ms ≈ 0`` (ideal case), speedup = ``skip_rate``.
        When passive frames are non-negligible (e.g., linear extrapolation math
        costs ~10 % of inner_latency_ms), the actual speedup is lower.

        Args:
            inner_latency_ms: Mean per-frame latency of the underlying tracker
                in milliseconds (e.g., from a baseline profiling run).
            passive_overhead_ms: Mean latency of passive frames in milliseconds.
                Use 0.0 to compute the theoretical maximum speedup.  Default: 0.0.

        Returns:
            Speedup factor > 0.  Values > 1.0 indicate the skip-tracker is
            faster than full-rate; < 1.0 indicates overhead dominates (which
            only happens when the inner tracker is cheaper than the skip logic).

        Raises:
            ValueError: If ``inner_latency_ms`` ≤ 0.

        Example::

            # MOSSE runs at 2 ms/frame; passive frames cost 0.01 ms (interpolation)
            fst = FrameSkipTracker(MOSSETracker(), skip_rate=4)
            print(fst.theoretical_speedup(inner_latency_ms=2.0, passive_overhead_ms=0.01))
            # ≈ 3.98  (very close to 4× because passive overhead is tiny)
        """
        if inner_latency_ms <= 0:
            raise ValueError(f"inner_latency_ms must be > 0, got {inner_latency_ms}")
        numerator = self.skip_rate
        denominator = 1.0 + (self.skip_rate - 1) * passive_overhead_ms / inner_latency_ms
        return numerator / denominator

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
