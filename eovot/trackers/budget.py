"""AdaptiveBudgetTracker — FPS-budget-constrained tracking for edge deployment.

Edge devices operate under strict latency SLAs: a robot controller may require
30 FPS, a surveillance pipeline 15 FPS, an autonomous system 60 FPS.  The
challenge is that the right skip rate to hit a given FPS budget depends on the
underlying tracker's speed, which varies by hardware and tracker type.

:class:`AdaptiveBudgetTracker` wraps any :class:`~.base.BaseTracker` and
dynamically adjusts its skip rate after every ``adapt_interval`` active frames
so that the effective throughput converges to ``fps_target``.

How it works
------------
During each *active frame* the wrapper:

1. Times the underlying tracker's ``update()`` call.
2. Maintains a rolling window of the last ``window`` active-frame latencies.
3. Estimates the current effective FPS::

       effective_fps = skip_rate × 1000 / mean_active_latency_ms

4. After ``adapt_interval`` active frames, compares ``effective_fps`` to
   ``fps_target`` and increments / decrements ``skip_rate`` within
   ``[min_skip, max_skip]`` to close the gap.

*Passive frames* (those between active frames) receive a linearly-extrapolated
bounding box from the last two active predictions, identical to
:class:`~.frame_skip.FrameSkipTracker` in ``"linear"`` mode.

Relationship to :class:`~.frame_skip.FrameSkipTracker`
-------------------------------------------------------
``FrameSkipTracker`` uses a *fixed* skip rate chosen before the run.
``AdaptiveBudgetTracker`` discovers the correct skip rate *online* by measuring
actual hardware latency, making it portable across devices without manual
per-device tuning.  It is therefore the recommended choice for production
edge deployment where the device performance is not known a priori.

Example::

    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.budget import AdaptiveBudgetTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    # Constrain KCF to deliver at most 60 FPS effective throughput
    tracker = AdaptiveBudgetTracker(KCFTracker(), fps_target=60.0)

    dataset = SyntheticDataset(num_sequences=5, num_frames=100)
    engine  = BenchmarkEngine(verbose=True)
    result  = engine.run(tracker, dataset, dataset_name="Synthetic-budget60")
    print(result.mean_fps, result.mean_iou)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional

import numpy as np

from .base import BaseTracker, BBox


class AdaptiveBudgetTracker(BaseTracker):
    """Wrap any tracker and dynamically adjust skip rate to hit an FPS target.

    The skip rate starts at ``initial_skip`` and is updated every
    ``adapt_interval`` active frames.  Between active frames, a linear
    extrapolation of the last two active positions is returned (zero compute
    overhead, one addition per dimension).

    Args:
        tracker:        Any :class:`BaseTracker` to wrap.
        fps_target:     Target effective frames-per-second throughput.
                        Must be positive.  The adapter tries to hold the
                        effective FPS within ±10 % of this value at steady
                        state.
        min_skip:       Minimum allowed skip rate (≥ 1).  ``1`` means every
                        frame is an active frame.
        max_skip:       Maximum allowed skip rate.  Caps compute savings at
                        ``1 / max_skip`` of full-rate cost.  Default: ``8``.
        initial_skip:   Skip rate used before enough data has been collected
                        for the first adaptation step.  Default: ``1``.
        adapt_interval: Number of *active* frames between skip-rate updates.
                        Smaller values react faster but are noisier.
                        Default: ``10``.
        window:         Size of the rolling-average window for latency
                        estimation.  Default: ``20`` active frames.

    Raises:
        ValueError: If ``fps_target`` ≤ 0, ``min_skip`` < 1, or
                    ``max_skip`` < ``min_skip``.

    Attributes:
        current_skip_rate:  Current skip rate (updated after each adaptation).
        active_frame_count: Number of frames on which the underlying tracker ran.
        skipped_frame_count: Number of frames that were propagated without a
            tracker update.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        fps_target: float = 30.0,
        min_skip: int = 1,
        max_skip: int = 8,
        initial_skip: int = 1,
        adapt_interval: int = 10,
        window: int = 20,
    ) -> None:
        if fps_target <= 0:
            raise ValueError(f"fps_target must be positive, got {fps_target}")
        if min_skip < 1:
            raise ValueError(f"min_skip must be >= 1, got {min_skip}")
        if max_skip < min_skip:
            raise ValueError(f"max_skip ({max_skip}) must be >= min_skip ({min_skip})")
        if adapt_interval < 1:
            raise ValueError(f"adapt_interval must be >= 1, got {adapt_interval}")

        name = f"{tracker.name}_budget{fps_target:.0f}fps"
        super().__init__(name=name)

        self._tracker = tracker
        self.fps_target = fps_target
        self._target_ms: float = 1000.0 / fps_target
        self.min_skip = min_skip
        self.max_skip = max_skip
        self._adapt_interval = adapt_interval

        self.current_skip_rate: int = max(min_skip, min(initial_skip, max_skip))
        self._latency_window: Deque[float] = deque(maxlen=window)

        self._frame_idx: int = 0
        self._active_count: int = 0
        self._last_bbox: Optional[BBox] = None
        self._prev_bbox: Optional[BBox] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the underlying tracker and reset all internal state.

        Args:
            frame: First frame as a ``(H, W, 3)`` BGR uint8 array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._frame_idx = 0
        self._active_count = 0
        self._last_bbox = bbox
        self._prev_bbox = bbox
        self._latency_window.clear()
        self.current_skip_rate = max(self.min_skip, min(self.current_skip_rate, self.max_skip))

    def update(self, frame: np.ndarray) -> BBox:
        """Return a bounding-box prediction and adapt the skip rate.

        Active frames (``frame_idx % skip_rate == 0``) call the underlying
        tracker and measure its latency.  Passive frames return an extrapolated
        position without calling the tracker.

        Args:
            frame: Current frame as a ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._last_bbox is None:
            raise RuntimeError(
                "AdaptiveBudgetTracker has not been initialised. "
                "Call initialize() first."
            )

        self._frame_idx += 1

        if self._frame_idx % self.current_skip_rate == 0:
            # Active frame: run the underlying tracker with latency measurement.
            t0 = time.perf_counter()
            prev = self._last_bbox
            self._last_bbox = self._tracker.update(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1_000.0
            self._prev_bbox = prev

            self._latency_window.append(elapsed_ms)
            self._active_count += 1

            # Adapt skip rate every adapt_interval active frames.
            if self._active_count % self._adapt_interval == 0 and self._latency_window:
                self._adapt()
        else:
            # Passive frame: linear extrapolation, zero tracker overhead.
            self._last_bbox = self._extrapolate()

        return self._last_bbox

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_frame_count(self) -> int:
        """Number of frames on which the underlying tracker actually ran."""
        return self._active_count

    @property
    def skipped_frame_count(self) -> int:
        """Number of frames that were propagated without a tracker update."""
        return self._frame_idx - self._active_count

    @property
    def effective_skip_fraction(self) -> float:
        """Fraction of frames that were skipped (0 = none, 1 = all)."""
        if self._frame_idx == 0:
            return 0.0
        return self.skipped_frame_count / self._frame_idx

    @property
    def underlying_tracker(self) -> BaseTracker:
        """The wrapped :class:`BaseTracker` instance."""
        return self._tracker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adapt(self) -> None:
        """Adjust ``current_skip_rate`` to move effective FPS towards ``fps_target``."""
        mean_ms = float(np.mean(self._latency_window))
        if mean_ms <= 0:
            return

        # Effective FPS = skip_rate / (mean_active_latency_ms / 1000)
        effective_fps = self.current_skip_rate * 1_000.0 / mean_ms

        # Allow ±10 % tolerance to avoid oscillation at steady state.
        if effective_fps < self.fps_target * 0.90:
            # Behind target: process fewer frames by increasing skip rate.
            self.current_skip_rate = min(self.current_skip_rate + 1, self.max_skip)
        elif effective_fps > self.fps_target * 1.10 and self.current_skip_rate > self.min_skip:
            # Comfortably ahead: process more frames by decreasing skip rate.
            self.current_skip_rate = max(self.current_skip_rate - 1, self.min_skip)

    def _extrapolate(self) -> BBox:
        """Linearly extrapolate the centre position from the last two active frames.

        Box dimensions (w, h) are held constant.  When only one active frame
        has been seen (velocity unknown), the last prediction is repeated.
        """
        if self._prev_bbox is None or self._last_bbox is None:
            return self._last_bbox or (0.0, 0.0, 0.0, 0.0)

        px, py, pw, ph = self._prev_bbox
        lx, ly, lw, lh = self._last_bbox

        vcx = (lx + lw / 2.0) - (px + pw / 2.0)
        vcy = (ly + lh / 2.0) - (py + ph / 2.0)

        new_cx = (lx + lw / 2.0) + vcx
        new_cy = (ly + lh / 2.0) + vcy
        return (new_cx - lw / 2.0, new_cy - lh / 2.0, lw, lh)

    def __repr__(self) -> str:
        return (
            f"AdaptiveBudgetTracker("
            f"tracker={self._tracker!r}, "
            f"fps_target={self.fps_target}, "
            f"skip_rate={self.current_skip_rate}, "
            f"min_skip={self.min_skip}, "
            f"max_skip={self.max_skip})"
        )
