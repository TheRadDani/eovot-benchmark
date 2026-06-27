"""Adaptive frame-skip wrapper for edge-constrained deployment.

On resource-limited hardware (Raspberry Pi, Jetson Nano, microcontrollers)
running a full tracker update on every frame often exceeds the compute budget.
:class:`FrameSkipAdapter` wraps any :class:`BaseTracker` and executes the
underlying tracker only every ``skip_interval`` frames, filling the remaining
frames with a cheap interpolation or velocity-extrapolation estimate.

This technique trades a small accuracy loss for a proportional reduction in
CPU/GPU usage — a fundamental building block for real-world edge deployment.

Reference
---------
Henriques et al. (2015) used frame skip implicitly in high-speed tracking;
the adaptive variant here is inspired by rate-control ideas from video codecs.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional

import numpy as np

from .base import BaseTracker, BBox

_ADAPTIVE_WINDOW = 10   # number of recent latencies used for adaptation
_MAX_SKIP = 16          # safety cap on dynamically-computed skip interval


class FrameSkipAdapter(BaseTracker):
    """Wraps any ``BaseTracker`` with configurable frame-skip for edge deployment.

    Two operating modes:

    * **Static** (default): the underlying tracker runs exactly every
      ``skip_interval`` frames; other frames receive an interpolated estimate.
    * **Adaptive** (``target_fps`` is set): the skip interval is adjusted
      online after every real tracker call so that effective throughput
      converges toward ``target_fps``.  The skip interval is recomputed as::

          skip ≈ ceil(tracker_fps / target_fps)

      and capped at :data:`_MAX_SKIP` to prevent stale predictions.

    Skipped frames are filled by one of two strategies controlled by the
    ``interpolation`` parameter:

    * ``"linear"`` — linearly blend the two most recently tracked positions.
      Accurate for slow, steady motion.
    * ``"velocity"`` — extrapolate using the displacement between the last two
      tracked positions.  Handles faster, consistent motion better.

    The adapter is itself a :class:`BaseTracker`, so it plugs into
    :class:`~eovot.benchmark.engine.BenchmarkEngine` without any changes::

        from eovot.trackers.frame_skip import FrameSkipAdapter
        from eovot.trackers.mosse import MOSSETracker
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        # Static: run MOSSE only every 3rd frame
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=3)

        # Adaptive: tune skip so effective throughput ≈ 30 FPS
        adapter = FrameSkipAdapter(MOSSETracker(), target_fps=30.0)

        engine = BenchmarkEngine(verbose=True)
        result = engine.run(adapter, SyntheticDataset(), dataset_name="demo")

    Args:
        tracker: The wrapped :class:`BaseTracker` instance.
        skip_interval: Frames between real tracker updates (1 = no skip).
            Ignored when ``target_fps`` is provided.  Default: ``2``.
        target_fps: When set, the skip interval is adjusted online to meet
            this throughput target (frames per second).  Default: ``None``.
        interpolation: Fill strategy for skipped frames: ``"linear"``
            (default) or ``"velocity"``.
        name: Optional display name for benchmark reports.  Defaults to
            ``"<inner_name>-Skip<N>"``.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        skip_interval: int = 2,
        target_fps: Optional[float] = None,
        interpolation: str = "linear",
        name: Optional[str] = None,
    ) -> None:
        if skip_interval < 1:
            raise ValueError(f"skip_interval must be >= 1, got {skip_interval}.")
        if interpolation not in ("linear", "velocity"):
            raise ValueError(
                f"interpolation must be 'linear' or 'velocity', got {interpolation!r}."
            )

        display_name = name or f"{tracker.name}-Skip{skip_interval}"
        super().__init__(name=display_name)

        self._tracker = tracker
        self._base_skip = skip_interval
        self._target_fps = target_fps
        self._interpolation = interpolation

        # Runtime state — reset in initialize()
        self._frame_count: int = 0
        self._frames_since_update: int = 0
        self._last_bbox: Optional[BBox] = None
        self._prev_bbox: Optional[BBox] = None
        self._current_skip: int = skip_interval
        self._update_latencies: Deque[float] = deque(maxlen=_ADAPTIVE_WINDOW)

    # ------------------------------------------------------------------ #
    # BaseTracker interface                                                #
    # ------------------------------------------------------------------ #

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the inner tracker and reset all frame-skip state.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._last_bbox = bbox
        self._prev_bbox = bbox
        self._frame_count = 0
        self._frames_since_update = 0
        self._current_skip = self._base_skip
        self._update_latencies.clear()

    def update(self, frame: np.ndarray) -> BBox:
        """Return a predicted bounding box, running the inner tracker only when needed.

        On non-update frames the prediction is filled by the configured
        interpolation strategy without calling the inner tracker at all.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._last_bbox is None:
            raise RuntimeError(
                "FrameSkipAdapter not initialised. Call initialize() before update()."
            )

        self._frame_count += 1
        self._frames_since_update += 1

        if self._frames_since_update >= self._current_skip:
            return self._do_tracker_update(frame)

        return self._fill_skipped()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _do_tracker_update(self, frame: np.ndarray) -> BBox:
        """Run the inner tracker, record latency, and adapt skip if needed."""
        t0 = time.perf_counter()
        new_bbox = self._tracker.update(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1_000.0

        self._update_latencies.append(elapsed_ms)
        self._prev_bbox = self._last_bbox
        self._last_bbox = new_bbox
        self._frames_since_update = 0

        if self._target_fps is not None and len(self._update_latencies) >= 5:
            self._adapt_skip()

        return new_bbox

    def _adapt_skip(self) -> None:
        """Recompute skip interval so throughput converges to ``target_fps``."""
        mean_ms = float(np.mean(self._update_latencies))
        tracker_fps = 1_000.0 / max(mean_ms, 1e-6)
        ideal_skip = max(1, int(np.ceil(tracker_fps / max(self._target_fps, 1e-6))))
        self._current_skip = min(ideal_skip, _MAX_SKIP)

    def _fill_skipped(self) -> BBox:
        """Return an estimated bbox for the current skipped frame."""
        if self._interpolation == "velocity":
            return self._velocity_extrapolate()
        return self._linear_interpolate()

    def _linear_interpolate(self) -> BBox:
        """Linearly blend previous and last tracked positions."""
        alpha = self._frames_since_update / max(self._current_skip, 1)
        x0, y0, w0, h0 = self._prev_bbox
        x1, y1, w1, h1 = self._last_bbox
        return (
            x0 + alpha * (x1 - x0),
            y0 + alpha * (y1 - y0),
            w0 + alpha * (w1 - w0),
            h0 + alpha * (h1 - h0),
        )

    def _velocity_extrapolate(self) -> BBox:
        """Extrapolate from last tracked position using inter-frame velocity."""
        x0, y0, w0, h0 = self._prev_bbox
        x1, y1, w1, h1 = self._last_bbox
        # Velocity per frame, estimated from the last skip window
        scale = 1.0 / max(self._current_skip, 1)
        dx = (x1 - x0) * scale
        dy = (y1 - y0) * scale
        dw = (w1 - w0) * scale
        dh = (h1 - h0) * scale
        t = float(self._frames_since_update)
        return (x1 + dx * t, y1 + dy * t, w1 + dw * t, h1 + dh * t)

    # ------------------------------------------------------------------ #
    # Diagnostics / inspection                                             #
    # ------------------------------------------------------------------ #

    @property
    def effective_skip(self) -> int:
        """Current skip interval (may differ from ``skip_interval`` in adaptive mode)."""
        return self._current_skip

    @property
    def mean_update_latency_ms(self) -> Optional[float]:
        """Mean latency of real tracker calls in milliseconds, or ``None`` before first update."""
        return float(np.mean(self._update_latencies)) if self._update_latencies else None

    @property
    def tracker_call_fraction(self) -> float:
        """Fraction of frames (after init) that triggered a real tracker update.

        A value of ``1/skip_interval`` is expected in static mode; lower in
        adaptive mode when the target FPS is much less than the tracker's
        native throughput.
        """
        if self._frame_count == 0:
            return 1.0
        return len(self._update_latencies) / max(self._frame_count, 1)

    def __repr__(self) -> str:
        mode = (
            f"target_fps={self._target_fps}"
            if self._target_fps is not None
            else f"skip={self._current_skip}"
        )
        return (
            f"FrameSkipAdapter({self._tracker.name!r}, {mode}, "
            f"interpolation={self._interpolation!r})"
        )
