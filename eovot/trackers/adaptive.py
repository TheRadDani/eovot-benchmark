"""
LatencyBudgetTracker — adaptive compute-allocation wrapper for any BaseTracker.

The central challenge in edge deployment is that tracker latency varies with
scene complexity, motion speed, and hardware temperature.  A fixed skip rate
optimised for one condition under- or over-spends compute in another.

:class:`LatencyBudgetTracker` closes this gap by measuring actual per-frame
latency and dynamically adjusting the frame-skip rate to steer FPS toward a
user-specified target.  When the tracker is too slow (projected FPS drops
below the budget), skip rate increases (coarser but cheaper).  When headroom
opens up, skip rate decreases (finer, more accurate).

This makes the tracker *self-regulating*: it achieves the best accuracy
possible within the imposed computational budget, adapting to workload and
thermal throttling without user intervention.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive import LatencyBudgetTracker, AdaptiveConfig

    config  = AdaptiveConfig(target_fps=30.0, min_skip=1, max_skip=6)
    tracker = LatencyBudgetTracker(MOSSETracker(), config)

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    engine  = BenchmarkEngine(verbose=False)
    dataset = SyntheticDataset(num_sequences=5)
    result  = engine.run(tracker, dataset, dataset_name="Syn")
    print(f"Mean FPS  : {result.mean_fps:.1f}  (target: {config.target_fps})")
    print(f"Adaptations: {tracker.state.total_adaptations}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from .base import BaseTracker, BBox


@dataclass
class AdaptiveConfig:
    """Policy parameters for :class:`LatencyBudgetTracker`.

    Args:
        target_fps:      Desired output frame rate in frames per second.
        min_skip:        Lowest allowed skip rate (1 = every frame processed).
        max_skip:        Highest allowed skip rate (caps compute reduction).
        ema_alpha:       EMA smoothing weight for per-frame latency estimates.
                         Larger values react faster to sudden latency changes.
        tolerance:       Fractional FPS band around *target_fps* treated as
                         "within budget" — adaptation is suppressed inside
                         ``target × [1-tol, 1+tol]``.
        cooldown_frames: Minimum active frames between consecutive skip-rate
                         changes.  Prevents oscillation on noisy latency signals.
    """

    target_fps: float
    min_skip: int = 1
    max_skip: int = 8
    ema_alpha: float = 0.20
    tolerance: float = 0.10
    cooldown_frames: int = 10

    def __post_init__(self) -> None:
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if self.min_skip < 1:
            raise ValueError("min_skip must be >= 1")
        if self.max_skip < self.min_skip:
            raise ValueError("max_skip must be >= min_skip")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.tolerance < 0:
            raise ValueError("tolerance must be >= 0")


@dataclass
class AdaptationEvent:
    """Records a single skip-rate change for post-hoc analysis."""

    frame_idx: int
    old_skip: int
    new_skip: int
    ema_latency_ms: float
    projected_fps: float


@dataclass
class AdaptiveState:
    """Snapshot of the current adaptation state.

    Returned by :attr:`LatencyBudgetTracker.state` for inspection and logging.
    """

    current_skip: int
    ema_latency_ms: float
    projected_fps: float
    frames_since_adapt: int
    total_adaptations: int
    history: List[AdaptationEvent] = field(default_factory=list)


class LatencyBudgetTracker(BaseTracker):
    """Wrapper that adapts frame-skip rate at runtime to hit a target FPS.

    On each *active* frame (one the inner tracker processes) the wall-clock
    latency is measured and folded into an exponential moving average (EMA).
    If the EMA-projected FPS falls outside the ``target ± tolerance`` band and
    the cooldown has elapsed, the skip rate is incremented or decremented by 1.

    Skipped (passive) frames return the last known bounding box without
    calling the underlying tracker, identical to
    :class:`~eovot.trackers.frame_skip.FrameSkipTracker` in ``"repeat"`` mode.

    The adaptation history is available via :attr:`state` for benchmarking
    and analysis — e.g. to understand how often thermal throttling forced a
    quality reduction on a given device.

    Args:
        tracker: The inner :class:`BaseTracker` to wrap.
        config:  Adaptation policy.  If omitted a default policy targeting
                 30 FPS with skip range [1, 8] is used.

    Example::

        cfg     = AdaptiveConfig(target_fps=25.0, min_skip=1, max_skip=4)
        adapted = LatencyBudgetTracker(KCFTracker(), cfg)
    """

    def __init__(
        self,
        tracker: BaseTracker,
        config: Optional[AdaptiveConfig] = None,
    ) -> None:
        if config is None:
            config = AdaptiveConfig(target_fps=30.0)
        super().__init__(name=f"Adaptive({tracker.name})")
        self._inner = tracker
        self._cfg = config

        self._skip: int = config.min_skip
        # Seed EMA at the per-frame budget so the first active frame does not
        # over-react before enough measurements have been collected.
        self._ema_ms: float = 1000.0 / config.target_fps
        self._frame_idx: int = 0
        self._active_frame_count: int = 0
        self._frames_since_adapt: int = 0
        self._total_adaptations: int = 0
        self._history: List[AdaptationEvent] = []
        self._last_bbox: Optional[BBox] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame, bbox: BBox) -> None:
        """Initialise the inner tracker and reset adaptation state.

        Args:
            frame: First video frame as a ``(H, W, 3)`` BGR uint8 array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._inner.initialize(frame, bbox)
        self._last_bbox = bbox
        self._frame_idx = 0
        self._active_frame_count = 0
        self._frames_since_adapt = 0
        self._skip = self._cfg.min_skip

    def update(self, frame) -> BBox:
        """Return a bounding-box prediction, adapting skip rate as needed.

        Active frames (those whose 1-based index is divisible by the current
        skip rate) call the inner tracker, measure latency, and potentially
        update the skip rate.  Passive frames return the cached bounding box
        without invoking the inner tracker.

        Args:
            frame: Current frame as a ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._last_bbox is None:
            raise RuntimeError(
                "LatencyBudgetTracker has not been initialised.  "
                "Call initialize() first."
            )

        self._frame_idx += 1

        if self._frame_idx % self._skip == 0:
            t0 = time.perf_counter()
            bbox = self._inner.update(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1_000.0

            self._last_bbox = bbox
            self._active_frame_count += 1
            self._frames_since_adapt += 1
            self._update_ema(elapsed_ms)
            self._maybe_adapt()
        # Passive frame: return cached bbox unchanged

        return self._last_bbox

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def state(self) -> AdaptiveState:
        """Current adaptation state snapshot (history is a shallow copy)."""
        return AdaptiveState(
            current_skip=self._skip,
            ema_latency_ms=self._ema_ms,
            projected_fps=self.projected_fps,
            frames_since_adapt=self._frames_since_adapt,
            total_adaptations=self._total_adaptations,
            history=list(self._history),
        )

    @property
    def underlying_tracker(self) -> BaseTracker:
        """The wrapped :class:`BaseTracker` instance."""
        return self._inner

    @property
    def projected_fps(self) -> float:
        """Estimated output FPS at the current skip rate and EMA latency.

        Since the inner tracker runs every ``skip`` frames, the effective FPS
        seen by the output stream is ``skip × (1000 / ema_latency_ms)``.
        """
        return (self._skip * 1_000.0) / max(self._ema_ms, 1e-9)

    @property
    def active_frame_count(self) -> int:
        """Number of frames on which the inner tracker actually ran."""
        return self._active_frame_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_ema(self, latency_ms: float) -> None:
        alpha = self._cfg.ema_alpha
        self._ema_ms = alpha * latency_ms + (1.0 - alpha) * self._ema_ms

    def _maybe_adapt(self) -> None:
        """Increment or decrement skip rate if FPS is outside the target band."""
        if self._frames_since_adapt < self._cfg.cooldown_frames:
            return

        proj = self.projected_fps
        lo = self._cfg.target_fps * (1.0 - self._cfg.tolerance)
        hi = self._cfg.target_fps * (1.0 + self._cfg.tolerance)

        old_skip = self._skip
        if proj < lo and self._skip < self._cfg.max_skip:
            # Running too slow — skip more frames to save compute
            self._skip += 1
        elif proj > hi and self._skip > self._cfg.min_skip:
            # Extra headroom — reduce skip for better accuracy
            self._skip -= 1
        else:
            return  # within tolerance, no change needed

        self._history.append(
            AdaptationEvent(
                frame_idx=self._frame_idx,
                old_skip=old_skip,
                new_skip=self._skip,
                ema_latency_ms=self._ema_ms,
                projected_fps=proj,
            )
        )
        self._total_adaptations += 1
        self._frames_since_adapt = 0
