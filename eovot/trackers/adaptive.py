"""Adaptive tracker that switches strategy based on a runtime FPS budget.

``AdaptiveTracker`` wraps two :class:`~eovot.trackers.base.BaseTracker`
instances — a *fast* tracker (high FPS, lower accuracy) and an *accurate*
tracker (lower FPS, higher accuracy) — and dynamically selects which one
to invoke on each frame.  The selection policy is driven by a rolling FPS
estimate compared to a user-defined ``fps_budget``.

Motivation
----------
On edge devices, thermal throttling and competing processes cause available
compute to fluctuate.  A static tracker either over-spends compute when the
device is under load (causing dropped frames or overheating) or
under-utilises available compute at idle.  ``AdaptiveTracker`` gracefully
degrades to the fast tracker under load and upgrades to the accurate tracker
when headroom exists.

Switching policy
----------------
The policy is *conservative by default*:

* **Downgrade** (accurate → fast): triggered immediately when the rolling
  mean FPS drops below ``fps_budget × (1 - hysteresis_factor)``.
* **Upgrade**   (fast → accurate): requires the rolling mean FPS to exceed
  ``fps_budget × (1 + hysteresis_factor)`` for at least
  ``upgrade_window`` consecutive frames before switching.

The hysteresis band prevents rapid oscillation between trackers.

Example
-------
::

    from eovot.trackers.registry import build_tracker
    from eovot.trackers.adaptive import AdaptiveTracker, AdaptiveConfig

    fast     = build_tracker("MOSSE")
    accurate = build_tracker("KCF")

    cfg = AdaptiveConfig(fps_budget=25.0, hysteresis_factor=0.15, upgrade_window=5)
    tracker = AdaptiveTracker(fast_tracker=fast, accurate_tracker=accurate, config=cfg)

    # Use exactly like any other BaseTracker
    tracker.initialize(first_frame, init_bbox)
    for frame in subsequent_frames:
        bbox = tracker.update(frame)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from .base import BaseTracker, BBox


@dataclass
class AdaptiveConfig:
    """Configuration for :class:`AdaptiveTracker`.

    Attributes
    ----------
    fps_budget:
        Target frame-rate in frames per second.  The tracker aims to keep
        mean processing FPS at or above this value by downgrading to the
        fast tracker when load increases.
    window_size:
        Number of recent frames used to compute the rolling FPS estimate.
        Smaller values react faster but are noisier; larger values are
        smoother but slower to respond.  Default: 10.
    hysteresis_factor:
        Fractional dead-band around ``fps_budget`` to prevent rapid
        oscillation.  Downgrade threshold = ``fps_budget × (1 − f)``;
        upgrade threshold = ``fps_budget × (1 + f)``.  Default: 0.15.
    upgrade_window:
        Consecutive frames above the upgrade threshold required before
        switching from fast → accurate.  Prevents premature upgrades
        on brief latency dips.  Default: 5.
    start_accurate:
        If ``True``, begin with the accurate tracker and downgrade if
        needed.  If ``False`` (default), begin with the fast tracker.
    """

    fps_budget: float = 25.0
    window_size: int = 10
    hysteresis_factor: float = 0.15
    upgrade_window: int = 5
    start_accurate: bool = False

    def __post_init__(self) -> None:
        if self.fps_budget <= 0:
            raise ValueError(f"fps_budget must be > 0, got {self.fps_budget}")
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if not 0.0 <= self.hysteresis_factor < 1.0:
            raise ValueError(
                f"hysteresis_factor must be in [0, 1), got {self.hysteresis_factor}"
            )
        if self.upgrade_window < 1:
            raise ValueError(f"upgrade_window must be >= 1, got {self.upgrade_window}")

    @property
    def downgrade_threshold(self) -> float:
        return self.fps_budget * (1.0 - self.hysteresis_factor)

    @property
    def upgrade_threshold(self) -> float:
        return self.fps_budget * (1.0 + self.hysteresis_factor)


class AdaptiveTracker(BaseTracker):
    """FPS-budget-aware tracker that switches between fast and accurate modes.

    Parameters
    ----------
    fast_tracker:
        Low-latency baseline tracker used when the device is under load.
    accurate_tracker:
        Higher-accuracy tracker used when compute headroom is available.
    config:
        Switching policy parameters.  Defaults to 25 FPS budget with
        15% hysteresis.
    name:
        Optional tracker name shown in reports.  Defaults to
        ``"Adaptive(<fast>/<accurate>)"``.
    """

    def __init__(
        self,
        fast_tracker: BaseTracker,
        accurate_tracker: BaseTracker,
        config: Optional[AdaptiveConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        cfg = config or AdaptiveConfig()
        derived_name = name or f"Adaptive({fast_tracker.name}/{accurate_tracker.name})"
        super().__init__(name=derived_name)

        self.fast_tracker = fast_tracker
        self.accurate_tracker = accurate_tracker
        self.config = cfg

        # Runtime state — reset on each initialize() call
        self._use_accurate: bool = cfg.start_accurate
        self._frame_times: Deque[float] = deque(maxlen=cfg.window_size)
        self._upgrade_streak: int = 0
        self._switch_log: list = []  # list of (frame_idx, "fast"/"accurate", rolling_fps)
        self._frame_idx: int = 0
        self._last_t: Optional[float] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise both sub-trackers and reset all runtime state.

        Both trackers are initialised so that a switch can happen at any
        subsequent frame without re-initialisation delay.

        Parameters
        ----------
        frame: BGR ``(H, W, 3)`` uint8 array.
        bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._frame_times.clear()
        self._upgrade_streak = 0
        self._switch_log.clear()
        self._frame_idx = 0
        self._last_t = None
        self._use_accurate = self.config.start_accurate

        self.fast_tracker.initialize(frame, bbox)
        self.accurate_tracker.initialize(frame, bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict target location, selecting tracker based on rolling FPS.

        Parameters
        ----------
        frame: BGR ``(H, W, 3)`` uint8 array.

        Returns
        -------
        Predicted bounding box ``(x, y, w, h)``.
        """
        self._frame_idx += 1

        t_start = time.perf_counter()
        active = self.accurate_tracker if self._use_accurate else self.fast_tracker
        bbox = active.update(frame)
        elapsed = time.perf_counter() - t_start

        self._frame_times.append(elapsed)
        self._apply_policy()

        return bbox

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def current_mode(self) -> str:
        """Return ``"accurate"`` or ``"fast"`` — the mode used on the last frame."""
        return "accurate" if self._use_accurate else "fast"

    @property
    def rolling_fps(self) -> Optional[float]:
        """Rolling mean FPS over the last ``window_size`` frames, or None."""
        if not self._frame_times:
            return None
        mean_s = float(np.mean(list(self._frame_times)))
        return 1.0 / mean_s if mean_s > 0 else float("inf")

    def switch_log(self) -> list:
        """Return a copy of the mode-switch event log.

        Each entry is a dict::

            {"frame": int, "mode": "fast" | "accurate", "rolling_fps": float | None}
        """
        return list(self._switch_log)

    def summary(self) -> dict:
        """Aggregate statistics for the last sequence.

        Returns
        -------
        dict with keys:
            ``total_frames``, ``frames_accurate``, ``frames_fast``,
            ``pct_accurate``, ``num_switches``, ``final_mode``,
            ``final_rolling_fps``.
        """
        total = self._frame_idx
        switches = self._switch_log
        accurate_frames = sum(1 for e in switches if e["mode"] == "accurate")
        fast_frames = total - accurate_frames
        return {
            "total_frames": total,
            "frames_accurate": accurate_frames,
            "frames_fast": fast_frames,
            "pct_accurate": round(accurate_frames / total * 100.0, 2) if total else 0.0,
            "num_switches": len(switches),
            "final_mode": self.current_mode,
            "final_rolling_fps": round(self.rolling_fps, 2) if self.rolling_fps else None,
        }

    # ------------------------------------------------------------------
    # Internal policy
    # ------------------------------------------------------------------

    def _apply_policy(self) -> None:
        fps = self.rolling_fps
        if fps is None:
            return

        if self._use_accurate:
            # Downgrade immediately if FPS falls below lower threshold
            if fps < self.config.downgrade_threshold:
                self._use_accurate = False
                self._upgrade_streak = 0
                self._switch_log.append(
                    {"frame": self._frame_idx, "mode": "fast", "rolling_fps": round(fps, 2)}
                )
        else:
            # Upgrade only after sustained high-FPS frames
            if fps >= self.config.upgrade_threshold:
                self._upgrade_streak += 1
                if self._upgrade_streak >= self.config.upgrade_window:
                    self._use_accurate = True
                    self._upgrade_streak = 0
                    self._switch_log.append(
                        {"frame": self._frame_idx, "mode": "accurate", "rolling_fps": round(fps, 2)}
                    )
            else:
                self._upgrade_streak = 0
