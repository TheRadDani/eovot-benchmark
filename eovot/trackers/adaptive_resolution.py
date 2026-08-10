"""Adaptive-resolution tracker wrapper for edge deployment.

Wraps any :class:`~eovot.trackers.base.BaseTracker` with a dynamic
resolution controller that scales the input frame up or down each update
to meet a target latency (FPS) budget.

Edge devices often cannot sustain full-resolution tracking without
exceeding thermal or latency limits.  Rather than choosing a fixed
downscale factor at design time, :class:`AdaptiveResolutionTracker` uses
an exponential moving average (EMA) of recent frame latencies to decide
— on a frame-by-frame basis — which scale from a predefined pyramid best
keeps throughput within budget.

Algorithm
---------
1. Before each ``update()`` call, compute the EMA latency ``L_ema``.
2. Estimate the latency at each candidate scale as
   ``L_s ≈ L_ema × (s / s_ref)²`` (patch area scales quadratically).
3. Select the **highest scale whose estimated latency ≤ budget**.
4. Resize the frame to that scale, call the inner tracker, then
   map the returned bbox back to the original resolution.
5. Update ``L_ema`` with the measured latency.

The bounding boxes passed to the inner tracker during initialization are
also rescaled so the inner tracker always operates in a consistent
coordinate frame (its own downscaled view).

Scale history
-------------
After a benchmark run, :attr:`scale_history` contains the scale factor
used at each call: the first entry is the sentinel ``0.0`` (initialization
frame), and every subsequent entry is the scale chosen for that update.
Use it to analyze how often the tracker falls back to lower resolution.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive_resolution import AdaptiveResolutionTracker
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    inner = MOSSETracker()
    tracker = AdaptiveResolutionTracker(inner, target_fps=30.0)

    dataset = SyntheticDataset(num_sequences=5, num_frames=100)
    engine = BenchmarkEngine(verbose=True)
    result = engine.run(tracker, dataset, dataset_name="Synthetic")

    stats = tracker.efficiency_summary()
    print(stats)
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


@dataclass
class AdaptiveResolutionStats:
    """Post-run scale distribution statistics for :class:`AdaptiveResolutionTracker`."""

    target_fps: float
    """FPS budget the controller was configured to meet."""

    scale_counts: Dict[float, int]
    """Number of update frames processed at each scale factor."""

    mean_scale: float
    """Mean scale factor across all update frames."""

    min_scale: float
    """Lowest scale factor used in the run."""

    max_scale: float
    """Highest scale factor used in the run (≤ 1.0)."""

    scale_utilization: Dict[float, float]
    """Fraction of update frames at each scale factor."""

    def __str__(self) -> str:
        util_str = "  ".join(
            f"s={s:.2f}:{v:.1%}"
            for s, v in sorted(self.scale_utilization.items(), reverse=True)
        )
        return (
            f"AdaptiveResolutionStats("
            f"target_fps={self.target_fps:.0f}  "
            f"mean_scale={self.mean_scale:.3f}  "
            f"[{util_str}])"
        )


class AdaptiveResolutionTracker(BaseTracker):
    """Wraps any tracker with per-frame dynamic resolution scaling.

    The tracker measures its own latency and adjusts input frame resolution
    each update so throughput stays within ``target_fps``.  When the inner
    tracker runs faster than needed it uses the highest available scale;
    when it is too slow it drops to a coarser scale from the pyramid.

    Args:
        tracker:       Inner :class:`BaseTracker` to wrap.  All actual
                       tracking computations are delegated to it.
        target_fps:    FPS budget.  The controller picks the largest scale
                       whose estimated latency fits inside
                       ``1000 / target_fps`` ms.  Use ``float("inf")`` to
                       always operate at the largest scale (disables
                       adaptation).
        scale_pyramid: Candidate scale factors in ``(0, 1]``, provided in
                       any order — sorted descending internally.
                       Default: ``[1.0, 0.75, 0.5, 0.25]``.
        ema_alpha:     EMA smoothing coefficient for latency estimates.
                       Values close to 1.0 track rapid changes; values
                       close to 0.0 are more stable but sluggish.
                       Range ``(0, 1]``.  Default: ``0.3``.
        warmup_frames: Number of update frames to run at the maximum scale
                       before the controller starts adapting.  This lets
                       the EMA converge before making scale decisions.
                       Default: ``5``.

    Attributes:
        scale_history: Per-frame scale choices appended by :meth:`update`.
                       Entry ``0`` is the sentinel ``0.0`` for the
                       initialization frame; entries ``1…`` are the scale
                       chosen for each subsequent update call.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        target_fps: float = 30.0,
        scale_pyramid: Optional[Sequence[float]] = None,
        ema_alpha: float = 0.3,
        warmup_frames: int = 5,
    ) -> None:
        super().__init__(name=f"AdaptiveRes({tracker.name})")
        self._inner = tracker
        self.target_fps = float(target_fps)
        self.budget_ms = (
            float("inf")
            if self.target_fps == float("inf")
            else (1_000.0 / self.target_fps if self.target_fps > 0.0 else float("inf"))
        )
        self.scale_pyramid: Tuple[float, ...] = tuple(
            sorted(scale_pyramid or [1.0, 0.75, 0.5, 0.25], reverse=True)
        )
        if not self.scale_pyramid or any(s <= 0.0 or s > 1.0 for s in self.scale_pyramid):
            raise ValueError(
                "All scale_pyramid values must be in (0, 1]. "
                f"Got: {self.scale_pyramid}"
            )
        self.ema_alpha = float(ema_alpha)
        self.warmup_frames = int(warmup_frames)

        self._ema_latency_ms: Optional[float] = None
        self._current_scale: float = self.scale_pyramid[0]
        self._frame_count: int = 0
        self._init_hw: Optional[Tuple[int, int]] = None

        self.scale_history: List[float] = []

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialize the inner tracker at the maximum scale.

        Args:
            frame: First frame in original resolution ``(H, W, C)`` or
                   ``(H, W)``.
            bbox:  Initial bounding box ``(x, y, w, h)`` in original
                   pixel coordinates.
        """
        self._init_hw = (frame.shape[0], frame.shape[1])
        self._ema_latency_ms = None
        self._current_scale = self.scale_pyramid[0]
        self._frame_count = 0
        self.scale_history.clear()

        init_scale = self.scale_pyramid[0]
        scaled_frame, scaled_bbox = self._rescale(frame, bbox, init_scale)
        self._inner.initialize(scaled_frame, scaled_bbox)
        self.scale_history.append(0.0)  # sentinel for init frame

    def update(self, frame: np.ndarray) -> BBox:
        """Track the target in *frame* at the dynamically chosen scale.

        Args:
            frame: Current frame in original resolution.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` in **original** pixel
            coordinates (not the downscaled frame).

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._init_hw is None:
            raise RuntimeError(
                "AdaptiveResolutionTracker not initialised. "
                "Call initialize() first."
            )

        scale = self._choose_scale()
        scaled_frame, _ = self._rescale(frame, (0.0, 0.0, 1.0, 1.0), scale)

        t0 = time.perf_counter()
        scaled_pred = self._inner.update(scaled_frame)
        elapsed_ms = (time.perf_counter() - t0) * 1_000.0

        self._update_ema(elapsed_ms)
        self._current_scale = scale
        self._frame_count += 1
        self.scale_history.append(scale)

        return self._upscale_bbox(scaled_pred, scale)

    def reset(self) -> None:
        """Reset all adapter state; forwards ``reset()`` to the inner tracker
        if it supports the method."""
        self._ema_latency_ms = None
        self._current_scale = self.scale_pyramid[0]
        self._frame_count = 0
        self._init_hw = None
        self.scale_history.clear()
        if hasattr(self._inner, "reset"):
            self._inner.reset()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def efficiency_summary(self) -> AdaptiveResolutionStats:
        """Return scale distribution statistics for the current run.

        Only update frames are counted (the initialization-frame sentinel
        ``0.0`` is excluded).

        Returns:
            :class:`AdaptiveResolutionStats` with counts and fractions per
            scale level.

        Raises:
            ValueError: If no update frames have been processed yet.
        """
        update_scales = [s for s in self.scale_history if s > 0.0]
        if not update_scales:
            raise ValueError(
                "No update frames processed yet. Call update() at least once."
            )

        counts = dict(Counter(update_scales))
        total = len(update_scales)
        mean_scale = float(np.mean(update_scales))

        return AdaptiveResolutionStats(
            target_fps=self.target_fps,
            scale_counts=counts,
            mean_scale=mean_scale,
            min_scale=float(min(update_scales)),
            max_scale=float(max(update_scales)),
            scale_utilization={s: c / total for s, c in counts.items()},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _choose_scale(self) -> float:
        """Return the best scale for the next update call.

        During the warmup window (or before any latency data exists) always
        use the largest scale.  Afterwards, pick the highest scale whose
        cost estimate fits the latency budget.

        Cost is estimated as ``L_ema × (s / s_ref)²`` where ``s_ref`` is
        the scale used on the most recent frame.
        """
        if (
            self._ema_latency_ms is None
            or self._frame_count < self.warmup_frames
            or self.budget_ms == float("inf")
        ):
            return self.scale_pyramid[0]

        ref_scale = self._current_scale if self._current_scale > 0.0 else self.scale_pyramid[0]
        for s in self.scale_pyramid:
            estimated_ms = self._ema_latency_ms * (s / ref_scale) ** 2
            if estimated_ms <= self.budget_ms:
                return s

        return self.scale_pyramid[-1]

    def _update_ema(self, latency_ms: float) -> None:
        a = self.ema_alpha
        if self._ema_latency_ms is None:
            self._ema_latency_ms = latency_ms
        else:
            self._ema_latency_ms = a * latency_ms + (1.0 - a) * self._ema_latency_ms

    @staticmethod
    def _rescale(
        frame: np.ndarray, bbox: BBox, scale: float
    ) -> Tuple[np.ndarray, BBox]:
        """Resize *frame* and *bbox* by *scale* (noop when scale == 1.0)."""
        if abs(scale - 1.0) < 1e-6:
            return frame, bbox
        h, w = frame.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        x, y, bw, bh = bbox
        return resized, (x * scale, y * scale, bw * scale, bh * scale)

    @staticmethod
    def _upscale_bbox(bbox: BBox, scale: float) -> BBox:
        """Map *bbox* from scaled coordinates back to original resolution."""
        if abs(scale - 1.0) < 1e-6 or scale <= 0.0:
            return bbox
        inv = 1.0 / scale
        x, y, w, h = bbox
        return (x * inv, y * inv, w * inv, h * inv)
