"""Adaptive frame-skip controller for edge-constrained tracker deployment.

Wraps any :class:`~eovot.trackers.base.BaseTracker` with a motion-adaptive
inference scheduler that skips tracker updates on low-motion frames,
propagating the last bounding box instead.  This trades a small amount of
accuracy for a significant reduction in compute — a practical strategy for
running classical trackers at consistent throughput on CPU-constrained edge
devices such as the Raspberry Pi or Jetson Nano.

Skip strategy
~~~~~~~~~~~~~
At each frame, the controller estimates scene motion from the mean absolute
pixel difference between the current and previous grayscale frame.  When the
motion score falls below ``skip_threshold`` *and* fewer than
``min_active_interval`` frames have elapsed since the last active update, the
tracker update is skipped and the last known bounding box is returned.

Optionally, a constant-velocity predictor extrapolates the bounding-box
position during skipped frames (``use_velocity=True``), reducing drift when
the target moves at a roughly constant rate between active updates.

Throughput metrics captured
~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the run, :meth:`skip_stats` returns a :class:`SkipStats` object that
reports:

- ``total_frames``    — frames processed (update calls, excluding init).
- ``skipped_frames``  — frames where inference was skipped.
- ``skip_ratio``      — ``skipped / total``.
- ``inference_fps``   — tracker's own throughput (active frames only).
- ``effective_fps``   — overall pipeline throughput across all frames.
- ``mean_motion_score`` — mean frame-difference score.

These statistics complement :class:`~eovot.profiling.profiler.ProfilingResult`
and are useful for reporting throughput gains in edge deployment tables.

Typical usage::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive_skip import AdaptiveFrameSkipController

    base    = MOSSETracker()
    tracker = AdaptiveFrameSkipController(base, skip_threshold=8.0)

    tracker.initialize(first_frame, init_bbox)
    for frame in remaining_frames:
        bbox = tracker.update(frame)

    stats = tracker.skip_stats()
    print(f"Skip ratio: {stats.skip_ratio:.1%}   Effective FPS: {stats.effective_fps:.1f}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .base import BaseTracker, BBox


@dataclass
class SkipStats:
    """Statistics accumulated by :class:`AdaptiveFrameSkipController`.

    Attributes:
        total_frames:      Frames processed since initialization (update calls only).
        skipped_frames:    Frames where inference was skipped.
        active_frames:     Frames where the underlying tracker was actually invoked.
        skip_ratio:        ``skipped / total`` in ``[0, 1]``.
        inference_fps:     Throughput of the underlying tracker (active frames only).
        effective_fps:     Overall pipeline throughput across all frames.
        mean_motion_score: Mean frame-difference score across processed frames.
    """

    total_frames: int = 0
    skipped_frames: int = 0
    active_frames: int = 0
    skip_ratio: float = 0.0
    inference_fps: float = 0.0
    effective_fps: float = 0.0
    mean_motion_score: float = 0.0

    def __str__(self) -> str:
        return (
            f"SkipStats("
            f"total={self.total_frames}  "
            f"skipped={self.skipped_frames} ({self.skip_ratio:.1%})  "
            f"inference_fps={self.inference_fps:.1f}  "
            f"effective_fps={self.effective_fps:.1f}  "
            f"mean_motion={self.mean_motion_score:.2f})"
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "total_frames": self.total_frames,
            "skipped_frames": self.skipped_frames,
            "active_frames": self.active_frames,
            "skip_ratio": round(self.skip_ratio, 4),
            "inference_fps": round(self.inference_fps, 2),
            "effective_fps": round(self.effective_fps, 2),
            "mean_motion_score": round(self.mean_motion_score, 4),
        }


class AdaptiveFrameSkipController(BaseTracker):
    """Motion-adaptive inference scheduler wrapping any BaseTracker.

    Runs the underlying tracker only on frames where significant scene motion
    is detected, holding the last bounding box otherwise.  The resulting
    object satisfies the :class:`~eovot.trackers.base.BaseTracker` interface
    and can be passed directly to :class:`~eovot.benchmark.engine.BenchmarkEngine`
    without modification.

    The motion detector computes the mean absolute difference (MAD) between
    consecutive grayscale frames (range 0–255).  A threshold of 6 corresponds
    roughly to sub-pixel target motion and very slow camera shake; a threshold
    of 15 skips everything except moderately fast motion.

    Args:
        tracker:              Any :class:`BaseTracker` subclass to wrap.
        skip_threshold:       MAD threshold below which inference is skipped.
                              Increase to skip more aggressively.  Default ``6.0``.
        min_active_interval:  Minimum frames between consecutive active updates,
                              regardless of motion score.  Prevents the tracker
                              being skipped indefinitely on static scenes.
                              Default ``5``.
        use_velocity:         When ``True``, apply a constant-velocity prediction
                              to the bbox on skipped frames.  Useful for targets
                              with smooth, predictable motion.  Default ``False``.

    Example::

        from eovot.trackers.kcf import KCFTracker
        from eovot.trackers.adaptive_skip import AdaptiveFrameSkipController
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=5, num_frames=200)
        base    = KCFTracker()
        tracker = AdaptiveFrameSkipController(base, skip_threshold=8.0, use_velocity=True)

        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        print(tracker.skip_stats())
    """

    def __init__(
        self,
        tracker: BaseTracker,
        skip_threshold: float = 6.0,
        min_active_interval: int = 5,
        use_velocity: bool = False,
    ) -> None:
        if skip_threshold < 0:
            raise ValueError(f"skip_threshold must be non-negative, got {skip_threshold}.")
        if min_active_interval < 1:
            raise ValueError(f"min_active_interval must be >= 1, got {min_active_interval}.")

        super().__init__(name=f"AdaptiveSkip({tracker.name})")
        self._tracker = tracker
        self.skip_threshold = skip_threshold
        self.min_active_interval = min_active_interval
        self.use_velocity = use_velocity

        self._last_bbox: Optional[BBox] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_active_bbox: Optional[BBox] = None
        self._prev_prev_active_bbox: Optional[BBox] = None
        self._frames_since_active: int = 0

        self._total_frames: int = 0
        self._skipped_frames: int = 0
        self._inference_time_s: float = 0.0
        self._total_time_s: float = 0.0
        self._motion_scores: List[float] = []

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialize the underlying tracker and reset skip statistics.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._last_bbox = bbox
        self._prev_active_bbox = bbox
        self._prev_prev_active_bbox = None
        self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self._frames_since_active = 0

        self._total_frames = 0
        self._skipped_frames = 0
        self._inference_time_s = 0.0
        self._total_time_s = 0.0
        self._motion_scores.clear()

    def update(self, frame: np.ndarray) -> BBox:
        """Return the predicted bounding box, skipping inference when safe.

        The decision to skip is taken when both conditions hold:
        1. Frame-to-frame motion score < ``skip_threshold``.
        2. Fewer than ``min_active_interval`` frames have elapsed since the
           last active update.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._last_bbox is None:
            raise RuntimeError("update() called before initialize().")

        t_start = time.perf_counter()
        self._total_frames += 1
        self._frames_since_active += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion = self._motion_score(gray)
        self._motion_scores.append(motion)

        skip = (
            motion < self.skip_threshold
            and self._frames_since_active < self.min_active_interval
        )

        if skip:
            self._skipped_frames += 1
            bbox = self._velocity_predict() if self.use_velocity else self._last_bbox
        else:
            t_infer = time.perf_counter()
            bbox = self._tracker.update(frame)
            self._inference_time_s += time.perf_counter() - t_infer
            self._prev_prev_active_bbox = self._prev_active_bbox
            self._prev_active_bbox = bbox
            self._frames_since_active = 0

        self._last_bbox = bbox
        self._prev_gray = gray
        self._total_time_s += time.perf_counter() - t_start
        return bbox

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def skip_stats(self) -> SkipStats:
        """Return accumulated skip and throughput statistics.

        Can be called at any point after :meth:`initialize`.  Returns a
        snapshot of the counters accumulated so far.

        Returns:
            :class:`SkipStats` populated from the current run.
        """
        total = self._total_frames
        skipped = self._skipped_frames
        active = total - skipped
        skip_ratio = skipped / total if total > 0 else 0.0
        inference_fps = active / self._inference_time_s if self._inference_time_s > 0 else 0.0
        effective_fps = total / self._total_time_s if self._total_time_s > 0 else 0.0
        mean_motion = float(np.mean(self._motion_scores)) if self._motion_scores else 0.0

        return SkipStats(
            total_frames=total,
            skipped_frames=skipped,
            active_frames=active,
            skip_ratio=skip_ratio,
            inference_fps=inference_fps,
            effective_fps=effective_fps,
            mean_motion_score=mean_motion,
        )

    @property
    def wrapped_tracker(self) -> BaseTracker:
        """Return the underlying tracker being wrapped."""
        return self._tracker

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _motion_score(self, gray: np.ndarray) -> float:
        """Mean absolute pixel difference against the previous frame (0–255)."""
        if self._prev_gray is None or gray.shape != self._prev_gray.shape:
            return float("inf")
        return float(np.mean(np.abs(gray - self._prev_gray)))

    def _velocity_predict(self) -> BBox:
        """Extrapolate bbox with a one-step constant-velocity model.

        Uses the displacement between the last two active bounding boxes as
        the velocity estimate and adds it to the most recent active bbox.
        Falls back to holding position when only one active update has occurred.
        """
        if self._prev_prev_active_bbox is None or self._prev_active_bbox is None:
            assert self._last_bbox is not None
            return self._last_bbox
        x0, y0, w0, h0 = self._prev_prev_active_bbox
        x1, y1, w1, h1 = self._prev_active_bbox
        return (x1 + (x1 - x0), y1 + (y1 - y0), w1, h1)
