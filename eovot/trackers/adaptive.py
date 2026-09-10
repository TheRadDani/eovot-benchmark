"""Confidence-aware adaptive resolution tracker for edge deployment.

Wraps any :class:`~eovot.trackers.base.BaseTracker` and dynamically adjusts
the resolution at which each frame is processed based on a lightweight
confidence proxy computed from the prediction stream itself — no ground truth
required at inference time.

Motivation
----------
On constrained edge hardware (microcontrollers, SBCs, wearables), processing
every frame at full resolution is often unnecessary:

- When the tracker is confident (stable predictions, smooth motion), high
  resolution adds latency without improving accuracy.
- When the tracker is uncertain (abrupt scale change, fast/irregular motion),
  full resolution is needed to re-acquire the target.

``AdaptiveResolutionTracker`` implements a simple but effective policy:

    - Maintain a rolling exponential moving average (EMA) confidence score
      based on two complementary signals:

      * **Scale consistency** — sudden changes in predicted box area imply the
        tracker may be drifting.
      * **Velocity smoothness** — abrupt displacement jumps indicate the
        target moved unpredictably or the tracker snapped to a distractor.

    - Map the confidence EMA to a discrete resolution tier from ``scale_factors``.
      High confidence → aggressive downscale (fast, cheap).
      Low  confidence → full resolution (accurate, expensive).

    - After a configurable ``warmup_frames`` period, confidence estimates are
      considered stable enough to act on.

The wrapper is tracker-agnostic: it works with MOSSE, KCF, CamShift, or any
custom :class:`~eovot.trackers.base.BaseTracker` subclass.

Typical usage::

    from eovot.trackers.adaptive import AdaptiveResolutionTracker
    from eovot.trackers.kcf import KCFTracker

    base = KCFTracker()
    tracker = AdaptiveResolutionTracker(
        base_tracker=base,
        scale_factors=[1.0, 0.75, 0.5],
        high_conf_threshold=0.75,
        low_conf_threshold=0.35,
    )

    # Use exactly like any other BaseTracker
    tracker.initialize(frame, bbox)
    for frame in sequence:
        pred = tracker.update(frame)
        print(tracker.current_scale, tracker.mean_confidence)

Benchmarking integration::

    from eovot.benchmark.engine import BenchmarkEngine
    engine = BenchmarkEngine()
    result = engine.run(tracker, dataset, dataset_name="Synthetic")

    # Inspect adaptation statistics stored on the tracker
    print(tracker.adaptation_log[:5])
    # [{'frame': 6, 'confidence': 0.82, 'scale': 1.0},
    #  {'frame': 7, 'confidence': 0.91, 'scale': 0.75}, ...]
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox

# Default scale tiers: full → 75 % → 50 % width/height.
_DEFAULT_SCALE_FACTORS: Tuple[float, ...] = (1.0, 0.75, 0.5)


class AdaptiveResolutionTracker(BaseTracker):
    """Confidence-aware wrapper that adjusts per-frame resolution dynamically.

    Args:
        base_tracker:        Any :class:`~eovot.trackers.base.BaseTracker`
                             instance to wrap.  Its weights / state are shared.
        scale_factors:       Ordered list of scale factors (largest first).
                             Each factor is applied to **both** width and height
                             of every frame before passing it to ``base_tracker``.
                             Must contain at least one element; the first element
                             should be ``1.0`` (full resolution).
                             Default: ``[1.0, 0.75, 0.5]``.
        high_conf_threshold: Confidence EMA above which the tracker switches
                             to a more aggressive downscale tier.
                             Default: ``0.75``.
        low_conf_threshold:  Confidence EMA below which the tracker reverts to
                             the next higher-resolution tier.
                             Default: ``0.35``.
        warmup_frames:       Number of frames after initialization during which
                             the tracker always runs at full resolution while the
                             confidence EMA stabilises.  Default: ``10``.
        ema_alpha:           Smoothing factor for the confidence EMA.
                             Larger values react faster to confidence changes.
                             Range ``(0, 1]``.  Default: ``0.2``.
        scale_area_penalty:  Weight of the scale-change signal in the confidence
                             score (relative to the velocity-smoothness signal).
                             Default: ``0.5``.

    Attributes:
        current_scale:   Active scale factor applied to frames (read-only).
        mean_confidence: Current confidence EMA value in ``[0, 1]``.
        adaptation_log:  List of per-frame dicts recording frame index,
                         confidence, and scale factor selected.  Useful for
                         post-hoc analysis of adaptation behaviour.
        switch_count:    Total number of resolution-tier changes so far.
    """

    def __init__(
        self,
        base_tracker: BaseTracker,
        scale_factors: Optional[List[float]] = None,
        high_conf_threshold: float = 0.75,
        low_conf_threshold: float = 0.35,
        warmup_frames: int = 10,
        ema_alpha: float = 0.2,
        scale_area_penalty: float = 0.5,
    ) -> None:
        super().__init__(name=f"Adaptive({base_tracker.name})")
        if not (0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if not (0.0 <= low_conf_threshold < high_conf_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 ≤ low_conf_threshold < high_conf_threshold ≤ 1; "
                f"got low={low_conf_threshold}, high={high_conf_threshold}"
            )

        self._base = base_tracker
        self._scales: List[float] = sorted(
            scale_factors if scale_factors else list(_DEFAULT_SCALE_FACTORS), reverse=True
        )
        if not math.isclose(self._scales[0], 1.0):
            # Ensure full resolution is always available as the fallback tier.
            self._scales.insert(0, 1.0)

        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.warmup_frames = warmup_frames
        self._alpha = ema_alpha
        self._area_penalty_weight = scale_area_penalty

        # Runtime state
        self._frame_idx: int = 0
        self._scale_tier: int = 0              # index into self._scales
        self._confidence_ema: float = 0.5      # initialise to neutral
        self._prev_bbox: Optional[BBox] = None
        self._prev_velocity: Optional[float] = None
        self._area_history: Deque[float] = deque(maxlen=5)
        self._vel_history: Deque[float] = deque(maxlen=5)

        # Analysis outputs
        self.adaptation_log: List[Dict] = []
        self.switch_count: int = 0

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the wrapped tracker at full resolution.

        Also resets all internal confidence and adaptation state so the tracker
        can be safely re-initialised on a new sequence.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._reset_state()
        scaled_frame, scaled_bbox = self._apply_scale(frame, bbox, self._current_scale)
        self._base.initialize(scaled_frame, scaled_bbox)
        self._prev_bbox = bbox
        self._area_history.append(bbox[2] * bbox[3])
        self._frame_idx = 0

    def update(self, frame: np.ndarray) -> BBox:
        """Predict target location, adapting resolution based on confidence.

        The method:

        1. Runs ``base_tracker.update`` on the (possibly downscaled) frame.
        2. Rescales the prediction back to original coordinates.
        3. Updates the confidence EMA from scale-consistency and velocity-
           smoothness signals.
        4. (After warmup) adjusts the resolution tier for the next frame.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` in **original** coordinates.
        """
        self._frame_idx += 1
        scale = self._current_scale
        scaled_frame, _ = self._apply_scale(frame, (0, 0, 1, 1), scale)
        raw_pred = self._base.update(scaled_frame)
        pred = self._unscale_bbox(raw_pred, scale)

        confidence = self._compute_confidence(pred)
        self._update_ema(confidence)
        self._log_frame()

        if self._frame_idx >= self.warmup_frames:
            self._adapt_scale()

        self._prev_bbox = pred
        return pred

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def current_scale(self) -> float:
        """Currently active scale factor applied to input frames."""
        return self._current_scale

    @property
    def mean_confidence(self) -> float:
        """Exponential moving average confidence estimate (0 = low, 1 = high)."""
        return self._confidence_ema

    # ------------------------------------------------------------------
    # Confidence computation
    # ------------------------------------------------------------------

    def _compute_confidence(self, pred: BBox) -> float:
        """Compute a frame-level confidence score in ``[0, 1]``.

        Two complementary signals are blended:

        * **Area consistency** (``s_area``): how stable the predicted box area
          is relative to recent history.  Sudden area jumps are penalised.

        * **Velocity smoothness** (``s_vel``): how consistent the frame-to-frame
          displacement is.  Abrupt trajectory changes suggest target loss.

        Returns:
            Confidence in ``[0, 1]``.  ``1.0`` = fully confident (stable,
            smooth prediction).  ``0.0`` = no confidence (erratic prediction).
        """
        x, y, w, h = pred
        area = max(w * h, 1e-6)

        # --- Scale consistency ---
        if self._area_history:
            mean_hist_area = float(np.mean(self._area_history))
            area_ratio = area / mean_hist_area if mean_hist_area > 1e-6 else 1.0
            # Squash ratio: ratios near 1.0 → high score; far from 1.0 → low score.
            s_area = math.exp(-abs(math.log(area_ratio + 1e-9)) * 2.0)
        else:
            s_area = 0.5
        self._area_history.append(area)

        # --- Velocity smoothness ---
        s_vel = 0.5
        if self._prev_bbox is not None:
            cx, cy = x + w / 2.0, y + h / 2.0
            px, py = self._prev_bbox[0] + self._prev_bbox[2] / 2.0, \
                     self._prev_bbox[1] + self._prev_bbox[3] / 2.0
            vel = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            self._vel_history.append(vel)

            if len(self._vel_history) >= 2:
                mean_vel = float(np.mean(list(self._vel_history)[:-1]))
                # Normalise by mean velocity; sudden acceleration = low confidence.
                norm = mean_vel if mean_vel > 1e-6 else 1.0
                rel_change = abs(vel - mean_vel) / norm
                s_vel = math.exp(-rel_change * 1.5)

        weight_a = self._area_penalty_weight
        weight_v = 1.0 - weight_a
        return float(weight_a * s_area + weight_v * s_vel)

    # ------------------------------------------------------------------
    # EMA and tier adaptation
    # ------------------------------------------------------------------

    def _update_ema(self, confidence: float) -> None:
        self._confidence_ema = (
            self._alpha * confidence + (1.0 - self._alpha) * self._confidence_ema
        )

    def _adapt_scale(self) -> None:
        """Shift resolution tier up or down based on current EMA confidence."""
        old_tier = self._scale_tier
        n_tiers = len(self._scales)

        if self._confidence_ema >= self.high_conf_threshold:
            # Confident → try to use a smaller (cheaper) resolution.
            self._scale_tier = min(self._scale_tier + 1, n_tiers - 1)
        elif self._confidence_ema <= self.low_conf_threshold:
            # Uncertain → move toward a larger (more accurate) resolution.
            self._scale_tier = max(self._scale_tier - 1, 0)

        if self._scale_tier != old_tier:
            self.switch_count += 1

    # ------------------------------------------------------------------
    # Frame scaling helpers
    # ------------------------------------------------------------------

    @property
    def _current_scale(self) -> float:
        return self._scales[self._scale_tier]

    @staticmethod
    def _apply_scale(
        frame: np.ndarray, bbox: BBox, scale: float
    ) -> Tuple[np.ndarray, BBox]:
        """Resize *frame* by *scale* and scale *bbox* accordingly."""
        if math.isclose(scale, 1.0):
            return frame, bbox
        h, w = frame.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        x, y, bw, bh = bbox
        return scaled, (x * scale, y * scale, bw * scale, bh * scale)

    @staticmethod
    def _unscale_bbox(bbox: BBox, scale: float) -> BBox:
        """Map a bbox predicted on a scaled frame back to original coordinates."""
        if math.isclose(scale, 1.0):
            return bbox
        inv = 1.0 / scale
        x, y, w, h = bbox
        return (x * inv, y * inv, w * inv, h * inv)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._frame_idx = 0
        self._scale_tier = 0
        self._confidence_ema = 0.5
        self._prev_bbox = None
        self._prev_velocity = None
        self._area_history.clear()
        self._vel_history.clear()
        self.adaptation_log.clear()
        self.switch_count = 0

    def _log_frame(self) -> None:
        self.adaptation_log.append({
            "frame": self._frame_idx,
            "confidence": round(self._confidence_ema, 4),
            "scale": self._current_scale,
        })

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def adaptation_summary(self) -> Dict:
        """Return aggregate statistics about the adaptation behaviour.

        Returns:
            Dict with keys:

            - ``total_frames`` — frames processed (excluding initialisation).
            - ``switch_count`` — number of resolution-tier changes.
            - ``switch_rate`` — switches per frame.
            - ``scale_distribution`` — fraction of frames spent at each tier.
            - ``mean_confidence`` — time-averaged EMA confidence.
            - ``mean_scale`` — time-averaged scale factor (lower = more aggressive).
        """
        if not self.adaptation_log:
            return {}
        scales = [e["scale"] for e in self.adaptation_log]
        confs = [e["confidence"] for e in self.adaptation_log]
        n = len(scales)
        dist: Dict[float, float] = {}
        for s in self._scales:
            count = sum(1 for x in scales if math.isclose(x, s))
            dist[s] = round(count / n, 4)
        return {
            "total_frames": n,
            "switch_count": self.switch_count,
            "switch_rate": round(self.switch_count / max(n, 1), 4),
            "scale_distribution": dist,
            "mean_confidence": round(float(np.mean(confs)), 4),
            "mean_scale": round(float(np.mean(scales)), 4),
        }

    def __repr__(self) -> str:
        return (
            f"AdaptiveResolutionTracker("
            f"base={self._base.name!r}, "
            f"scales={self._scales}, "
            f"high={self.high_conf_threshold}, "
            f"low={self.low_conf_threshold})"
        )
