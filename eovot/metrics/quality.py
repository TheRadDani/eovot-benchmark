"""Ground-truth-free tracker quality estimation via Normalized Cross-Correlation.

In production deployments on edge devices, ground-truth bounding boxes are
unavailable at inference time.  The AccuracyMetrics engine can only be used
offline to assess historical runs.  This module provides a deployment-compatible
quality signal derived purely from the tracker's own predictions and the input
video stream.

The :class:`AppearanceQualityEstimator` stores a reference template patch on
tracker initialization (when the GT bbox is available) and computes the
Normalized Cross-Correlation (NCC) between that template and the region
predicted by the tracker at each subsequent frame.  NCC in ``[-1, 1]`` is
mapped to a quality score in ``[0, 1]``; values near 1 indicate the predicted
region looks like the initial template (tracker is on target), while values
near 0 indicate the appearance has diverged (likely drift or occlusion).

Design notes
~~~~~~~~~~~~
* Template is resized to a fixed ``(64, 64)`` greyscale patch to normalize
  across different target scales and aspect ratios.
* An Exponential Moving Average (EMA) of the raw NCC scores provides a
  smoothed quality signal less sensitive to single-frame noise.
* A "drift event" is flagged when the EMA score stays below
  ``drift_threshold`` for at least ``min_drift_frames`` consecutive frames,
  avoiding false positives from momentary appearance changes.
* No ground-truth dependency: the module runs identically during offline
  benchmarking and live on-device deployment.

Correlation with IoU
~~~~~~~~~~~~~~~~~~~~
On clean sequences the quality score correlates positively with IoU — when
the tracker drifts, both IoU and quality score drop.  On occluded sequences
the quality score drops during the gap (correct: appearance is masked) while
IoU may stay high if the tracker's momentum keeps it near the true location.
This asymmetry makes quality score a *complementary* signal to IoU, not a
proxy for it.

Typical usage::

    from eovot.metrics.quality import AppearanceQualityEstimator

    estimator = AppearanceQualityEstimator()
    estimator.initialize(frame0, gt_bbox)

    for frame, pred_bbox in zip(frames[1:], predictions[1:]):
        score = estimator.score(frame, pred_bbox)

    result = estimator.result(tracker_name="MOSSE", sequence_name="car1")
    print(result)
    # QualityResult[MOSSE on car1] mean_q=0.743 drift_events=1 min_q=0.231
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[float, float, float, float]

_TEMPLATE_SIZE: Tuple[int, int] = (64, 64)


# ---------------------------------------------------------------------------
# Low-level image utilities
# ---------------------------------------------------------------------------


def _crop_patch(frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    """Crop a greyscale patch from *frame* at *bbox* and resize to ``_TEMPLATE_SIZE``.

    Args:
        frame: BGR (or greyscale) frame, shape ``(H, W, 3)`` or ``(H, W)``.
        bbox:  ``(x, y, w, h)`` in pixel coordinates (may be fractional).

    Returns:
        Greyscale uint8 patch of shape ``_TEMPLATE_SIZE``, or ``None`` if the
        crop has zero area or falls entirely outside the frame.
    """
    x, y, w, h = [int(round(v)) for v in bbox]
    H, W = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + max(w, 1)), min(H, y + max(h, 1))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop, _TEMPLATE_SIZE, interpolation=cv2.INTER_LINEAR)


def _ncc(template: np.ndarray, patch: np.ndarray) -> float:
    """Normalized cross-correlation between two same-shape arrays.

    Both inputs are mean-subtracted before computing the inner product, so
    the result is insensitive to global brightness shifts.

    Args:
        template: Reference patch, shape ``_TEMPLATE_SIZE``.
        patch:    Current patch, same shape.

    Returns:
        NCC in ``[-1, 1]``.  Returns ``0.0`` when either patch is constant
        (near-zero L2 norm) — avoids division by zero on blank crops.
    """
    t = template.astype(np.float32) - template.mean()
    p = patch.astype(np.float32) - patch.mean()
    t_norm = float(np.linalg.norm(t))
    p_norm = float(np.linalg.norm(p))
    if t_norm < 1e-6 or p_norm < 1e-6:
        return 0.0
    return float(np.dot(t.ravel(), p.ravel()) / (t_norm * p_norm))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class QualityResult:
    """Appearance-quality summary for one tracker run on one sequence.

    Attributes:
        tracker_name:   Human-readable tracker identifier.
        sequence_name:  Sequence identifier.
        scores:         Raw NCC quality scores per frame, shape ``(N,)``.
                        Frame 0 (initialization) is always ``1.0``.
        ema_scores:     EMA-smoothed quality scores, same shape.
        drift_events:   Frame indices at which *new* drift events begin.
                        A drift event is triggered when the EMA score has
                        stayed below ``drift_threshold`` for at least
                        ``min_drift_frames`` consecutive frames.
        mean_quality:   Mean raw NCC score across all frames in ``[0, 1]``.
        min_quality:    Minimum raw NCC score across all frames in ``[0, 1]``.
    """

    tracker_name: str
    sequence_name: str
    scores: np.ndarray
    ema_scores: np.ndarray
    drift_events: List[int] = field(default_factory=list)
    mean_quality: float = 0.0
    min_quality: float = 0.0

    def __str__(self) -> str:
        return (
            f"QualityResult[{self.tracker_name} on {self.sequence_name}] "
            f"mean_q={self.mean_quality:.4f}  "
            f"min_q={self.min_quality:.4f}  "
            f"drift_events={len(self.drift_events)}"
        )


class AppearanceQualityEstimator:
    """Estimate per-frame tracker quality via Normalized Cross-Correlation.

    This class is stateful: call :meth:`initialize` once per sequence with
    the first frame and ground-truth bbox, then call :meth:`score` for every
    subsequent frame with the tracker's predicted bbox.  Finally, call
    :meth:`result` to retrieve the full :class:`QualityResult`.

    The estimator can be used:

    * **During benchmarking** — the :class:`~eovot.benchmark.engine.BenchmarkEngine`
      accepts an optional ``quality_estimator`` argument and attaches a
      :class:`QualityResult` to each :class:`~eovot.benchmark.engine.SequenceResult`.
    * **In production** — no ground-truth dependency; only the predicted bbox
      from the running tracker is required after the first frame.

    Args:
        ema_alpha:        EMA smoothing factor.  Closer to 1 → less smoothing;
                          closer to 0 → very slow response.  Default: ``0.3``.
        drift_threshold:  EMA score below which a frame is classified as
                          "failing".  Default: ``0.5``.
        min_drift_frames: Consecutive failing frames required before a drift
                          event is logged.  Prevents false positives from
                          single-frame noise.  Default: ``5``.

    Example::

        estimator = AppearanceQualityEstimator(drift_threshold=0.4)
        estimator.initialize(first_frame, gt_bbox)
        for frame, pred_bbox in zip(frames[1:], predictions[1:]):
            q = estimator.score(frame, pred_bbox)
        result = estimator.result("KCF", "seq_001")
        print(f"Drift events: {result.drift_events}")
    """

    def __init__(
        self,
        ema_alpha: float = 0.3,
        drift_threshold: float = 0.5,
        min_drift_frames: int = 5,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha!r}")
        if not 0.0 <= drift_threshold <= 1.0:
            raise ValueError(f"drift_threshold must be in [0, 1], got {drift_threshold!r}")
        if min_drift_frames < 1:
            raise ValueError(f"min_drift_frames must be >= 1, got {min_drift_frames!r}")

        self.ema_alpha = ema_alpha
        self.drift_threshold = drift_threshold
        self.min_drift_frames = min_drift_frames

        self._template: Optional[np.ndarray] = None
        self._scores: List[float] = []
        self._ema_scores: List[float] = []
        self._ema_current: float = 1.0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Store the reference template and reset internal state.

        Must be called once per sequence before any :meth:`score` calls.

        Args:
            frame: First frame (BGR uint8 array).
            bbox:  Ground-truth bounding box ``(x, y, w, h)`` on the first frame.
        """
        self._template = _crop_patch(frame, bbox)
        self._scores = [1.0]
        self._ema_scores = [1.0]
        self._ema_current = 1.0

    def score(self, frame: np.ndarray, bbox: BBox) -> float:
        """Compute the appearance quality score for the current frame prediction.

        Args:
            frame: Current BGR frame.
            bbox:  Tracker-predicted bounding box ``(x, y, w, h)``.

        Returns:
            Quality score in ``[0, 1]``.  ``1.0`` means the predicted region
            looks identical to the reference template; ``0.0`` indicates a
            complete appearance mismatch (drift or full occlusion).

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._template is None:
            raise RuntimeError("Call initialize() before score().")

        patch = _crop_patch(frame, bbox)
        if patch is None:
            raw = 0.0
        else:
            ncc_val = _ncc(self._template, patch)
            raw = float(np.clip((ncc_val + 1.0) / 2.0, 0.0, 1.0))

        self._ema_current = (
            self.ema_alpha * raw + (1.0 - self.ema_alpha) * self._ema_current
        )
        self._scores.append(raw)
        self._ema_scores.append(self._ema_current)
        return raw

    def result(
        self,
        tracker_name: str = "",
        sequence_name: str = "",
    ) -> QualityResult:
        """Build and return the :class:`QualityResult` from accumulated scores.

        Scans EMA scores to detect drift events (onset frame index reported).

        Args:
            tracker_name:  Human-readable tracker identifier for the result.
            sequence_name: Sequence identifier for the result.

        Returns:
            :class:`QualityResult` with per-frame data and aggregate stats.
        """
        scores_arr = np.array(self._scores, dtype=np.float64)
        ema_arr = np.array(self._ema_scores, dtype=np.float64)

        drift_events: List[int] = []
        in_drift = False
        consec_below = 0

        for i, ema_val in enumerate(ema_arr):
            if ema_val < self.drift_threshold:
                consec_below += 1
                if consec_below == self.min_drift_frames and not in_drift:
                    drift_events.append(i - self.min_drift_frames + 1)
                    in_drift = True
            else:
                consec_below = 0
                in_drift = False

        return QualityResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            scores=scores_arr,
            ema_scores=ema_arr,
            drift_events=drift_events,
            mean_quality=float(scores_arr.mean()) if len(scores_arr) > 0 else 0.0,
            min_quality=float(scores_arr.min()) if len(scores_arr) > 0 else 0.0,
        )

    def reset(self) -> None:
        """Clear accumulated state so the estimator can be reused for a new sequence."""
        self._template = None
        self._scores = []
        self._ema_scores = []
        self._ema_current = 1.0


# ---------------------------------------------------------------------------
# Batch analysis helper
# ---------------------------------------------------------------------------


def correlate_quality_with_iou(
    quality_scores: np.ndarray,
    ious: np.ndarray,
) -> float:
    """Compute the Pearson correlation between quality scores and ground-truth IoU.

    Useful for validating the quality estimator offline: a well-calibrated
    estimator should show strong positive correlation (> 0.6) on clean
    sequences and weaker correlation on occluded sequences where the
    appearance disappears even though the target remains trackable.

    Args:
        quality_scores: Per-frame NCC quality scores, shape ``(N,)``.
        ious:           Per-frame IoU values, shape ``(N,)``.

    Returns:
        Pearson r in ``[-1, 1]``, or ``0.0`` if either input is constant.
    """
    n = min(len(quality_scores), len(ious))
    if n < 2:
        return 0.0
    q = np.asarray(quality_scores[:n], dtype=np.float64)
    iou = np.asarray(ious[:n], dtype=np.float64)
    q_std = q.std()
    iou_std = iou.std()
    if q_std < 1e-9 or iou_std < 1e-9:
        return 0.0
    return float(np.corrcoef(q, iou)[0, 1])
