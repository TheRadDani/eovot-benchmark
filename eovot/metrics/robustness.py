"""Robustness and reliability metrics for visual object tracking.

Supplements the standard OTB success/precision metrics with tracking
robustness measures that quantify how gracefully a tracker degrades under
challenging conditions.

Provided metrics:

- **Failure detection**: Identifies frames where the tracker has lost the
  target (IoU below a configurable threshold for a sustained period).
- **Failure count / rate**: Summary statistics for the number of tracker
  losses per sequence and per 100 frames.
- **Mean IoU until first failure**: Accuracy during the portion of the
  sequence where tracking is still active — uncontaminated by post-failure
  drift.
- **Tracking success rate**: Fraction of frames where the tracker is
  considered "on target".
- **Normalized Precision**: LaSOT benchmark metric that normalises center
  distance by ground-truth object diagonal, making it scale-invariant and
  comparable across objects of different sizes.
- **Success rate at IoU=0.5**: Canonical Pascal VOC / COCO detection
  criterion; a single interpretable scalar complementing the full AUC.

References:
    Kristan et al., "The Seventh Visual Object Tracking VOT2019 Challenge
    Results." ICCV Workshops, 2019.

    Fan et al., "LaSOT: A High-quality Benchmark for Large-scale Single
    Object Tracking." CVPR, 2019.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RobustnessMetrics:
    """Tracking robustness and reliability summary for one sequence run.

    Attributes:
        failure_count: Number of detected tracking failures in the sequence.
        failure_rate: Failures per 100 frames (normalised for cross-sequence
            comparison regardless of sequence length).
        mean_iou_until_first_failure: Mean IoU over all frames *before* the
            first detected failure.  ``None`` if no failure was detected
            (full-sequence mean returned instead) or if failure occurs on
            the very first evaluated frame.
        tracking_success_rate: Fraction of frames where IoU exceeds the
            failure threshold — the proportion of time the tracker is
            considered "on target".
        failure_frames: List of frame indices at which each detected failure
            begins (0-indexed, relative to the evaluated frame array).
    """

    failure_count: int
    failure_rate: float
    mean_iou_until_first_failure: Optional[float]
    tracking_success_rate: float
    failure_frames: List[int]

    def __str__(self) -> str:
        return (
            f"RobustnessMetrics("
            f"failures={self.failure_count}, "
            f"failure_rate={self.failure_rate:.2f}/100fr, "
            f"success_rate={self.tracking_success_rate:.4f})"
        )

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        return {
            "failure_count": self.failure_count,
            "failure_rate_per_100_frames": round(self.failure_rate, 4),
            "mean_iou_until_first_failure": (
                round(self.mean_iou_until_first_failure, 4)
                if self.mean_iou_until_first_failure is not None
                else None
            ),
            "tracking_success_rate": round(self.tracking_success_rate, 4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Core robustness computation
# ──────────────────────────────────────────────────────────────────────────────


def compute_robustness(
    ious: np.ndarray,
    failure_threshold: float = 0.1,
    min_failure_length: int = 5,
) -> RobustnessMetrics:
    """Detect tracking failures and compute robustness statistics.

    A **failure** is defined as a run of at least ``min_failure_length``
    consecutive frames where IoU is strictly below ``failure_threshold``.
    Requiring a minimum run length prevents brief partial-occlusion events
    (which often recover quickly) from being counted as full tracker losses.

    Args:
        ious: Per-frame IoU values, shape ``(N,)``.
        failure_threshold: IoU below this value is classified as a candidate
            failure frame.  Default: ``0.1`` (standard VOT criterion).
        min_failure_length: Minimum consecutive candidate-failure frames
            required to declare a failure event.  Default: ``5``.

    Returns:
        :class:`RobustnessMetrics` populated with failure count, rate,
        mean-IoU-until-failure, success rate, and failure frame indices.

    Example::

        ious = np.array([0.8, 0.75, 0.05, 0.02, 0.0, 0.0, 0.0, 0.7, 0.68])
        r = compute_robustness(ious, failure_threshold=0.1, min_failure_length=3)
        print(r.failure_count)  # 1
        print(r.failure_frames)  # [2]
    """
    if len(ious) == 0:
        return RobustnessMetrics(
            failure_count=0,
            failure_rate=0.0,
            mean_iou_until_first_failure=None,
            tracking_success_rate=0.0,
            failure_frames=[],
        )

    n = len(ious)
    failed = ious < failure_threshold  # bool mask: True = below threshold

    # Detect contiguous failure runs via a single linear scan.
    failure_start_frames: List[int] = []
    in_failure = False
    run_start = 0

    for i, f in enumerate(failed):
        if f and not in_failure:
            in_failure = True
            run_start = i
        elif not f and in_failure:
            if (i - run_start) >= min_failure_length:
                failure_start_frames.append(run_start)
            in_failure = False

    # Handle a failure run that extends to the last frame.
    if in_failure and (n - run_start) >= min_failure_length:
        failure_start_frames.append(run_start)

    failure_count = len(failure_start_frames)

    # Normalise to failures per 100 frames for cross-sequence comparison.
    failure_rate = (failure_count / n) * 100.0

    # Mean IoU in the pre-failure portion of the sequence.
    first_fail = failure_start_frames[0] if failure_start_frames else None
    if first_fail is not None and first_fail > 0:
        mean_iou_until_fail: Optional[float] = float(ious[:first_fail].mean())
    elif first_fail == 0:
        mean_iou_until_fail = None  # Failure from the very first frame
    else:
        # No failure detected — mean over the full sequence.
        mean_iou_until_fail = float(ious.mean())

    tracking_success_rate = float((~failed).mean())

    return RobustnessMetrics(
        failure_count=failure_count,
        failure_rate=failure_rate,
        mean_iou_until_first_failure=mean_iou_until_fail,
        tracking_success_rate=tracking_success_rate,
        failure_frames=failure_start_frames,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Normalised Precision (LaSOT protocol)
# ──────────────────────────────────────────────────────────────────────────────


def normalized_precision_curve(
    preds: np.ndarray,
    gts: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised Precision curve used by the LaSOT benchmark.

    Unlike raw precision (which uses absolute pixel distances), normalised
    precision divides the center-to-center distance by the diagonal of the
    ground-truth bounding box.  This makes the metric scale-invariant and
    directly comparable across sequences containing objects of vastly
    different sizes.

    Args:
        preds: ``(N, 4)`` predicted boxes in ``(x, y, w, h)`` format.
        gts:   ``(N, 4)`` ground-truth boxes in ``(x, y, w, h)`` format.
        thresholds: Normalised distance thresholds to sweep.  Defaults to
            51 linearly spaced values from 0 to 0.5, matching the LaSOT
            evaluation protocol.

    Returns:
        ``(thresholds, precision_rates)`` — both ``(T,)`` float arrays.

    Reference:
        Fan et al., "LaSOT: A High-quality Benchmark for Large-scale Single
        Object Tracking." CVPR, 2019.

    Example::

        thr, rates = normalized_precision_curve(preds, gts)
        norm_prec_auc = float(np.trapz(rates, thr) / thr[-1])
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.5, 51)

    n = min(len(preds), len(gts))
    if n == 0:
        return thresholds, np.zeros_like(thresholds)

    norm_dists = np.empty(n, dtype=np.float64)
    for i in range(n):
        px, py, pw, ph = preds[i]
        gx, gy, gw, gh = gts[i]
        if gw <= 0 or gh <= 0:
            norm_dists[i] = np.inf
            continue
        dx = (px + pw / 2.0) - (gx + gw / 2.0)
        dy = (py + ph / 2.0) - (gy + gh / 2.0)
        dist = np.sqrt(dx * dx + dy * dy)
        diag = np.sqrt(gw * gw + gh * gh)
        norm_dists[i] = dist / diag

    rates = np.array([(norm_dists < t).mean() for t in thresholds], dtype=np.float64)
    return thresholds, rates


# ──────────────────────────────────────────────────────────────────────────────
# Scalar helpers
# ──────────────────────────────────────────────────────────────────────────────


def success_rate_at_threshold(ious: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of frames where IoU strictly exceeds *threshold*.

    IoU > 0.5 is the canonical "correct detection" criterion from Pascal VOC
    and COCO.  Reporting this alongside the full success-curve AUC provides
    a human-interpretable accuracy score at the most common operating point.

    Args:
        ious: Per-frame IoU values, shape ``(N,)``.
        threshold: IoU threshold. Default: ``0.5``.

    Returns:
        Success rate in ``[0, 1]``.
    """
    if len(ious) == 0:
        return 0.0
    return float((ious > threshold).mean())
