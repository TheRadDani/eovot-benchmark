"""Ensemble tracker — combines predictions from multiple base trackers.

Runs N trackers in parallel on every frame and fuses their bounding-box
predictions into a single output using a weighted box average.  Optionally
rejects outlier predictions before averaging via an IoU-based consensus
filter.

Motivation
----------
Single-tracker baselines are noisy on difficult sequences.  An ensemble
provides:

* **An upper-bound reference** — the weighted fusion reliably outperforms
  the weakest member, giving a ceiling for what a single-tracker can achieve.
* **Robustness analysis** — per-frame variance across ensemble members
  flags frames where all trackers disagree, which is a proxy for genuine
  tracking difficulty and correlates with low IoU against ground truth.
* **Modular research scaffold** — researchers can swap tracker weights or
  the fusion function without changing the benchmark infrastructure.

Fusion algorithm
----------------
Given N predicted boxes ``{b_1, ..., b_N}`` with weights ``{w_i}``:

1. **Outlier filtering** (optional): Remove boxes whose IoU with the
   weight-averaged centroid estimate is below ``outlier_threshold``.
   This prevents one catastrophically-failing tracker from dragging the
   ensemble toward an empty part of the frame.

2. **Weighted centre-of-mass**: Final box centre is the weighted mean of
   the surviving centres; final dimensions are the weighted mean of the
   surviving width/height values.

Computational cost
------------------
The ensemble runs every member tracker sequentially, so wall-clock time is
O(N) times a single tracker.  For edge deployment, keep N small (2–3) and
prefer fast trackers (MOSSE, KCF, KalmanFilter) as members.

Example::

    from eovot.trackers.ensemble import EnsembleTracker
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker

    ensemble = EnsembleTracker(
        trackers=[MOSSETracker(), KCFTracker()],
        weights=[0.6, 0.4],
    )
    ensemble.initialize(frame, bbox)
    for frame in remaining_frames:
        fused_bbox = ensemble.update(frame)
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import BaseTracker, BBox


def _box_iou(a: BBox, b: BBox) -> float:
    """Axis-aligned IoU between two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _weighted_box_average(
    boxes: List[BBox],
    weights: List[float],
) -> BBox:
    """Return a weighted average of bounding boxes.

    Args:
        boxes:   List of ``(x, y, w, h)`` tuples.
        weights: Non-negative weights (need not sum to 1; normalised here).

    Returns:
        Weighted-average bounding box ``(x, y, w, h)``.
    """
    total = sum(weights)
    if total <= 0:
        # Degenerate: return first box unchanged
        return boxes[0]
    x = sum(b[0] * w for b, w in zip(boxes, weights)) / total
    y = sum(b[1] * w for b, w in zip(boxes, weights)) / total
    w = sum(b[2] * w for b, w in zip(boxes, weights)) / total
    h = sum(b[3] * w for b, w in zip(boxes, weights)) / total
    return (float(x), float(y), float(w), float(h))


class EnsembleTracker(BaseTracker):
    """Fuse predictions from multiple trackers into one bounding box.

    Args:
        trackers: Two or more :class:`~eovot.trackers.base.BaseTracker`
            instances.  They must all be freshly created (not yet
            initialised).
        weights: Optional per-tracker fusion weights; must be the same
            length as *trackers*.  All weights must be > 0.
            Default: uniform (equal weights for every tracker).
        outlier_threshold: IoU threshold below which a member's prediction
            is excluded from the weighted average.  The reference for
            computing IoU is the unfiltered weighted-average box.
            Set to ``0.0`` to disable outlier rejection.  Default: ``0.1``.

    Raises:
        ValueError: If fewer than two trackers are provided.
        ValueError: If *weights* length does not match *trackers* length.

    Example::

        from eovot.trackers.ensemble import EnsembleTracker
        from eovot.trackers.mosse import MOSSETracker
        from eovot.trackers.kcf import KCFTracker
        from eovot.trackers.lk_optical_flow import LKOpticalFlowTracker

        ensemble = EnsembleTracker(
            trackers=[MOSSETracker(), KCFTracker(), LKOpticalFlowTracker()],
            weights=[0.5, 0.3, 0.2],
        )
    """

    def __init__(
        self,
        trackers: Sequence[BaseTracker],
        weights: Optional[Sequence[float]] = None,
        outlier_threshold: float = 0.1,
    ) -> None:
        if len(trackers) < 2:
            raise ValueError(
                f"EnsembleTracker requires at least 2 trackers, got {len(trackers)}."
            )
        if weights is not None:
            if len(weights) != len(trackers):
                raise ValueError(
                    f"len(weights)={len(weights)} must match len(trackers)={len(trackers)}."
                )
            if any(w <= 0 for w in weights):
                raise ValueError("All tracker weights must be positive.")

        member_names = "+".join(t.name for t in trackers)
        super().__init__(name=f"Ensemble({member_names})")

        self._trackers: List[BaseTracker] = list(trackers)
        self._weights: List[float] = (
            list(weights) if weights is not None
            else [1.0] * len(trackers)
        )
        self._outlier_threshold = float(outlier_threshold)
        self._last_individual: List[BBox] = []

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise all member trackers on the first frame.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8 array.
            bbox:  Initial bounding box ``(x, y, w, h)``.
        """
        for tracker in self._trackers:
            tracker.initialize(frame, bbox)
        self._last_individual = [bbox] * len(self._trackers)

    def update(self, frame: np.ndarray) -> BBox:
        """Update all members and return the fused bounding box.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8 array.

        Returns:
            Fused bounding box ``(x, y, w, h)``.
        """
        predictions: List[BBox] = [t.update(frame) for t in self._trackers]
        self._last_individual = predictions

        # Step 1: find the consensus reference — the box with the highest
        # total IoU sum against all other boxes.  This is robust to outliers
        # because a single outlier box has low IoU with all consensus members,
        # so it will never win the consensus vote.
        if self._outlier_threshold > 0.0 and len(predictions) > 2:
            iou_sums = [
                sum(_box_iou(predictions[i], predictions[j]) for j in range(len(predictions)) if j != i)
                for i in range(len(predictions))
            ]
            ref_idx = int(np.argmax(iou_sums))
            reference = predictions[ref_idx]

            surviving = [
                (box, w)
                for box, w in zip(predictions, self._weights)
                if _box_iou(box, reference) >= self._outlier_threshold
            ]
            if surviving:
                kept_boxes, kept_weights = zip(*surviving)
                fused = _weighted_box_average(list(kept_boxes), list(kept_weights))
            else:
                fused = _weighted_box_average(predictions, self._weights)
        else:
            # For 2-tracker ensembles or when outlier filtering is off,
            # use the simple weighted average.
            fused = _weighted_box_average(predictions, self._weights)

        return fused

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def last_individual_predictions(self) -> List[BBox]:
        """Individual member predictions from the most recent :meth:`update` call.

        Returns an empty list before the first update.  Useful for
        computing per-frame ensemble variance or diagnosing which member
        is failing on a given sequence.

        Example::

            fused = ensemble.update(frame)
            for t, bbox in zip(ensemble.trackers, ensemble.last_individual_predictions):
                print(t.name, bbox)
        """
        return list(self._last_individual)

    @property
    def trackers(self) -> List[BaseTracker]:
        """Read-only view of the member tracker list."""
        return list(self._trackers)

    def prediction_variance(self) -> float:
        """Mean squared deviation of member centres from the last fused prediction.

        Returns ``0.0`` before the first update or if no predictions are stored.

        This metric quantifies ensemble disagreement.  High variance on a
        frame correlates with genuine tracking difficulty (fast motion,
        occlusion, background clutter).
        """
        if not self._last_individual:
            return 0.0
        centres = np.array(
            [[b[0] + b[2] / 2.0, b[1] + b[3] / 2.0] for b in self._last_individual],
            dtype=np.float64,
        )
        mean_centre = centres.mean(axis=0)
        return float(np.mean(np.sum((centres - mean_centre) ** 2, axis=1)))
