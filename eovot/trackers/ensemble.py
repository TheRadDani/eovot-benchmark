"""
ConsensusTracker — multi-tracker ensemble with bounding-box fusion.

Running a single tracker on edge hardware is fast but fragile: one bad
appearance model or a moment of occlusion causes drift that propagates for
many frames.  An ensemble of lightweight trackers is often more robust than
any individual member, at a compute cost that scales linearly with ensemble
size.

:class:`ConsensusTracker` wraps an arbitrary list of :class:`BaseTracker`
instances and combines their per-frame predictions into a single bounding box
using one of three fusion strategies:

``"mean"``
    Simple unweighted average of all predicted centres and dimensions.
    Fast, symmetric, suitable when all member trackers are equally reliable.

``"iou_weighted"``
    Weight each prediction by how well it agrees with the rest of the ensemble.
    Agreement is measured as the mean pairwise IoU between a candidate box and
    every other member's prediction.  Outlier trackers (e.g. ones that have
    drifted) receive low weight; the consensus view receives high weight.

``"smallest_box"``
    Return the prediction from the member that produced the smallest bounding
    box — a conservative heuristic that tends to avoid over-expansion after
    occlusion.

The ensemble exposes the individual member predictions (``last_predictions``)
after each update, enabling post-hoc analysis of how much members agree.
The inter-member agreement score (``agreement``) is the mean pairwise IoU
over all pairs — useful as a proxy confidence signal for downstream
decision-making (e.g. trigger a re-detection when agreement drops).

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.ensemble import ConsensusTracker

    ensemble = ConsensusTracker(
        [MOSSETracker(), KCFTracker()],
        fusion="iou_weighted",
    )

    # Drop-in replacement for any BaseTracker
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    engine  = BenchmarkEngine(verbose=False)
    dataset = SyntheticDataset(num_sequences=3)
    result  = engine.run(ensemble, dataset, dataset_name="Syn")
    print(f"Mean IoU: {result.mean_iou:.4f}")
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox

FusionMode = Literal["mean", "iou_weighted", "smallest_box"]


def _bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-Union between two ``(x, y, w, h)`` boxes.

    Returns ``0.0`` when either box has zero area.
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _mean_fusion(predictions: List[BBox]) -> BBox:
    """Unweighted mean of all predicted bounding boxes."""
    arr = np.array(predictions, dtype=np.float64)
    mean = arr.mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]), float(mean[3]))


def _iou_weighted_fusion(predictions: List[BBox]) -> BBox:
    """Fuse predictions weighted by mean pairwise IoU agreement.

    Each prediction's weight is its mean IoU with every other prediction.
    A tracker that disagrees with the rest gets a low weight; the majority
    view gets a proportionally higher weight.  Falls back to mean fusion
    when all predictions are identical or the weights degenerate.
    """
    n = len(predictions)
    if n == 1:
        return predictions[0]

    weights = np.zeros(n, dtype=np.float64)
    for i in range(n):
        total_iou = 0.0
        for j in range(n):
            if i != j:
                total_iou += _bbox_iou(predictions[i], predictions[j])
        weights[i] = total_iou / (n - 1)

    w_sum = weights.sum()
    if w_sum < 1e-12:
        return _mean_fusion(predictions)

    weights /= w_sum
    arr = np.array(predictions, dtype=np.float64)
    fused = np.einsum("i,ij->j", weights, arr)
    return (float(fused[0]), float(fused[1]), float(fused[2]), float(fused[3]))


def _smallest_box_fusion(predictions: List[BBox]) -> BBox:
    """Return the prediction with the smallest bounding-box area."""
    return min(predictions, key=lambda b: b[2] * b[3])


def _pairwise_mean_iou(predictions: List[BBox]) -> float:
    """Mean IoU over all unique pairs of predictions.

    Returns 1.0 when the ensemble has only one member (trivial agreement).
    """
    n = len(predictions)
    if n < 2:
        return 1.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _bbox_iou(predictions[i], predictions[j])
            count += 1
    return total / count if count > 0 else 0.0


class ConsensusTracker(BaseTracker):
    """Ensemble of trackers fused into a single prediction.

    All member trackers are driven identically (same frames, same initial
    bbox) and their per-frame predictions are combined using the chosen
    *fusion* strategy.

    Args:
        trackers: Non-empty list of :class:`BaseTracker` instances.  Each
            member is driven independently; their internal states do not
            interact.
        fusion:   Prediction fusion strategy.  One of:

                  * ``"mean"``         — simple coordinate average.
                  * ``"iou_weighted"`` — weighted by pairwise IoU agreement.
                  * ``"smallest_box"`` — most conservative spatial estimate.

                  Default: ``"iou_weighted"``.

    Raises:
        ValueError: If *trackers* is empty or *fusion* is unrecognised.

    Properties:
        last_predictions: List of individual member predictions from the
            most recent :meth:`update` call (empty before first update).
        agreement:        Mean pairwise IoU among ``last_predictions``
            (1.0 before first update / single-member ensemble).
    """

    _FUSION_MODES = {"mean", "iou_weighted", "smallest_box"}

    def __init__(
        self,
        trackers: List[BaseTracker],
        fusion: FusionMode = "iou_weighted",
    ) -> None:
        if not trackers:
            raise ValueError("trackers list must not be empty")
        if fusion not in self._FUSION_MODES:
            raise ValueError(
                f"Unknown fusion mode '{fusion}'. "
                f"Choose from {sorted(self._FUSION_MODES)}."
            )

        names = "+".join(t.name for t in trackers)
        super().__init__(name=f"Consensus({names})[{fusion}]")
        self._trackers = list(trackers)
        self._fusion: FusionMode = fusion
        self._last_predictions: List[BBox] = []
        self._agreement: float = 1.0

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise all member trackers on the same first frame.

        Args:
            frame: First video frame as a ``(H, W, 3)`` BGR uint8 array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        for tracker in self._trackers:
            tracker.initialize(frame, bbox)
        self._last_predictions = []
        self._agreement = 1.0

    def update(self, frame: np.ndarray) -> BBox:
        """Run all member trackers and return their fused prediction.

        Args:
            frame: Current frame as a ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Fused bounding box ``(x, y, w, h)``.
        """
        predictions: List[BBox] = [t.update(frame) for t in self._trackers]
        self._last_predictions = predictions
        self._agreement = _pairwise_mean_iou(predictions)
        return self._fuse(predictions)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def last_predictions(self) -> List[BBox]:
        """Individual member predictions from the most recent ``update()`` call."""
        return list(self._last_predictions)

    @property
    def agreement(self) -> float:
        """Mean pairwise IoU among member predictions after the last update.

        Values close to 1.0 indicate high consensus; values near 0.0
        indicate the ensemble is diverging — a signal to consider
        re-initialising or falling back to a slower, more reliable tracker.
        """
        return self._agreement

    @property
    def members(self) -> List[BaseTracker]:
        """Ordered list of member :class:`BaseTracker` instances."""
        return list(self._trackers)

    @property
    def fusion_mode(self) -> FusionMode:
        """The active fusion strategy."""
        return self._fusion

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fuse(self, predictions: List[BBox]) -> BBox:
        if self._fusion == "mean":
            return _mean_fusion(predictions)
        if self._fusion == "iou_weighted":
            return _iou_weighted_fusion(predictions)
        if self._fusion == "smallest_box":
            return _smallest_box_fusion(predictions)
        raise RuntimeError(f"Unknown fusion mode: {self._fusion!r}")  # unreachable
