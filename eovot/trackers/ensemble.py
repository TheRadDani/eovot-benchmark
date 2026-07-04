"""Ensemble tracker — fuse predictions from N base trackers into one.

Running multiple trackers in parallel and combining their predictions can
improve robustness over any single algorithm:

* When one tracker drifts or fails, the others can anchor the fused result.
* Different trackers have complementary strengths: MOSSE is fast but
  sensitive to illumination changes; CSRT is accurate but slow; KCF sits
  between them.  A fusion of all three amortises their failure modes.

This module provides :class:`EnsembleTracker`, a
:class:`~eovot.trackers.base.BaseTracker`-compatible wrapper with two
fusion strategies:

``"mean"`` — weighted centroid average of all component predictions.
    Fast and smooth.  A single drifted tracker pulls the fused box
    proportionally to its weight, so a low-confidence component can be
    down-weighted at construction time.

``"vote"`` — select the prediction with the highest mean pairwise IoU
    against all other predictions (consensus selection).
    More robust to a single outlier: if one tracker drifts badly, the
    remaining trackers will agree with each other and their shared box
    wins.  O(N²) IoU comparisons per frame, but N is typically ≤ 5.

Typical usage::

    from eovot.trackers.ensemble import EnsembleTracker
    from eovot.trackers.registry import build_tracker

    ensemble = EnsembleTracker(
        base_trackers=[
            build_tracker("MOSSE"),
            build_tracker("KCF", learning_rate=0.075),
            build_tracker("CSRT"),
        ],
        fusion="vote",
    )
    ensemble.initialize(frame, (x, y, w, h))
    for frame in video:
        bbox = ensemble.update(frame)

    # Weighted mean: trust CSRT twice as much as MOSSE/KCF
    ensemble = EnsembleTracker(
        base_trackers=[build_tracker("MOSSE"), build_tracker("KCF"), build_tracker("CSRT")],
        fusion="mean",
        weights=[1.0, 1.0, 2.0],
    )
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BaseTracker, BBox


class EnsembleTracker(BaseTracker):
    """Fuse predictions from N base trackers into a single robust output.

    Args:
        base_trackers: Non-empty list of fully-constructed
            :class:`~eovot.trackers.base.BaseTracker` instances.
        fusion: Fusion strategy — ``"mean"`` (weighted average) or
            ``"vote"`` (consensus selection by pairwise IoU).
        weights: Positive floats, one per tracker in *base_trackers*.
            Used only by the ``"mean"`` strategy.  Defaults to equal
            weights.  The values are normalised internally so only their
            relative magnitudes matter.
        name: Human-readable name for reports.  Auto-generated from
            component names and fusion strategy when omitted.

    Raises:
        ValueError: If *base_trackers* is empty, *fusion* is unrecognised,
            or *weights* length does not match *base_trackers*.
    """

    _VALID_FUSION = frozenset({"mean", "vote"})

    def __init__(
        self,
        base_trackers: List[BaseTracker],
        fusion: str = "mean",
        weights: Optional[List[float]] = None,
        name: Optional[str] = None,
    ) -> None:
        if not base_trackers:
            raise ValueError("base_trackers must not be empty.")
        if fusion not in self._VALID_FUSION:
            raise ValueError(
                f"Unknown fusion '{fusion}'. Valid options: {sorted(self._VALID_FUSION)}"
            )
        if weights is not None:
            if len(weights) != len(base_trackers):
                raise ValueError(
                    f"len(weights)={len(weights)} != "
                    f"len(base_trackers)={len(base_trackers)}"
                )
            if any(w <= 0 for w in weights):
                raise ValueError(
                    "All weights must be strictly positive, got: "
                    + str(weights)
                )

        component_names = "+".join(t.name for t in base_trackers)
        default_name = f"Ensemble({component_names},{fusion})"
        super().__init__(name=name if name is not None else default_name)

        self.base_trackers: List[BaseTracker] = list(base_trackers)
        self.fusion: str = fusion
        self._weights: np.ndarray = (
            np.array(weights, dtype=np.float64)
            if weights is not None
            else np.ones(len(base_trackers), dtype=np.float64)
        )
        # Normalise so weights sum to 1 (only matters for "mean", but harmless)
        self._weights /= self._weights.sum()

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise every component tracker on the first frame.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        for tracker in self.base_trackers:
            tracker.initialize(frame, bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Run all component trackers and return the fused bounding box.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 array.

        Returns:
            Fused bounding box ``(x, y, w, h)``.
        """
        predictions: List[BBox] = [t.update(frame) for t in self.base_trackers]

        if len(predictions) == 1:
            return predictions[0]

        if self.fusion == "mean":
            return self._mean_fusion(predictions)
        return self._vote_fusion(predictions)

    # ------------------------------------------------------------------
    # Fusion strategies
    # ------------------------------------------------------------------

    def _mean_fusion(self, predictions: List[BBox]) -> BBox:
        """Weighted centroid average of all component predictions.

        Each box ``(x, y, w, h)`` is blended in proportion to its weight.
        The result is the component-wise weighted sum of all four coordinates.

        Args:
            predictions: List of ``(x, y, w, h)`` tuples, one per tracker.

        Returns:
            Weighted-average bounding box ``(x, y, w, h)``.
        """
        preds = np.array(predictions, dtype=np.float64)  # (N, 4)
        blended = (preds * self._weights[:, np.newaxis]).sum(axis=0)  # (4,)
        return (float(blended[0]), float(blended[1]),
                float(blended[2]), float(blended[3]))

    def _vote_fusion(self, predictions: List[BBox]) -> BBox:
        """Select the prediction with the highest mean pairwise IoU (consensus).

        For each candidate prediction, compute its average IoU against all
        other predictions.  The candidate with the highest mean pairwise IoU
        is the "consensus" box — the one most agreed upon by the ensemble —
        and is returned as-is.

        Robustness property: if a single tracker drifts far from the target,
        its prediction will have low IoU against the other N-1 trackers and
        will not be selected, provided the remaining trackers still agree.

        Args:
            predictions: List of ``(x, y, w, h)`` tuples, one per tracker.

        Returns:
            The prediction (unchanged) that maximises mean pairwise IoU.
        """
        n = len(predictions)
        mean_pairwise = np.zeros(n, dtype=np.float64)
        for i in range(n):
            iou_sum = 0.0
            for j in range(n):
                if i != j:
                    iou_sum += _iou(predictions[i], predictions[j])
            mean_pairwise[i] = iou_sum / (n - 1)

        best = int(np.argmax(mean_pairwise))
        return predictions[best]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        components = ", ".join(repr(t) for t in self.base_trackers)
        return (
            f"EnsembleTracker("
            f"name={self.name!r}, "
            f"fusion={self.fusion!r}, "
            f"base_trackers=[{components}])"
        )


# ---------------------------------------------------------------------------
# Module-level IoU (avoids circular import with eovot.metrics)
# ---------------------------------------------------------------------------

def _iou(pred: BBox, gt: BBox) -> float:
    """Axis-aligned bounding-box IoU for ``(x, y, w, h)`` boxes."""
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    if pw <= 0 or ph <= 0 or gw <= 0 or gh <= 0:
        return 0.0
    ix1 = max(px, gx)
    iy1 = max(py, gy)
    ix2 = min(px + pw, gx + gw)
    iy2 = min(py + ph, gy + gh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return float(inter / union) if union > 0 else 0.0
