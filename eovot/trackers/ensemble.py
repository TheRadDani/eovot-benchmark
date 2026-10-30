"""Consensus-based ensemble tracker for EOVOT.

Research motivation
-------------------
Individual lightweight trackers (MOSSE, KCF, MedianFlow, …) each have
different failure modes: MOSSE drifts on scale change, KCF struggles with
fast motion, MedianFlow fails on heavy occlusion.  By running several cheap
trackers in parallel and selecting the output that maximises *consensus*
with all others, the ensemble achieves improved robustness without requiring
a more expensive backbone.

Algorithm
---------
On each frame, every member tracker produces a predicted box.  The ensemble
selects the prediction whose mean pairwise IoU against **all other**
predictions is highest — this is the box most "agreed upon" across members.
In the degenerate case of a single member the sole prediction is returned.

The selector is purely geometric (no learned component) so it generalises
across datasets and requires no fine-tuning.  The computational overhead is
``O(M²)`` pairwise IoU comparisons where ``M`` is the number of members —
negligible for ``M ≤ 8`` because each comparison is an arithmetic vector
operation.

This design follows the classical multi-expert selection framework from
MEEM (Zhang et al., ECCV 2014) generalised to any set of trackers.

Example
-------
::

    from eovot.trackers.ensemble import EnsembleTracker
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.median_flow import MedianFlowTracker

    tracker = EnsembleTracker(
        members=[MOSSETracker(), KCFTracker(), MedianFlowTracker()],
        name="Ensemble-MKM",
    )
    tracker.initialize(frame0, bbox0)
    pred = tracker.update(frame1)

CLI / config usage::

    trackers:
      - name: Ensemble
        params:
          member_names: [MOSSE, KCF, MedianFlow]

Registry alias ``"Ensemble"`` builds a default three-member ensemble
(MOSSE + KCF + MedianFlow) when ``member_names`` is not overridden.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox


def _iou(a: BBox, b: BBox) -> float:
    """Compute IoU between two ``(x, y, w, h)`` boxes."""
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


class EnsembleTracker(BaseTracker):
    """Consensus-based ensemble that selects the prediction with highest
    mean pairwise IoU against all other member predictions.

    Args:
        members: List of :class:`~eovot.trackers.base.BaseTracker` instances
            to run in parallel.  Must contain at least one member.
        name: Human-readable label used in benchmark reports.
            Defaults to ``"Ensemble(<member names>)"``.

    Raises:
        ValueError: If ``members`` is empty.
    """

    def __init__(
        self,
        members: List[BaseTracker],
        name: Optional[str] = None,
    ) -> None:
        if not members:
            raise ValueError("EnsembleTracker requires at least one member tracker.")
        if name is None:
            name = "Ensemble(" + "+".join(m.name for m in members) + ")"
        super().__init__(name=name)
        self.members = members

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise all member trackers on the first frame.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        for member in self.members:
            member.initialize(frame, bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location via consensus selection.

        Each member tracker produces a prediction.  The prediction with
        the highest mean pairwise IoU against all others is returned.
        When all pairwise IoUs are zero (e.g. first frame on some trackers)
        the first member's prediction is used as the fallback.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            The consensus bounding box ``(x, y, w, h)``.
        """
        predictions: List[BBox] = [m.update(frame) for m in self.members]

        if len(predictions) == 1:
            return predictions[0]

        scores = self._consensus_scores(predictions)
        best_idx = int(np.argmax(scores))
        return predictions[best_idx]

    # ------------------------------------------------------------------
    # Consensus computation
    # ------------------------------------------------------------------

    @staticmethod
    def _consensus_scores(predictions: List[BBox]) -> np.ndarray:
        """Compute per-prediction mean pairwise IoU against all others.

        Args:
            predictions: List of ``(x, y, w, h)`` boxes, one per member.

        Returns:
            ``(M,)`` float array of consensus scores; higher = more agreed upon.
        """
        m = len(predictions)
        scores = np.zeros(m, dtype=np.float64)
        for i in range(m):
            total = 0.0
            for j in range(m):
                if i != j:
                    total += _iou(predictions[i], predictions[j])
            scores[i] = total / (m - 1)
        return scores

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_names(
        cls,
        member_names: List[str],
        name: Optional[str] = None,
        **member_kwargs,
    ) -> "EnsembleTracker":
        """Build an ensemble from a list of tracker registry names.

        Args:
            member_names: Registry keys, e.g. ``["MOSSE", "KCF", "MedianFlow"]``.
            name: Optional human-readable label for the ensemble.
            **member_kwargs: Keyword arguments forwarded to every member's constructor.

        Returns:
            A new :class:`EnsembleTracker` with the requested members.

        Example::

            tracker = EnsembleTracker.from_names(["MOSSE", "KCF", "MedianFlow"])
        """
        from .registry import build_tracker

        members = [build_tracker(n, **member_kwargs) for n in member_names]
        return cls(members=members, name=name)
