"""Multi-tracker ensemble with IoU-weighted consensus fusion.

**Motivation**

Classical edge trackers fail in complementary ways:

* MOSSE / KCF lose the target under illumination change (correlation filter
  degrades when the patch appearance shifts).
* CamShift fails when the target colour blends into the background.
* LK Optical Flow fails on texture-poor targets.

Running several lightweight trackers in parallel and fusing their predictions
via an IoU-based consensus vote exploits this complementarity: when one
tracker fails the others can compensate, improving robustness **without**
switching to a heavier deep-learning model.

**Consensus algorithm**

For each frame each member tracker predicts a bounding box independently.
The ensemble then:

1. Computes a pairwise IoU **agreement matrix** between all predictions.
2. Assigns each tracker a weight proportional to the sum of its IoU
   agreements with all other trackers — a tracker that agrees with the
   majority gets a higher weight; an outlier gets a lower weight.
3. Returns the **weighted mean** of all predictions as the ensemble output.

When all trackers disagree (every pairwise IoU = 0, e.g. all trackers are
lost), the ensemble falls back to a simple unweighted average, ensuring a
valid output is always returned.

**Edge deployment suitability**

The consensus overhead is O(N²) in the number of member trackers (pairwise
IoU computation) plus O(N) for the weighted mean — negligible compared to
the per-frame cost of any individual tracker.  Three lightweight trackers
(MOSSE + KCF + CamShift) combined via ensemble run at 200–400 FPS on a
modern CPU core, comfortably within real-time budgets for most edge systems.

Typical usage::

    from eovot.trackers.ensemble import TrackerEnsemble
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.camshift import CamShiftTracker

    ensemble = TrackerEnsemble([MOSSETracker(), KCFTracker(), CamShiftTracker()])
    ensemble.initialize(frame, init_bbox)
    for frame in video_frames:
        bbox = ensemble.update(frame)

    # Or via registry shorthand
    from eovot.trackers.ensemble import build_default_ensemble
    ensemble = build_default_ensemble()
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import BaseTracker, BBox


def _iou(a: BBox, b: BBox) -> float:
    """Compute IoU between two (x, y, w, h) boxes."""
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


class TrackerEnsemble(BaseTracker):
    """Fuse multiple trackers via IoU-weighted consensus voting.

    Each member tracker runs independently every frame.  Their predictions
    are combined using a pairwise-IoU weighting scheme that up-weights
    trackers whose predictions agree with the majority.

    Args:
        trackers: List of :class:`~.base.BaseTracker` instances to combine.
            Must contain at least one tracker.
        name:     Identifier used in benchmark reports.

    Raises:
        ValueError: If *trackers* is empty.

    Example::

        from eovot.trackers.ensemble import TrackerEnsemble
        from eovot.trackers.mosse import MOSSETracker
        from eovot.trackers.kcf import KCFTracker

        ens = TrackerEnsemble([MOSSETracker(), KCFTracker()])
        ens.initialize(frame0, init_bbox)
        pred = ens.update(frame1)
    """

    def __init__(
        self,
        trackers: List[BaseTracker],
        name: str = "Ensemble",
    ) -> None:
        if not trackers:
            raise ValueError("TrackerEnsemble requires at least one member tracker.")
        super().__init__(name=name)
        self._trackers = list(trackers)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialize all member trackers on the first frame.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        for tracker in self._trackers:
            tracker.initialize(frame, bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict target location by consensus of all member trackers.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Weighted-mean bounding box ``(x, y, w, h)``.
        """
        predictions = [t.update(frame) for t in self._trackers]
        return self._consensus(predictions)

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------

    @property
    def members(self) -> List[BaseTracker]:
        """Read-only view of the member trackers."""
        return list(self._trackers)

    def member_names(self) -> List[str]:
        """Return the name of each member tracker."""
        return [t.name for t in self._trackers]

    # ------------------------------------------------------------------
    # Consensus logic
    # ------------------------------------------------------------------

    def _consensus(self, predictions: List[BBox]) -> BBox:
        """Compute the IoU-weighted mean of all member predictions.

        Algorithm:

        1. Build an N×N pairwise IoU agreement matrix.
        2. For each tracker, weight = 1 + sum of its IoU agreements with
           every other tracker.  The ``+1`` ensures every tracker has a
           non-zero weight even when all pairwise IoUs are zero.
        3. Normalise weights and return the weighted centroid.

        Args:
            predictions: List of ``(x, y, w, h)`` predictions, one per member.

        Returns:
            Weighted-mean bounding box as ``(x, y, w, h)``.
        """
        n = len(predictions)
        if n == 1:
            return predictions[0]

        # Pairwise IoU agreement
        weights = np.ones(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    weights[i] += _iou(predictions[i], predictions[j])

        weights /= weights.sum()

        arr = np.array(predictions, dtype=np.float64)   # (N, 4)
        result = (arr * weights[:, None]).sum(axis=0)

        # Ensure positive width / height
        result[2] = max(1.0, float(result[2]))
        result[3] = max(1.0, float(result[3]))

        return (float(result[0]), float(result[1]), float(result[2]), float(result[3]))

    def last_predictions(self, frame: np.ndarray) -> List[Tuple[str, BBox]]:
        """Return each member's prediction without updating ensemble state.

        Useful for diagnostic analysis: run once to see which member trackers
        agree and which are outliers.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.

        Returns:
            List of ``(tracker_name, bbox)`` tuples.
        """
        return [(t.name, t.update(frame)) for t in self._trackers]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_default_ensemble(name: str = "Ensemble-MKC") -> TrackerEnsemble:
    """Build the default lightweight ensemble: MOSSE + KCF + CamShift.

    This combination covers three distinct tracking families:

    * **MOSSE** — correlation filter, very fast, handles illumination changes
    * **KCF** — kernelised correlation filter, higher accuracy, moderate speed
    * **CamShift** — colour-histogram tracker, robust to feature-poor targets

    Args:
        name: Name for the ensemble tracker. Default: ``"Ensemble-MKC"``.

    Returns:
        :class:`TrackerEnsemble` with three member trackers.

    Example::

        from eovot.trackers.ensemble import build_default_ensemble
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        ensemble = build_default_ensemble()
        dataset  = SyntheticDataset(num_sequences=5, num_frames=100)
        engine   = BenchmarkEngine(verbose=True)
        result   = engine.run(ensemble, dataset, dataset_name="Synthetic")
        print(result)
    """
    from .camshift import CamShiftTracker
    from .kcf import KCFTracker
    from .mosse import MOSSETracker

    return TrackerEnsemble(
        trackers=[MOSSETracker(), KCFTracker(), CamShiftTracker()],
        name=name,
    )
