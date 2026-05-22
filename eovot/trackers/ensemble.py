"""EnsembleTracker — combine multiple BaseTracker instances via prediction fusion.

Motivation
----------
Individual trackers exhibit complementary failure modes: MOSSE is fast but
drifts on complex backgrounds; KCF is more discriminative but slower; deep
trackers are accurate but memory-hungry.  An ensemble that intelligently
merges their predictions can be more robust than any single member, and
studying ensemble behaviour helps researchers understand the sources of
tracker error.

Fusion strategies
-----------------
- ``"mean"``     — unweighted coordinate average of all predictions.
                   Fast; best when all sub-trackers are equally reliable.
- ``"median"``   — coordinate-wise median.  Robust to one badly drifted
                   tracker without penalising the rest.  Recommended default.
- ``"nms_vote"`` — weighted box fusion: predictions that overlap with many
                   others (consensus) receive higher weight, giving a
                   majority-vote result that down-weights outlier trackers.

Usage::

    from eovot.trackers.ensemble import EnsembleTracker
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker

    ensemble = EnsembleTracker(
        trackers=[MOSSETracker(), KCFTracker()],
        strategy="median",
        name="MOSSE+KCF",
    )
    ensemble.initialize(first_frame, init_bbox)
    bbox = ensemble.update(next_frame)

Parallel execution
------------------
Set ``n_jobs > 1`` to run sub-tracker updates in a thread pool.  Note that
OpenCV-based trackers hold the GIL during C++ calls, so threading yields
benefit mainly for Python-heavy trackers or when ``n_jobs`` matches the
number of CPU cores and the GIL is released by the underlying C extension.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal, Optional

import numpy as np

from .base import BaseTracker, BBox

FusionStrategy = Literal["mean", "median", "nms_vote"]

_VALID_STRATEGIES = ("mean", "median", "nms_vote")


class EnsembleTracker(BaseTracker):
    """Combine multiple trackers by fusing their bounding-box predictions.

    Args:
        trackers: List of :class:`BaseTracker` instances to combine.  Must
            contain at least one tracker.
        strategy: Prediction fusion strategy — ``"mean"``, ``"median"``, or
            ``"nms_vote"``.  Default: ``"median"``.
        n_jobs:   Number of worker threads for parallel ``update`` calls.
            ``1`` (default) runs sub-trackers sequentially; higher values
            use a :class:`~concurrent.futures.ThreadPoolExecutor`.
        name:     Human-readable label used in benchmark reports.
            Default: ``"Ensemble"``.

    Raises:
        ValueError: If *trackers* is empty or *strategy* is unknown.

    Example::

        ens = EnsembleTracker(
            [MOSSETracker(), KCFTracker(), MOSSETracker(learning_rate=0.05)],
            strategy="nms_vote",
            name="VotingEnsemble",
        )
    """

    def __init__(
        self,
        trackers: List[BaseTracker],
        strategy: FusionStrategy = "median",
        n_jobs: int = 1,
        name: str = "Ensemble",
    ) -> None:
        if not trackers:
            raise ValueError("EnsembleTracker requires at least one sub-tracker.")
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown fusion strategy {strategy!r}. "
                f"Choose from {_VALID_STRATEGIES}."
            )
        super().__init__(name=name)
        self.trackers = trackers
        self.strategy: FusionStrategy = strategy
        self.n_jobs = max(1, n_jobs)

    # ------------------------------------------------------------------ #
    # BaseTracker interface                                                #
    # ------------------------------------------------------------------ #

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialize every sub-tracker with the same first frame and bbox.

        Args:
            frame: BGR image ``(H, W, 3)`` uint8.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        for tracker in self.trackers:
            tracker.initialize(frame, bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Collect predictions from all sub-trackers and fuse them.

        Args:
            frame: Next BGR frame ``(H, W, 3)`` uint8.

        Returns:
            Fused bounding box ``(x, y, w, h)``.
        """
        preds = self._collect_predictions(frame)
        return self._fuse(preds)

    # ------------------------------------------------------------------ #
    # Prediction collection                                               #
    # ------------------------------------------------------------------ #

    def _collect_predictions(self, frame: np.ndarray) -> np.ndarray:
        """Return an ``(M, 4)`` array of sub-tracker predictions."""
        if self.n_jobs == 1:
            return np.stack(
                [np.asarray(t.update(frame), dtype=np.float64) for t in self.trackers],
                axis=0,
            )

        results: List[Optional[np.ndarray]] = [None] * len(self.trackers)
        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            futures = {
                pool.submit(t.update, frame): i
                for i, t in enumerate(self.trackers)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = np.asarray(fut.result(), dtype=np.float64)
        return np.stack(results, axis=0)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Fusion strategies                                                   #
    # ------------------------------------------------------------------ #

    def _fuse(self, preds: np.ndarray) -> BBox:
        """Merge ``(M, 4)`` predictions into a single box."""
        if self.strategy == "mean":
            box = preds.mean(axis=0)
        elif self.strategy == "median":
            box = np.median(preds, axis=0)
        else:
            box = self._nms_vote(preds)
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

    @staticmethod
    def _nms_vote(preds: np.ndarray) -> np.ndarray:
        """Weighted box fusion: weight each prediction by its mean pairwise IoU.

        A prediction that overlaps strongly with the majority of other
        predictions (high consensus) receives high weight; an outlier
        prediction receives near-zero weight.  Falls back to uniform
        weighting when all predictions are non-overlapping.
        """
        M = len(preds)
        if M == 1:
            return preds[0].copy()

        x1 = preds[:, 0]
        y1 = preds[:, 1]
        x2 = preds[:, 0] + preds[:, 2]
        y2 = preds[:, 1] + preds[:, 3]
        areas = preds[:, 2] * preds[:, 3]

        weights = np.zeros(M, dtype=np.float64)
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                ix1 = max(x1[i], x1[j])
                iy1 = max(y1[i], y1[j])
                ix2 = min(x2[i], x2[j])
                iy2 = min(y2[i], y2[j])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = areas[i] + areas[j] - inter
                weights[i] += inter / (union + 1e-6)

        total = weights.sum()
        if total < 1e-8:
            # Non-overlapping outliers — fall back to uniform (= median-like)
            weights = np.ones(M, dtype=np.float64)
            total = float(M)
        weights /= total

        return (preds * weights[:, np.newaxis]).sum(axis=0)

    # ------------------------------------------------------------------ #
    # Repr                                                                #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        sub = ", ".join(t.name for t in self.trackers)
        return (
            f"EnsembleTracker(name={self.name!r}, "
            f"strategy={self.strategy!r}, "
            f"trackers=[{sub}])"
        )
