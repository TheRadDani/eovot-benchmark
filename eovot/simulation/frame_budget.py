"""Frame budget simulation for edge-constrained tracker evaluation.

On resource-limited devices a tracker frequently cannot process every
incoming camera frame within the capture period.  This module models
that scenario by running the tracker on a controlled *fraction* of frames
and propagating the last known bounding box for skipped frames.

The simulation answers two practical research questions:

1. **Accuracy degradation** — how much does IoU / AUC drop when the
   tracker processes only 50 %, 25 %, or 10 % of frames?
2. **Pareto-optimal deployment** — for a given device FPS cap, which
   tracker achieves the best accuracy?

The zero-motion propagation model is deliberately conservative: it assumes
the target does *not* move on skipped frames.  Real-world propagation
strategies (Kalman filtering, optical flow) would outperform this baseline,
making the curves a lower bound on achievable accuracy.

Typical usage::

    from eovot.simulation.frame_budget import FrameBudgetSimulator

    sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5, 0.25, 0.1])
    curve = sim.simulate(tracker, sequence, native_fps=200.0)
    FrameBudgetSimulator.print_curve(curve)

    # Serialise for analysis or plotting
    import json
    print(json.dumps(curve.to_dict(), indent=2))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..datasets.base import Sequence
from ..metrics.accuracy import AccuracyMetrics, MetricsEngine
from ..trackers.base import BaseTracker


@dataclass
class BudgetPoint:
    """Accuracy measurement at a single frame processing rate.

    Attributes:
        budget_rate:      Fraction of non-initialisation frames processed,
                          in the range ``(0, 1]``.  ``1.0`` means every frame;
                          ``0.5`` means every other frame.
        effective_fps:    Estimated real-world throughput at this budget,
                          computed as ``native_fps × budget_rate``.
                          Set to ``0.0`` when ``native_fps`` is not provided.
        accuracy:         :class:`~eovot.metrics.accuracy.AccuracyMetrics`
                          (mIoU, success AUC, precision AUC) measured at this
                          budget level.
        frames_processed: Number of frames the tracker actually ran on
                          (including the initialisation frame).
        frames_total:     Total frames in the sequence.
    """

    budget_rate: float
    effective_fps: float
    accuracy: AccuracyMetrics
    frames_processed: int
    frames_total: int

    @property
    def skip_ratio(self) -> float:
        """Fraction of frames that were *skipped* (complement of budget_rate)."""
        return 1.0 - self.budget_rate


@dataclass
class BudgetCurve:
    """Accuracy-vs-budget trade-off curve for one tracker on one sequence.

    Attributes:
        tracker_name:  Name of the evaluated tracker.
        sequence_name: Name of the evaluated sequence.
        native_fps:    Unthrottled tracker throughput (FPS) at full budget.
                       Used to compute :attr:`BudgetPoint.effective_fps`.
        points:        One :class:`BudgetPoint` per simulated budget rate,
                       ordered from highest to lowest budget.
    """

    tracker_name: str
    sequence_name: str
    native_fps: float
    points: List[BudgetPoint] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialise the full curve to a JSON-compatible dict."""
        return {
            "tracker_name": self.tracker_name,
            "sequence_name": self.sequence_name,
            "native_fps": round(self.native_fps, 2),
            "points": [
                {
                    "budget_rate": p.budget_rate,
                    "effective_fps": round(p.effective_fps, 2),
                    "mean_iou": round(p.accuracy.mean_iou, 4),
                    "success_auc": round(p.accuracy.success_auc, 4),
                    "precision_auc": round(p.accuracy.precision_auc, 4),
                    "frames_processed": p.frames_processed,
                    "frames_total": p.frames_total,
                }
                for p in self.points
            ],
        }


class FrameBudgetSimulator:
    """Simulate tracker accuracy under frame budget (temporal subsampling) constraints.

    For each budget rate in *budget_rates*, the simulator:

    1. Re-initialises the tracker on the first frame of the sequence.
    2. Selects a uniformly distributed subset of frames to process.
    3. For skipped frames, copies the last predicted bounding box (zero-motion).
    4. Computes :class:`~eovot.metrics.accuracy.AccuracyMetrics` over the
       full sequence including propagated boxes.

    Re-initialisation per budget level ensures each point is an independent,
    fair measurement.

    Args:
        budget_rates: Frame processing fractions to simulate.  Each value
            must be in ``(0, 1]``.  Default: ``[1.0, 0.75, 0.5, 0.25, 0.1]``.
        native_fps:   Unthrottled tracker throughput used to compute effective
            FPS for each budget point.  Can be overridden per :meth:`simulate`
            call.  Default: ``0.0`` (effective FPS reported as 0.0).

    Raises:
        ValueError: If any rate in *budget_rates* is outside ``(0, 1]``.

    Example::

        sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5, 0.25, 0.1], native_fps=200.0)
        curve = sim.simulate(tracker, sequence)
        FrameBudgetSimulator.print_curve(curve)
    """

    DEFAULT_RATES: List[float] = [1.0, 0.75, 0.5, 0.25, 0.1]

    def __init__(
        self,
        budget_rates: Optional[List[float]] = None,
        native_fps: float = 0.0,
    ) -> None:
        rates = budget_rates if budget_rates is not None else list(self.DEFAULT_RATES)
        invalid = [r for r in rates if not (0 < r <= 1.0)]
        if invalid:
            raise ValueError(
                f"All budget_rates must be in (0, 1].  "
                f"Invalid values: {invalid}"
            )
        # Store highest-to-lowest so curves read naturally (full → minimal budget)
        self.budget_rates: List[float] = sorted(rates, reverse=True)
        self._native_fps = native_fps
        self._metrics = MetricsEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        tracker: BaseTracker,
        sequence: Sequence,
        native_fps: Optional[float] = None,
    ) -> BudgetCurve:
        """Evaluate *tracker* on *sequence* across all configured budget rates.

        The tracker is re-initialised from frame 0 for each budget rate to
        ensure measurements are independent.

        Args:
            tracker:    Tracker to evaluate.  Must implement
                        :class:`~eovot.trackers.base.BaseTracker`.
            sequence:   Dataset sequence to evaluate on.
            native_fps: Overrides the constructor-level ``native_fps`` for
                        effective-FPS computation.  Pass the unthrottled FPS
                        measured by a full benchmark run.

        Returns:
            :class:`BudgetCurve` with one :class:`BudgetPoint` per budget rate.
        """
        fps = native_fps if native_fps is not None else self._native_fps
        frames = list(sequence)
        gt = np.array(sequence.ground_truth, dtype=np.float64)
        n_total = len(frames)

        curve = BudgetCurve(
            tracker_name=tracker.name,
            sequence_name=sequence.name,
            native_fps=fps,
        )

        for rate in self.budget_rates:
            preds = self._run_with_budget(tracker, frames, gt, rate)
            n_eval = min(len(preds), len(gt))
            acc = self._metrics.compute_all(preds[:n_eval], gt[:n_eval])

            n_update_frames = max(1, round(rate * max(n_total - 1, 1)))
            point = BudgetPoint(
                budget_rate=rate,
                effective_fps=fps * rate,
                accuracy=acc,
                frames_processed=n_update_frames + 1,  # +1 for init frame
                frames_total=n_total,
            )
            curve.points.append(point)

        return curve

    def simulate_dataset(
        self,
        tracker: BaseTracker,
        sequences: List[Sequence],
        native_fps: Optional[float] = None,
    ) -> List[BudgetCurve]:
        """Evaluate *tracker* on a list of sequences and return one curve each.

        Args:
            tracker:    Tracker to evaluate.
            sequences:  Iterable of :class:`~eovot.datasets.base.Sequence`.
            native_fps: Unthrottled tracker FPS (forwarded to :meth:`simulate`).

        Returns:
            List of :class:`BudgetCurve` objects in the same order as *sequences*.
        """
        return [self.simulate(tracker, seq, native_fps=native_fps) for seq in sequences]

    # ------------------------------------------------------------------
    # Pretty-print helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_curve(curve: BudgetCurve) -> None:
        """Print a formatted summary table of one budget curve.

        Args:
            curve: Output from :meth:`simulate`.
        """
        fps_hdr = f"Native FPS: {curve.native_fps:.1f}" if curve.native_fps > 0 else "Native FPS: N/A"
        print(f"\nFrame Budget Curve: {curve.tracker_name} on {curve.sequence_name}")
        print(fps_hdr)
        print("-" * 72)
        print(
            f"{'Budget':>8}  {'Eff. FPS':>10}  "
            f"{'mIoU':>8}  {'S-AUC':>8}  {'P-AUC':>8}  {'Frames':>12}"
        )
        print("-" * 72)
        for p in curve.points:
            fps_str = f"{p.effective_fps:.1f}" if curve.native_fps > 0 else "N/A"
            print(
                f"{p.budget_rate:>7.0%}  {fps_str:>10}  "
                f"{p.accuracy.mean_iou:>8.4f}  "
                f"{p.accuracy.success_auc:>8.4f}  "
                f"{p.accuracy.precision_auc:>8.4f}  "
                f"{p.frames_processed}/{p.frames_total}"
            )
        print("-" * 72)

    @staticmethod
    def aggregate_curves(curves: List[BudgetCurve]) -> Dict[float, AccuracyMetrics]:
        """Average accuracy metrics across multiple curves at each budget rate.

        Useful for reporting dataset-level results: average the per-sequence
        curves to get a single curve representing overall tracker performance.

        Args:
            curves: List of curves from :meth:`simulate` or :meth:`simulate_dataset`.
                    All curves must share the same budget rates.

        Returns:
            Dict mapping budget_rate → mean :class:`~eovot.metrics.accuracy.AccuracyMetrics`.
        """
        if not curves:
            return {}

        rate_to_ious: Dict[float, List[float]] = {}
        rate_to_sauc: Dict[float, List[float]] = {}
        rate_to_pauc: Dict[float, List[float]] = {}

        for curve in curves:
            for pt in curve.points:
                rate_to_ious.setdefault(pt.budget_rate, []).append(pt.accuracy.mean_iou)
                rate_to_sauc.setdefault(pt.budget_rate, []).append(pt.accuracy.success_auc)
                rate_to_pauc.setdefault(pt.budget_rate, []).append(pt.accuracy.precision_auc)

        result: Dict[float, AccuracyMetrics] = {}
        for rate in sorted(rate_to_ious, reverse=True):
            result[rate] = AccuracyMetrics(
                mean_iou=float(np.mean(rate_to_ious[rate])),
                success_auc=float(np.mean(rate_to_sauc[rate])),
                precision_auc=float(np.mean(rate_to_pauc[rate])),
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_with_budget(
        self,
        tracker: BaseTracker,
        frames: List[np.ndarray],
        gt: np.ndarray,
        budget_rate: float,
    ) -> np.ndarray:
        """Run the tracker on a uniformly subsampled set of frames.

        Skipped frames receive the last predicted bounding box (zero-motion
        propagation).  The tracker is always re-initialised on frame 0 from
        the ground-truth box.

        Args:
            tracker:     Tracker instance to re-initialise.
            frames:      Sequence frames as BGR numpy arrays.
            gt:          Ground-truth boxes, shape ``(N, 4)``.
            budget_rate: Fraction of non-init frames to process.

        Returns:
            Predicted bounding boxes, shape ``(N, 4)``.
        """
        n = len(frames)
        if n == 0:
            return np.empty((0, 4), dtype=np.float64)

        preds = np.empty((n, 4), dtype=np.float64)

        init_bbox: BaseTracker.BBox = tuple(gt[0].tolist())  # type: ignore[attr-defined]
        tracker.initialize(frames[0], init_bbox)
        preds[0] = gt[0]

        if n == 1:
            return preds

        process_mask = self._build_process_mask(n - 1, budget_rate)
        last_bbox = np.array(gt[0], dtype=np.float64)

        for i in range(1, n):
            if process_mask[i - 1]:
                bbox = tracker.update(frames[i])
                last_bbox = np.array(bbox, dtype=np.float64)
            preds[i] = last_bbox

        return preds

    @staticmethod
    def _build_process_mask(n_frames: int, rate: float) -> np.ndarray:
        """Build a boolean mask for uniform frame subsampling.

        Selects approximately ``round(rate × n_frames)`` indices, spaced as
        evenly as possible across ``[0, n_frames - 1]`` using ``np.linspace``.

        Args:
            n_frames: Number of non-initialisation frames.
            rate:     Fraction to process, in ``(0, 1]``.

        Returns:
            Boolean array of length ``n_frames``; ``True`` means process.
        """
        if n_frames == 0:
            return np.empty(0, dtype=bool)
        mask = np.zeros(n_frames, dtype=bool)
        n_process = max(1, round(rate * n_frames))
        indices = np.round(np.linspace(0, n_frames - 1, n_process)).astype(int)
        mask[indices] = True
        return mask
