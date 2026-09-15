"""Multi-objective Pareto frontier analysis for VOT benchmark results.

This module operates in the *raw* metric space — (accuracy, latency, memory) —
without collapsing objectives into a composite score.  It complements the
EES-based composite scoring in ``eovot.metrics.efficiency``.

Key concepts
~~~~~~~~~~~~
**Pareto dominance**: Tracker A dominates B iff A is at least as good on every
objective and strictly better on at least one.  The Pareto-optimal set contains
all non-dominated trackers.

**Hypervolume indicator** (HV): The volume of objective space dominated by the
Pareto front with respect to a reference point.  Larger HV = better front.
This is the de-facto standard for quantifying multi-objective result quality
and enables objective comparison across algorithmic designs or paper results.

Usage example::

    from eovot.analysis.pareto import ParetoFrontier

    frontier = ParetoFrontier.from_benchmark_results(
        results, accuracy_key="success_auc"
    )

    print("Pareto-optimal trackers (accuracy vs latency):")
    for p in frontier.frontier_2d:
        print(f"  {p.tracker_name}: AUC={p.accuracy:.3f}, lat={p.latency_ms:.1f}ms")

    # Best tracker under a 15 ms latency budget
    best = frontier.best_under_latency(15.0)
    if best:
        print(f"Best under 15ms: {best.tracker_name} AUC={best.accuracy:.3f}")

    # Hypervolume of the Pareto front
    hv = frontier.hypervolume_2d(ref_accuracy=0.0, ref_latency=200.0)
    print(f"Hypervolume indicator: {hv:.4f}")

    print(frontier.to_markdown())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class TrackerPoint:
    """A single data-point in accuracy-efficiency space for one tracker run.

    All coordinates are non-negative scalars.  Higher ``accuracy`` is better;
    lower ``latency_ms`` and ``memory_mb`` are better.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        accuracy: Accuracy scalar, e.g. success AUC or mean IoU, in ``[0, 1]``.
        latency_ms: Mean per-frame processing latency in milliseconds.
        memory_mb: Peak RSS memory footprint in megabytes.
        fps: Frames per second (derived from ``latency_ms``; kept for convenience).
        dataset: Dataset name this measurement was collected on.
    """

    tracker_name: str
    accuracy: float
    latency_ms: float
    memory_mb: float
    fps: float = 0.0
    dataset: str = ""

    @property
    def efficiency_ratio(self) -> float:
        """Accuracy per millisecond — a quick edge-suitability scalar (higher = better)."""
        return self.accuracy / self.latency_ms if self.latency_ms > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset,
            "accuracy": round(self.accuracy, 4),
            "latency_ms": round(self.latency_ms, 3),
            "fps": round(self.fps, 2),
            "memory_mb": round(self.memory_mb, 2),
            "efficiency_ratio": round(self.efficiency_ratio, 6),
        }


def _dominates_2d(a: TrackerPoint, b: TrackerPoint) -> bool:
    """Return True iff *a* Pareto-dominates *b* in (maximize accuracy, minimize latency)."""
    return (
        a.accuracy >= b.accuracy
        and a.latency_ms <= b.latency_ms
        and (a.accuracy > b.accuracy or a.latency_ms < b.latency_ms)
    )


def _dominates_3d(a: TrackerPoint, b: TrackerPoint) -> bool:
    """Return True iff *a* dominates *b* in (maximize accuracy, minimize latency, minimize memory)."""
    at_least_as_good = (
        a.accuracy >= b.accuracy
        and a.latency_ms <= b.latency_ms
        and a.memory_mb <= b.memory_mb
    )
    strictly_better = (
        a.accuracy > b.accuracy
        or a.latency_ms < b.latency_ms
        or a.memory_mb < b.memory_mb
    )
    return at_least_as_good and strictly_better


def _filter_pareto(points: List[TrackerPoint], dominator) -> List[TrackerPoint]:
    """Return the subset of *points* that are not dominated by any other point."""
    front = [
        p for p in points
        if not any(dominator(o, p) for o in points if o is not p)
    ]
    return sorted(front, key=lambda p: p.accuracy, reverse=True)


class ParetoFrontier:
    """Compute and query the Pareto-optimal tracker set in multi-objective space.

    Objectives evaluated:

    * **Accuracy** (maximise): ``success_auc``, ``mean_iou``, or ``precision_auc``.
    * **Latency** (minimise): mean per-frame processing time in milliseconds.
    * **Memory** (minimise): peak RSS footprint in megabytes.

    The **2-D frontier** (accuracy vs latency) is the primary view for
    edge-deployment papers.  The **3-D frontier** (accuracy, latency, memory)
    is relevant for embedded and microcontroller targets.

    The **hypervolume indicator** quantifies the overall quality of the Pareto
    front with a single scalar, enabling objective comparisons across different
    algorithmic designs and experimental sessions.

    Example usage — see module docstring.
    """

    def __init__(self, points: List[TrackerPoint]) -> None:
        self._points: List[TrackerPoint] = list(points)
        self._frontier_2d: Optional[List[TrackerPoint]] = None
        self._frontier_3d: Optional[List[TrackerPoint]] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_benchmark_results(
        cls,
        results: Sequence,
        accuracy_key: str = "success_auc",
    ) -> "ParetoFrontier":
        """Build a :class:`ParetoFrontier` from :class:`~eovot.benchmark.engine.BenchmarkResult` objects.

        Args:
            results: Iterable of ``BenchmarkResult`` instances (one per tracker run).
            accuracy_key: Which accuracy metric to use.  One of ``"success_auc"``
                (default), ``"mean_iou"``, or ``"precision_auc"``.

        Returns:
            A :class:`ParetoFrontier` with one :class:`TrackerPoint` per result.
        """
        points: List[TrackerPoint] = []
        for r in results:
            summary = r.summary()
            fps = float(summary.get("mean_fps", 0.0))
            lat = 1000.0 / fps if fps > 0 else float("inf")
            acc = float(summary.get(accuracy_key) or summary.get("mean_iou", 0.0))
            mem = float(summary.get("peak_memory_mb", 0.0))
            points.append(TrackerPoint(
                tracker_name=summary["tracker"],
                dataset=summary.get("dataset", ""),
                accuracy=acc,
                latency_ms=lat,
                memory_mb=mem,
                fps=fps,
            ))
        return cls(points)

    # ------------------------------------------------------------------
    # Pareto frontiers
    # ------------------------------------------------------------------

    @property
    def all_points(self) -> List[TrackerPoint]:
        """All tracker points (dominated and non-dominated), sorted by accuracy descending."""
        return sorted(self._points, key=lambda p: p.accuracy, reverse=True)

    @property
    def frontier_2d(self) -> List[TrackerPoint]:
        """Non-dominated set in the (accuracy, latency) plane, sorted by accuracy descending.

        Tracker A is on the 2-D front iff no other tracker has both higher accuracy
        and lower latency.
        """
        if self._frontier_2d is None:
            self._frontier_2d = _filter_pareto(self._points, _dominates_2d)
        return list(self._frontier_2d)

    @property
    def frontier_3d(self) -> List[TrackerPoint]:
        """Non-dominated set in (accuracy, latency, memory) space, sorted by accuracy descending.

        Adds peak memory as a third constraint.  Useful for devices with strict
        RAM budgets such as microcontrollers.
        """
        if self._frontier_3d is None:
            self._frontier_3d = _filter_pareto(self._points, _dominates_3d)
        return list(self._frontier_3d)

    # ------------------------------------------------------------------
    # Constraint-based deployment queries
    # ------------------------------------------------------------------

    def best_under_latency(self, max_latency_ms: float) -> Optional[TrackerPoint]:
        """Return the highest-accuracy tracker satisfying the latency budget.

        Args:
            max_latency_ms: Maximum acceptable mean per-frame latency in milliseconds.

        Returns:
            :class:`TrackerPoint` with highest accuracy where
            ``latency_ms <= max_latency_ms``, or ``None`` if none qualify.
        """
        candidates = [p for p in self._points if p.latency_ms <= max_latency_ms]
        return max(candidates, key=lambda p: p.accuracy) if candidates else None

    def best_under_memory(self, max_memory_mb: float) -> Optional[TrackerPoint]:
        """Return the highest-accuracy tracker satisfying the memory budget.

        Args:
            max_memory_mb: Maximum acceptable peak memory in megabytes.

        Returns:
            :class:`TrackerPoint` with highest accuracy where
            ``memory_mb <= max_memory_mb``, or ``None`` if none qualify.
        """
        candidates = [p for p in self._points if p.memory_mb <= max_memory_mb]
        return max(candidates, key=lambda p: p.accuracy) if candidates else None

    def best_under_constraints(
        self,
        max_latency_ms: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
    ) -> Optional[TrackerPoint]:
        """Return the highest-accuracy tracker satisfying both budgets.

        Args:
            max_latency_ms: Optional latency ceiling in milliseconds.
            max_memory_mb: Optional memory ceiling in megabytes.

        Returns:
            Best qualifying :class:`TrackerPoint`, or ``None`` if none qualify.
        """
        candidates = list(self._points)
        if max_latency_ms is not None:
            candidates = [p for p in candidates if p.latency_ms <= max_latency_ms]
        if max_memory_mb is not None:
            candidates = [p for p in candidates if p.memory_mb <= max_memory_mb]
        return max(candidates, key=lambda p: p.accuracy) if candidates else None

    # ------------------------------------------------------------------
    # Hypervolume indicator
    # ------------------------------------------------------------------

    def hypervolume_2d(
        self,
        ref_accuracy: float = 0.0,
        ref_latency: float = 200.0,
    ) -> float:
        """Compute the 2-D hypervolume indicator of the Pareto front.

        The hypervolume is the area in (accuracy, 1/latency_ms) space that is
        dominated by the front and bounded below-left by the reference point.
        A **larger** value indicates a better-quality Pareto front.

        The reference point must be weakly worse than every front point:
        ``ref_accuracy`` should be <= the minimum front accuracy and
        ``ref_latency`` should be >= the maximum front latency.
        Sensible defaults: ``ref_accuracy=0.0``, ``ref_latency=200.0`` ms
        (20 x the reciprocal of a 30 FPS real-time target).

        Implementation uses the O(n log n) sweep-line algorithm for 2-D
        hypervolume under a staircase-monotone Pareto front.

        Args:
            ref_accuracy: Lower bound for accuracy in the reference point.
            ref_latency:  Upper bound for latency in the reference point (ms).

        Returns:
            Non-negative hypervolume scalar.  Returns ``0.0`` for an empty
            front or when no front point strictly improves on both objectives.
        """
        front = [
            p for p in self.frontier_2d
            if p.accuracy > ref_accuracy and p.latency_ms < ref_latency
        ]
        if not front:
            return 0.0

        ref_speed = 1.0 / ref_latency  # reference in speed (1/ms) space

        # Sort by accuracy ascending for the column sweep
        pts = sorted(front, key=lambda p: p.accuracy)
        hv = 0.0
        prev_acc = ref_accuracy
        for p in pts:
            speed = 1.0 / p.latency_ms
            width = p.accuracy - prev_acc
            height = speed - ref_speed
            if width > 0 and height > 0:
                hv += width * height
            prev_acc = p.accuracy
        return float(hv)

    # ------------------------------------------------------------------
    # Ranking and reporting
    # ------------------------------------------------------------------

    def rank_by_efficiency(self) -> List[TrackerPoint]:
        """Return all points ranked by accuracy-per-millisecond descending."""
        return sorted(self._points, key=lambda p: p.efficiency_ratio, reverse=True)

    def summary_table(self) -> List[Dict]:
        """Return a list of dicts — one row per tracker — with Pareto membership flags.

        Each dict contains the fields from :meth:`TrackerPoint.to_dict` plus
        ``on_frontier_2d`` and ``on_frontier_3d`` booleans, making it easy
        to highlight Pareto-optimal trackers in downstream reports.

        Returns:
            List sorted by accuracy descending.
        """
        f2 = {p.tracker_name for p in self.frontier_2d}
        f3 = {p.tracker_name for p in self.frontier_3d}
        rows = []
        for p in sorted(self._points, key=lambda x: x.accuracy, reverse=True):
            row = p.to_dict()
            row["on_frontier_2d"] = p.tracker_name in f2
            row["on_frontier_3d"] = p.tracker_name in f3
            rows.append(row)
        return rows

    def to_markdown(self) -> str:
        """Render a summary table as a Markdown string for publication or reports.

        Columns: Tracker, Dataset, Accuracy, Latency (ms), FPS, Mem (MB),
        2D-Front, 3D-Front.  Pareto-optimal rows are marked with ✓.

        Returns:
            Multi-line Markdown table string.
        """
        rows = self.summary_table()
        header = (
            "| Tracker | Dataset | Accuracy | Latency (ms) | FPS | Mem (MB) "
            "| 2D-Front | 3D-Front |"
        )
        sep = (
            "|---------|---------|--------:|------------:|----:|--------:"
            "|:--------:|:--------:|"
        )
        lines = [header, sep]
        for r in rows:
            f2 = "✓" if r["on_frontier_2d"] else ""
            f3 = "✓" if r["on_frontier_3d"] else ""
            lines.append(
                f"| {r['tracker']} | {r['dataset']} "
                f"| {r['accuracy']:.4f} | {r['latency_ms']:.2f} "
                f"| {r['fps']:.1f} | {r['memory_mb']:.1f} "
                f"| {f2} | {f3} |"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._points)

    def __repr__(self) -> str:
        return (
            f"ParetoFrontier(n={len(self._points)}, "
            f"frontier_2d={len(self.frontier_2d)}, "
            f"frontier_3d={len(self.frontier_3d)})"
        )
