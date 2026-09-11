"""Accuracy-efficiency Pareto front analysis for edge deployment.

Identifies which trackers are Pareto-optimal in the accuracy-latency (or
accuracy-FPS) space.  For edge deployment, no single metric suffices: a
tracker must balance accuracy against the device's hard latency budget.
The Pareto front isolates the trackers where no other tracker is strictly
better on *all* objectives simultaneously — these are the principled
trade-off choices.

Key concepts
============
* **Pareto-optimal**: A tracker T is Pareto-optimal if no other tracker
  beats T on both accuracy *and* FPS.
* **Dominated**: A tracker is dominated if another tracker has at least as
  high accuracy *and* at least as high FPS, with at least one strictly
  better.
* **Edge Efficiency Score (EES)**: A composite scalar weighting accuracy
  against how well the tracker meets a device's FPS budget::

      speed_score = min(fps / target_fps, 1.0)
      EES = accuracy_weight * (accuracy / max_accuracy)
            + fps_weight * speed_score

  A tracker that exceeds the target FPS is not further rewarded — once
  the constraint is satisfied, accuracy is the tiebreaker.

Usage::

    from eovot.analysis.pareto import ParetoAnalyzer, TrackerPoint

    points = [
        TrackerPoint("MOSSE",  accuracy=0.42, fps=320.0),
        TrackerPoint("KCF",    accuracy=0.55, fps=180.0),
        TrackerPoint("CSRT",   accuracy=0.67, fps=25.0),
        TrackerPoint("Ours",   accuracy=0.58, fps=95.0),
    ]
    analyzer = ParetoAnalyzer(points)
    print(analyzer.summary_table())

    for entry in analyzer.rank_for_device(target_fps=30.0):
        print(entry)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrackerPoint:
    """Scalar accuracy and efficiency descriptor for a single tracker.

    Args:
        name:     Tracker identifier (must be unique within an analysis).
        accuracy: Accuracy scalar — typically success AUC or mean IoU
                  (higher is better).
        fps:      Measured throughput in frames per second (higher is
                  better).
        extra:    Optional metadata dict (dataset name, device, memory,
                  etc.) — stored but not used in comparisons.
    """

    name: str
    accuracy: float
    fps: float
    latency_mean_ms: float = field(init=False)
    extra: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.latency_mean_ms = 1000.0 / self.fps if self.fps > 0 else float("inf")

    @classmethod
    def from_benchmark_result(cls, result: object) -> "TrackerPoint":
        """Construct from a :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Uses ``mean_success_auc`` as the accuracy metric when available,
        falling back to ``mean_iou``.
        """
        accuracy = getattr(result, "mean_success_auc", None)
        if accuracy is None:
            accuracy = float(getattr(result, "mean_iou", 0.0))
        fps = float(getattr(result, "mean_fps", 0.0))
        return cls(
            name=str(getattr(result, "tracker_name", "unknown")),
            accuracy=float(accuracy),
            fps=fps,
            extra={"dataset": getattr(result, "dataset_name", "")},
        )


@dataclass
class ParetoEntry:
    """Enriched result for one tracker after Pareto analysis."""

    point: TrackerPoint
    is_pareto_optimal: bool
    pareto_rank: int
    dominated_by: List[str] = field(default_factory=list)
    ees: Optional[float] = None

    @property
    def name(self) -> str:
        return self.point.name

    @property
    def accuracy(self) -> float:
        return self.point.accuracy

    @property
    def fps(self) -> float:
        return self.point.fps

    @property
    def latency_ms(self) -> float:
        return self.point.latency_mean_ms

    def __repr__(self) -> str:
        if self.is_pareto_optimal:
            status = "Pareto-optimal"
        else:
            status = f"dominated by {self.dominated_by}"
        return (
            f"ParetoEntry({self.name!r}, acc={self.accuracy:.4f}, "
            f"fps={self.fps:.1f}, {status})"
        )


def _dominators(candidate: TrackerPoint, others: List[TrackerPoint]) -> List[str]:
    """Return names of trackers in *others* that strictly dominate *candidate*."""
    result = []
    for other in others:
        if other.name == candidate.name:
            continue
        better_or_equal_acc = other.accuracy >= candidate.accuracy
        better_or_equal_fps = other.fps >= candidate.fps
        strictly_better = (
            other.accuracy > candidate.accuracy
            or other.fps > candidate.fps
        )
        if better_or_equal_acc and better_or_equal_fps and strictly_better:
            result.append(other.name)
    return result


def compute_pareto_front(points: List[TrackerPoint]) -> List[ParetoEntry]:
    """Compute layered Pareto dominance ranks for a list of tracker points.

    Objectives: maximise accuracy, maximise FPS.  A tracker is
    Pareto-optimal (rank 1) when no other tracker is at least as good on
    *all* objectives and strictly better on at least one.  Dominated
    trackers are iteratively assigned to higher ranks by removing each
    non-dominated layer in turn.

    Args:
        points: Tracker performance points to analyse.

    Returns:
        List of :class:`ParetoEntry` objects, one per input tracker,
        annotated with dominance status and Pareto rank.
    """
    remaining = list(points)
    entries: Dict[str, ParetoEntry] = {}
    rank = 1

    while remaining:
        # Find non-dominated subset of the current remaining set
        dominated_in_round: set = set()
        for p in remaining:
            if _dominators(p, remaining):
                dominated_in_round.add(p.name)

        front = [p for p in remaining if p.name not in dominated_in_round]
        dominated = [p for p in remaining if p.name in dominated_in_round]

        for p in front:
            entries[p.name] = ParetoEntry(
                point=p,
                is_pareto_optimal=(rank == 1),
                pareto_rank=rank,
            )

        # Assign dominators from the *full* point list for informational use
        for p in dominated:
            if p.name not in entries:
                dom = _dominators(p, points)
                entries[p.name] = ParetoEntry(
                    point=p,
                    is_pareto_optimal=False,
                    pareto_rank=rank + 1,
                    dominated_by=dom,
                )

        remaining = [p for p in remaining if p.name in dominated_in_round]
        rank += 1

    return list(entries.values())


class ParetoAnalyzer:
    """Analyse tracker performance in the accuracy-efficiency Pareto space.

    Args:
        points: List of :class:`TrackerPoint` objects, one per tracker.
                At least one point is required.

    Raises:
        ValueError: If *points* is empty.
    """

    def __init__(self, points: List[TrackerPoint]) -> None:
        if not points:
            raise ValueError("ParetoAnalyzer requires at least one TrackerPoint.")
        self._points = points
        self._entries: List[ParetoEntry] = compute_pareto_front(points)

    @property
    def pareto_entries(self) -> List[ParetoEntry]:
        """All entries sorted by Pareto rank, then descending accuracy."""
        return sorted(
            self._entries,
            key=lambda e: (e.pareto_rank, -e.accuracy),
        )

    @property
    def pareto_optimal(self) -> List[ParetoEntry]:
        """Only the Pareto-optimal (rank-1) entries."""
        return [e for e in self._entries if e.is_pareto_optimal]

    def rank_for_device(
        self,
        target_fps: float,
        accuracy_weight: float = 0.7,
        fps_weight: float = 0.3,
    ) -> List[ParetoEntry]:
        """Rank trackers for a device with a hard FPS requirement.

        Computes the Edge Efficiency Score (EES) for each tracker::

            speed_score = min(fps / target_fps, 1.0)
            EES = accuracy_weight * (accuracy / max_accuracy)
                  + fps_weight * speed_score

        Trackers that cannot meet *target_fps* are included but penalised
        (their speed score is below 1.0).

        Args:
            target_fps:      Hard framerate budget of the target device.
            accuracy_weight: Weight for the accuracy component (default 0.7).
            fps_weight:      Weight for the speed component (default 0.3).

        Returns:
            Entries sorted by EES descending (best choice first).

        Raises:
            ValueError: If *target_fps* is not positive, or weights are out
                        of ``[0, 1]``.
        """
        if target_fps <= 0.0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if not (0.0 <= accuracy_weight <= 1.0 and 0.0 <= fps_weight <= 1.0):
            raise ValueError("accuracy_weight and fps_weight must be in [0, 1].")

        max_acc = max(e.accuracy for e in self._entries) or 1.0
        for entry in self._entries:
            speed_score = min(entry.fps / target_fps, 1.0)
            norm_acc = entry.accuracy / max_acc
            entry.ees = accuracy_weight * norm_acc + fps_weight * speed_score

        return sorted(self._entries, key=lambda e: -(e.ees or 0.0))

    def summary_table(self) -> str:
        """Return a formatted ASCII table of all trackers with Pareto status.

        Returns:
            Multi-line string suitable for direct ``print()`` output.
        """
        header = (
            f"{'Tracker':<22} {'Accuracy':>10} {'FPS':>8} "
            f"{'Lat(ms)':>9} {'Rank':>5} {'Status':<16} {'EES':>6}"
        )
        sep = "-" * len(header)
        rows = [header, sep]
        for e in self.pareto_entries:
            status = "Pareto-optimal" if e.is_pareto_optimal else f"rank {e.pareto_rank}"
            ees_str = f"{e.ees:.4f}" if e.ees is not None else "  N/A"
            rows.append(
                f"{e.name:<22} {e.accuracy:>10.4f} {e.fps:>8.1f} "
                f"{e.latency_ms:>9.2f} {e.pareto_rank:>5} {status:<16} {ees_str:>6}"
            )
        return "\n".join(rows)

    def __repr__(self) -> str:
        return f"ParetoAnalyzer({len(self._points)} trackers)"
