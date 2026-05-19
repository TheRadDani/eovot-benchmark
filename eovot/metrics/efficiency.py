"""Edge efficiency metrics: composite scoring and Pareto-front analysis.

Operationalises EOVOT's core evaluation thesis — tracker quality must be
measured across *both* accuracy (IoU) and efficiency (FPS, memory)
simultaneously, not accuracy alone.

Edge Efficiency Score (EES)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
A single scalar rewarding high accuracy at low latency within an acceptable
memory envelope::

    EES = mean_iou × log1p(fps) / (1 + peak_memory_mb / memory_budget_mb)

* ``log1p(fps)`` applies diminishing returns to raw throughput — going from
  5 → 50 FPS matters far more than 500 → 550 FPS on resource-constrained
  edge hardware.
* The denominator soft-penalises trackers that exceed ``memory_budget_mb``
  without hard-cutting them: a tracker 2× over budget scores half the memory
  factor, not zero.

Pareto Front
~~~~~~~~~~~~
The Pareto front identifies the subset of trackers where no other tracker
dominates *both* accuracy and EES simultaneously.  Pareto-optimal trackers
represent the true accuracy–efficiency frontier and are the natural
comparison set for edge deployment decisions.

Typical usage::

    from eovot.metrics.efficiency import EfficiencyMetricsEngine

    engine = EfficiencyMetricsEngine(memory_budget_mb=512.0)
    ranking = engine.rank_trackers(benchmark_results)
    for entry in ranking:
        print(f"{entry.tracker_name:12s}  EES={entry.ees:.4f}  "
              f"mIoU={entry.mean_iou:.4f}  FPS={entry.fps:.1f}  "
              f"Pareto={'yes' if entry.on_pareto_front else ' no'}")
    print(engine.to_markdown_table(ranking))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


@dataclass
class EfficiencyEntry:
    """Per-tracker efficiency summary produced by :class:`EfficiencyMetricsEngine`.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        dataset_name: Dataset on which the tracker was evaluated.
        mean_iou: Mean IoU across all evaluated frames.
        fps: Mean frames-per-second throughput.
        peak_memory_mb: Peak RSS memory footprint in megabytes.
        ees: Edge Efficiency Score — composite accuracy × throughput / memory scalar.
        on_pareto_front: ``True`` when no other tracker in the comparison set
            dominates this one in both accuracy and EES.
    """

    tracker_name: str
    dataset_name: str
    mean_iou: float
    fps: float
    peak_memory_mb: float
    ees: float
    on_pareto_front: bool = field(default=False)

    def __str__(self) -> str:
        pareto = "yes" if self.on_pareto_front else "no"
        return (
            f"EfficiencyEntry({self.tracker_name}  "
            f"EES={self.ees:.4f}  mIoU={self.mean_iou:.4f}  "
            f"FPS={self.fps:.1f}  mem={self.peak_memory_mb:.1f} MB  "
            f"pareto={pareto})"
        )


class EfficiencyMetricsEngine:
    """Compute edge efficiency scores and Pareto fronts for tracker comparison.

    Args:
        memory_budget_mb: Acceptable peak-memory ceiling in megabytes.
            Trackers within this budget receive full memory credit.
            Default: ``512.0`` MB (typical constrained edge device).
    """

    def __init__(self, memory_budget_mb: float = 512.0) -> None:
        if memory_budget_mb <= 0:
            raise ValueError(f"memory_budget_mb must be positive, got {memory_budget_mb}.")
        self.memory_budget_mb = memory_budget_mb

    def edge_efficiency_score(
        self,
        mean_iou: float,
        fps: float,
        peak_memory_mb: float,
    ) -> float:
        """Compute the Edge Efficiency Score for one (accuracy, efficiency) point.

        Formula::

            EES = mean_iou × log1p(fps) / (1 + peak_memory_mb / memory_budget_mb)

        Args:
            mean_iou: Mean IoU over the evaluated sequence set, in ``[0, 1]``.
            fps: Mean frames per second (must be > 0).
            peak_memory_mb: Peak RSS memory footprint in megabytes.

        Returns:
            EES scalar ≥ 0.  Higher values indicate better edge suitability.
            Returns ``0.0`` for non-positive fps or negative IoU.
        """
        if fps <= 0 or mean_iou < 0:
            return 0.0
        throughput_factor = math.log1p(fps)
        memory_penalty = 1.0 + peak_memory_mb / self.memory_budget_mb
        return (mean_iou * throughput_factor) / memory_penalty

    def compute_pareto_front(
        self, entries: List[EfficiencyEntry]
    ) -> List[EfficiencyEntry]:
        """Mark Pareto-optimal trackers in the (mIoU, EES) objective space.

        Tracker A dominates tracker B iff ``A.mean_iou >= B.mean_iou`` AND
        ``A.ees >= B.ees`` with at least one strict inequality.  Non-dominated
        trackers are Pareto-optimal.

        Args:
            entries: List of :class:`EfficiencyEntry` objects to analyse.
                The ``on_pareto_front`` flag is updated **in-place**.

        Returns:
            The same list with ``on_pareto_front`` flags set.
        """
        for i, candidate in enumerate(entries):
            dominated = False
            for j, other in enumerate(entries):
                if i == j:
                    continue
                if (
                    other.mean_iou >= candidate.mean_iou
                    and other.ees >= candidate.ees
                    and (other.mean_iou > candidate.mean_iou or other.ees > candidate.ees)
                ):
                    dominated = True
                    break
            candidate.on_pareto_front = not dominated
        return entries

    def rank_trackers(
        self, results: List["BenchmarkResult"]
    ) -> List[EfficiencyEntry]:
        """Build a ranked list of :class:`EfficiencyEntry` from benchmark results.

        Computes EES for each result, identifies the Pareto front, and sorts
        the list by EES descending so the best edge-deployable tracker is first.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult` per
                tracker/dataset combination.

        Returns:
            List of :class:`EfficiencyEntry` sorted by EES (highest first),
            with ``on_pareto_front`` flags set.
        """
        entries: List[EfficiencyEntry] = []
        for r in results:
            ees = self.edge_efficiency_score(
                mean_iou=r.mean_iou,
                fps=r.mean_fps,
                peak_memory_mb=r.peak_memory_mb,
            )
            entries.append(
                EfficiencyEntry(
                    tracker_name=r.tracker_name,
                    dataset_name=r.dataset_name,
                    mean_iou=r.mean_iou,
                    fps=r.mean_fps,
                    peak_memory_mb=r.peak_memory_mb,
                    ees=ees,
                )
            )
        self.compute_pareto_front(entries)
        entries.sort(key=lambda e: e.ees, reverse=True)
        return entries

    def to_markdown_table(self, entries: List[EfficiencyEntry]) -> str:
        """Format a ranked efficiency table as a Markdown string.

        Args:
            entries: Output of :meth:`rank_trackers`.

        Returns:
            Multi-line Markdown table string ready to embed in reports or READMEs.
        """
        lines = [
            "| Rank | Tracker | Dataset | mIoU | FPS | Mem (MB) | EES | Pareto |",
            "|------|---------|---------|-----:|----:|---------:|----:|:------:|",
        ]
        for rank, e in enumerate(entries, start=1):
            pareto = "✓" if e.on_pareto_front else ""
            lines.append(
                f"| {rank} | {e.tracker_name} | {e.dataset_name} "
                f"| {e.mean_iou:.4f} | {e.fps:.1f} | {e.peak_memory_mb:.1f} "
                f"| {e.ees:.4f} | {pareto} |"
            )
        return "\n".join(lines)
