"""FPS-budget accuracy analysis for edge deployment.

Answers the question: *"Given a fixed throughput budget, what is the best
accuracy this tracker can deliver on this hardware?"*

This module complements :mod:`eovot.analysis.skip_analysis` (which sweeps
fixed skip rates) and :mod:`eovot.analysis.resolution_analysis` (which sweeps
resolution scales) with a budget-driven view:

- The caller specifies desired throughput targets in FPS.
- For each target, an :class:`~eovot.trackers.budget.AdaptiveBudgetTracker`
  wraps the base tracker and adapts its skip rate online to hit the target.
- The resulting accuracy/FPS pairs form the **FPS-budget accuracy frontier**:
  the Pareto-optimal curve of achievable (accuracy, throughput) trade-offs.

Typical usage::

    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.analysis.budget_analysis import BudgetAnalyzer

    dataset  = SyntheticDataset(num_sequences=5, num_frames=120)
    engine   = BenchmarkEngine(verbose=False)
    analyzer = BudgetAnalyzer(engine)

    result = analyzer.analyze(
        KCFTracker(),
        dataset,
        dataset_name="Synthetic",
        fps_targets=[15.0, 30.0, 60.0, 120.0],
    )
    print(result.to_markdown_table())

    # Best budget that keeps mIoU ≥ 0.65
    fps, iou, skip = result.optimal_budget(min_iou=0.65)
    print(f"Use fps_target={fps}: mean_fps={fps} at mIoU={iou:.3f} (skip={skip})")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..trackers.budget import AdaptiveBudgetTracker

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
    from ..datasets.base import BaseDataset
    from ..trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BudgetEntry:
    """Per-budget-target accuracy and throughput summary."""

    fps_target: float
    """The FPS target passed to :class:`~eovot.trackers.budget.AdaptiveBudgetTracker`."""

    mean_iou: float
    """Mean IoU across all evaluated frames at this budget."""

    mean_fps: float
    """Measured effective FPS (may differ from target due to adaptation lag)."""

    final_skip_rate: int
    """Skip rate that the adaptive tracker settled on by the end of the run."""

    peak_memory_mb: float
    """Peak RSS memory footprint in megabytes."""

    success_auc: Optional[float] = None
    """Success-curve AUC, if computed."""

    iou_degradation: float = 0.0
    """Absolute drop in mean IoU vs. the baseline (fps_target = +inf / skip=1)."""

    fps_gap: float = 0.0
    """Difference between measured FPS and the target (positive = above target)."""

    def __str__(self) -> str:
        auc_str = f"  AUC={self.success_auc:.3f}" if self.success_auc is not None else ""
        return (
            f"fps_target={self.fps_target:.0f}  "
            f"mIoU={self.mean_iou:.4f} (Δ={self.iou_degradation:+.4f})  "
            f"FPS={self.mean_fps:.1f} (gap={self.fps_gap:+.1f})  "
            f"skip={self.final_skip_rate}{auc_str}"
        )


@dataclass
class BudgetResult:
    """Full sweep result from :class:`BudgetAnalyzer`.

    Contains one :class:`BudgetEntry` per tested FPS target, ordered by
    ``fps_target`` ascending.  The first entry (lowest target) is the most
    compute-efficient; the last entry (highest target / unconstrained) is the
    most accurate.
    """

    tracker_name: str
    dataset_name: str
    entries: List[BudgetEntry] = field(default_factory=list)

    def optimal_budget(self, min_iou: float = 0.0) -> Tuple[float, float, int]:
        """Return the highest-FPS budget that still keeps mIoU ≥ ``min_iou``.

        Args:
            min_iou: Minimum acceptable mean IoU.

        Returns:
            ``(fps_target, mean_iou, final_skip_rate)`` for the winning entry.

        Raises:
            ValueError: If no entry satisfies ``min_iou`` (even the
                unconstrained run is below the target).
        """
        candidates = [e for e in self.entries if e.mean_iou >= min_iou]
        if not candidates:
            raise ValueError(
                f"No budget entry achieves mIoU ≥ {min_iou:.3f}. "
                f"Best achieved: {max(e.mean_iou for e in self.entries):.4f}"
            )
        best = max(candidates, key=lambda e: e.fps_target)
        return best.fps_target, best.mean_iou, best.final_skip_rate

    def to_markdown_table(self) -> str:
        """Format the sweep results as a Markdown table.

        Returns:
            Multi-line Markdown string with columns for FPS target, measured
            FPS, mIoU, IoU degradation, skip rate, and (if available)
            success AUC.
        """
        has_auc = any(e.success_auc is not None for e in self.entries)
        header = (
            "| fps_target | mean_fps | mIoU   | ΔmIoU   | skip | peak_mem_MB"
        )
        sep = (
            "|------------|----------|--------|---------|------|------------"
        )
        if has_auc:
            header += " | success_AUC"
            sep += "|------------"
        header += " |"
        sep += "|"

        rows = [header, sep]
        for e in self.entries:
            row = (
                f"| {e.fps_target:>10.0f} "
                f"| {e.mean_fps:>8.1f} "
                f"| {e.mean_iou:>6.4f} "
                f"| {e.iou_degradation:>+7.4f} "
                f"| {e.final_skip_rate:>4d} "
                f"| {e.peak_memory_mb:>11.1f}"
            )
            if has_auc:
                auc_str = f"{e.success_auc:.4f}" if e.success_auc is not None else "  N/A  "
                row += f" | {auc_str:>11}"
            row += " |"
            rows.append(row)
        return "\n".join(rows)

    def __str__(self) -> str:
        lines = [
            f"BudgetResult[{self.tracker_name} on {self.dataset_name}]",
            f"  {len(self.entries)} budget points:",
        ]
        for e in self.entries:
            lines.append(f"  {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class BudgetAnalyzer:
    """Sweep FPS targets to characterise the accuracy–throughput frontier.

    Args:
        engine: A configured :class:`~eovot.benchmark.engine.BenchmarkEngine`
            instance.  The same engine (and profiling settings) is reused
            across all budget points, so energy profiling is inherited if the
            engine was constructed with ``tdp_watts``.
    """

    def __init__(self, engine: "BenchmarkEngine") -> None:
        self._engine = engine

    def analyze(
        self,
        tracker: "BaseTracker",
        dataset: "BaseDataset",
        dataset_name: str = "unknown",
        fps_targets: Optional[List[float]] = None,
        max_sequences: Optional[int] = None,
        min_skip: int = 1,
        max_skip: int = 8,
    ) -> BudgetResult:
        """Run the tracker under each FPS budget and collect accuracy/FPS pairs.

        A fresh :class:`~eovot.trackers.budget.AdaptiveBudgetTracker` is
        constructed for each ``fps_target`` so adaptations from one run do
        not pollute the next.

        Args:
            tracker:        Base tracker to evaluate.
            dataset:        Evaluation dataset.
            dataset_name:   Human-readable dataset label.
            fps_targets:    List of FPS budgets to test, in ascending order.
                            Default: ``[10, 20, 30, 60, 120, 240]``.
            max_sequences:  Limit evaluation to this many sequences per budget
                            point (useful for quick sweeps).
            min_skip:       Minimum skip rate for all adaptive wrappers.
            max_skip:       Maximum skip rate for all adaptive wrappers.

        Returns:
            :class:`BudgetResult` with one :class:`BudgetEntry` per target.
        """
        if fps_targets is None:
            fps_targets = [10.0, 20.0, 30.0, 60.0, 120.0, 240.0]

        result = BudgetResult(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
        )

        # Baseline: unconstrained run (fps_target so high skip_rate stays at 1)
        _UNCONSTRAINED = float("inf")
        baseline_iou: Optional[float] = None

        for fps_target in sorted(fps_targets):
            budget_tracker = AdaptiveBudgetTracker(
                tracker,
                fps_target=fps_target,
                min_skip=min_skip,
                max_skip=max_skip,
                initial_skip=1,
            )
            br: "BenchmarkResult" = self._engine.run(
                budget_tracker,
                dataset,
                dataset_name=dataset_name,
                max_sequences=max_sequences,
            )

            if baseline_iou is None:
                baseline_iou = br.mean_iou

            sauc = br.mean_success_auc
            entry = BudgetEntry(
                fps_target=fps_target,
                mean_iou=br.mean_iou,
                mean_fps=br.mean_fps,
                final_skip_rate=budget_tracker.current_skip_rate,
                peak_memory_mb=br.peak_memory_mb,
                success_auc=sauc,
                iou_degradation=br.mean_iou - baseline_iou,
                fps_gap=br.mean_fps - fps_target,
            )
            result.entries.append(entry)

        return result
