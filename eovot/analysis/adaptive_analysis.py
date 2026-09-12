"""Sweep-and-report analysis for AdaptiveBudgetTracker FPS budgets.

:class:`AdaptiveBudgetAnalyzer` runs a wrapped tracker under several
``target_fps`` budgets and records accuracy vs. efficiency trade-off data.
Results are surfaced through :class:`AdaptiveBudgetReport` which supports
Markdown rendering and the convenience query
:meth:`AdaptiveBudgetReport.optimal_fps_at_iou_budget`.

Typical usage::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.analysis.adaptive_analysis import AdaptiveBudgetAnalyzer

    def factory():
        return MOSSETracker()

    analyzer = AdaptiveBudgetAnalyzer(
        tracker_factory=factory,
        fps_budgets=[10, 20, 30, 60],
        max_skip=7,
        verbose=True,
    )
    dataset = SyntheticDataset(num_sequences=5)
    report = analyzer.run(dataset, dataset_name="Synthetic")
    print(report.to_markdown())
    best_fps = report.optimal_fps_at_iou_budget(min_iou=0.45)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..benchmark.engine import BenchmarkEngine
from ..trackers.adaptive import AdaptiveBudgetTracker
from ..trackers.base import BaseTracker


@dataclass
class AdaptiveBudgetEntry:
    """Aggregated results for one (target_fps, tracker) configuration.

    Attributes:
        target_fps:          The FPS budget passed to :class:`AdaptiveBudgetTracker`.
        mean_iou:            Mean IoU over all sequences under this budget.
        actual_fps:          Wall-clock throughput achieved (frames/s).
        mean_skip_fraction:  Average fraction of frames served from cache
            (0.0 = no skipping, 1.0 = fully skipped).
        adaptation_count:    Total AIMD adjustments summed across all sequences.
        num_sequences:       Number of sequences evaluated.
    """

    target_fps: float
    mean_iou: float
    actual_fps: float
    mean_skip_fraction: float
    adaptation_count: int
    num_sequences: int


@dataclass
class AdaptiveBudgetReport:
    """Full adaptive-budget sweep report.

    Attributes:
        tracker_name:  Name of the wrapped tracker (without the adaptive suffix).
        dataset_name:  Dataset label used during evaluation.
        entries:       One :class:`AdaptiveBudgetEntry` per FPS budget, sorted
            by ``target_fps`` ascending.
    """

    tracker_name: str
    dataset_name: str
    entries: List[AdaptiveBudgetEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Return a Markdown table summarising the sweep.

        Returns:
            Multi-line Markdown string with a header row and one data row per
            :class:`AdaptiveBudgetEntry`, sorted by ``target_fps``.
        """
        lines = [
            f"## Adaptive Budget Report — {self.tracker_name} on {self.dataset_name}\n",
            "| Target FPS | mIoU | Actual FPS | Skip Fraction | Adaptations | Sequences |",
            "|------------|-----:|-----------:|--------------:|------------:|----------:|",
        ]
        for e in sorted(self.entries, key=lambda x: x.target_fps):
            lines.append(
                f"| {e.target_fps:.0f} "
                f"| {e.mean_iou:.4f} "
                f"| {e.actual_fps:.1f} "
                f"| {e.mean_skip_fraction:.2%} "
                f"| {e.adaptation_count} "
                f"| {e.num_sequences} |"
            )
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialise the report to a plain dictionary.

        Returns:
            Dict with keys ``"tracker_name"``, ``"dataset_name"``, and
            ``"entries"`` (list of per-entry dicts).
        """
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "entries": [
                {
                    "target_fps": e.target_fps,
                    "mean_iou": e.mean_iou,
                    "actual_fps": e.actual_fps,
                    "mean_skip_fraction": e.mean_skip_fraction,
                    "adaptation_count": e.adaptation_count,
                    "num_sequences": e.num_sequences,
                }
                for e in self.entries
            ],
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def optimal_fps_at_iou_budget(self, min_iou: float) -> Optional[float]:
        """Return the lowest target_fps that still meets the mIoU floor.

        Finds the entry with the smallest ``target_fps`` whose ``mean_iou``
        is at least ``min_iou``, i.e. the most aggressive FPS reduction that
        does not violate the accuracy budget.

        Args:
            min_iou: Minimum acceptable mean IoU (0.0 – 1.0).

        Returns:
            The qualifying ``target_fps`` value, or ``None`` if no entry meets
            the requirement.
        """
        qualifying = [e for e in self.entries if e.mean_iou >= min_iou]
        if not qualifying:
            return None
        return min(qualifying, key=lambda e: e.target_fps).target_fps

    def best_iou_entry(self) -> Optional[AdaptiveBudgetEntry]:
        """Return the entry with the highest mean IoU."""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.mean_iou)

    def fastest_entry(self) -> Optional[AdaptiveBudgetEntry]:
        """Return the entry with the highest achieved FPS."""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.actual_fps)


class AdaptiveBudgetAnalyzer:
    """Sweep multiple FPS budgets and build an :class:`AdaptiveBudgetReport`.

    For each value in ``fps_budgets`` the analyzer constructs a fresh
    :class:`AdaptiveBudgetTracker` wrapping a new tracker instance (obtained
    from ``tracker_factory``) and runs a :class:`~eovot.benchmark.engine.BenchmarkEngine`
    evaluation.  After the run the tracker's ``adaptation_stats`` are aggregated
    across sequences and stored in an :class:`AdaptiveBudgetEntry`.

    Args:
        tracker_factory:  Zero-argument callable that returns a fresh
            :class:`~eovot.trackers.base.BaseTracker` instance.  Called once
            per FPS budget so each run starts from a clean state.
        fps_budgets:      Iterable of target FPS values to evaluate.
        max_skip:         Forwarded to :class:`AdaptiveBudgetTracker`.  Default 7.
        warmup_frames:    Forwarded to :class:`AdaptiveBudgetTracker`.  Default 5.
        window_size:      Forwarded to :class:`AdaptiveBudgetTracker`.  Default 8.
        verbose:          If ``True``, print progress lines.  Default ``False``.
    """

    def __init__(
        self,
        tracker_factory: Callable[[], BaseTracker],
        fps_budgets: List[float],
        max_skip: int = 7,
        warmup_frames: int = 5,
        window_size: int = 8,
        verbose: bool = False,
    ) -> None:
        self.tracker_factory = tracker_factory
        self.fps_budgets = list(fps_budgets)
        self.max_skip = max_skip
        self.warmup_frames = warmup_frames
        self.window_size = window_size
        self.verbose = verbose

    def run(self, dataset, dataset_name: str = "unknown") -> AdaptiveBudgetReport:
        """Evaluate all FPS budgets and return a consolidated report.

        Args:
            dataset:       Any :class:`~eovot.datasets.base.BaseDataset`
                instance to evaluate on.
            dataset_name:  Human-readable dataset label used in the report.

        Returns:
            :class:`AdaptiveBudgetReport` with one entry per budget.
        """
        engine = BenchmarkEngine(verbose=False)
        base_tracker = self.tracker_factory()
        report = AdaptiveBudgetReport(
            tracker_name=base_tracker.name,
            dataset_name=dataset_name,
        )

        for fps_budget in self.fps_budgets:
            inner = self.tracker_factory()
            adaptive = AdaptiveBudgetTracker(
                tracker=inner,
                target_fps=fps_budget,
                max_skip=self.max_skip,
                warmup_frames=self.warmup_frames,
                window_size=self.window_size,
            )

            if self.verbose:
                print(f"[AdaptiveBudgetAnalyzer] target_fps={fps_budget:.0f} …", flush=True)

            result = engine.run(
                tracker=adaptive,
                dataset=dataset,
                dataset_name=dataset_name,
            )

            summary = result.to_dict().get("summary", {})
            stats = adaptive.adaptation_stats

            entry = AdaptiveBudgetEntry(
                target_fps=fps_budget,
                mean_iou=float(summary.get("mean_iou", 0.0)),
                actual_fps=float(summary.get("mean_fps", 0.0)),
                mean_skip_fraction=float(stats.get("mean_skip_fraction", 0.0)),
                adaptation_count=int(stats.get("adaptation_count", 0)),
                num_sequences=int(summary.get("num_sequences", 0)),
            )
            report.entries.append(entry)

            if self.verbose:
                print(
                    f"  → mIoU={entry.mean_iou:.4f}, "
                    f"fps={entry.actual_fps:.1f}, "
                    f"skip={entry.mean_skip_fraction:.1%}"
                )

        return report
