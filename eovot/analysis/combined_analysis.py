"""Combined frame-skip × resolution-scale tradeoff analysis.

Edge deployments often apply *both* frame skipping and resolution downscaling
simultaneously.  Running them in sequence — first optimise skip rate, then
optimise scale — is suboptimal because the two strategies interact: skipping
frames at a low resolution is not the same as skipping frames at full
resolution and then downscaling results.

:class:`CombinedEdgeAnalyzer` sweeps a 2-D grid of
``(skip_rate, scale_factor)`` pairs, wrapping the base tracker as
``FrameSkipTracker(ResolutionScaledTracker(base))``, and records the
resulting ``(mIoU, FPS)`` for each configuration.  It then identifies the
Pareto-optimal configurations — those where no other configuration achieves
both higher accuracy *and* higher FPS simultaneously — so you can pick the
operating point that best fits your edge budget.

Typical usage::

    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.analysis.combined_analysis import CombinedEdgeAnalyzer

    dataset  = SyntheticDataset(num_sequences=5)
    engine   = BenchmarkEngine(verbose=False)
    analyzer = CombinedEdgeAnalyzer(engine)

    result = analyzer.analyze(
        KCFTracker(),
        dataset,
        dataset_name="Synthetic",
        skip_rates=[1, 2, 3],
        scale_factors=[1.0, 0.75, 0.5],
    )
    print(result.to_markdown_table())

    # Best config that keeps mIoU above 0.60
    cfg = result.optimal_config(min_iou=0.60)
    if cfg is not None:
        print(f"skip={cfg.skip_rate}, scale={cfg.scale_factor}: "
              f"{cfg.fps:.0f} FPS at mIoU={cfg.mean_iou:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
    from ..datasets.base import BaseDataset
    from ..trackers.base import BaseTracker


@dataclass
class CombinedEntry:
    """Result for one (skip_rate, scale_factor) configuration.

    Attributes:
        skip_rate:    Frame-skip rate (1 = no skipping).
        scale_factor: Resolution scale factor (1.0 = full resolution).
        mean_iou:     Mean IoU across all sequences.
        fps:          Mean frames-per-second (wall-clock throughput).
        on_pareto:    True when no other config dominates on both axes.
        benchmark_result: Full :class:`~eovot.benchmark.engine.BenchmarkResult`
            for further analysis; may be ``None`` when loading from a dict.
    """

    skip_rate: int
    scale_factor: float
    mean_iou: float
    fps: float
    on_pareto: bool = False
    benchmark_result: Optional["BenchmarkResult"] = field(default=None, repr=False)

    def label(self) -> str:
        """Short human-readable label, e.g. ``"skip=2 scale=0.50"``."""
        return f"skip={self.skip_rate} scale={self.scale_factor:.2f}"


@dataclass
class CombinedResult:
    """Full sweep result from :class:`CombinedEdgeAnalyzer`.

    Attributes:
        tracker_name: Name of the base tracker that was wrapped.
        dataset_name: Name of the evaluated dataset.
        entries:      All (skip_rate, scale_factor) results, sorted by
                      descending Pareto status then descending mIoU.
    """

    tracker_name: str
    dataset_name: str
    entries: List[CombinedEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def pareto_entries(self) -> List[CombinedEntry]:
        """Return only Pareto-optimal entries, sorted by descending FPS."""
        return sorted(
            [e for e in self.entries if e.on_pareto],
            key=lambda e: e.fps,
            reverse=True,
        )

    def optimal_config(self, min_iou: float) -> Optional[CombinedEntry]:
        """Return the highest-FPS config whose mIoU is at least *min_iou*.

        Searches all entries (not only Pareto-optimal ones) so that a budget
        constraint can be met even when the Pareto front does not extend to
        the required accuracy.

        Args:
            min_iou: Minimum acceptable mean IoU (inclusive).

        Returns:
            :class:`CombinedEntry` with highest FPS subject to the IoU
            constraint, or ``None`` if no config meets the constraint.
        """
        candidates = [e for e in self.entries if e.mean_iou >= min_iou]
        return max(candidates, key=lambda e: e.fps) if candidates else None

    def best_accuracy(self) -> Optional[CombinedEntry]:
        """Return the entry with the highest mean IoU."""
        return max(self.entries, key=lambda e: e.mean_iou) if self.entries else None

    def best_fps(self) -> Optional[CombinedEntry]:
        """Return the entry with the highest FPS."""
        return max(self.entries, key=lambda e: e.fps) if self.entries else None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_table(self) -> str:
        """Format all entries as a Markdown table, Pareto-optimal rows starred.

        Returns:
            Multi-line Markdown string ready for writing to a ``.md`` file or
            embedding in a GitHub comment.
        """
        if not self.entries:
            return f"No results for {self.tracker_name} on {self.dataset_name}.\n"

        lines = [
            f"## Combined Edge Sweep: {self.tracker_name} on {self.dataset_name}\n",
            "| Skip | Scale | mIoU | FPS | Pareto |",
            "|-----:|------:|-----:|----:|:------:|",
        ]
        sorted_entries = sorted(
            self.entries,
            key=lambda e: (not e.on_pareto, -e.mean_iou),
        )
        for e in sorted_entries:
            star = "★" if e.on_pareto else ""
            lines.append(
                f"| {e.skip_rate} | {e.scale_factor:.2f} "
                f"| {e.mean_iou:.4f} | {e.fps:.1f} | {star} |"
            )
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable plain dict."""
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "entries": [
                {
                    "skip_rate": e.skip_rate,
                    "scale_factor": round(e.scale_factor, 4),
                    "mean_iou": round(e.mean_iou, 4),
                    "fps": round(e.fps, 2),
                    "on_pareto": e.on_pareto,
                }
                for e in self.entries
            ],
        }


class CombinedEdgeAnalyzer:
    """Sweep a 2-D (skip_rate × scale_factor) grid and find Pareto-optimal configs.

    Wraps the base tracker as ``FrameSkipTracker(ResolutionScaledTracker(base))``
    for each grid point, runs the benchmark, and returns a
    :class:`CombinedResult` with Pareto flags set.

    Args:
        engine: :class:`~eovot.benchmark.engine.BenchmarkEngine` instance used
            to run each configuration.

    Example::

        engine   = BenchmarkEngine(verbose=False)
        analyzer = CombinedEdgeAnalyzer(engine)
        result   = analyzer.analyze(
            MOSSETracker(),
            dataset,
            dataset_name="Synthetic",
            skip_rates=[1, 2, 4],
            scale_factors=[1.0, 0.5, 0.25],
        )
        print(result.to_markdown_table())
    """

    def __init__(self, engine: "BenchmarkEngine") -> None:
        self._engine = engine

    def analyze(
        self,
        tracker: "BaseTracker",
        dataset: "BaseDataset",
        dataset_name: str = "Unknown",
        skip_rates: Optional[List[int]] = None,
        scale_factors: Optional[List[float]] = None,
        max_sequences: Optional[int] = None,
    ) -> CombinedResult:
        """Run the combined sweep and return a :class:`CombinedResult`.

        For each ``(skip_rate, scale_factor)`` pair a fresh wrapper stack is
        constructed around a *copy* of the base tracker class (instantiated
        with the same type, no parameters) — the original ``tracker`` object
        is only used to infer the tracker class and name.

        Args:
            tracker:        Base tracker to wrap and evaluate.
            dataset:        Dataset to evaluate on.
            dataset_name:   Human-readable dataset label for reports.
            skip_rates:     List of skip rates to sweep.  Defaults to
                            ``[1, 2, 3, 4]``.
            scale_factors:  List of scale factors to sweep.  Defaults to
                            ``[1.0, 0.75, 0.5, 0.25]``.
            max_sequences:  Optional cap on the number of sequences to
                            evaluate (useful for quick sanity checks).

        Returns:
            :class:`CombinedResult` with Pareto flags set on all entries.
        """
        from ..trackers.frame_skip import FrameSkipTracker
        from ..trackers.scale import ResolutionScaledTracker

        if skip_rates is None:
            skip_rates = [1, 2, 3, 4]
        if scale_factors is None:
            scale_factors = [1.0, 0.75, 0.5, 0.25]

        entries: List[CombinedEntry] = []

        for skip_rate in skip_rates:
            for scale_factor in scale_factors:
                wrapped = self._make_wrapper(tracker, skip_rate, scale_factor)
                br = self._engine.run(
                    tracker=wrapped,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    max_sequences=max_sequences,
                )
                entries.append(
                    CombinedEntry(
                        skip_rate=skip_rate,
                        scale_factor=scale_factor,
                        mean_iou=br.mean_iou,
                        fps=br.mean_fps,
                        benchmark_result=br,
                    )
                )

        _mark_pareto(entries)
        return CombinedResult(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
            entries=entries,
        )

    @staticmethod
    def _make_wrapper(
        base: "BaseTracker",
        skip_rate: int,
        scale_factor: float,
    ) -> "BaseTracker":
        """Construct FrameSkipTracker(ResolutionScaledTracker(fresh_base)).

        A fresh instance of the base tracker class is created each time so
        that internal tracker state does not leak between configurations.
        """
        from ..trackers.frame_skip import FrameSkipTracker
        from ..trackers.scale import ResolutionScaledTracker

        fresh_base = base.__class__()
        scaled = ResolutionScaledTracker(fresh_base, scale_factor=scale_factor)
        if skip_rate == 1:
            return scaled
        return FrameSkipTracker(scaled, skip_rate=skip_rate)


def _mark_pareto(entries: List[CombinedEntry]) -> None:
    """Set ``on_pareto=True`` on Pareto-optimal entries in-place.

    A config *A* dominates config *B* when A.mean_iou >= B.mean_iou AND
    A.fps >= B.fps, with at least one strict inequality.  A config is
    Pareto-optimal when no other config dominates it.
    """
    for candidate in entries:
        dominated = False
        for other in entries:
            if other is candidate:
                continue
            if (
                other.mean_iou >= candidate.mean_iou
                and other.fps >= candidate.fps
                and (other.mean_iou > candidate.mean_iou or other.fps > candidate.fps)
            ):
                dominated = True
                break
        candidate.on_pareto = not dominated
