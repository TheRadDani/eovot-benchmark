"""Joint 2-D edge optimization: frame-skip × resolution-scale grid search.

Edge deployment rarely offers a single degree of freedom.  On a constrained
device you can reduce compute along two independent axes simultaneously:

* **Frame skipping** — process only 1 in every *k* frames (temporal decimation)
* **Resolution scaling** — downscale every frame by factor *s* (spatial decimation)

The combined throughput gain is approximately ``skip_rate / scale_factor²``.
Neither axis fully dominates the other: skipping at rate 4 and scaling by 0.5
may reach the same FPS target as many other combinations, but at very different
accuracy costs depending on the tracker and scene content.

:class:`JointEdgeAnalyzer` sweeps the full Cartesian product of
``skip_rates × scale_factors``, runs a :class:`~eovot.benchmark.engine.BenchmarkEngine`
evaluation for each point, and returns a :class:`JointEdgeResult` that:

- Reports accuracy + FPS + EES for every grid point
- Identifies the Pareto front in the (mIoU, FPS) space
- Answers ``optimal_point(fps_budget, min_iou)`` queries
- Renders a Markdown grid table and LaTeX fragment for papers

Design note
~~~~~~~~~~~
The search is exact (no surrogate model) because the grid is typically small
(≤ 4 skip rates × ≤ 4 scale factors = 16 points) and each evaluation is fast
on a synthetic or sub-sampled dataset.  For larger grids, pass
``max_sequences`` to the engine to cap cost.

Typical usage::

    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.analysis.joint_analysis import JointEdgeAnalyzer

    dataset  = SyntheticDataset(num_sequences=5, num_frames=100)
    engine   = BenchmarkEngine(verbose=False)
    analyzer = JointEdgeAnalyzer(engine)

    result = analyzer.analyze(
        KCFTracker(),
        dataset,
        dataset_name="Synthetic",
        skip_rates=[1, 2, 4],
        scale_factors=[1.0, 0.75, 0.5],
    )

    # Ranked grid
    print(result.to_markdown_table())

    # Best operating point within budget
    skip, scale, entry = result.optimal_point(fps_budget=60.0, min_iou=0.50)
    print(f"Recommended: skip={skip}, scale={scale} → {entry.fps:.0f} FPS / mIoU={entry.mean_iou:.3f}")

    # Pareto-optimal points only
    for pt in result.pareto_front():
        print(pt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..trackers.frame_skip import FrameSkipTracker
from ..trackers.scale import ResolutionScaledTracker

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
    from ..datasets.base import BaseDataset
    from ..trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class JointGridEntry:
    """Results for a single (skip_rate, scale_factor) operating point.

    Attributes:
        skip_rate: Temporal decimation factor (1 = no skipping).
        scale_factor: Spatial scale applied to both frame dimensions.
        mean_iou: Mean IoU across all evaluated frames.
        fps: Mean frames-per-second throughput.
        peak_memory_mb: Peak RSS memory footprint in megabytes.
        success_auc: Area under the Success Curve, or ``None`` if unavailable.
        ees: Edge Efficiency Score (accuracy × log1p(fps) / memory_penalty).
        iou_vs_baseline: Absolute IoU drop relative to (skip=1, scale=1.0).
        fps_vs_baseline: FPS multiplier relative to (skip=1, scale=1.0).
        on_pareto_front: ``True`` when no other point dominates in (mIoU, FPS).
        pixel_throughput_ratio: Fraction of pixels processed per unit time
            compared to the baseline, ≈ ``1 / (skip_rate × scale_factor²)``.
            Reflects theoretical compute savings.
    """

    skip_rate: int
    scale_factor: float
    mean_iou: float
    fps: float
    peak_memory_mb: float
    success_auc: Optional[float]
    ees: float
    iou_vs_baseline: float = 0.0
    fps_vs_baseline: float = 1.0
    on_pareto_front: bool = False
    pixel_throughput_ratio: float = 1.0

    def __str__(self) -> str:
        pareto = " [Pareto]" if self.on_pareto_front else ""
        return (
            f"skip={self.skip_rate} scale={self.scale_factor:.2f}  "
            f"mIoU={self.mean_iou:.4f} ({self.iou_vs_baseline:+.4f})  "
            f"FPS={self.fps:.1f} ({self.fps_vs_baseline:.2f}×)  "
            f"EES={self.ees:.4f}{pareto}"
        )


@dataclass
class JointEdgeResult:
    """Full 2-D grid search result returned by :class:`JointEdgeAnalyzer`.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Name of the dataset used.
        grid: Nested dict ``{skip_rate: {scale_factor: JointGridEntry}}``.
        baseline: The (skip=1, scale=1.0) reference entry, or ``None`` if it
            was not included in the search grid.
    """

    tracker_name: str
    dataset_name: str
    grid: Dict[int, Dict[float, JointGridEntry]] = field(default_factory=dict)
    baseline: Optional[JointGridEntry] = None

    # ------------------------------------------------------------------
    # Flat iteration helpers
    # ------------------------------------------------------------------

    def entries(self) -> List[JointGridEntry]:
        """Return all grid entries as a flat list."""
        out: List[JointGridEntry] = []
        for skip_dict in self.grid.values():
            out.extend(skip_dict.values())
        return out

    def pareto_front(self) -> List[JointGridEntry]:
        """Return only the Pareto-optimal operating points, sorted by FPS."""
        return sorted(
            [e for e in self.entries() if e.on_pareto_front],
            key=lambda e: e.fps,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def optimal_point(
        self,
        fps_budget: float,
        min_iou: float = 0.0,
    ) -> Tuple[int, float, JointGridEntry]:
        """Find the operating point with the highest EES within constraints.

        Selects the point satisfying ``fps >= fps_budget`` AND
        ``mean_iou >= min_iou``, then maximises EES.

        Args:
            fps_budget: Minimum required frames-per-second.
            min_iou:    Minimum acceptable mean IoU (default 0.0 — no floor).

        Returns:
            ``(skip_rate, scale_factor, JointGridEntry)`` for the best point,
            or the highest-FPS point when no entry meets both constraints.
        """
        candidates = [
            e for e in self.entries()
            if e.fps >= fps_budget and e.mean_iou >= min_iou
        ]
        if not candidates:
            # Relax min_iou and return the highest-FPS point available
            candidates = sorted(self.entries(), key=lambda e: e.fps, reverse=True)
        best = max(candidates, key=lambda e: e.ees)
        return best.skip_rate, best.scale_factor, best

    def best_accuracy(self) -> JointGridEntry:
        """Return the entry with the highest mean IoU."""
        return max(self.entries(), key=lambda e: e.mean_iou)

    def best_fps(self) -> JointGridEntry:
        """Return the entry with the highest FPS."""
        return max(self.entries(), key=lambda e: e.fps)

    def best_ees(self) -> JointGridEntry:
        """Return the entry with the highest Edge Efficiency Score."""
        return max(self.entries(), key=lambda e: e.ees)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_table(self) -> str:
        """Render the full grid as a Markdown table.

        Rows are skip rates; columns are scale factors.  Each cell shows
        ``mIoU / FPS`` so the tradeoff is immediately visible.  Pareto-
        optimal cells are marked with a ★.

        Returns:
            Multi-line Markdown string.
        """
        skip_rates = sorted(self.grid.keys())
        scale_factors = sorted(
            next(iter(self.grid.values())).keys()
        ) if self.grid else []

        header = (
            f"## Joint Edge Optimization: {self.tracker_name} on {self.dataset_name}\n\n"
        )

        # Column headers
        col_header = "| skip \\ scale |"
        for sf in scale_factors:
            col_header += f" scale={sf:.2f} |"
        sep = "|" + "---|" * (len(scale_factors) + 1)

        lines = [header, col_header, sep]

        for sr in skip_rates:
            row = f"| skip={sr} |"
            for sf in scale_factors:
                entry = self.grid.get(sr, {}).get(sf)
                if entry is None:
                    row += " — |"
                else:
                    star = "★" if entry.on_pareto_front else ""
                    row += f" {entry.mean_iou:.3f}/{entry.fps:.0f}fps{star} |"
            lines.append(row)

        lines.append(
            "\n*Cells show mIoU/FPS. ★ = Pareto-optimal in (mIoU, FPS) space.*"
        )
        return "\n".join(lines)

    def to_ranked_markdown(self) -> str:
        """Render a flat ranked table sorted by EES (best first).

        Returns:
            Multi-line Markdown string with one row per grid point.
        """
        entries = sorted(self.entries(), key=lambda e: e.ees, reverse=True)
        lines = [
            f"### {self.tracker_name} — Ranked Operating Points\n",
            "| Rank | skip | scale | mIoU | FPS | EES | Δ IoU | FPS× | Pareto |",
            "|------|------|-------|-----:|----:|----:|------:|-----:|:------:|",
        ]
        for rank, e in enumerate(entries, start=1):
            pareto = "★" if e.on_pareto_front else ""
            lines.append(
                f"| {rank} | {e.skip_rate} | {e.scale_factor:.2f} "
                f"| {e.mean_iou:.4f} | {e.fps:.1f} | {e.ees:.4f} "
                f"| {e.iou_vs_baseline:+.4f} | {e.fps_vs_baseline:.2f}× | {pareto} |"
            )
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Render the grid as a LaTeX ``tabular`` fragment.

        Best mIoU in each (skip, scale) row is typeset in bold.

        Returns:
            LaTeX string ready for ``\\input{}`` in a paper.
        """
        skip_rates = sorted(self.grid.keys())
        scale_factors = sorted(
            next(iter(self.grid.values())).keys()
        ) if self.grid else []

        n_cols = len(scale_factors)
        col_spec = "l" + "c" * n_cols

        header_row = "skip $\\downarrow$ / scale $\\rightarrow$"
        for sf in scale_factors:
            header_row += f" & {sf:.2f}"
        header_row += " \\\\"

        # Best mIoU per column for bolding
        best_iou_per_col: Dict[float, float] = {}
        for sf in scale_factors:
            vals = [
                self.grid[sr][sf].mean_iou
                for sr in skip_rates
                if sf in self.grid.get(sr, {})
            ]
            best_iou_per_col[sf] = max(vals) if vals else 0.0

        rows = [
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            header_row,
            "\\midrule",
        ]

        for sr in skip_rates:
            row = f"skip={sr}"
            for sf in scale_factors:
                entry = self.grid.get(sr, {}).get(sf)
                if entry is None:
                    row += " & —"
                else:
                    cell = f"{entry.mean_iou:.3f}/{entry.fps:.0f}"
                    if abs(entry.mean_iou - best_iou_per_col.get(sf, 0.0)) < 1e-6:
                        cell = f"\\textbf{{{cell}}}"
                    row += f" & {cell}"
            row += " \\\\"
            rows.append(row)

        rows += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class JointEdgeAnalyzer:
    """Sweep all (skip_rate × scale_factor) combinations for a tracker.

    Wraps :class:`~eovot.benchmark.engine.BenchmarkEngine` to run the full
    Cartesian product and collect accuracy + efficiency metrics at every point.

    Args:
        engine: A configured :class:`~eovot.benchmark.engine.BenchmarkEngine`.
            Pass ``verbose=False`` to suppress per-sequence output during the
            grid search.
        memory_budget_mb: Memory ceiling used in the Edge Efficiency Score
            denominator.  Default: ``512.0`` MB.
    """

    def __init__(
        self,
        engine: "BenchmarkEngine",
        memory_budget_mb: float = 512.0,
    ) -> None:
        self._engine = engine
        self._memory_budget_mb = memory_budget_mb

    def analyze(
        self,
        tracker: "BaseTracker",
        dataset: "BaseDataset",
        dataset_name: str = "unknown",
        skip_rates: Optional[List[int]] = None,
        scale_factors: Optional[List[float]] = None,
    ) -> JointEdgeResult:
        """Run the full 2-D grid search.

        Args:
            tracker:       Base tracker to wrap with skip + scale adapters.
            dataset:       Dataset to evaluate on.
            dataset_name:  Label used in result headers and reports.
            skip_rates:    Temporal decimation factors to sweep.
                           Default: ``[1, 2, 4]``.
            scale_factors: Spatial scale factors to sweep.
                           Default: ``[1.0, 0.75, 0.5]``.

        Returns:
            :class:`JointEdgeResult` with a complete grid of evaluated points.
        """
        if skip_rates is None:
            skip_rates = [1, 2, 4]
        if scale_factors is None:
            scale_factors = [1.0, 0.75, 0.5]

        result = JointEdgeResult(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
        )

        all_entries: List[JointGridEntry] = []

        for skip in skip_rates:
            result.grid[skip] = {}
            for scale in scale_factors:
                wrapped = self._wrap(tracker, skip, scale)
                bench = self._engine.run(wrapped, dataset, dataset_name=dataset_name)
                ees = self._ees(bench.mean_iou, bench.mean_fps, bench.peak_memory_mb)
                entry = JointGridEntry(
                    skip_rate=skip,
                    scale_factor=scale,
                    mean_iou=bench.mean_iou,
                    fps=bench.mean_fps,
                    peak_memory_mb=bench.peak_memory_mb,
                    success_auc=bench.mean_success_auc,
                    ees=ees,
                    pixel_throughput_ratio=1.0 / (skip * scale * scale),
                )
                result.grid[skip][scale] = entry
                all_entries.append(entry)

        # Store the baseline for reference
        baseline_key = (1, 1.0)
        if 1 in result.grid and 1.0 in result.grid[1]:
            result.baseline = result.grid[1][1.0]

        # Compute relative metrics vs baseline
        baseline_entry = result.baseline
        if baseline_entry is not None:
            for entry in all_entries:
                entry.iou_vs_baseline = entry.mean_iou - baseline_entry.mean_iou
                entry.fps_vs_baseline = (
                    entry.fps / baseline_entry.fps if baseline_entry.fps > 0 else 1.0
                )

        # Mark Pareto front in (mIoU, FPS) space
        self._mark_pareto(all_entries)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap(
        tracker: "BaseTracker", skip_rate: int, scale_factor: float
    ) -> "BaseTracker":
        """Apply skip and/or scale wrappers to a tracker."""
        t = tracker
        if scale_factor < 1.0:
            t = ResolutionScaledTracker(t, scale_factor=scale_factor)
        if skip_rate > 1:
            t = FrameSkipTracker(t, skip_rate=skip_rate)
        return t

    def _ees(self, mean_iou: float, fps: float, peak_memory_mb: float) -> float:
        """Edge Efficiency Score: accuracy × log1p(fps) / memory_penalty."""
        if fps <= 0 or mean_iou < 0:
            return 0.0
        memory_penalty = 1.0 + peak_memory_mb / self._memory_budget_mb
        return (mean_iou * math.log1p(fps)) / memory_penalty

    @staticmethod
    def _mark_pareto(entries: List[JointGridEntry]) -> None:
        """Set ``on_pareto_front`` in-place for entries in (mIoU, FPS) space.

        Entry A dominates B when A.mean_iou >= B.mean_iou AND A.fps >= B.fps
        with at least one strict inequality.
        """
        for i, candidate in enumerate(entries):
            dominated = False
            for j, other in enumerate(entries):
                if i == j:
                    continue
                if (
                    other.mean_iou >= candidate.mean_iou
                    and other.fps >= candidate.fps
                    and (
                        other.mean_iou > candidate.mean_iou
                        or other.fps > candidate.fps
                    )
                ):
                    dominated = True
                    break
            candidate.on_pareto_front = not dominated
