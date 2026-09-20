"""Joint skip-rate × resolution-scale optimizer for edge deployment.

Both :class:`~eovot.trackers.frame_skip.FrameSkipTracker` and
:class:`~eovot.trackers.scale.ResolutionScaledTracker` trade accuracy for
speed — but they address orthogonal costs:

* **Frame skipping** reduces *temporal* compute: run the tracker every *k*
  frames instead of every frame.  Cost reduction ≈ *k*×, latency spike
  on fast-moving targets.
* **Resolution scaling** reduces *spatial* compute: process a smaller image.
  Cost reduction ≈ (1/s)², graceful accuracy degradation on low-texture
  scenes but sensitive to small objects.

Combined, the approximate throughput gain is ``skip_rate / scale_factor²``,
but the accuracy impact is non-linear: a large skip rate on a fast target
hurts more than resolution loss; resolution loss on a small target hurts
more than skipping.  Running the two analyzers independently cannot reveal
the joint optimum.

:class:`JointOptimizationAnalyzer` sweeps every ``(skip_rate, scale_factor)``
combination and identifies the **Pareto-optimal** operating points in the
``(mIoU, FPS)`` plane — the set of configs where no other config is strictly
better on both axes simultaneously.  This gives practitioners concrete,
data-driven deployment recommendations:

    "For Raspberry Pi 4 at 25 FPS, use skip=2, scale=0.5  (mIoU=0.72)"
    "For Jetson Nano at 60 FPS, use skip=3, scale=0.5  (mIoU=0.64)"

Example::

    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.analysis.joint_analysis import JointOptimizationAnalyzer

    dataset  = SyntheticDataset(num_sequences=5)
    engine   = BenchmarkEngine(verbose=False)
    analyzer = JointOptimizationAnalyzer(engine)

    report = analyzer.analyze(
        KCFTracker(), dataset,
        dataset_name="Synthetic",
        skip_rates=[1, 2, 3],
        scale_factors=[1.0, 0.75, 0.5],
    )
    print(report.to_markdown_table())

    skip, scale, iou, fps = report.optimal_config(min_iou=0.70)
    print(f"Optimal: skip={skip}, scale={scale:.2f} → {fps:.0f} FPS  mIoU={iou:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
    from ..datasets.base import BaseDataset
    from ..trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class JointEntry:
    """Single ``(skip_rate, scale_factor)`` operating point.

    Attributes:
        skip_rate:       Frame skip rate (≥ 1).
        scale_factor:    Spatial resolution scale (0 < s ≤ 1.0).
        mean_iou:        Mean IoU across all evaluated sequences.
        mean_fps:        Mean frames per second.
        peak_memory_mb:  Peak RSS memory usage in MiB.
        success_auc:     Area under the success curve, or ``None``.
        precision_auc:   Area under the precision curve, or ``None``.
        is_pareto:       ``True`` if this entry is on the Pareto front
                         (not dominated in both ``mean_iou`` and ``mean_fps``).
    """

    skip_rate: int
    scale_factor: float
    mean_iou: float
    mean_fps: float
    peak_memory_mb: float
    success_auc: Optional[float] = None
    precision_auc: Optional[float] = None
    is_pareto: bool = False

    @property
    def label(self) -> str:
        """Short human-readable config label."""
        return f"skip={self.skip_rate}/scale={self.scale_factor:.2f}"

    def __str__(self) -> str:
        pareto_mark = " *" if self.is_pareto else ""
        return (
            f"{self.label:<28}  "
            f"mIoU={self.mean_iou:.4f}  "
            f"FPS={self.mean_fps:>8.1f}  "
            f"mem={self.peak_memory_mb:.1f} MB{pareto_mark}"
        )


@dataclass
class JointOptimizationResult:
    """Full 2-D sweep result for one tracker on one dataset.

    Attributes:
        tracker_name:      Name of the evaluated tracker.
        dataset_name:      Name of the dataset.
        entries:           All ``(skip_rate, scale_factor)`` results.
        benchmark_results: Raw :class:`~eovot.benchmark.engine.BenchmarkResult`
                           objects keyed by ``(skip_rate, scale_factor)``.
    """

    tracker_name: str
    dataset_name: str
    entries: List[JointEntry] = field(default_factory=list)
    benchmark_results: Dict[Tuple[int, float], "BenchmarkResult"] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Pareto helpers
    # ------------------------------------------------------------------

    @property
    def pareto_front(self) -> List[JointEntry]:
        """Entries not dominated in ``(mIoU, FPS)`` by any other entry."""
        return [e for e in self.entries if e.is_pareto]

    def optimal_config(
        self,
        min_iou: float = 0.0,
        min_fps: float = 0.0,
    ) -> Tuple[int, float, float, float]:
        """Return the config with highest FPS subject to accuracy and speed floors.

        Among all entries satisfying ``mean_iou >= min_iou`` AND
        ``mean_fps >= min_fps``, returns the one with the highest FPS.

        Args:
            min_iou: Minimum acceptable mean IoU (default 0.0 — no constraint).
            min_fps: Minimum acceptable FPS (default 0.0 — no constraint).

        Returns:
            ``(skip_rate, scale_factor, mean_iou, mean_fps)`` of the best entry.

        Raises:
            ValueError: If no entry satisfies both constraints.
        """
        candidates = [
            e
            for e in self.entries
            if e.mean_iou >= min_iou and e.mean_fps >= min_fps
        ]
        if not candidates:
            best_iou = max(self.entries, key=lambda e: e.mean_iou).mean_iou
            raise ValueError(
                f"No config meets mIoU≥{min_iou} AND FPS≥{min_fps}. "
                f"Best available mIoU: {best_iou:.4f}"
            )
        best = max(candidates, key=lambda e: e.mean_fps)
        return best.skip_rate, best.scale_factor, best.mean_iou, best.mean_fps

    def entry_for(
        self, skip_rate: int, scale_factor: float
    ) -> Optional[JointEntry]:
        """Look up a specific ``(skip_rate, scale_factor)`` entry.

        Args:
            skip_rate:    Skip rate to look up.
            scale_factor: Scale factor to look up (matched within 1e-6).

        Returns:
            The matching :class:`JointEntry`, or ``None`` if not found.
        """
        for e in self.entries:
            if e.skip_rate == skip_rate and abs(e.scale_factor - scale_factor) < 1e-6:
                return e
        return None

    def to_grid(self) -> Dict[int, Dict[float, JointEntry]]:
        """Return results as a nested ``{skip_rate: {scale_factor: entry}}`` dict."""
        grid: Dict[int, Dict[float, JointEntry]] = {}
        for e in self.entries:
            grid.setdefault(e.skip_rate, {})[e.scale_factor] = e
        return grid

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_table(self) -> str:
        """Render the sweep as a Markdown table, sorted by FPS descending.

        Pareto-optimal entries are marked with a ✓ in the last column.

        Returns:
            Multi-line Markdown string ready for embedding in reports.
        """
        has_auc = any(
            e.success_auc is not None or e.precision_auc is not None
            for e in self.entries
        )
        header_cols = ["Config", "mIoU", "FPS", "Mem (MB)", "Pareto"]
        sep_cols = ["------", "-----", "---", "--------", "------"]
        if has_auc:
            header_cols.insert(-1, "Success AUC")
            header_cols.insert(-1, "Precision AUC")
            sep_cols.insert(-1, "-----------")
            sep_cols.insert(-1, "-------------")

        header = "| " + " | ".join(header_cols) + " |"
        sep = "| " + " | ".join(sep_cols) + " |"

        rows = []
        for e in sorted(self.entries, key=lambda x: x.mean_fps, reverse=True):
            pareto_mark = "✓" if e.is_pareto else ""
            cols = [
                f"`{e.label}`",
                f"{e.mean_iou:.4f}",
                f"{e.mean_fps:.1f}",
                f"{e.peak_memory_mb:.1f}",
            ]
            if has_auc:
                cols.append(
                    f"{e.success_auc:.4f}" if e.success_auc is not None else "—"
                )
                cols.append(
                    f"{e.precision_auc:.4f}" if e.precision_auc is not None else "—"
                )
            cols.append(pareto_mark)
            rows.append("| " + " | ".join(cols) + " |")

        return "\n".join([header, sep] + rows)

    def __str__(self) -> str:
        pareto_count = len(self.pareto_front)
        lines = [
            f"JointOptimizationResult [{self.tracker_name}] on [{self.dataset_name}]",
            f"  configs evaluated : {len(self.entries)}",
            f"  Pareto-optimal    : {pareto_count}",
            "-" * 72,
        ]
        for e in sorted(self.entries, key=lambda x: x.mean_fps, reverse=True):
            lines.append(f"  {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class JointOptimizationAnalyzer:
    """Sweep ``(skip_rate, scale_factor)`` pairs and identify the Pareto front.

    Args:
        engine: A configured :class:`~eovot.benchmark.engine.BenchmarkEngine`.

    Example::

        engine   = BenchmarkEngine(verbose=False)
        analyzer = JointOptimizationAnalyzer(engine)
        result   = analyzer.analyze(
            KCFTracker(),
            SyntheticDataset(num_sequences=5),
            skip_rates=[1, 2, 3],
            scale_factors=[1.0, 0.75, 0.5],
        )
        print(result.to_markdown_table())
        skip, scale, iou, fps = result.optimal_config(min_iou=0.75)
    """

    def __init__(self, engine: "BenchmarkEngine") -> None:
        self._engine = engine

    def analyze(
        self,
        tracker: "BaseTracker",
        dataset: "BaseDataset",
        dataset_name: str = "dataset",
        skip_rates: Optional[List[int]] = None,
        scale_factors: Optional[List[float]] = None,
        max_sequences: Optional[int] = None,
        verbose: bool = False,
    ) -> JointOptimizationResult:
        """Run the 2-D sweep and return the full result with Pareto labels.

        For each ``(skip_rate, scale_factor)`` pair:

        1. Wrap *tracker* in :class:`~eovot.trackers.scale.ResolutionScaledTracker`
           when ``scale_factor < 1.0``.
        2. Further wrap in :class:`~eovot.trackers.frame_skip.FrameSkipTracker`
           when ``skip_rate > 1``.
        3. Run via :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
        4. Collect ``(mIoU, FPS, peak_memory)`` into a :class:`JointEntry`.

        After all runs, :func:`_mark_pareto` labels entries on the Pareto front.

        Args:
            tracker:        Base tracker to wrap.  Its ``initialize()`` is
                            called by the engine at each new sequence.
            dataset:        Dataset to evaluate on.
            dataset_name:   Human-readable label for reports.
            skip_rates:     Skip rates to sweep.  Default: ``[1, 2, 3, 4]``.
            scale_factors:  Resolution scales to sweep (0 < s ≤ 1.0).
                            Default: ``[1.0, 0.75, 0.5, 0.25]``.
            max_sequences:  Limit to first *N* sequences for quick sweeps.
            verbose:        Print ``[i/total] skip=X, scale=Y …`` to stdout.

        Returns:
            :class:`JointOptimizationResult` with ``is_pareto`` flags set.

        Raises:
            ValueError: If any ``skip_rate < 1`` or any ``scale_factor``
                        outside ``(0, 1]``.
        """
        from ..trackers.frame_skip import FrameSkipTracker
        from ..trackers.scale import ResolutionScaledTracker

        if skip_rates is None:
            skip_rates = [1, 2, 3, 4]
        if scale_factors is None:
            scale_factors = [1.0, 0.75, 0.5, 0.25]

        skip_rates = sorted(set(skip_rates))
        scale_factors = sorted(set(scale_factors), reverse=True)  # full-res first

        if any(r < 1 for r in skip_rates):
            raise ValueError("All skip_rates must be >= 1.")
        if any(not (0.0 < s <= 1.0) for s in scale_factors):
            raise ValueError("All scale_factors must be in (0, 1].")

        result = JointOptimizationResult(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
        )

        total = len(skip_rates) * len(scale_factors)
        run_idx = 0

        for skip in skip_rates:
            for scale in scale_factors:
                run_idx += 1
                if verbose:
                    print(
                        f"  [{run_idx}/{total}] skip={skip}, "
                        f"scale={scale:.2f} ...",
                        flush=True,
                    )

                # Build composite wrapped tracker
                wrapped = tracker
                if scale < 1.0:
                    wrapped = ResolutionScaledTracker(wrapped, scale_factor=scale)
                if skip > 1:
                    wrapped = FrameSkipTracker(wrapped, skip_rate=skip)

                bench = self._engine.run(
                    tracker=wrapped,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    max_sequences=max_sequences,
                )
                result.benchmark_results[(skip, scale)] = bench

                entry = JointEntry(
                    skip_rate=skip,
                    scale_factor=scale,
                    mean_iou=bench.mean_iou,
                    mean_fps=bench.mean_fps,
                    peak_memory_mb=bench.peak_memory_mb,
                    success_auc=getattr(bench, "mean_success_auc", None),
                    precision_auc=getattr(bench, "mean_precision_auc", None),
                )
                result.entries.append(entry)

        _mark_pareto(result.entries)
        return result


def _mark_pareto(entries: List[JointEntry]) -> None:
    """Set ``is_pareto=True`` on entries not dominated in ``(mIoU, FPS)``.

    An entry *A* is dominated by *B* when ``B.mean_iou >= A.mean_iou`` AND
    ``B.mean_fps >= A.mean_fps`` with at least one strict inequality.
    Entries on the Pareto front are those where no such *B* exists.
    """
    for candidate in entries:
        dominated = any(
            other.mean_iou >= candidate.mean_iou
            and other.mean_fps >= candidate.mean_fps
            and (
                other.mean_iou > candidate.mean_iou
                or other.mean_fps > candidate.mean_fps
            )
            for other in entries
            if other is not candidate
        )
        candidate.is_pareto = not dominated
