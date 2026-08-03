"""Resolution-scale accuracy–speed tradeoff analysis.

Frame-resolution downscaling is a complementary edge-optimization strategy to
frame skipping (:mod:`eovot.analysis.skip_analysis`).  Whereas frame skipping
reduces the number of processed frames (temporal decimation), resolution scaling
reduces the number of pixels per frame (spatial decimation).

The throughput gain of scale factor *s* ≈ 1/s² in compute terms (processing a
0.5× frame requires ~4× fewer pixel operations), while the IoU degradation
depends on how much spatial information the tracker needs.

:class:`ResolutionScaleAnalyzer` sweeps a list of scale factors, runs the
tracker at each resolution using :class:`~eovot.benchmark.engine.BenchmarkEngine`,
and packages the resulting accuracy/FPS pairs into a :class:`ScaleResult` that
can be reported as a Markdown table or queried for the optimal scale under a
given IoU budget.

Typical usage::

    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer

    dataset  = SyntheticDataset(num_sequences=5)
    engine   = BenchmarkEngine(verbose=False)
    analyzer = ResolutionScaleAnalyzer(engine)

    result = analyzer.analyze(
        KCFTracker(),
        dataset,
        dataset_name="Synthetic",
        scale_factors=[1.0, 0.75, 0.5, 0.25],
    )
    print(result.to_markdown_table())

    # Best scale that keeps IoU above 0.80
    scale, iou, fps = result.optimal_scale(min_iou=0.80)
    print(f"Use scale={scale}: {fps:.0f} FPS at mIoU={iou:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..trackers.scale import ResolutionScaledTracker

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
    from ..datasets.base import BaseDataset
    from ..trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScaleEntry:
    """Per-scale-factor accuracy and throughput summary."""

    scale_factor: float
    """Linear resolution scale applied to both frame dimensions."""

    mean_iou: float
    mean_fps: float
    peak_memory_mb: float
    success_auc: Optional[float]
    precision_auc: Optional[float]
    iou_degradation: float = 0.0
    """Absolute IoU drop vs. the scale_factor=1.0 baseline."""
    fps_gain: float = 1.0
    """Throughput multiplier relative to the scale_factor=1.0 baseline."""
    pixel_reduction: float = 1.0
    """Fraction of pixels processed relative to full resolution (= scale²)."""

    def __str__(self) -> str:
        auc_str = f"  AUC={self.success_auc:.3f}" if self.success_auc is not None else ""
        return (
            f"scale={self.scale_factor:.2f}  "
            f"pixels={self.pixel_reduction:.2f}×  "
            f"mIoU={self.mean_iou:.4f}  (Δ={self.iou_degradation:+.4f})  "
            f"FPS={self.mean_fps:.1f}  ({self.fps_gain:.2f}×){auc_str}"
        )


@dataclass
class ScaleResult:
    """Full resolution-scale analysis result for one tracker on one dataset.

    Attributes:
        tracker_name: Name of the tracker (without the ``_scale{s}`` suffix).
        dataset_name: Dataset on which the analysis was run.
        entries: One :class:`ScaleEntry` per scale factor, in descending order
            of scale (1.0 first).
        benchmark_results: Raw :class:`~eovot.benchmark.engine.BenchmarkResult`
            objects keyed by scale factor — available for deeper analysis.
    """

    tracker_name: str
    dataset_name: str
    entries: List[ScaleEntry] = field(default_factory=list)
    benchmark_results: Dict[float, "BenchmarkResult"] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def baseline(self) -> Optional[ScaleEntry]:
        """The scale_factor=1.0 (full-resolution) entry, or ``None`` if absent."""
        for e in self.entries:
            if abs(e.scale_factor - 1.0) < 1e-9:
                return e
        return None

    def entry_for(self, scale_factor: float) -> Optional[ScaleEntry]:
        """Return the entry closest to *scale_factor*, or ``None`` if empty."""
        if not self.entries:
            return None
        return min(self.entries, key=lambda e: abs(e.scale_factor - scale_factor))

    def optimal_scale(self, min_iou: float) -> Tuple[float, float, float]:
        """Find the lowest scale factor that keeps mean IoU ≥ *min_iou*.

        Among entries whose ``mean_iou >= min_iou``, returns the one with the
        highest FPS (lowest resolution = most aggressive downscaling while
        still meeting the accuracy budget).

        Args:
            min_iou: Minimum acceptable mean IoU in ``[0, 1]``.

        Returns:
            ``(scale_factor, mean_iou, mean_fps)`` for the optimal entry.

        Raises:
            ValueError: If no entry satisfies the constraint.
        """
        candidates = [e for e in self.entries if e.mean_iou >= min_iou]
        if not candidates:
            raise ValueError(
                f"No scale factor achieves mIoU >= {min_iou}. "
                f"Best: {max(self.entries, key=lambda e: e.mean_iou).mean_iou:.4f}"
            )
        best = max(candidates, key=lambda e: e.mean_fps)
        return best.scale_factor, best.mean_iou, best.mean_fps

    def fps_at_iou_budget(self, iou_budget: float) -> Optional[float]:
        """Maximum FPS achievable within an absolute IoU degradation budget.

        Args:
            iou_budget: Maximum tolerable absolute IoU drop from the baseline.
                E.g. ``0.05`` means accept at most 5 % absolute IoU loss.

        Returns:
            Maximum FPS, or ``None`` if no baseline entry is present.
        """
        base = self.baseline
        if base is None:
            return None
        min_iou = base.mean_iou - iou_budget
        try:
            _, _, fps = self.optimal_scale(min_iou=max(0.0, min_iou))
            return fps
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_table(self) -> str:
        """Format the analysis as a Markdown table.

        Returns:
            Multi-line Markdown table string ready for embedding in reports.
        """
        has_auc = any(e.success_auc is not None for e in self.entries)
        header_cols = [
            "Scale", "Pixels", "mIoU", "ΔmIoU", "FPS", "FPS Gain", "Mem (MB)",
        ]
        sep_cols = ["-----", "------", "-----", "------", "---", "--------", "--------"]
        if has_auc:
            header_cols += ["Success AUC", "Precision AUC"]
            sep_cols += ["----------", "-------------"]

        header = "| " + " | ".join(header_cols) + " |"
        sep = "| " + " | ".join(sep_cols) + " |"

        rows = []
        for e in self.entries:
            label = f"{e.scale_factor:.2f}" + (" (full)" if abs(e.scale_factor - 1.0) < 1e-9 else "")
            delta = f"{e.iou_degradation:+.4f}" if abs(e.iou_degradation) > 1e-9 else "—"
            cols = [
                label,
                f"{e.pixel_reduction:.2f}×",
                f"{e.mean_iou:.4f}",
                delta,
                f"{e.mean_fps:.1f}",
                f"{e.fps_gain:.2f}×",
                f"{e.peak_memory_mb:.1f}",
            ]
            if has_auc:
                cols.append(f"{e.success_auc:.4f}" if e.success_auc is not None else "—")
                cols.append(f"{e.precision_auc:.4f}" if e.precision_auc is not None else "—")
            rows.append("| " + " | ".join(cols) + " |")

        return "\n".join([header, sep] + rows)

    def __str__(self) -> str:
        lines = [
            f"ResolutionScaleAnalysis [{self.tracker_name}] on [{self.dataset_name}]",
            "-" * 72,
        ]
        for e in self.entries:
            lines.append(f"  {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ResolutionScaleAnalyzer:
    """Run a tracker at multiple resolution scales and report the accuracy–FPS tradeoff.

    Args:
        engine: A configured :class:`~eovot.benchmark.engine.BenchmarkEngine`
            instance.  Reused for every scale-factor run.

    Example::

        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        from eovot.trackers.mosse import MOSSETracker
        from eovot.datasets.synthetic import SyntheticDataset

        engine   = BenchmarkEngine(verbose=False)
        analyzer = ResolutionScaleAnalyzer(engine)
        result   = analyzer.analyze(
            MOSSETracker(),
            SyntheticDataset(num_sequences=4),
            dataset_name="Synthetic",
            scale_factors=[1.0, 0.75, 0.5, 0.25],
        )
        print(result.to_markdown_table())
        scale, iou, fps = result.optimal_scale(min_iou=0.70)
    """

    def __init__(self, engine: "BenchmarkEngine") -> None:
        self._engine = engine

    def analyze(
        self,
        tracker: "BaseTracker",
        dataset: "BaseDataset",
        dataset_name: str = "dataset",
        scale_factors: Optional[List[float]] = None,
        max_sequences: Optional[int] = None,
    ) -> ScaleResult:
        """Benchmark *tracker* at each resolution scale and return the analysis.

        For ``scale_factor=1.0`` the tracker is used directly (no wrapping).
        For all other factors it is wrapped in
        :class:`~eovot.trackers.scale.ResolutionScaledTracker`.

        Args:
            tracker:       The :class:`~eovot.trackers.base.BaseTracker` to analyse.
            dataset:       Dataset to evaluate on.
            dataset_name:  Human-readable label used in results.
            scale_factors: List of linear scale factors to sweep.  Values must be
                in ``(0, 1]``.  Duplicate values are deduplicated.
                Default: ``[1.0, 0.75, 0.5, 0.25]``.
            max_sequences: Limit evaluation to the first *N* sequences.

        Returns:
            :class:`ScaleResult` with one :class:`ScaleEntry` per scale factor,
            sorted from largest (full resolution) to smallest.

        Raises:
            ValueError: If any scale factor is outside ``(0, 1]``.
        """
        if scale_factors is None:
            scale_factors = [1.0, 0.75, 0.5, 0.25]

        # Deduplicate and sort descending (full resolution first).
        seen: dict = {}
        for s in scale_factors:
            key = round(s, 6)
            if key not in seen:
                seen[key] = s
        scale_factors = sorted(seen.values(), reverse=True)

        if any(s <= 0 or s > 1.0 for s in scale_factors):
            raise ValueError(
                "All scale_factors must be in (0, 1].  "
                f"Got: {scale_factors}"
            )

        result = ScaleResult(tracker_name=tracker.name, dataset_name=dataset_name)

        baseline_iou: Optional[float] = None
        baseline_fps: Optional[float] = None

        for scale in scale_factors:
            if abs(scale - 1.0) < 1e-9:
                active_tracker = tracker
            else:
                active_tracker = ResolutionScaledTracker(tracker, scale_factor=scale)

            bench = self._engine.run(
                tracker=active_tracker,
                dataset=dataset,
                dataset_name=dataset_name,
                max_sequences=max_sequences,
            )
            result.benchmark_results[scale] = bench

            if abs(scale - 1.0) < 1e-9:
                baseline_iou = bench.mean_iou
                baseline_fps = bench.mean_fps

            iou_deg = 0.0 if baseline_iou is None else (bench.mean_iou - baseline_iou)
            fps_gain = 1.0 if baseline_fps is None or baseline_fps == 0 else (
                bench.mean_fps / baseline_fps
            )

            result.entries.append(
                ScaleEntry(
                    scale_factor=scale,
                    mean_iou=bench.mean_iou,
                    mean_fps=bench.mean_fps,
                    peak_memory_mb=bench.peak_memory_mb,
                    success_auc=bench.mean_success_auc,
                    precision_auc=bench.mean_precision_auc,
                    iou_degradation=iou_deg,
                    fps_gain=fps_gain,
                    pixel_reduction=scale ** 2,
                )
            )

        return result
