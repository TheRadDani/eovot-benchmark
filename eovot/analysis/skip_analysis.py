"""Frame-skip accuracy–speed tradeoff analysis.

Edge devices often cannot sustain full-frame-rate tracking.  Frame skipping
is a principled way to trade accuracy for throughput: instead of running the
tracker on every frame, skip *k-1* frames and propagate the last prediction.
This cuts compute by ~k× at the cost of some IoU degradation — the key
question is *how much* degradation at each skip rate.

:class:`FrameSkipAnalyzer` sweeps a list of skip rates, runs the tracker at
each rate using :class:`~eovot.benchmark.engine.BenchmarkEngine`, and
packages the resulting accuracy/FPS pairs into a :class:`SkipRateResult`
that can be reported as a Markdown table or queried for the optimal skip rate
under a given IoU budget.

Typical usage::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.analysis.skip_analysis import FrameSkipAnalyzer

    dataset  = SyntheticDataset(num_sequences=5)
    engine   = BenchmarkEngine(verbose=False)
    analyzer = FrameSkipAnalyzer(engine)

    result = analyzer.analyze(
        MOSSETracker(),
        dataset,
        dataset_name="Synthetic",
        skip_rates=[1, 2, 3, 4, 6],
    )
    print(result)
    print(result.to_markdown_table())

    # Best skip rate that keeps IoU above 0.80
    rate, iou, fps = result.optimal_rate(min_iou=0.80)
    print(f"Use skip_rate={rate}: {fps:.0f} FPS at mIoU={iou:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..trackers.frame_skip import FrameSkipTracker, SkipMode

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
    from ..datasets.base import BaseDataset
    from ..trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SkipRateEntry:
    """Per-skip-rate accuracy and throughput summary."""

    skip_rate: int
    mean_iou: float
    mean_fps: float
    peak_memory_mb: float
    success_auc: Optional[float]
    precision_auc: Optional[float]
    iou_degradation: float = 0.0
    """Absolute drop in mean IoU vs. the skip_rate=1 baseline."""
    fps_gain: float = 1.0
    """Throughput multiplier relative to the skip_rate=1 baseline."""

    def __str__(self) -> str:
        auc_str = f"  AUC={self.success_auc:.3f}" if self.success_auc is not None else ""
        return (
            f"skip={self.skip_rate}  mIoU={self.mean_iou:.4f}"
            f"  (Δ={self.iou_degradation:+.4f})  "
            f"FPS={self.mean_fps:.1f}  ({self.fps_gain:.2f}×){auc_str}"
        )


@dataclass
class SkipRateResult:
    """Full frame-skip analysis result for one tracker on one dataset.

    Attributes:
        tracker_name: Name of the tracker (without the ``_skip{k}`` suffix).
        dataset_name: Dataset on which the analysis was run.
        mode: Skip propagation mode (``"repeat"`` or ``"linear"``).
        entries: One :class:`SkipRateEntry` per skip rate, in ascending order.
        benchmark_results: Raw :class:`~eovot.benchmark.engine.BenchmarkResult`
            objects keyed by skip rate — available for deeper analysis.
    """

    tracker_name: str
    dataset_name: str
    mode: SkipMode
    entries: List[SkipRateEntry] = field(default_factory=list)
    benchmark_results: Dict[int, "BenchmarkResult"] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def baseline(self) -> Optional[SkipRateEntry]:
        """The skip_rate=1 (no-skip) entry, or ``None`` if not present."""
        for e in self.entries:
            if e.skip_rate == 1:
                return e
        return None

    def entry_for(self, skip_rate: int) -> Optional[SkipRateEntry]:
        """Return the entry for *skip_rate*, or ``None``."""
        for e in self.entries:
            if e.skip_rate == skip_rate:
                return e
        return None

    def optimal_rate(self, min_iou: float) -> Tuple[int, float, float]:
        """Find the highest skip rate that keeps mean IoU ≥ *min_iou*.

        Among entries whose ``mean_iou >= min_iou``, returns the one with the
        highest FPS.

        Args:
            min_iou: Minimum acceptable mean IoU (in ``[0, 1]``).

        Returns:
            ``(skip_rate, mean_iou, mean_fps)`` for the optimal entry.

        Raises:
            ValueError: If no entry satisfies the constraint.
        """
        candidates = [e for e in self.entries if e.mean_iou >= min_iou]
        if not candidates:
            raise ValueError(
                f"No skip rate achieves mIoU >= {min_iou}. "
                f"Best: {max(self.entries, key=lambda e: e.mean_iou).mean_iou:.4f}"
            )
        best = max(candidates, key=lambda e: e.mean_fps)
        return best.skip_rate, best.mean_iou, best.mean_fps

    def fps_at_iou_budget(self, iou_budget: float) -> Optional[float]:
        """Return the maximum FPS achievable within an absolute IoU budget.

        Args:
            iou_budget: Maximum tolerable IoU drop from the baseline
                (e.g. ``0.05`` means accept at most 5 % absolute IoU loss).

        Returns:
            Maximum FPS, or ``None`` if no baseline result is available.
        """
        base = self.baseline
        if base is None:
            return None
        min_iou = base.mean_iou - iou_budget
        try:
            _, _, fps = self.optimal_rate(min_iou=max(0.0, min_iou))
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
        header_cols = ["Skip Rate", "mIoU", "ΔmIoU", "FPS", "FPS Gain", "Mem (MB)"]
        sep_cols = ["---------", "-----", "------", "---", "--------", "--------"]
        if has_auc:
            header_cols += ["Success AUC", "Precision AUC"]
            sep_cols += ["----------", "-------------"]

        header = "| " + " | ".join(header_cols) + " |"
        sep = "| " + " | ".join(sep_cols) + " |"

        rows = []
        for e in self.entries:
            label = f"{e.skip_rate}" + (" (baseline)" if e.skip_rate == 1 else "")
            delta = f"{e.iou_degradation:+.4f}" if e.iou_degradation != 0 else "—"
            cols = [
                label,
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
            f"FrameSkipAnalysis [{self.tracker_name}] on [{self.dataset_name}] "
            f"mode={self.mode}",
            "-" * 70,
        ]
        for e in self.entries:
            lines.append(f"  {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class FrameSkipAnalyzer:
    """Run a tracker at multiple skip rates and report the accuracy–FPS tradeoff.

    Args:
        engine: A configured :class:`~eovot.benchmark.engine.BenchmarkEngine`
            instance.  Reused for every skip-rate run.

    Example::

        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.analysis.skip_analysis import FrameSkipAnalyzer
        from eovot.trackers.kcf import KCFTracker
        from eovot.datasets.synthetic import SyntheticDataset

        engine   = BenchmarkEngine(verbose=False)
        analyzer = FrameSkipAnalyzer(engine)
        result   = analyzer.analyze(
            KCFTracker(),
            SyntheticDataset(num_sequences=4),
            dataset_name="Synthetic",
            skip_rates=[1, 2, 3, 4],
        )
        print(result.to_markdown_table())
        rate, iou, fps = result.optimal_rate(min_iou=0.85)
    """

    def __init__(self, engine: "BenchmarkEngine") -> None:
        self._engine = engine

    def analyze(
        self,
        tracker: "BaseTracker",
        dataset: "BaseDataset",
        dataset_name: str = "dataset",
        skip_rates: Optional[List[int]] = None,
        mode: SkipMode = "repeat",
        max_sequences: Optional[int] = None,
    ) -> SkipRateResult:
        """Benchmark *tracker* at each skip rate and return the analysis.

        For ``skip_rate=1`` the tracker is used directly (no wrapping).
        For all other rates it is wrapped in
        :class:`~eovot.trackers.frame_skip.FrameSkipTracker`.

        The tracker's ``initialize()`` is called by the engine at the start
        of every sequence, so internal state is correctly reset between runs.

        Args:
            tracker: The :class:`~eovot.trackers.base.BaseTracker` to analyse.
                Must implement ``initialize()`` and ``update()``.
            dataset: Dataset to evaluate on.
            dataset_name: Human-readable label used in results.
            skip_rates: List of skip rates to sweep.  Must contain at least
                one entry ≥ 1.  Default: ``[1, 2, 3, 4]``.
            mode: Propagation mode for skipped frames
                (``"repeat"`` or ``"linear"``).  Default: ``"repeat"``.
            max_sequences: Limit evaluation to the first *N* sequences.
                Useful for quick sweeps during development.

        Returns:
            :class:`SkipRateResult` with one :class:`SkipRateEntry` per rate.

        Raises:
            ValueError: If *skip_rates* is empty or contains values < 1.
        """
        from ..benchmark.engine import BenchmarkEngine

        if skip_rates is None:
            skip_rates = [1, 2, 3, 4]
        skip_rates = sorted(set(skip_rates))
        if not skip_rates or any(r < 1 for r in skip_rates):
            raise ValueError("All skip rates must be >= 1.")

        result = SkipRateResult(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
            mode=mode,
        )

        baseline_iou: Optional[float] = None
        baseline_fps: Optional[float] = None

        for rate in skip_rates:
            if rate == 1:
                active_tracker = tracker
            else:
                active_tracker = FrameSkipTracker(tracker, skip_rate=rate, mode=mode)

            bench = self._engine.run(
                tracker=active_tracker,
                dataset=dataset,
                dataset_name=dataset_name,
                max_sequences=max_sequences,
            )
            result.benchmark_results[rate] = bench

            if rate == 1:
                baseline_iou = bench.mean_iou
                baseline_fps = bench.mean_fps

            iou_deg = 0.0 if baseline_iou is None else (bench.mean_iou - baseline_iou)
            fps_gain = 1.0 if baseline_fps is None or baseline_fps == 0 else (
                bench.mean_fps / baseline_fps
            )

            result.entries.append(
                SkipRateEntry(
                    skip_rate=rate,
                    mean_iou=bench.mean_iou,
                    mean_fps=bench.mean_fps,
                    peak_memory_mb=bench.peak_memory_mb,
                    success_auc=bench.mean_success_auc,
                    precision_auc=bench.mean_precision_auc,
                    iou_degradation=iou_deg,
                    fps_gain=fps_gain,
                )
            )

        return result

    def compare_modes(
        self,
        tracker: "BaseTracker",
        dataset: "BaseDataset",
        dataset_name: str = "dataset",
        skip_rates: Optional[List[int]] = None,
        max_sequences: Optional[int] = None,
    ) -> Dict[SkipMode, SkipRateResult]:
        """Run the analysis under both ``"repeat"`` and ``"linear"`` modes.

        Useful for choosing which propagation strategy is better suited to
        the motion characteristics of a given dataset.

        Args:
            tracker:        Tracker to wrap.
            dataset:        Evaluation dataset.
            dataset_name:   Human-readable label.
            skip_rates:     Skip rates to sweep (default: ``[1, 2, 3, 4]``).
            max_sequences:  Limit to first *N* sequences.

        Returns:
            Dict with keys ``"repeat"`` and ``"linear"``, each mapping to a
            :class:`SkipRateResult`.
        """
        return {
            mode: self.analyze(
                tracker, dataset, dataset_name=dataset_name,
                skip_rates=skip_rates, mode=mode,
                max_sequences=max_sequences,
            )
            for mode in ("repeat", "linear")
        }
