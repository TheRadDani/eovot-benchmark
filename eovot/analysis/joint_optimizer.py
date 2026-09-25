"""Joint frame-skip × resolution optimizer for edge deployment.

Single-axis analyses (:mod:`~eovot.analysis.skip_analysis` and
:mod:`~eovot.analysis.resolution_analysis`) explore the deployment space along
independent axes.  In practice, engineers tune both simultaneously: skip every
*k*-th frame **and** downscale frames by factor *s*, achieving a combined
throughput gain of roughly *k / s²* in compute terms.

This module implements :class:`JointDeploymentOptimizer`, which:

1. Sweeps the full ``(skip_rate, scale_factor)`` grid.
2. Runs each configuration through :class:`~eovot.benchmark.engine.BenchmarkEngine`.
3. Identifies Pareto-optimal ``(IoU, FPS)`` configurations — no other setting
   dominates both accuracy **and** throughput simultaneously.
4. Recommends the optimal ``(skip, scale)`` pair for a user-specified FPS
   budget or minimum IoU constraint.

This closes the gap between EOVOT's per-axis analyses and real-world edge
deployment, where both axes are adjusted together.

Typical usage::

    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.analysis.joint_optimizer import JointDeploymentOptimizer

    engine   = BenchmarkEngine(verbose=False)
    dataset  = SyntheticDataset(num_sequences=5, num_frames=200)
    tracker  = KCFTracker()

    optimizer = JointDeploymentOptimizer(engine)
    result = optimizer.optimize(
        tracker,
        dataset,
        dataset_name="Synthetic",
        skip_rates=[1, 2, 3, 4],
        scale_factors=[1.0, 0.75, 0.5, 0.25],
    )

    # Best config that keeps IoU above 0.80
    rec = result.recommend(min_iou=0.80)
    print(f"skip={rec.skip_rate} scale={rec.scale_factor}  "
          f"FPS={rec.mean_fps:.0f}  mIoU={rec.mean_iou:.3f}")

    # Full grid as Markdown table
    print(result.to_markdown_table())

    # Pareto-optimal frontier only
    for entry in result.pareto_front():
        print(entry)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..trackers.frame_skip import FrameSkipTracker
from ..trackers.scale import ResolutionScaledTracker

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
    from ..datasets.base import BaseDataset
    from ..trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Per-configuration entry
# ---------------------------------------------------------------------------

@dataclass
class JointEntry:
    """Result for one ``(skip_rate, scale_factor)`` configuration.

    Attributes:
        skip_rate:        Frame-skip rate applied (1 = full rate).
        scale_factor:     Frame downscale factor applied (1.0 = full resolution).
        mean_iou:         Mean IoU across all evaluated frames.
        mean_fps:         Mean throughput in frames per second.
        peak_memory_mb:   Peak RSS memory during the run.
        fps_gain:         Throughput multiplier relative to the baseline
                          configuration ``(skip=1, scale=1.0)``.
        iou_degradation:  Absolute drop in mean IoU vs. the baseline
                          (positive = accuracy loss).
        on_pareto_front:  ``True`` if no other configuration dominates both
                          accuracy **and** throughput simultaneously.
        success_auc:      Mean success-curve AUC (``None`` if not computed).
        config_label:     Human-readable ``"skip=k scale=s"`` label.
    """

    skip_rate: int
    scale_factor: float
    mean_iou: float
    mean_fps: float
    peak_memory_mb: float
    fps_gain: float = 1.0
    iou_degradation: float = 0.0
    on_pareto_front: bool = False
    success_auc: Optional[float] = None

    @property
    def config_label(self) -> str:
        return f"skip={self.skip_rate} scale={self.scale_factor:.2f}"

    def __str__(self) -> str:
        pareto_tag = " [Pareto]" if self.on_pareto_front else ""
        return (
            f"JointEntry({self.config_label}  "
            f"mIoU={self.mean_iou:.4f}  Δ={self.iou_degradation:+.4f}  "
            f"FPS={self.mean_fps:.1f}  gain={self.fps_gain:.2f}×"
            f"{pareto_tag})"
        )


# ---------------------------------------------------------------------------
# Optimization result
# ---------------------------------------------------------------------------

class JointOptimizationResult:
    """Grid of :class:`JointEntry` objects from a joint sweep.

    Created by :meth:`JointDeploymentOptimizer.optimize`.

    Args:
        entries:        All ``(skip_rate, scale_factor)`` configurations
                        evaluated, in the order they were run.
        tracker_name:   Name of the evaluated tracker.
        dataset_name:   Name of the evaluated dataset.
    """

    def __init__(
        self,
        entries: List[JointEntry],
        tracker_name: str,
        dataset_name: str,
    ) -> None:
        self.entries = entries
        self.tracker_name = tracker_name
        self.dataset_name = dataset_name
        self._mark_pareto()

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __repr__(self) -> str:
        n_pareto = sum(1 for e in self.entries if e.on_pareto_front)
        return (
            f"JointOptimizationResult[{self.tracker_name} on {self.dataset_name}] "
            f"configs={len(self.entries)} pareto_front={n_pareto}"
        )

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def pareto_front(self) -> List[JointEntry]:
        """Return all Pareto-optimal configurations, sorted by FPS ascending.

        A configuration is Pareto-optimal when no other configuration has
        both strictly higher (or equal) mean IoU **and** strictly higher
        (or equal) FPS.
        """
        return sorted(
            [e for e in self.entries if e.on_pareto_front],
            key=lambda e: e.mean_fps,
        )

    def recommend(
        self,
        min_iou: Optional[float] = None,
        fps_budget: Optional[float] = None,
    ) -> Optional[JointEntry]:
        """Recommend the best ``(skip, scale)`` configuration.

        Selection logic:

        1. If *min_iou* is given: among configurations satisfying
           ``mean_iou >= min_iou``, pick the one with the highest FPS.
        2. If *fps_budget* is given: among configurations satisfying
           ``mean_fps <= fps_budget``, pick the one with the highest mean IoU.
        3. If both are given: apply both constraints, then maximise FPS.
        4. If neither is given: return the entry on the Pareto front with
           the best harmonic mean of normalised IoU and normalised FPS.

        Args:
            min_iou:    Minimum acceptable mean IoU (exclusive filter).
            fps_budget: Maximum acceptable FPS (inclusive filter).  Useful
                when the deployment target imposes a power/compute ceiling.

        Returns:
            The recommended :class:`JointEntry`, or ``None`` if no entry
            satisfies the constraints.
        """
        candidates = list(self.entries)
        if min_iou is not None:
            candidates = [e for e in candidates if e.mean_iou >= min_iou]
        if fps_budget is not None:
            candidates = [e for e in candidates if e.mean_fps <= fps_budget]
        if not candidates:
            return None
        if min_iou is not None:
            return max(candidates, key=lambda e: e.mean_fps)
        if fps_budget is not None:
            return max(candidates, key=lambda e: e.mean_iou)
        # Unconstrained: harmonic mean of normalised axes
        max_iou = max(e.mean_iou for e in candidates) or 1.0
        max_fps = max(e.mean_fps for e in candidates) or 1.0
        def _hmean(e: JointEntry) -> float:
            ni = e.mean_iou / max_iou
            nf = e.mean_fps / max_fps
            return 2.0 * ni * nf / (ni + nf + 1e-9)
        return max(candidates, key=_hmean)

    def baseline(self) -> Optional[JointEntry]:
        """Return the ``(skip=1, scale=1.0)`` baseline entry if present."""
        for e in self.entries:
            if e.skip_rate == 1 and abs(e.scale_factor - 1.0) < 1e-6:
                return e
        return None

    def to_markdown_table(self) -> str:
        """Render a Markdown grid table of ``skip × scale`` results.

        Rows are skip rates; columns are scale factors.  Each cell shows
        ``mIoU / FPS``; Pareto-optimal cells are marked with ``*``.

        Returns:
            Multi-line Markdown string.
        """
        skip_rates = sorted({e.skip_rate for e in self.entries})
        scale_factors = sorted({e.scale_factor for e in self.entries}, reverse=True)
        entry_map = {(e.skip_rate, e.scale_factor): e for e in self.entries}

        scale_hdrs = " | ".join(f"scale={s:.2f}" for s in scale_factors)
        header = f"| skip \\ scale | {scale_hdrs} |"
        sep = "|-" + "-|-".join(["------"] * (len(scale_factors) + 1)) + "-|"

        lines = [
            f"## Joint Optimization Grid: {self.tracker_name} on {self.dataset_name}",
            "",
            f"Format: `mIoU / FPS` — `*` = Pareto-optimal",
            "",
            header,
            sep,
        ]
        for skip in skip_rates:
            cells = []
            for scale in scale_factors:
                e = entry_map.get((skip, scale))
                if e is None:
                    cells.append("  —   ")
                else:
                    tag = "*" if e.on_pareto_front else " "
                    cells.append(f"{e.mean_iou:.3f}/{e.mean_fps:5.0f}{tag}")
            lines.append(f"| skip={skip:<2} | " + " | ".join(cells) + " |")

        lines.append("")
        pf = self.pareto_front()
        if pf:
            lines.append("**Pareto front** (IoU × FPS trade-offs):")
            for e in pf:
                lines.append(f"- {e.config_label}: mIoU={e.mean_iou:.4f}  FPS={e.mean_fps:.1f}  "
                              f"gain={e.fps_gain:.2f}×  Δ={e.iou_degradation:+.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise the full result to a plain dict for JSON export."""
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "num_configs": len(self.entries),
            "entries": [
                {
                    "skip_rate": e.skip_rate,
                    "scale_factor": round(e.scale_factor, 4),
                    "mean_iou": round(e.mean_iou, 4),
                    "mean_fps": round(e.mean_fps, 2),
                    "peak_memory_mb": round(e.peak_memory_mb, 2),
                    "fps_gain": round(e.fps_gain, 4),
                    "iou_degradation": round(e.iou_degradation, 4),
                    "on_pareto_front": e.on_pareto_front,
                    "success_auc": round(e.success_auc, 4) if e.success_auc is not None else None,
                }
                for e in self.entries
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_pareto(self) -> None:
        """Label Pareto-optimal entries.

        An entry *e* is dominated if there exists another entry *f* such that
        ``f.mean_iou >= e.mean_iou`` and ``f.mean_fps >= e.mean_fps`` with at
        least one strict inequality.
        """
        for e in self.entries:
            e.on_pareto_front = not any(
                (f.mean_iou >= e.mean_iou and f.mean_fps >= e.mean_fps)
                and (f.mean_iou > e.mean_iou or f.mean_fps > e.mean_fps)
                for f in self.entries
                if f is not e
            )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class JointDeploymentOptimizer:
    """Sweep the ``(skip_rate, scale_factor)`` grid to find optimal edge configs.

    Wraps the base tracker in
    :class:`~eovot.trackers.frame_skip.FrameSkipTracker` and
    :class:`~eovot.trackers.scale.ResolutionScaledTracker` for each
    ``(skip, scale)`` combination, runs a fresh benchmark, and packages all
    results into a :class:`JointOptimizationResult`.

    Args:
        engine:   A pre-configured :class:`~eovot.benchmark.engine.BenchmarkEngine`
                  instance.  Verbose mode is suppressed inside the sweep to
                  avoid flooding stdout; pass ``verbose=True`` to the engine
                  before construction to re-enable it.
        skip_mode: Propagation mode for the
                   :class:`~eovot.trackers.frame_skip.FrameSkipTracker`.
                   ``"repeat"`` (default) is cheaper; ``"linear"`` adds
                   velocity-based extrapolation.

    Example::

        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.analysis.joint_optimizer import JointDeploymentOptimizer

        engine   = BenchmarkEngine(verbose=False)
        dataset  = SyntheticDataset(num_sequences=5)
        optimizer = JointDeploymentOptimizer(engine)

        result = optimizer.optimize(
            MOSSETracker(), dataset, dataset_name="Synthetic",
            skip_rates=[1, 2, 4], scale_factors=[1.0, 0.5],
        )
        print(result.to_markdown_table())
    """

    def __init__(
        self,
        engine: "BenchmarkEngine",
        skip_mode: str = "repeat",
    ) -> None:
        self._engine = engine
        self._skip_mode = skip_mode

    def optimize(
        self,
        tracker: "BaseTracker",
        dataset: "BaseDataset",
        dataset_name: str = "unknown",
        skip_rates: Optional[List[int]] = None,
        scale_factors: Optional[List[float]] = None,
        max_sequences: Optional[int] = None,
    ) -> JointOptimizationResult:
        """Run the full ``(skip_rate × scale_factor)`` grid.

        The baseline configuration ``(skip=1, scale=1.0)`` is always included
        regardless of the provided lists, as it is needed to compute relative
        FPS gain and IoU degradation.

        Args:
            tracker:        The base tracker to wrap.
            dataset:        Dataset to evaluate on.
            dataset_name:   Human-readable label for reports.
            skip_rates:     List of skip rates to evaluate.
                Default: ``[1, 2, 3, 4]``.
            scale_factors:  List of scale factors to evaluate.
                Default: ``[1.0, 0.75, 0.5, 0.25]``.
            max_sequences:  Cap on the number of sequences per configuration.

        Returns:
            :class:`JointOptimizationResult` with Pareto front marked.
        """
        if skip_rates is None:
            skip_rates = [1, 2, 3, 4]
        if scale_factors is None:
            scale_factors = [1.0, 0.75, 0.5, 0.25]

        # Ensure baseline is always evaluated
        all_skips = sorted(set(skip_rates) | {1})
        all_scales = sorted(set(scale_factors) | {1.0}, reverse=True)

        # Suppress per-sequence output during the sweep
        orig_verbose = self._engine.verbose
        self._engine.verbose = False

        entries: List[JointEntry] = []
        total = len(all_skips) * len(all_scales)
        idx = 0

        try:
            for skip in all_skips:
                for scale in all_scales:
                    idx += 1
                    label = f"skip={skip} scale={scale:.2f}"
                    print(f"  [{idx:>2}/{total}] {label}  …", end="", flush=True)
                    bench_result = self._run_config(
                        tracker, dataset, dataset_name, skip, scale, max_sequences
                    )
                    entry = JointEntry(
                        skip_rate=skip,
                        scale_factor=scale,
                        mean_iou=bench_result.mean_iou,
                        mean_fps=bench_result.mean_fps,
                        peak_memory_mb=bench_result.peak_memory_mb,
                        success_auc=bench_result.mean_success_auc,
                    )
                    entries.append(entry)
                    print(f"  mIoU={entry.mean_iou:.3f}  FPS={entry.mean_fps:.1f}")
        finally:
            self._engine.verbose = orig_verbose

        # Compute relative metrics vs. baseline
        baseline = next(
            (e for e in entries if e.skip_rate == 1 and abs(e.scale_factor - 1.0) < 1e-6),
            None,
        )
        if baseline is not None:
            for e in entries:
                e.fps_gain = e.mean_fps / max(baseline.mean_fps, 1e-6)
                e.iou_degradation = baseline.mean_iou - e.mean_iou

        return JointOptimizationResult(
            entries=entries,
            tracker_name=tracker.name,
            dataset_name=dataset_name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_config(
        self,
        tracker: "BaseTracker",
        dataset: "BaseDataset",
        dataset_name: str,
        skip_rate: int,
        scale_factor: float,
        max_sequences: Optional[int],
    ) -> "BenchmarkResult":
        """Wrap *tracker* and run one configuration."""
        import copy

        # Deep-copy the tracker to get a fresh internal state for each run.
        t = copy.deepcopy(tracker)

        if skip_rate > 1:
            t = FrameSkipTracker(t, skip_rate=skip_rate, mode=self._skip_mode)
        if abs(scale_factor - 1.0) > 1e-6:
            t = ResolutionScaledTracker(t, scale_factor=scale_factor)

        return self._engine.run(t, dataset, dataset_name=dataset_name,
                                max_sequences=max_sequences)
