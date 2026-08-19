"""Accuracy–Efficiency Pareto Sweep for edge deployment hyperparameter selection.

A central challenge in deploying tracking algorithms to edge hardware is
choosing the right trade-off between accuracy and computational cost.  Two
orthogonal knobs are available in EOVOT:

1. **Frame skipping** (``skip_rate ∈ {1, 2, 3, 4, …}``) — run the tracker
   every k-th frame and propagate the last prediction otherwise.
2. **Resolution scaling** (``scale_factor ∈ (0, 1]``) — downscale each frame
   before running the tracker, then map the prediction back to original
   coordinates.

:class:`ParetoSweep` exhaustively evaluates all combinations of these two
knobs for a given base tracker and dataset, then computes the Edge Efficiency
Score (EES) for each operating point.  The result is a 2-D grid from which
the Pareto-optimal operating points can be read off, giving researchers a
principled answer to: *"For this tracker on this device, which (skip_rate,
scale_factor) pair gives the best accuracy–efficiency trade-off?"*

The sweep also simulates results on registered edge device profiles using
:class:`~eovot.profiling.device_sim.DeviceSimulator`, enabling fleet-wide
deployment analysis from a single benchmark run.

Typical usage
-------------
::

    from eovot.experiment.pareto_sweep import ParetoSweep
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, num_frames=100)
    sweep = ParetoSweep(
        skip_rates=[1, 2, 3, 4],
        scale_factors=[1.0, 0.75, 0.5],
    )
    result = sweep.run(
        base_tracker_name="KCF",
        dataset=dataset,
        dataset_name="Synthetic",
    )
    print(result.to_markdown())
    result.save("results/pareto_sweep.csv")
    # Inspect the best operating point
    best = result.pareto_front[0]
    print(f"Best: skip={best.skip_rate}  scale={best.scale_factor:.2f}  "
          f"mIoU={best.mean_iou:.3f}  EES={best.ees:.3f}")

Script usage
------------
::

    python scripts/pareto_sweep.py \\
        --tracker KCF \\
        --synthetic --synthetic-sequences 5 \\
        --skip-rates 1 2 3 4 \\
        --scale-factors 1.0 0.75 0.5 \\
        --output results/pareto_sweep.csv
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..metrics.efficiency import EfficiencyMetricsEngine
from ..trackers.frame_skip import FrameSkipTracker
from ..trackers.scale import ResolutionScaledTracker
from ..trackers.registry import build_tracker


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    """A single evaluated (skip_rate, scale_factor) operating point.

    Attributes:
        skip_rate: Frame-skipping factor (1 = no skipping).
        scale_factor: Linear resolution scaling factor (1.0 = full resolution).
        mean_iou: Mean IoU across all sequences in the sweep dataset.
        mean_fps: Mean throughput in frames per second.
        peak_memory_mb: Peak RSS memory footprint in megabytes.
        ees: Edge Efficiency Score computed with :class:`EfficiencyMetricsEngine`.
        success_auc: Mean success-curve AUC across all sequences (None if not computed).
        on_pareto_front: True when this point is Pareto-optimal in (mIoU, EES) space.
        elapsed_s: Wall-clock time for this operating point evaluation.
    """

    skip_rate: int
    scale_factor: float
    mean_iou: float
    mean_fps: float
    peak_memory_mb: float
    ees: float
    success_auc: Optional[float] = None
    on_pareto_front: bool = False
    elapsed_s: float = 0.0

    @property
    def label(self) -> str:
        """Short human-readable label for plots and tables."""
        return f"skip={self.skip_rate} scale={self.scale_factor:.2f}"

    def __str__(self) -> str:
        pareto = " [Pareto]" if self.on_pareto_front else ""
        return (
            f"SweepPoint({self.label}  "
            f"mIoU={self.mean_iou:.4f}  FPS={self.mean_fps:.1f}  "
            f"EES={self.ees:.4f}  mem={self.peak_memory_mb:.1f} MB{pareto})"
        )


@dataclass
class ParetoSweepResult:
    """Complete results for a :class:`ParetoSweep` run.

    Attributes:
        tracker_name: Base tracker name (e.g. ``"KCF"``).
        dataset_name: Dataset label used in the sweep.
        memory_budget_mb: Memory budget used for EES computation.
        points: All evaluated (skip_rate, scale_factor) operating points,
            sorted by EES descending.
        pareto_front: Subset of ``points`` that are Pareto-optimal.
        total_elapsed_s: Total wall-clock time for the whole sweep.
    """

    tracker_name: str
    dataset_name: str
    memory_budget_mb: float
    points: List[SweepPoint] = field(default_factory=list)
    pareto_front: List[SweepPoint] = field(default_factory=list)
    total_elapsed_s: float = 0.0

    def best_point(self) -> Optional[SweepPoint]:
        """Return the Pareto-optimal point with the highest EES, or None."""
        return self.pareto_front[0] if self.pareto_front else None

    def to_markdown(self) -> str:
        """Format the sweep grid as a Markdown table sorted by EES descending.

        Returns:
            Multi-line Markdown string ready to embed in README or reports.
        """
        lines = [
            "| Skip | Scale | mIoU | AUC | FPS | Mem (MB) | EES | Pareto |",
            "|-----:|------:|-----:|----:|----:|---------:|----:|:------:|",
        ]
        for p in self.points:
            auc_str = f"{p.success_auc:.4f}" if p.success_auc is not None else "—"
            pareto = "✓" if p.on_pareto_front else ""
            lines.append(
                f"| {p.skip_rate} | {p.scale_factor:.2f} "
                f"| {p.mean_iou:.4f} | {auc_str} "
                f"| {p.mean_fps:.1f} | {p.peak_memory_mb:.1f} "
                f"| {p.ees:.4f} | {pareto} |"
            )
        header = (
            f"## Pareto Sweep: {self.tracker_name} on {self.dataset_name}\n"
            f"Memory budget: {self.memory_budget_mb} MB  |  "
            f"Total time: {self.total_elapsed_s:.1f}s  |  "
            f"Pareto-optimal points: {len(self.pareto_front)}\n"
        )
        return header + "\n".join(lines)

    def save(self, path: Union[str, Path]) -> Path:
        """Save the sweep results to a CSV file.

        Args:
            path: Destination path.  ``.csv`` extension appended if absent.

        Returns:
            The resolved path that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".csv")
        p.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "tracker", "dataset", "skip_rate", "scale_factor",
            "mean_iou", "success_auc", "mean_fps", "peak_memory_mb",
            "ees", "on_pareto_front", "elapsed_s",
        ]
        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for pt in self.points:
                writer.writerow({
                    "tracker": self.tracker_name,
                    "dataset": self.dataset_name,
                    "skip_rate": pt.skip_rate,
                    "scale_factor": pt.scale_factor,
                    "mean_iou": round(pt.mean_iou, 6),
                    "success_auc": round(pt.success_auc, 6) if pt.success_auc is not None else "",
                    "mean_fps": round(pt.mean_fps, 3),
                    "peak_memory_mb": round(pt.peak_memory_mb, 3),
                    "ees": round(pt.ees, 6),
                    "on_pareto_front": int(pt.on_pareto_front),
                    "elapsed_s": round(pt.elapsed_s, 3),
                })
        return p


# ---------------------------------------------------------------------------
# Sweep engine
# ---------------------------------------------------------------------------

class ParetoSweep:
    """Evaluate a tracker over a grid of (skip_rate, scale_factor) combinations.

    Runs the :class:`~eovot.benchmark.engine.BenchmarkEngine` for each
    combination, collects the key efficiency metrics, identifies the
    Pareto-optimal operating points, and returns a :class:`ParetoSweepResult`
    that can be saved to CSV or formatted as Markdown.

    Args:
        skip_rates: List of frame-skip factors to sweep.  ``1`` means no
            skipping (full rate).  Default: ``[1, 2, 3, 4]``.
        scale_factors: List of linear resolution scale factors to sweep.
            All values must be in ``(0, 1]``.  Default: ``[1.0, 0.75, 0.5]``.
        memory_budget_mb: Memory budget for EES computation.  Trackers within
            this limit receive full memory credit.  Default: ``512.0``.
        tdp_watts: If provided, enables energy profiling with this TDP value.
            Passed to :class:`~eovot.benchmark.engine.BenchmarkEngine`.
        verbose: Print per-operating-point progress.  Default: ``True``.

    Raises:
        ValueError: If ``skip_rates`` or ``scale_factors`` is empty, or if
            any scale factor is outside ``(0, 1]``.
    """

    def __init__(
        self,
        skip_rates: Optional[List[int]] = None,
        scale_factors: Optional[List[float]] = None,
        memory_budget_mb: float = 512.0,
        tdp_watts: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        self.skip_rates = skip_rates if skip_rates is not None else [1, 2, 3, 4]
        self.scale_factors = scale_factors if scale_factors is not None else [1.0, 0.75, 0.5]
        self.memory_budget_mb = memory_budget_mb
        self.tdp_watts = tdp_watts
        self.verbose = verbose

        if not self.skip_rates:
            raise ValueError("skip_rates must be non-empty.")
        if not self.scale_factors:
            raise ValueError("scale_factors must be non-empty.")
        bad = [s for s in self.scale_factors if not (0.0 < s <= 1.0)]
        if bad:
            raise ValueError(f"All scale_factors must be in (0, 1]; got {bad}.")

        self._ees_engine = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)

    def run(
        self,
        base_tracker_name: str,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        base_tracker_params: Optional[Dict] = None,
        max_sequences: Optional[int] = None,
    ) -> ParetoSweepResult:
        """Execute the full sweep and return a :class:`ParetoSweepResult`.

        Each (skip_rate, scale_factor) combination is evaluated independently.
        The order is skip_rate × scale_factor so that the no-skip full-resolution
        baseline (skip=1, scale=1.0) is always evaluated first.

        Args:
            base_tracker_name: Registry key for the base tracker (e.g. ``"KCF"``).
            dataset: Dataset to evaluate on.  All sequences are used unless
                ``max_sequences`` is given.
            dataset_name: Human-readable dataset label for reports.
            base_tracker_params: Keyword arguments forwarded to the tracker's
                constructor.  Default: ``{}`` (use tracker defaults).
            max_sequences: Evaluate only the first N sequences.  ``None``
                means use all.

        Returns:
            :class:`ParetoSweepResult` with all points evaluated, Pareto
            front identified, and points sorted by EES descending.
        """
        params = base_tracker_params or {}
        total_combinations = len(self.skip_rates) * len(self.scale_factors)
        sweep_start = time.perf_counter()

        result = ParetoSweepResult(
            tracker_name=base_tracker_name,
            dataset_name=dataset_name,
            memory_budget_mb=self.memory_budget_mb,
        )

        if self.verbose:
            print(
                f"\nPareto sweep: {base_tracker_name} on {dataset_name}  "
                f"({total_combinations} combinations)"
            )
            print(
                f"  skip_rates={self.skip_rates}  "
                f"scale_factors={self.scale_factors}"
            )
            print("-" * 64)

        combo_idx = 0
        for skip_rate in self.skip_rates:
            for scale_factor in self.scale_factors:
                combo_idx += 1
                t0 = time.perf_counter()

                tracker = self._build_tracker(
                    base_tracker_name, params, skip_rate, scale_factor
                )

                engine = BenchmarkEngine(verbose=False, tdp_watts=self.tdp_watts)
                bench: BenchmarkResult = engine.run(
                    tracker=tracker,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    max_sequences=max_sequences,
                )

                elapsed = time.perf_counter() - t0
                ees = self._ees_engine.edge_efficiency_score(
                    mean_iou=bench.mean_iou,
                    fps=bench.mean_fps,
                    peak_memory_mb=bench.peak_memory_mb,
                )

                point = SweepPoint(
                    skip_rate=skip_rate,
                    scale_factor=scale_factor,
                    mean_iou=bench.mean_iou,
                    mean_fps=bench.mean_fps,
                    peak_memory_mb=bench.peak_memory_mb,
                    ees=ees,
                    success_auc=bench.mean_success_auc,
                    elapsed_s=elapsed,
                )
                result.points.append(point)

                if self.verbose:
                    print(
                        f"  [{combo_idx:>2}/{total_combinations}] "
                        f"skip={skip_rate}  scale={scale_factor:.2f}  "
                        f"mIoU={bench.mean_iou:.3f}  "
                        f"FPS={bench.mean_fps:.1f}  "
                        f"EES={ees:.3f}  "
                        f"({elapsed:.1f}s)"
                    )

        # Identify Pareto front and sort by EES.
        self._mark_pareto_front(result.points)
        result.points.sort(key=lambda p: p.ees, reverse=True)
        result.pareto_front = [p for p in result.points if p.on_pareto_front]
        result.total_elapsed_s = time.perf_counter() - sweep_start

        if self.verbose:
            print("-" * 64)
            print(
                f"Sweep complete in {result.total_elapsed_s:.1f}s. "
                f"Pareto-optimal points: {len(result.pareto_front)}"
            )
            best = result.best_point()
            if best is not None:
                print(
                    f"Best operating point: {best.label}  "
                    f"mIoU={best.mean_iou:.4f}  EES={best.ees:.4f}"
                )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tracker(
        name: str,
        params: Dict,
        skip_rate: int,
        scale_factor: float,
    ):
        """Construct the possibly-wrapped tracker for one operating point."""
        base = build_tracker(name, **params)
        if scale_factor < 1.0:
            base = ResolutionScaledTracker(base, scale_factor=scale_factor)
        if skip_rate > 1:
            base = FrameSkipTracker(base, skip_rate=skip_rate, mode="linear")
        return base

    @staticmethod
    def _mark_pareto_front(points: List[SweepPoint]) -> None:
        """Set ``on_pareto_front`` in-place for each point.

        A point P is Pareto-optimal if no other point Q satisfies
        ``Q.mean_iou >= P.mean_iou`` AND ``Q.ees >= P.ees`` with at least
        one strict inequality.
        """
        for i, p in enumerate(points):
            dominated = False
            for j, q in enumerate(points):
                if i == j:
                    continue
                if (
                    q.mean_iou >= p.mean_iou
                    and q.ees >= p.ees
                    and (q.mean_iou > p.mean_iou or q.ees > p.ees)
                ):
                    dominated = True
                    break
            p.on_pareto_front = not dominated
