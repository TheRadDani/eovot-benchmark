"""Tracker hyperparameter sweep engine for EOVOT.

Provides two complementary APIs for grid-searching tracker constructor
parameters:

:class:`HyperparamSweep` (``run_1d`` / ``run_2d``)
    Axis-oriented API for 1D and 2D sweeps with per-axis sensitivity scores
    (normalised AUC range).  Useful for publication-grade ablation tables and
    understanding which parameters matter most.

:class:`HyperparamSweeper` (``sweep``)
    Cartesian-product API for arbitrary-arity grids with Pareto-front
    analysis (accuracy–FPS dominance) and FPS-budget queries
    (``best_at_fps_budget``).  Useful for optimising a tracker for a target
    hardware constraint.

Both use :class:`~eovot.benchmark.engine.BenchmarkEngine` under the hood
and work with any tracker registered in
:data:`~eovot.trackers.registry.TRACKER_REGISTRY`.

Typical usage::

    from eovot.analysis.hyperparam_sweep import HyperparamSweep, SweepAxis
    from eovot.analysis.hyperparam_sweep import HyperparamSweeper
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, num_frames=80, seed=42)

    # --- Axis-oriented 1D sweep with sensitivity -------------------------
    sweep = HyperparamSweep("KCF", dataset, "Synthetic", verbose=True)
    result = sweep.run_1d(SweepAxis("learning_rate", [0.02, 0.05, 0.125, 0.2]))
    print(result.to_markdown())
    print("Optimal:", result.optimal_config)
    print("Sensitivity:", result.sensitivity)

    # --- Pareto-front grid sweep with FPS budget -------------------------
    sweeper = HyperparamSweeper("KCF", dataset, "Synthetic", verbose=True)
    report = sweeper.sweep({"learning_rate": [0.05, 0.10, 0.15]})
    print(report.to_markdown())
    best = report.best_at_fps_budget(fps_min=200.0)
    print("Best config at 200 FPS:", best)
"""

from __future__ import annotations

import csv
import io
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..trackers.registry import build_tracker


# ===========================================================================
# Axis-oriented API: HyperparamSweep, SweepAxis, SweepResult, SweepPoint
# ===========================================================================

@dataclass
class SweepAxis:
    """One parameter axis in a hyperparameter sweep.

    Args:
        param_name: Keyword argument name accepted by the tracker constructor.
        values: Ordered list of values to sweep over.
        display_name: Optional human-readable label used in reports and tables.
    """

    param_name: str
    values: List[Any]
    display_name: Optional[str] = None

    def label(self) -> str:
        """Return the display name, falling back to param_name."""
        return self.display_name or self.param_name


@dataclass
class SweepPoint:
    """Evaluation metrics for one hyperparameter configuration.

    Attributes:
        config: Dict of ``{param_name: value}`` for this evaluation.
        mean_iou: Mean IoU across all evaluated frames.
        success_auc: Area under the success curve.
        precision_auc: Normalised precision-curve AUC.
        fps: Mean frames per second.
        peak_memory_mb: Peak RSS memory footprint in megabytes.
    """

    config: Dict[str, Any]
    mean_iou: float
    success_auc: float
    precision_auc: float
    fps: float
    peak_memory_mb: float


@dataclass
class SweepResult:
    """Complete result of a 1D or 2D hyperparameter sweep.

    Attributes:
        tracker_name: Registry name of the swept tracker.
        axes: The :class:`SweepAxis` objects that define the parameter grid.
        points: One :class:`SweepPoint` per evaluated configuration.
        dataset_name: Human-readable label for the evaluation dataset.
    """

    tracker_name: str
    axes: List[SweepAxis]
    points: List[SweepPoint]
    dataset_name: str = "unknown"

    @property
    def optimal_config(self) -> Dict[str, Any]:
        """Hyperparameter configuration with the highest success AUC."""
        if not self.points:
            return {}
        return max(self.points, key=lambda p: p.success_auc).config

    @property
    def sensitivity(self) -> Dict[str, float]:
        """Per-axis sensitivity: normalised AUC range (marginalised over other axes).

        Returns a dict ``{param_name: sensitivity_score}`` where a score of
        0.0 means the parameter has no effect and 1.0 means it can cause a
        100 % swing relative to the mean performance.
        """
        if not self.points:
            return {ax.param_name: 0.0 for ax in self.axes}

        mean_auc = float(np.mean([p.success_auc for p in self.points]))
        if mean_auc == 0.0:
            return {ax.param_name: 0.0 for ax in self.axes}

        result: Dict[str, float] = {}
        for ax in self.axes:
            by_val: Dict[Any, List[float]] = {}
            for pt in self.points:
                v = pt.config[ax.param_name]
                by_val.setdefault(v, []).append(pt.success_auc)
            marginal = [float(np.mean(aucs)) for aucs in by_val.values()]
            if len(marginal) < 2:
                result[ax.param_name] = 0.0
            else:
                result[ax.param_name] = round((max(marginal) - min(marginal)) / mean_auc, 4)
        return result

    def to_markdown(self) -> str:
        """Format sweep results as a human-readable Markdown report."""
        lines: List[str] = [
            f"## Hyperparameter Sweep: `{self.tracker_name}` on {self.dataset_name}\n",
        ]

        # Sensitivity summary
        sens = self.sensitivity
        lines.append("### Parameter Sensitivity (normalised AUC range)\n")
        for param, s in sorted(sens.items(), key=lambda kv: -kv[1]):
            bar = "█" * max(1, int(s * 20))
            lines.append(f"- `{param}`: **{s:.4f}**  {bar}")

        opt = self.optimal_config
        opt_str = ", ".join(f"{k}={v}" for k, v in opt.items())
        lines.append(f"\n**Optimal config**: `{opt_str}`\n")

        # Results table
        if len(self.axes) == 1:
            ax = self.axes[0]
            lines += [
                f"### 1D Sweep: `{ax.label()}`\n",
                f"| {ax.label()} | mIoU | Success AUC | Precision AUC | FPS | Mem (MB) |",
                "|" + "---------|" * 6,
            ]
            for pt in sorted(self.points, key=lambda p: str(p.config[ax.param_name])):
                v = pt.config[ax.param_name]
                lines.append(
                    f"| {v} | {pt.mean_iou:.4f} | {pt.success_auc:.4f} "
                    f"| {pt.precision_auc:.4f} | {pt.fps:.1f} | {pt.peak_memory_mb:.1f} |"
                )
        elif len(self.axes) == 2:
            ax0, ax1 = self.axes
            lines += [
                f"### 2D Sweep: `{ax0.label()}` × `{ax1.label()}` (Success AUC)\n",
                f"| {ax0.label()} \\ {ax1.label()} | "
                + " | ".join(str(v) for v in ax1.values)
                + " |",
                "|" + "---------|" * (len(ax1.values) + 1),
            ]
            grid: Dict[Tuple, float] = {
                (pt.config[ax0.param_name], pt.config[ax1.param_name]): pt.success_auc
                for pt in self.points
            }
            for v0 in ax0.values:
                row = " | ".join(
                    f"{grid.get((v0, v1), float('nan')):.4f}" for v1 in ax1.values
                )
                lines.append(f"| {v0} | {row} |")

        return "\n".join(lines)


class HyperparamSweep:
    """Grid-sweep hyperparameters of any registered EOVOT tracker.

    Each configuration is evaluated on the full dataset (or a capped subset)
    using :class:`~eovot.benchmark.engine.BenchmarkEngine`.  For exploratory
    sweeps, set ``max_sequences=5`` to get fast results; for publication-grade
    sensitivity analysis, use the full dataset.

    Args:
        tracker_name: Registry key (e.g. ``"KCF"``).  Must exist in
            :data:`~eovot.trackers.registry.TRACKER_REGISTRY`.
        dataset: Dataset to evaluate each configuration on.
        dataset_name: Human-readable label used in reports.
        max_sequences: Cap on sequences per configuration.  ``None`` uses all.
        fixed_params: Hyperparameters held constant during the sweep.
        verbose: Print per-configuration progress.  Default: ``False``.
    """

    def __init__(
        self,
        tracker_name: str,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> None:
        self.tracker_name = tracker_name
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.max_sequences = max_sequences
        self.fixed_params: Dict[str, Any] = fixed_params or {}
        self.verbose = verbose
        self._engine = BenchmarkEngine(verbose=False)

    def _evaluate(self, params: Dict[str, Any]) -> SweepPoint:
        """Evaluate one parameter configuration and return a SweepPoint."""
        merged = {**self.fixed_params, **params}
        tracker = build_tracker(self.tracker_name, **merged)
        result = self._engine.run(
            tracker=tracker,
            dataset=self.dataset,
            dataset_name=self.dataset_name,
            max_sequences=self.max_sequences,
        )
        return SweepPoint(
            config=params,
            mean_iou=result.mean_iou,
            success_auc=result.mean_success_auc or 0.0,
            precision_auc=result.mean_precision_auc or 0.0,
            fps=result.mean_fps,
            peak_memory_mb=result.peak_memory_mb,
        )

    def run_1d(self, axis: SweepAxis) -> SweepResult:
        """Evaluate every value in a single parameter axis.

        Args:
            axis: :class:`SweepAxis` defining the parameter name and candidate
                values.

        Returns:
            :class:`SweepResult` with one :class:`SweepPoint` per value,
            plus sensitivity scores and an optimal-config lookup.
        """
        if not axis.values:
            raise ValueError(f"SweepAxis '{axis.param_name}' has no values to sweep.")

        points: List[SweepPoint] = []
        total = len(axis.values)
        for idx, val in enumerate(axis.values):
            if self.verbose:
                print(
                    f"  [{idx + 1}/{total}] {axis.param_name}={val} ...",
                    end=" ",
                    flush=True,
                )
            pt = self._evaluate({axis.param_name: val})
            points.append(pt)
            if self.verbose:
                print(f"AUC={pt.success_auc:.4f}  FPS={pt.fps:.1f}")

        return SweepResult(
            tracker_name=self.tracker_name,
            axes=[axis],
            points=points,
            dataset_name=self.dataset_name,
        )

    def run_2d(self, axis0: SweepAxis, axis1: SweepAxis) -> SweepResult:
        """Evaluate all combinations of two parameter axes (full grid).

        Args:
            axis0: Outer-loop axis.
            axis1: Inner-loop axis.

        Returns:
            :class:`SweepResult` with one :class:`SweepPoint` per grid cell
            ``(len(axis0.values) × len(axis1.values))``.
        """
        if not axis0.values or not axis1.values:
            raise ValueError("Both SweepAxis objects must have non-empty value lists.")

        points: List[SweepPoint] = []
        total = len(axis0.values) * len(axis1.values)
        count = 0
        for v0 in axis0.values:
            for v1 in axis1.values:
                count += 1
                params = {axis0.param_name: v0, axis1.param_name: v1}
                if self.verbose:
                    print(
                        f"  [{count}/{total}] "
                        f"{axis0.param_name}={v0}  {axis1.param_name}={v1} ...",
                        end=" ",
                        flush=True,
                    )
                pt = self._evaluate(params)
                points.append(pt)
                if self.verbose:
                    print(f"AUC={pt.success_auc:.4f}")

        return SweepResult(
            tracker_name=self.tracker_name,
            axes=[axis0, axis1],
            points=points,
            dataset_name=self.dataset_name,
        )


# ===========================================================================
# Pareto-front grid API: HyperparamSweeper, SweepEntry, SweepReport
# ===========================================================================

@dataclass
class SweepEntry:
    """Result for one parameter configuration (Pareto-front API).

    Attributes:
        params:         The exact constructor kwargs used for this config.
        result:         Full benchmark result object.
        mean_iou:       Mean IoU across all sequences.
        success_auc:    Success-curve AUC (falls back to mean_iou when absent).
        mean_fps:       Mean frames-per-second across all sequences.
        peak_memory_mb: Peak RSS memory across all sequences (MB).
        is_pareto:      True if no other config dominates this one on both
                        success_auc and mean_fps simultaneously.
    """

    params: Dict[str, Any]
    result: BenchmarkResult
    mean_iou: float
    success_auc: float
    mean_fps: float
    peak_memory_mb: float
    is_pareto: bool = False

    def params_label(self) -> str:
        """Short human-readable label, e.g. ``lr=0.125``."""
        parts = [f"{k}={v}" for k, v in sorted(self.params.items())]
        return " ".join(parts) if parts else "(default)"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "params": self.params,
            "mean_iou": round(self.mean_iou, 4),
            "success_auc": round(self.success_auc, 4),
            "mean_fps": round(self.mean_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "is_pareto": self.is_pareto,
        }
        return d


@dataclass
class SweepReport:
    """Results of a complete Pareto-front hyperparameter sweep.

    Attributes:
        tracker_name: Base tracker class name (e.g. ``"KCF"``).
        dataset_name: Human-readable dataset identifier.
        param_names:  Sorted list of swept parameter names.
        entries:      All sweep entries, sorted by ``success_auc`` descending.
    """

    tracker_name: str
    dataset_name: str
    param_names: List[str]
    entries: List[SweepEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def best_config(self) -> Optional[SweepEntry]:
        """Return the entry with the highest success AUC (or mIoU fallback)."""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.success_auc)

    def best_at_fps_budget(self, fps_min: float) -> Optional[SweepEntry]:
        """Return the highest-accuracy config that meets a minimum FPS floor.

        Args:
            fps_min: Minimum acceptable throughput in frames per second.

        Returns:
            :class:`SweepEntry` with highest ``success_auc`` among configs
            where ``mean_fps >= fps_min``, or ``None`` if no config qualifies.
        """
        candidates = [e for e in self.entries if e.mean_fps >= fps_min]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.success_auc)

    def pareto_front(self) -> List[SweepEntry]:
        """Return entries on the accuracy–speed Pareto front."""
        return [e for e in self.entries if e.is_pareto]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render a Markdown table sorted by success AUC descending.

        Pareto-optimal rows are annotated with ``✓`` so readers can quickly
        spot the efficiency frontier.
        """
        if not self.entries:
            return f"# Sweep: {self.tracker_name} on {self.dataset_name}\n\nNo results.\n"

        param_cols = self.param_names or ["params"]
        header_params = " | ".join(f"{p}" for p in param_cols)
        sep_params = " | ".join("-" * max(len(p), 5) for p in param_cols)

        lines = [
            f"# Hyperparameter Sweep: {self.tracker_name} on {self.dataset_name}\n",
            f"| Rank | {header_params} | Success AUC | mIoU | FPS | Mem (MB) | Pareto |",
            f"|------|{sep_params}|------------:|-----:|----:|---------:|:------:|",
        ]

        sorted_entries = sorted(self.entries, key=lambda e: e.success_auc, reverse=True)
        for rank, entry in enumerate(sorted_entries, start=1):
            if self.param_names:
                param_vals = " | ".join(
                    str(entry.params.get(p, "—")) for p in self.param_names
                )
            else:
                param_vals = entry.params_label()
            pareto_mark = "✓" if entry.is_pareto else ""
            lines.append(
                f"| {rank} | {param_vals} "
                f"| {entry.success_auc:.4f} "
                f"| {entry.mean_iou:.4f} "
                f"| {entry.mean_fps:.1f} "
                f"| {entry.peak_memory_mb:.1f} "
                f"| {pareto_mark} |"
            )

        lines.append("")
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Return sweep results as a CSV string."""
        buf = io.StringIO()
        fieldnames = self.param_names + [
            "success_auc", "mean_iou", "mean_fps", "peak_memory_mb", "is_pareto"
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for entry in sorted(self.entries, key=lambda e: e.success_auc, reverse=True):
            row: Dict[str, Any] = {p: entry.params.get(p, "") for p in self.param_names}
            row["success_auc"] = round(entry.success_auc, 4)
            row["mean_iou"] = round(entry.mean_iou, 4)
            row["mean_fps"] = round(entry.mean_fps, 2)
            row["peak_memory_mb"] = round(entry.peak_memory_mb, 2)
            row["is_pareto"] = entry.is_pareto
            writer.writerow(row)
        return buf.getvalue()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire report to a plain dict (JSON-compatible)."""
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "param_names": self.param_names,
            "entries": [e.to_dict() for e in self.entries],
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Save the report as a JSON file.

        Args:
            path: Destination file path. ``.json`` appended if no extension.

        Returns:
            Resolved path that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return p


class HyperparamSweeper:
    """Grid-search tracker hyperparameters, rank by Pareto-front analysis.

    Evaluates all Cartesian-product combinations of *param_grid* and marks
    Pareto-optimal configs (no other config is simultaneously more accurate
    AND faster).  Provides an FPS-budget query to find the best config that
    satisfies a given throughput constraint.

    Args:
        tracker_name: Registry name of the tracker to sweep (e.g. ``"KCF"``).
        dataset:      Dataset to evaluate on.
        dataset_name: Human-readable label for reports (defaults to class name).
        verbose:      Print per-config benchmark progress.
        tdp_watts:    If set, enables energy profiling (passed to
                      :class:`~eovot.benchmark.engine.BenchmarkEngine`).

    Example::

        sweeper = HyperparamSweeper("KCF", dataset, "OTB100")
        report = sweeper.sweep({"learning_rate": [0.05, 0.1, 0.125, 0.15]})
        print(report.to_markdown())
        best = report.best_at_fps_budget(fps_min=300)
    """

    def __init__(
        self,
        tracker_name: str,
        dataset: BaseDataset,
        dataset_name: str = "",
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
    ) -> None:
        self.tracker_name = tracker_name
        self.dataset = dataset
        self.dataset_name = dataset_name or type(dataset).__name__
        self.verbose = verbose
        self._engine = BenchmarkEngine(verbose=verbose, tdp_watts=tdp_watts)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def sweep(
        self,
        param_grid: Dict[str, Sequence[Any]],
        base_params: Optional[Dict[str, Any]] = None,
        max_sequences: Optional[int] = None,
    ) -> SweepReport:
        """Run the grid search and return a :class:`SweepReport`.

        All combinations of *param_grid* values are evaluated (full Cartesian
        product).  For a 1-D sweep (one parameter, many values) the product
        is a simple loop.

        Args:
            param_grid:     Mapping from parameter name to list of candidate
                            values.  All combinations are tried.
            base_params:    Fixed parameters always passed to the tracker
                            constructor.  Values in *param_grid* override
                            these for each trial.
            max_sequences:  Limit evaluation to this many sequences (useful
                            for fast development sweeps).

        Returns:
            :class:`SweepReport` with all entries and Pareto flags set.
        """
        base = dict(base_params or {})
        param_names = sorted(param_grid.keys())
        value_lists = [param_grid[p] for p in param_names]
        combos: List[Tuple] = list(itertools.product(*value_lists))

        n = len(combos)
        if self.verbose:
            print(
                f"\n{'=' * 60}\n"
                f"  HyperparamSweeper: {self.tracker_name} — {n} configs\n"
                f"  Dataset: {self.dataset_name}\n"
                f"{'=' * 60}"
            )

        entries: List[SweepEntry] = []
        for idx, combo in enumerate(combos):
            trial_params = {**base, **dict(zip(param_names, combo))}

            if self.verbose:
                label = ", ".join(f"{k}={v}" for k, v in zip(param_names, combo))
                print(f"\n[{idx + 1}/{n}] {label}")

            try:
                tracker = build_tracker(self.tracker_name, **trial_params)
            except Exception as exc:
                if self.verbose:
                    print(f"  SKIP — build_tracker failed: {exc}")
                continue

            try:
                br = self._engine.run(
                    tracker=tracker,
                    dataset=self.dataset,
                    dataset_name=self.dataset_name,
                    max_sequences=max_sequences,
                )
            except Exception as exc:
                if self.verbose:
                    print(f"  SKIP — benchmark failed: {exc}")
                continue

            sauc = br.mean_success_auc
            entry = SweepEntry(
                params=dict(zip(param_names, combo)),
                result=br,
                mean_iou=br.mean_iou,
                success_auc=sauc if sauc is not None else br.mean_iou,
                mean_fps=br.mean_fps,
                peak_memory_mb=br.peak_memory_mb,
            )
            entries.append(entry)

        _mark_pareto(entries)

        report = SweepReport(
            tracker_name=self.tracker_name,
            dataset_name=self.dataset_name,
            param_names=param_names,
            entries=sorted(entries, key=lambda e: e.success_auc, reverse=True),
        )

        if self.verbose:
            print(f"\n{'=' * 60}")
            best = report.best_config()
            if best:
                print(f"  Best config: {best.params_label()}")
                print(f"  success_auc={best.success_auc:.4f}  FPS={best.mean_fps:.1f}")
            print(f"  Pareto-optimal configs: {len(report.pareto_front())}/{n}")
            print(f"{'=' * 60}\n")

        return report


# ---------------------------------------------------------------------------
# Pareto helpers (used by HyperparamSweeper)
# ---------------------------------------------------------------------------

def _dominates(a: SweepEntry, b: SweepEntry) -> bool:
    """Return True if *a* weakly dominates *b* on (success_auc, fps)."""
    return a.success_auc >= b.success_auc and a.mean_fps >= b.mean_fps and (
        a.success_auc > b.success_auc or a.mean_fps > b.mean_fps
    )


def _mark_pareto(entries: List[SweepEntry]) -> None:
    """Set ``is_pareto=True`` on entries that are not dominated by any other."""
    for i, candidate in enumerate(entries):
        dominated = any(
            _dominates(other, candidate)
            for j, other in enumerate(entries)
            if i != j
        )
        candidate.is_pareto = not dominated
