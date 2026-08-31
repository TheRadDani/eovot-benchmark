"""Hyperparameter sensitivity sweep for EOVOT trackers.

Runs a 1D or 2D grid sweep over any hyperparameter(s) of a registered
tracker, evaluating each configuration on a dataset and computing per-axis
sensitivity scores.  Enables researchers to:

- Find optimal hyperparameters for a given dataset
- Quantify which parameters matter most (high sensitivity = critical)
- Report robustness to mistuning (flat AUC curve = robust to detuning)
- Generate ablation tables for papers

Sensitivity is the normalised range of success-AUC across one parameter
axis (marginalised over the other axis in the 2D case)::

    sensitivity = (max_AUC - min_AUC) / mean_AUC

A value of 0.0 means the parameter has no effect; 1.0 means it can shift
performance by 100 % of the mean.

Typical usage::

    from eovot.analysis.hyperparam_sweep import HyperparamSweep, SweepAxis
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, num_frames=80, seed=42)
    sweep = HyperparamSweep(
        tracker_name="KCF",
        dataset=dataset,
        dataset_name="Synthetic",
        max_sequences=5,
        verbose=True,
    )
    result = sweep.run_1d(SweepAxis("learning_rate", [0.02, 0.05, 0.075, 0.125, 0.2]))
    print(result.to_markdown())
    print("Optimal:", result.optimal_config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..benchmark.engine import BenchmarkEngine
from ..datasets.base import BaseDataset
from ..trackers.registry import build_tracker


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
