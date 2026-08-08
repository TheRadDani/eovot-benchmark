"""Tracker hyperparameter sensitivity analysis for EOVOT.

Systematically evaluates how a tracker's accuracy, latency, and memory usage
change as its configuration parameters vary — enabling data-driven
hyperparameter selection tuned for specific edge-deployment constraints.

Motivation
----------
Every tracker in EOVOT has hyperparameters that encode implicit accuracy–speed
trade-offs:

* **MOSSE** ``learning_rate`` — high values adapt faster but produce noisier
  predictions; low values are stable but lag behind fast motion.
* **KCF** ``lambda_`` — stronger regularisation reduces FPS-hungry model updates
  at the cost of accuracy on complex scenes.
* **Kalman** ``process_noise`` — controls how quickly the kinematic state
  accepts acceleration, directly affecting prediction lag vs. noise.

These parameters are typically set by copying defaults from published papers,
which were tuned for desktop hardware.  On constrained edge devices the
optimal trade-off is different: a setting that wastes 5 ms per frame on a
laptop wastes 50 ms on a Raspberry Pi 4.

``ParameterSweep`` runs a tracker with every combination of parameter values
in a user-supplied grid, measures all EOVOT metrics for each combination, and
returns a ranked ``SweepResult`` with the Pareto-optimal configurations.

Typical usage::

    from eovot.analysis.parameter_sweep import ParameterSweep
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=10, num_frames=100, motion="linear")

    sweep = ParameterSweep(tracker_name="MOSSE", dataset=dataset)
    result = sweep.run(
        param_grid={"learning_rate": [0.05, 0.125, 0.25, 0.5]},
        dataset_name="Synthetic",
    )

    print(result.to_markdown())

    # Find the Pareto-optimal configs (best accuracy per FPS budget)
    for cfg in result.pareto_configs(budget_fps=30.0):
        print(cfg)
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SweepEntry:
    """Results for one parameter configuration in a sweep.

    Attributes:
        params:       The parameter dict used to construct the tracker.
        tracker_name: Tracker identifier (base name from the registry).
        dataset_name: Dataset label used in reports.
        mean_iou:     Mean IoU across all evaluated frames.
        success_auc:  Area under the success curve (IoU thresholds 0 → 1).
        precision_auc: Normalised AUC of the precision curve (0 → 50 px).
        mean_fps:     Mean frames per second across all sequences.
        mean_latency_ms: Mean per-frame latency in milliseconds.
        peak_memory_mb: Peak RSS memory footprint in megabytes.
        num_sequences: Number of sequences evaluated.
        wall_time_s:  Total wall-clock time for this sweep entry (seconds).
        on_pareto_front: ``True`` when no other entry in the sweep dominates
            this one in both accuracy (mean_iou) and speed (mean_fps).
    """

    params: Dict[str, Any]
    tracker_name: str
    dataset_name: str
    mean_iou: float
    success_auc: float
    precision_auc: float
    mean_fps: float
    mean_latency_ms: float
    peak_memory_mb: float
    num_sequences: int
    wall_time_s: float = 0.0
    on_pareto_front: bool = False

    def params_label(self) -> str:
        """Short string of param key=value pairs, suitable for table cells."""
        return ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))

    def __str__(self) -> str:
        pareto = " [Pareto]" if self.on_pareto_front else ""
        return (
            f"SweepEntry({self.params_label()}){pareto}  "
            f"mIoU={self.mean_iou:.4f}  AUC={self.success_auc:.4f}  "
            f"FPS={self.mean_fps:.1f}  lat={self.mean_latency_ms:.2f}ms  "
            f"mem={self.peak_memory_mb:.1f}MB"
        )


@dataclass
class SweepResult:
    """Full results of a parameter sweep for one tracker / dataset combination.

    Attributes:
        tracker_name: Tracker identifier.
        dataset_name: Dataset label.
        param_names:  Ordered list of swept parameter names.
        entries:      One :class:`SweepEntry` per parameter combination,
                      sorted by mean IoU descending.
    """

    tracker_name: str
    dataset_name: str
    param_names: List[str]
    entries: List[SweepEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def best_by_accuracy(self) -> Optional[SweepEntry]:
        """Return the entry with the highest mean IoU."""
        return max(self.entries, key=lambda e: e.mean_iou) if self.entries else None

    def best_by_fps(self) -> Optional[SweepEntry]:
        """Return the entry with the highest mean FPS."""
        return max(self.entries, key=lambda e: e.mean_fps) if self.entries else None

    def pareto_configs(self, budget_fps: Optional[float] = None) -> List[SweepEntry]:
        """Return Pareto-optimal entries in the (accuracy, FPS) space.

        Args:
            budget_fps: If given, filter to entries where ``mean_fps >= budget_fps``
                before computing the Pareto front.  Useful for selecting the
                most accurate configuration that still meets a real-time SLA.

        Returns:
            List of Pareto-optimal :class:`SweepEntry` objects, sorted by
            mean IoU descending.
        """
        candidates = (
            [e for e in self.entries if e.mean_fps >= budget_fps]
            if budget_fps is not None
            else self.entries
        )
        pareto = [e for e in candidates if e.on_pareto_front]
        return sorted(pareto, key=lambda e: e.mean_iou, reverse=True)

    def sensitivity(self, param: str) -> List[Tuple[Any, float, float]]:
        """Return the per-value accuracy and FPS for a single swept parameter.

        Only meaningful when the sweep has exactly one parameter (1-D sweep)
        or when all other parameters are held constant at their default values.

        Args:
            param: Parameter name to analyse.

        Returns:
            List of ``(value, mean_iou, mean_fps)`` tuples, sorted by value.
        """
        seen: Dict[Any, List[SweepEntry]] = {}
        for e in self.entries:
            val = e.params.get(param)
            seen.setdefault(val, []).append(e)
        rows = []
        for val, group in seen.items():
            mean_iou = float(np.mean([e.mean_iou for e in group]))
            mean_fps = float(np.mean([e.mean_fps for e in group]))
            rows.append((val, mean_iou, mean_fps))
        rows.sort(key=lambda r: (r[0] is None, r[0]))
        return rows

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Format the sweep results as a Markdown table, best-first.

        Returns:
            Multi-line Markdown string ready for embedding in reports.
        """
        if not self.entries:
            return "*(no results)*\n"

        header = (
            f"## Parameter Sweep: {self.tracker_name} on {self.dataset_name}\n\n"
            "| Rank | Parameters | mIoU | Success AUC | Precision AUC | FPS "
            "| Latency (ms) | Mem (MB) | Pareto |\n"
            "|------|-----------|-----:|------------:|--------------:|----:"
            "|-------------:|---------:|:------:|\n"
        )
        rows = []
        sorted_entries = sorted(self.entries, key=lambda e: e.mean_iou, reverse=True)
        for rank, e in enumerate(sorted_entries, start=1):
            pareto = "✓" if e.on_pareto_front else ""
            rows.append(
                f"| {rank} | `{e.params_label()}` "
                f"| {e.mean_iou:.4f} | {e.success_auc:.4f} | {e.precision_auc:.4f} "
                f"| {e.mean_fps:.1f} | {e.mean_latency_ms:.2f} "
                f"| {e.peak_memory_mb:.1f} | {pareto} |"
            )
        return header + "\n".join(rows) + "\n"

    def to_sensitivity_markdown(self, param: str) -> str:
        """Format a 1-D sensitivity table for one parameter.

        Args:
            param: Parameter name to report on.

        Returns:
            Markdown table showing how accuracy and FPS vary with *param*.
        """
        rows = self.sensitivity(param)
        if not rows:
            return f"*(no data for parameter '{param}')*\n"

        lines = [
            f"## Sensitivity: `{param}` — {self.tracker_name} on {self.dataset_name}\n",
            f"| {param} | Mean IoU | Mean FPS |",
            "|---------|----------:|----------:|",
        ]
        for val, iou, fps in rows:
            lines.append(f"| {val} | {iou:.4f} | {fps:.1f} |")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> Dict:
        """Serialise to a plain dict suitable for JSON export."""
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "param_names": self.param_names,
            "entries": [
                {
                    "params": e.params,
                    "mean_iou": round(e.mean_iou, 6),
                    "success_auc": round(e.success_auc, 6),
                    "precision_auc": round(e.precision_auc, 6),
                    "mean_fps": round(e.mean_fps, 3),
                    "mean_latency_ms": round(e.mean_latency_ms, 3),
                    "peak_memory_mb": round(e.peak_memory_mb, 2),
                    "num_sequences": e.num_sequences,
                    "wall_time_s": round(e.wall_time_s, 3),
                    "on_pareto_front": e.on_pareto_front,
                }
                for e in self.entries
            ],
        }


# ---------------------------------------------------------------------------
# Core sweep engine
# ---------------------------------------------------------------------------

class ParameterSweep:
    """Systematically evaluate a tracker across a grid of hyperparameter values.

    Args:
        tracker_name: Registry key of the tracker to sweep (e.g. ``"MOSSE"``).
        dataset:      :class:`~eovot.datasets.base.BaseDataset` instance to
                      evaluate on.  A :class:`~eovot.datasets.synthetic.SyntheticDataset`
                      with 10–20 sequences is recommended for fast sweeps.
        verbose:      Print per-configuration progress to stdout.
                      Default: ``True``.

    Example::

        from eovot.analysis.parameter_sweep import ParameterSweep
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=10, num_frames=80, seed=42)
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=True)

        result = sweep.run(
            param_grid={
                "learning_rate": [0.05, 0.125, 0.25, 0.5],
                "sigma": [1.0, 2.0, 4.0],
            },
            dataset_name="Synthetic",
            max_sequences=10,
        )

        print(result.to_markdown())
        print(result.to_sensitivity_markdown("learning_rate"))
    """

    def __init__(
        self,
        tracker_name: str,
        dataset,
        verbose: bool = True,
    ) -> None:
        self.tracker_name = tracker_name
        self.dataset = dataset
        self.verbose = verbose

    def run(
        self,
        param_grid: Dict[str, Sequence[Any]],
        dataset_name: str = "dataset",
        max_sequences: Optional[int] = None,
    ) -> SweepResult:
        """Run the sweep over every combination in *param_grid*.

        Args:
            param_grid:    Mapping of parameter name to list of values to try.
                           All combinations are evaluated (Cartesian product).
            dataset_name:  Human-readable dataset label for reports.
            max_sequences: Cap on the number of sequences evaluated per
                           configuration — useful for fast iteration.
                           ``None`` evaluates all sequences.

        Returns:
            :class:`SweepResult` with one :class:`SweepEntry` per combination,
            sorted by mean IoU descending, with Pareto-front flags set.

        Raises:
            ValueError: If *param_grid* is empty.
        """
        if not param_grid:
            raise ValueError("param_grid must contain at least one parameter.")

        from ..benchmark.engine import BenchmarkEngine
        from ..trackers.registry import build_tracker

        param_names = list(param_grid.keys())
        value_lists = [param_grid[k] for k in param_names]
        combinations: List[Dict[str, Any]] = [
            dict(zip(param_names, combo))
            for combo in itertools.product(*value_lists)
        ]

        n_total = len(combinations)
        if self.verbose:
            print(
                f"\n[sweep] {self.tracker_name} on {dataset_name} — "
                f"{n_total} configurations"
            )
            print("-" * 60)

        engine = BenchmarkEngine(verbose=False)
        entries: List[SweepEntry] = []

        for idx, params in enumerate(combinations, start=1):
            tracker = build_tracker(self.tracker_name, **params)
            t0 = time.perf_counter()
            result = engine.run(
                tracker,
                self.dataset,
                dataset_name=dataset_name,
                max_sequences=max_sequences,
            )
            wall_time = time.perf_counter() - t0

            mean_iou = result.mean_iou
            success_auc = result.mean_success_auc or mean_iou
            precision_auc = result.mean_precision_auc or 0.0
            mean_fps = result.mean_fps
            peak_mem = result.peak_memory_mb

            # Mean latency across all sequence profiling results
            all_lats = [
                sr.profiling.latency_mean_ms for sr in result.sequence_results
            ]
            mean_lat = float(np.mean(all_lats)) if all_lats else 0.0

            entry = SweepEntry(
                params=dict(params),
                tracker_name=self.tracker_name,
                dataset_name=dataset_name,
                mean_iou=mean_iou,
                success_auc=success_auc,
                precision_auc=precision_auc,
                mean_fps=mean_fps,
                mean_latency_ms=mean_lat,
                peak_memory_mb=peak_mem,
                num_sequences=len(result.sequence_results),
                wall_time_s=wall_time,
            )
            entries.append(entry)

            if self.verbose:
                print(
                    f"  [{idx:>3}/{n_total}] {entry.params_label():<40s} "
                    f"mIoU={mean_iou:.4f}  FPS={mean_fps:.1f}"
                )

        # Mark Pareto-optimal configurations in (mIoU, FPS) space
        _mark_pareto(entries)

        entries.sort(key=lambda e: e.mean_iou, reverse=True)

        if self.verbose:
            print("-" * 60)
            best = entries[0] if entries else None
            if best:
                print(
                    f"  Best config: {best.params_label()}  "
                    f"mIoU={best.mean_iou:.4f}  FPS={best.mean_fps:.1f}"
                )

        return SweepResult(
            tracker_name=self.tracker_name,
            dataset_name=dataset_name,
            param_names=param_names,
            entries=entries,
        )

    def sensitivity_1d(
        self,
        param_name: str,
        values: Sequence[Any],
        fixed_params: Optional[Dict[str, Any]] = None,
        dataset_name: str = "dataset",
        max_sequences: Optional[int] = None,
    ) -> SweepResult:
        """Run a 1-D sensitivity sweep over a single parameter.

        Convenience wrapper around :meth:`run` that sweeps a single parameter
        while holding all others at their ``fixed_params`` values.

        Args:
            param_name:    Name of the parameter to sweep.
            values:        Sequence of values to try for *param_name*.
            fixed_params:  Other tracker parameters to hold constant.
                           Passed directly to the tracker constructor.
            dataset_name:  Human-readable dataset label.
            max_sequences: Cap on the number of sequences per configuration.

        Returns:
            :class:`SweepResult` (same as :meth:`run`).
        """
        base = dict(fixed_params or {})
        grid = {**{k: [v] for k, v in base.items()}, param_name: list(values)}
        return self.run(grid, dataset_name=dataset_name, max_sequences=max_sequences)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mark_pareto(entries: List[SweepEntry]) -> None:
    """Mark Pareto-optimal entries in the (mean_iou, mean_fps) space.

    Entry A dominates entry B iff ``A.mean_iou >= B.mean_iou`` AND
    ``A.mean_fps >= B.mean_fps`` with at least one strict inequality.
    Non-dominated entries have ``on_pareto_front`` set to ``True`` in-place.

    Args:
        entries: List of :class:`SweepEntry` objects.  Modified in-place.
    """
    for i, candidate in enumerate(entries):
        dominated = False
        for j, other in enumerate(entries):
            if i == j:
                continue
            if (
                other.mean_iou >= candidate.mean_iou
                and other.mean_fps >= candidate.mean_fps
                and (other.mean_iou > candidate.mean_iou or other.mean_fps > candidate.mean_fps)
            ):
                dominated = True
                break
        candidate.on_pareto_front = not dominated
