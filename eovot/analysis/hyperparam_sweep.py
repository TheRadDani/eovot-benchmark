"""Tracker hyperparameter sweep engine for EOVOT.

:class:`HyperparamSweeper` runs a systematic grid search over a tracker's
constructor parameters, evaluating each configuration with
:class:`~eovot.benchmark.engine.BenchmarkEngine` and ranking results by
accuracy–speed trade-off.

Knowing which parameter setting maximises accuracy at a given FPS budget is a
prerequisite for fair cross-tracker comparisons in research papers: using the
default configuration of one tracker while tuning another introduces
confounded baselines.  This module makes the sweep reproducible and
exportable.

Typical usage::

    from eovot.analysis.hyperparam_sweep import HyperparamSweeper
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, num_frames=50)
    sweeper = HyperparamSweeper("KCF", dataset, dataset_name="Synthetic")
    report = sweeper.sweep(
        param_grid={"learning_rate": [0.05, 0.075, 0.1, 0.125, 0.15]},
    )

    print(report.to_markdown())
    best = report.best_at_fps_budget(fps_min=200.0)
    print(best)
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


@dataclass
class SweepEntry:
    """Result for one parameter configuration.

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
    """Results of a complete hyperparameter sweep.

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

        Pareto-optimal rows are annotated with a ``*`` marker so readers
        can quickly spot the efficiency frontier.
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
    """Grid-search tracker hyperparameters and rank by accuracy–speed trade-off.

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
# Pareto helpers
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
