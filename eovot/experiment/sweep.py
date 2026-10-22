"""Hyperparameter grid-search sweep for EOVOT trackers.

Evaluates every combination of tracker parameters in a user-supplied grid,
collects benchmark results for each configuration, and identifies the best
parameter set.  Results are saved as JSON and Markdown tables.

Typical usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.experiment.sweep import HyperparameterSweeper

    dataset = SyntheticDataset(num_sequences=10, num_frames=100, motion="linear")
    sweeper = HyperparameterSweeper(output_dir="results/sweeps")

    result = sweeper.sweep(
        tracker_name="KCF",
        param_grid={
            "learning_rate": [0.05, 0.075, 0.125, 0.2],
            "padding":        [1.0, 1.5, 2.0],
        },
        dataset=dataset,
        dataset_name="Synthetic-Linear",
    )

    print(result.to_markdown_table())
    print("Best params:", result.best_by_success_auc.params)

The resulting Markdown table is suitable for pasting directly into a paper
or README with ``★`` marking the best configuration.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SweepEntry:
    """Benchmark result for one parameter configuration.

    Attributes:
        params:      The exact parameter dict passed to the tracker constructor.
        result:      :class:`~eovot.benchmark.engine.BenchmarkResult` from the run.
        wall_time_s: Wall-clock seconds taken for this configuration.
    """

    params: Dict[str, Any]
    result: Any  # BenchmarkResult; typed as Any to avoid circular import
    wall_time_s: float

    @property
    def mean_iou(self) -> float:
        return self.result.mean_iou

    @property
    def success_auc(self) -> Optional[float]:
        return self.result.mean_success_auc

    @property
    def mean_fps(self) -> float:
        return self.result.mean_fps

    def __str__(self) -> str:
        sauc = f"  AUC={self.success_auc:.4f}" if self.success_auc is not None else ""
        return (
            f"SweepEntry({_params_str(self.params)}) "
            f"mIoU={self.mean_iou:.4f}{sauc}  "
            f"FPS={self.mean_fps:.1f}  ({self.wall_time_s:.1f}s)"
        )


@dataclass
class SweepResult:
    """Full results from a :class:`HyperparameterSweeper` run.

    Attributes:
        tracker_name: Name of the swept tracker.
        dataset_name: Name of the dataset used.
        param_grid:   The original grid dict passed to :meth:`~HyperparameterSweeper.sweep`.
        entries:      One :class:`SweepEntry` per evaluated configuration, in
            the order they were evaluated.
    """

    tracker_name: str
    dataset_name: str
    param_grid: Dict[str, List[Any]]
    entries: List[SweepEntry] = field(default_factory=list)

    @property
    def best_by_success_auc(self) -> Optional[SweepEntry]:
        """Return the entry with the highest success AUC (falls back to mIoU).

        Returns ``None`` when ``entries`` is empty.
        """
        if not self.entries:
            return None
        return max(
            self.entries,
            key=lambda e: e.success_auc if e.success_auc is not None else e.mean_iou,
        )

    @property
    def best_by_iou(self) -> Optional[SweepEntry]:
        """Return the entry with the highest mean IoU."""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.mean_iou)

    def to_dict(self) -> Dict:
        """Serialise sweep results to a JSON-compatible dict.

        Returns:
            Dict with keys ``tracker``, ``dataset``, ``param_grid``,
            ``num_configs``, ``best_params``, and ``rows`` (one dict per
            configuration with all metric scalars).
        """
        rows = []
        for e in self.entries:
            row: Dict[str, Any] = dict(e.params)
            row["mean_iou"] = round(e.mean_iou, 4)
            row["mean_fps"] = round(e.mean_fps, 2)
            row["wall_time_s"] = round(e.wall_time_s, 2)
            if e.success_auc is not None:
                row["success_auc"] = round(e.success_auc, 4)
            rows.append(row)

        best = self.best_by_success_auc
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "param_grid": self.param_grid,
            "num_configs": len(self.entries),
            "best_params": best.params if best else {},
            "rows": rows,
        }

    def to_markdown_table(self) -> str:
        """Format sweep results as a Markdown table sorted by success AUC.

        The best configuration is marked with ``★``.

        Returns:
            Multi-line Markdown string ready to embed in a README or paper.
        """
        if not self.entries:
            return "No results.\n"

        param_names = sorted(self.param_grid.keys())
        has_sauc = any(e.success_auc is not None for e in self.entries)

        headers = list(param_names) + ["mIoU"]
        if has_sauc:
            headers.append("Success AUC")
        headers += ["FPS", "Time (s)"]

        sep = lambda h: "---:" if h != headers[0] else "---"
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(sep(h) for h in headers) + " |"

        best = self.best_by_success_auc
        sorted_entries = sorted(
            self.entries,
            key=lambda e: (e.success_auc if e.success_auc is not None else e.mean_iou),
            reverse=True,
        )

        table_rows = []
        for e in sorted_entries:
            vals = [str(e.params.get(p, "")) for p in param_names]
            vals.append(f"{e.mean_iou:.4f}")
            if has_sauc:
                vals.append(
                    f"{e.success_auc:.4f}" if e.success_auc is not None else "—"
                )
            vals.append(f"{e.mean_fps:.1f}")
            vals.append(f"{e.wall_time_s:.1f}")
            marker = " ★" if (best and e.params == best.params) else ""
            table_rows.append("| " + " | ".join(vals) + f"{marker} |")

        return "\n".join([header_row, sep_row] + table_rows)


class HyperparameterSweeper:
    """Grid-search hyperparameter sweep for any registered EOVOT tracker.

    Evaluates every combination of parameter values from *param_grid* on
    *dataset*, collects :class:`~eovot.benchmark.engine.BenchmarkResult` for
    each configuration, and saves a JSON + Markdown summary.

    Args:
        output_dir: Directory where sweep outputs (JSON, Markdown) are written.
            Created automatically if it does not exist. Default: ``"results/sweeps"``.
        verbose:    Print per-configuration progress. Default: ``True``.
        tdp_watts:  If set, enables CPU energy estimation with this TDP value.

    Example::

        sweeper = HyperparameterSweeper(output_dir="results/sweeps")
        result = sweeper.sweep(
            tracker_name="KCF",
            param_grid={"learning_rate": [0.05, 0.075, 0.125]},
            dataset=SyntheticDataset(num_sequences=10, num_frames=100),
            dataset_name="Synthetic",
        )
        print(result.to_markdown_table())
    """

    def __init__(
        self,
        output_dir: str = "results/sweeps",
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.tdp_watts = tdp_watts

    def sweep(
        self,
        tracker_name: str,
        param_grid: Dict[str, List[Any]],
        dataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> SweepResult:
        """Run a grid search over *param_grid* and return :class:`SweepResult`.

        Args:
            tracker_name:   Name of a registered tracker (see
                :func:`~eovot.trackers.registry.available_trackers`).
            param_grid:     Mapping ``{param_name: [val1, val2, ...]}``.
                All Cartesian combinations are evaluated.  An empty dict
                evaluates the tracker with its default parameters once.
            dataset:        :class:`~eovot.datasets.base.BaseDataset` instance
                to evaluate each configuration on.  The same dataset object is
                reused across all configurations.
            dataset_name:   Human-readable label for filenames and reports.
            max_sequences:  Cap on the number of sequences evaluated per
                configuration.  ``None`` evaluates all sequences.

        Returns:
            :class:`SweepResult` with one :class:`SweepEntry` per configuration,
            sorted in evaluation order (not by metric).
        """
        from ..benchmark.engine import BenchmarkEngine
        from ..trackers.registry import build_tracker

        sweep_result = SweepResult(
            tracker_name=tracker_name,
            dataset_name=dataset_name,
            param_grid=param_grid,
        )

        configs = _generate_configs(param_grid)
        n_total = len(configs)

        if self.verbose:
            seq_cap = max_sequences or "all"
            print(
                f"\nHyperparameter Sweep: {tracker_name} on {dataset_name}\n"
                f"  {n_total} configs  ×  (up to {seq_cap} sequences each)"
            )
            print("-" * 60)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        engine = BenchmarkEngine(verbose=False, tdp_watts=self.tdp_watts)

        for i, params in enumerate(configs):
            tracker = build_tracker(tracker_name, **params)
            t0 = time.perf_counter()
            bench_result = engine.run(
                tracker=tracker,
                dataset=dataset,
                dataset_name=dataset_name,
                max_sequences=max_sequences,
            )
            wall_time = time.perf_counter() - t0

            entry = SweepEntry(params=params, result=bench_result, wall_time_s=wall_time)
            sweep_result.entries.append(entry)

            if self.verbose:
                sauc_str = (
                    f"  AUC={entry.success_auc:.3f}"
                    if entry.success_auc is not None
                    else ""
                )
                print(
                    f"  [{i + 1:>3}/{n_total}] {_params_str(params):<40s} "
                    f"mIoU={entry.mean_iou:.4f}{sauc_str}  "
                    f"FPS={entry.mean_fps:.1f}  ({wall_time:.1f}s)"
                )

        self._save(sweep_result, f"sweep-{tracker_name}-{dataset_name}")

        if self.verbose:
            best = sweep_result.best_by_success_auc
            if best:
                print(f"\n  Best config → {_params_str(best.params)}")
                if best.success_auc is not None:
                    print(f"    success_auc={best.success_auc:.4f}  mIoU={best.mean_iou:.4f}")

        return sweep_result

    def _save(self, result: SweepResult, name: str) -> None:
        """Persist sweep results as JSON and Markdown."""
        d = result.to_dict()

        json_path = self.output_dir / f"{name}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2)

        md_path = self.output_dir / f"{name}.md"
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"# Hyperparameter Sweep: {result.tracker_name} "
                f"on {result.dataset_name}\n\n"
            )
            fh.write(f"**Configurations evaluated:** {d['num_configs']}\n\n")
            if d["best_params"]:
                fh.write(f"**Best params (by success AUC):** `{d['best_params']}`\n\n")
            fh.write(result.to_markdown_table())
            fh.write("\n")

        if self.verbose:
            print(f"\n  [JSON] {json_path}")
            print(f"  [MD]   {md_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_configs(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Expand *param_grid* into a flat list of all parameter combinations.

    Args:
        param_grid: ``{param: [val1, val2, ...]}``.

    Returns:
        List of dicts, one per Cartesian combination of values.  An empty
        grid returns ``[{}]`` (one config: all defaults).
    """
    if not param_grid:
        return [{}]
    keys = sorted(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    return [
        dict(zip(keys, combo)) for combo in itertools.product(*value_lists)
    ]


def _params_str(params: Dict[str, Any]) -> str:
    """Format a params dict as a compact key=value string."""
    return "  ".join(f"{k}={v}" for k, v in sorted(params.items()))
