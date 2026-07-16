"""Hyperparameter sweep engine for tracker configuration tuning.

Enables systematic grid or random search over tracker parameter spaces —
learning rates, regularization constants, patch sizes, etc. — to find
configurations that maximise a benchmark metric on a given dataset.

This is important for reproducible research: the default tracker hyperparameters
are rarely optimal for a specific benchmark, and unreported tuning is a common
source of unfair comparisons in the VOT literature.

Design
------
:class:`TrackerHparamSweep` wraps :class:`~eovot.benchmark.engine.BenchmarkEngine`
and :class:`~eovot.trackers.registry.build_tracker`, delegating the evaluation
loop so the sweep is compatible with any tracker registered in
:data:`~eovot.trackers.registry.TRACKER_REGISTRY`.

Example (grid search)::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.experiment.hparam_sweep import TrackerHparamSweep, SweepConfig

    ds = SyntheticDataset(num_sequences=5, num_frames=50, seed=42)
    sweep = TrackerHparamSweep()
    result = sweep.run(
        SweepConfig(
            tracker_name="KCF",
            param_grid={
                "learning_rate": [0.05, 0.075, 0.10, 0.15],
                "lambda_": [1e-5, 1e-4, 1e-3],
            },
        ),
        dataset=ds,
        dataset_name="synthetic",
    )
    print(result.to_markdown())
    print("Best params:", result.best_params)

Example (random search)::

    sweep = TrackerHparamSweep(mode="random", n_random=20, seed=0)
    result = sweep.run(
        SweepConfig(
            tracker_name="MOSSE",
            param_grid={
                "learning_rate": [0.05, 0.075, 0.10, 0.125, 0.15, 0.20],
                "patch_size": [(96, 96), (112, 112), (128, 128), (160, 160)],
            },
            metric="success_auc",
        ),
        dataset=ds,
        dataset_name="synthetic",
    )
"""

from __future__ import annotations

import csv
import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..trackers.registry import build_tracker


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """Configuration for a hyperparameter sweep.

    Attributes:
        tracker_name: Key in :data:`~eovot.trackers.registry.TRACKER_REGISTRY`,
            e.g. ``"KCF"``, ``"MOSSE"``, ``"DCF"``.
        param_grid: Mapping from parameter name to a list of candidate values.
            Grid search evaluates every Cartesian product; random search draws
            *n_random* random combinations.
        metric: Benchmark metric to maximise. Supported:
            ``"mean_iou"``, ``"success_auc"``, ``"precision_auc"``, ``"mean_fps"``.
        max_sequences: Cap on sequences per evaluation (for speed). ``None``
            uses the full dataset.
    """

    tracker_name: str
    param_grid: Dict[str, List[Any]]
    metric: str = "success_auc"
    max_sequences: Optional[int] = None

    def __post_init__(self) -> None:
        valid = {"mean_iou", "success_auc", "precision_auc", "mean_fps"}
        if self.metric not in valid:
            raise ValueError(
                f"metric must be one of {sorted(valid)}, got {self.metric!r}"
            )
        if not self.param_grid:
            raise ValueError("param_grid must not be empty")


@dataclass
class SweepEntry:
    """One evaluated hyperparameter combination.

    Attributes:
        params: The hyperparameter dict that was evaluated.
        score: Value of the target metric for this configuration.
        benchmark_result: Full :class:`~eovot.benchmark.engine.BenchmarkResult`
            from the evaluation run.
    """

    params: Dict[str, Any]
    score: float
    benchmark_result: BenchmarkResult


@dataclass
class SweepResult:
    """Complete outcome of a hyperparameter sweep.

    Attributes:
        tracker_name: Name of the swept tracker.
        metric: Metric that was optimised.
        best_params: Hyperparameter combination with the highest ``metric`` score.
        best_score: Score achieved by ``best_params``.
        entries: All evaluated configurations, sorted by score (highest first).
    """

    tracker_name: str
    metric: str
    best_params: Dict[str, Any]
    best_score: float
    entries: List[SweepEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "tracker": self.tracker_name,
            "metric": self.metric,
            "best_params": self.best_params,
            "best_score": round(self.best_score, 6),
            "n_configs_evaluated": len(self.entries),
            "all_results": [
                {"params": e.params, "score": round(e.score, 6)}
                for e in self.entries
            ],
        }

    def to_markdown(self) -> str:
        """Format sweep results as a Markdown table, best first."""
        if not self.entries:
            return "No results."

        # Collect all parameter names
        param_names = sorted(
            {k for e in self.entries for k in e.params}
        )

        header_params = " | ".join(f"**{p}**" for p in param_names)
        lines = [
            f"## Hyperparameter Sweep — {self.tracker_name}",
            f"Optimised metric: `{self.metric}`",
            "",
            f"| Rank | {header_params} | {self.metric} |",
            "|------|" + "|".join("-----" for _ in param_names) + "|-------|",
        ]
        for rank, e in enumerate(self.entries[:20], 1):  # show top 20
            param_vals = " | ".join(
                str(e.params.get(p, "—")) for p in param_names
            )
            lines.append(f"| {rank} | {param_vals} | {e.score:.4f} |")

        if len(self.entries) > 20:
            lines.append(f"| … | (showing top 20 of {len(self.entries)}) | |")

        lines += [
            "",
            f"**Best**: `{self.best_params}` → `{self.metric}` = **{self.best_score:.4f}**",
        ]
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """Write all sweep entries to a CSV file.

        Args:
            path: Destination file path (created or overwritten).
        """
        if not self.entries:
            return
        param_names = sorted({k for e in self.entries for k in e.params})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=param_names + [self.metric])
            writer.writeheader()
            for e in self.entries:
                row = {p: e.params.get(p, "") for p in param_names}
                row[self.metric] = round(e.score, 6)
                writer.writerow(row)

    def save_json(self, path: str) -> None:
        """Save the sweep summary as JSON.

        Args:
            path: Destination file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Sweep engine
# ---------------------------------------------------------------------------

class TrackerHparamSweep:
    """Grid or random search over tracker hyperparameter spaces.

    Args:
        mode: ``"grid"`` evaluates every combination in ``param_grid``
            (Cartesian product). ``"random"`` samples ``n_random``
            combinations uniformly at random.
        n_random: Number of random configurations to evaluate when
            ``mode="random"``. Ignored in grid mode.
        seed: Random seed for reproducible random sweeps. ``None`` uses
            the system clock (non-reproducible).
        verbose: If ``True``, print per-configuration progress to stdout.
        tdp_watts: Forwarded to :class:`~eovot.benchmark.engine.BenchmarkEngine`
            for energy profiling. ``None`` disables energy profiling.

    Example::

        sweep = TrackerHparamSweep(mode="grid", verbose=True)
        result = sweep.run(config, dataset, "my_dataset")
        result.save_json("sweep_results/kcf_sweep.json")
    """

    def __init__(
        self,
        mode: str = "grid",
        n_random: int = 20,
        seed: Optional[int] = None,
        verbose: bool = False,
        tdp_watts: Optional[float] = None,
    ) -> None:
        if mode not in ("grid", "random"):
            raise ValueError(f"mode must be 'grid' or 'random', got {mode!r}")
        self.mode = mode
        self.n_random = n_random
        self.seed = seed
        self.verbose = verbose
        self._engine = BenchmarkEngine(verbose=False, tdp_watts=tdp_watts)

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def run(
        self,
        config: SweepConfig,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
    ) -> SweepResult:
        """Evaluate all (or a random subset of) hyperparameter combinations.

        Args:
            config: Sweep configuration including tracker name, grid, and metric.
            dataset: Dataset to benchmark on.
            dataset_name: Display name for the dataset (used in progress output).

        Returns:
            :class:`SweepResult` with all entries sorted by score, best first.
        """
        combinations = self._get_combinations(config)
        n_total = len(combinations)

        if self.verbose:
            print(
                f"\nHparam sweep: {config.tracker_name} on {dataset_name}  "
                f"({n_total} configurations, mode={self.mode})"
            )

        entries: List[SweepEntry] = []

        for i, params in enumerate(combinations, 1):
            score, bench_result = self._evaluate(
                config.tracker_name, params, dataset, dataset_name,
                config.metric, config.max_sequences,
            )
            entries.append(SweepEntry(params=params, score=score, benchmark_result=bench_result))
            if self.verbose:
                print(f"  [{i:>3}/{n_total}] {params}  →  {config.metric}={score:.4f}")

        entries.sort(key=lambda e: e.score, reverse=True)

        best = entries[0] if entries else None
        return SweepResult(
            tracker_name=config.tracker_name,
            metric=config.metric,
            best_params=best.params if best else {},
            best_score=best.score if best else 0.0,
            entries=entries,
        )

    # ------------------------------------------------------------------
    # Convenience: run multiple trackers and compare
    # ------------------------------------------------------------------

    def run_multi(
        self,
        configs: List[SweepConfig],
        dataset: BaseDataset,
        dataset_name: str = "unknown",
    ) -> Dict[str, SweepResult]:
        """Run a sweep for each tracker and return a dict of results.

        Args:
            configs: One :class:`SweepConfig` per tracker.
            dataset: Shared dataset used for all sweeps.
            dataset_name: Dataset display name.

        Returns:
            Dict mapping ``tracker_name`` → :class:`SweepResult`.
        """
        return {
            cfg.tracker_name: self.run(cfg, dataset, dataset_name)
            for cfg in configs
        }

    def comparison_table(self, results: Dict[str, SweepResult]) -> str:
        """Format best-per-tracker results as a Markdown comparison table.

        Args:
            results: Output of :meth:`run_multi`.

        Returns:
            Markdown table string comparing the best configuration per tracker.
        """
        if not results:
            return "No results."
        metric = next(iter(results.values())).metric
        lines = [
            "## Best Hyperparameters per Tracker",
            f"Metric: `{metric}`",
            "",
            "| Tracker | Best Score | Best Params |",
            "|---------|------------|-------------|",
        ]
        for name, r in sorted(results.items(), key=lambda x: x[1].best_score, reverse=True):
            lines.append(
                f"| {name} | {r.best_score:.4f} | `{r.best_params}` |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_combinations(
        self, config: SweepConfig
    ) -> List[Dict[str, Any]]:
        """Generate the list of hyperparameter dicts to evaluate."""
        names = list(config.param_grid.keys())
        values = list(config.param_grid.values())
        all_combos = [
            dict(zip(names, combo))
            for combo in itertools.product(*values)
        ]

        if self.mode == "grid":
            return all_combos

        # Random sampling without replacement (or with replacement if needed)
        rng = random.Random(self.seed)
        n = min(self.n_random, len(all_combos))
        return rng.sample(all_combos, n)

    def _get_metric(self, result: BenchmarkResult, metric: str) -> float:
        """Extract a scalar metric value from a BenchmarkResult."""
        if metric == "mean_iou":
            return result.mean_iou
        if metric == "success_auc":
            val = result.mean_success_auc
            return val if val is not None else result.mean_iou
        if metric == "precision_auc":
            val = result.mean_precision_auc
            return val if val is not None else result.mean_iou
        if metric == "mean_fps":
            return result.mean_fps
        return result.mean_iou

    def _evaluate(
        self,
        tracker_name: str,
        params: Dict[str, Any],
        dataset: BaseDataset,
        dataset_name: str,
        metric: str,
        max_sequences: Optional[int],
    ) -> Tuple[float, BenchmarkResult]:
        """Build, run, and score a single tracker configuration."""
        tracker = build_tracker(tracker_name, **params)
        bench_result = self._engine.run(
            tracker, dataset, dataset_name, max_sequences=max_sequences
        )
        score = self._get_metric(bench_result, metric)
        return score, bench_result
