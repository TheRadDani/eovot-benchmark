"""Tracker hyperparameter sweep and sensitivity analysis for EOVOT.

Enables researchers to systematically explore how a tracker's configuration
affects accuracy, throughput, and edge-deployment suitability — without
writing custom loops.  The analysis outputs:

* A ranked table of all tried configurations (mIoU, FPS, EES).
* The Pareto-optimal front across accuracy and edge efficiency.
* Per-parameter sensitivity scores showing which hyperparameters matter most.
* CSV and Markdown exports for notebooks and papers.

Why this matters
----------------
Tracker papers typically report a single configuration tuned on a held-out
dataset.  For *edge deployment*, the optimal trade-off point depends on the
target device — a configuration that maximises mIoU on a workstation may be
too slow on a Raspberry Pi.  By sweeping configurations and extracting the
Pareto front, researchers can identify which settings to publish for each
device class.

Typical usage::

    from eovot.analysis.hyperparam_sweep import HyperparamSweeper
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=10, num_frames=100)

    sweeper = HyperparamSweeper(dataset=dataset, dataset_name="Synthetic")

    # Grid search over KCF parameters
    results = sweeper.grid_search(
        tracker_cls=KCFTracker,
        param_grid={"learning_rate": [0.075, 0.125, 0.200]},
    )

    print(sweeper.to_markdown(results))
    sweeper.to_csv(results, "kcf_sweep.csv")

    pareto = sweeper.pareto_front(results)
    print(f"Pareto-optimal configs: {[r.params for r in pareto]}")
"""

from __future__ import annotations

import csv
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

from ..benchmark.engine import BenchmarkEngine
from ..metrics.efficiency import EfficiencyMetricsEngine
from ..trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SweepEntry:
    """Result for a single hyperparameter configuration.

    Attributes:
        params: The exact kwargs passed to the tracker constructor.
        mean_iou: Mean IoU across all evaluated sequences.
        success_auc: Mean success-curve AUC (if computed, else ``None``).
        mean_fps: Mean frames-per-second throughput.
        peak_memory_mb: Peak RSS memory footprint in megabytes.
        ees: Edge Efficiency Score — ``mean_iou × log1p(fps) / (1 + mem / budget)``.
        wall_time_s: Total wall-clock seconds for this configuration's run.
        on_pareto_front: Set to ``True`` after :meth:`~HyperparamSweeper.pareto_front`
            computes the front; defaults to ``False``.
    """

    params: Dict[str, Any]
    mean_iou: float
    success_auc: Optional[float]
    mean_fps: float
    peak_memory_mb: float
    ees: float
    wall_time_s: float
    on_pareto_front: bool = False

    def __str__(self) -> str:
        return (
            f"SweepEntry(params={self.params}  "
            f"mIoU={self.mean_iou:.4f}  "
            f"FPS={self.mean_fps:.1f}  "
            f"EES={self.ees:.4f}  "
            f"mem={self.peak_memory_mb:.1f} MiB)"
        )


# ---------------------------------------------------------------------------
# Sweeper
# ---------------------------------------------------------------------------

class HyperparamSweeper:
    """Run a tracker with multiple hyperparameter configurations and compare.

    Args:
        dataset: Dataset to evaluate on.  A small synthetic dataset with 5–20
            sequences is recommended for fast sweeps.
        dataset_name: Human-readable label used in reports.
        max_sequences: Limit each run to the first *N* sequences.  ``None``
            evaluates the full dataset.
        memory_budget_mb: Memory budget for the Edge Efficiency Score
            denominator.  Default: ``512`` MiB.
        verbose: Print per-run progress.  Default: ``True``.

    Example::

        sweeper = HyperparamSweeper(
            dataset=SyntheticDataset(num_sequences=10, num_frames=100),
            dataset_name="Synthetic",
            max_sequences=5,
        )
        results = sweeper.grid_search(KCFTracker, {"learning_rate": [0.075, 0.125]})
    """

    def __init__(
        self,
        dataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
        memory_budget_mb: float = 512.0,
        verbose: bool = True,
    ) -> None:
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.max_sequences = max_sequences
        self.memory_budget_mb = memory_budget_mb
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Core search routines
    # ------------------------------------------------------------------

    def grid_search(
        self,
        tracker_cls: Type[BaseTracker],
        param_grid: Dict[str, List[Any]],
    ) -> List[SweepEntry]:
        """Exhaustive grid search over all combinations in *param_grid*.

        Each combination is evaluated independently via :class:`BenchmarkEngine`.
        Results are returned sorted by EES (descending).

        Args:
            tracker_cls: Tracker class to instantiate (subclass of
                :class:`~eovot.trackers.base.BaseTracker`).
            param_grid: Dict mapping parameter name → list of values to try.
                All combinations are evaluated.  E.g.
                ``{"lr": [0.1, 0.2], "sigma": [0.5, 1.0]}`` → 4 runs.

        Returns:
            List of :class:`SweepEntry` sorted by EES (highest first).
        """
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]
        configs = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

        if self.verbose:
            print(
                f"\n[Sweep] {tracker_cls.__name__} — "
                f"{len(configs)} configs × {self.dataset_name}"
            )
            print("-" * 60)

        entries: List[SweepEntry] = []
        for i, params in enumerate(configs):
            entry = self._evaluate_config(tracker_cls, params, idx=i, total=len(configs))
            entries.append(entry)

        entries.sort(key=lambda e: e.ees, reverse=True)
        return entries

    def random_search(
        self,
        tracker_cls: Type[BaseTracker],
        param_distributions: Dict[str, Any],
        n_trials: int = 10,
        seed: Optional[int] = None,
    ) -> List[SweepEntry]:
        """Random hyperparameter search by sampling from distributions.

        Each distribution is any object with a ``rvs()`` method (e.g.
        ``scipy.stats.uniform(0.05, 0.20)``), a list (uniform discrete), or a
        two-tuple ``(low, high)`` for continuous uniform sampling.

        Args:
            tracker_cls: Tracker class to instantiate.
            param_distributions: Dict mapping parameter name → distribution.
            n_trials: Number of random configurations to evaluate.
            seed: Optional random seed for reproducibility.

        Returns:
            List of :class:`SweepEntry` sorted by EES (highest first).
        """
        rng = np.random.default_rng(seed)

        configs: List[Dict[str, Any]] = []
        for _ in range(n_trials):
            params: Dict[str, Any] = {}
            for key, dist in param_distributions.items():
                if hasattr(dist, "rvs"):
                    params[key] = float(dist.rvs(random_state=int(rng.integers(0, 2**31))))
                elif isinstance(dist, (list, tuple)) and len(dist) == 2 and not isinstance(dist[0], (list, tuple)):
                    lo, hi = float(dist[0]), float(dist[1])
                    params[key] = float(rng.uniform(lo, hi))
                elif isinstance(dist, list):
                    params[key] = dist[int(rng.integers(0, len(dist)))]
                else:
                    params[key] = dist
            configs.append(params)

        if self.verbose:
            print(
                f"\n[RandomSearch] {tracker_cls.__name__} — "
                f"{n_trials} trials × {self.dataset_name}"
            )
            print("-" * 60)

        entries: List[SweepEntry] = []
        for i, params in enumerate(configs):
            entry = self._evaluate_config(tracker_cls, params, idx=i, total=n_trials)
            entries.append(entry)

        entries.sort(key=lambda e: e.ees, reverse=True)
        return entries

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def pareto_front(self, entries: List[SweepEntry]) -> List[SweepEntry]:
        """Identify Pareto-optimal configurations over (mean_iou, ees).

        A configuration is Pareto-optimal if no other configuration is
        strictly better on *both* mean_iou and EES simultaneously.
        Marks ``entry.on_pareto_front = True`` for each optimal entry.

        Args:
            entries: Sweep results from :meth:`grid_search` or
                :meth:`random_search`.

        Returns:
            The subset of *entries* that are Pareto-optimal (also mutated
            in-place via ``on_pareto_front = True``).
        """
        # Reset all flags.
        for e in entries:
            e.on_pareto_front = False

        dominated = [False] * len(entries)
        for i, a in enumerate(entries):
            for j, b in enumerate(entries):
                if i == j:
                    continue
                # b dominates a when b is >= a on both axes and > on at least one.
                if (b.mean_iou >= a.mean_iou and b.ees >= a.ees) and (
                    b.mean_iou > a.mean_iou or b.ees > a.ees
                ):
                    dominated[i] = True
                    break

        front = []
        for e, dom in zip(entries, dominated):
            if not dom:
                e.on_pareto_front = True
                front.append(e)
        return front

    def sensitivity_scores(self, entries: List[SweepEntry]) -> Dict[str, float]:
        """Estimate per-parameter importance from variance in EES.

        For each hyperparameter, computes the standard deviation of mean
        EES values grouped by that parameter's setting.  A high score means
        that parameter has a large effect on edge efficiency.  Parameters
        with only one distinct value return 0.0.

        Args:
            entries: Sweep results.

        Returns:
            Dict mapping parameter name → sensitivity score (std of group means).
        """
        if not entries:
            return {}

        all_keys = set()
        for e in entries:
            all_keys.update(e.params.keys())

        scores: Dict[str, float] = {}
        for key in all_keys:
            groups: Dict[Any, List[float]] = {}
            for e in entries:
                val = e.params.get(key, None)
                groups.setdefault(val, []).append(e.ees)

            if len(groups) < 2:
                scores[key] = 0.0
                continue
            group_means = [np.mean(v) for v in groups.values()]
            scores[key] = float(np.std(group_means))

        return dict(sorted(scores.items(), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_markdown(self, entries: List[SweepEntry], title: str = "") -> str:
        """Render the sweep results as a Markdown table.

        Args:
            entries: Sweep results (sorted by EES before display).
            title: Optional table heading line.

        Returns:
            Markdown string ready for README or GitHub comment.
        """
        if not entries:
            return "*No sweep results.*"

        header_line = f"## Hyperparameter Sweep — {title or self.dataset_name}\n\n" if title or self.dataset_name else ""

        all_keys = sorted({k for e in entries for k in e.params})
        param_cols = " | ".join(all_keys) + " | " if all_keys else ""
        header = f"| # | {param_cols}mIoU | FPS | Mem (MiB) | EES | Pareto |"
        sep_param = " | ".join(["-" * max(len(k), 5) for k in all_keys]) + " | " if all_keys else ""
        separator = f"| - | {sep_param}---- | --- | --------- | --- | ------ |"

        rows = []
        for i, e in enumerate(entries, 1):
            param_vals = " | ".join(str(e.params.get(k, "")) for k in all_keys)
            param_str = f"{param_vals} | " if all_keys else ""
            star = "✓" if e.on_pareto_front else ""
            rows.append(
                f"| {i} | {param_str}"
                f"{e.mean_iou:.4f} | {e.mean_fps:.1f} | {e.peak_memory_mb:.1f} | "
                f"{e.ees:.4f} | {star} |"
            )

        return header_line + "\n".join([header, separator] + rows)

    def to_csv(self, entries: List[SweepEntry], path: str) -> Path:
        """Write sweep results to a CSV file.

        Args:
            entries: Sweep results.
            path: Output file path.

        Returns:
            Resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        all_keys = sorted({k for e in entries for k in e.params})
        fieldnames = all_keys + ["mean_iou", "success_auc", "mean_fps",
                                  "peak_memory_mb", "ees", "wall_time_s", "on_pareto_front"]

        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                row = {k: e.params.get(k, "") for k in all_keys}
                row.update({
                    "mean_iou": round(e.mean_iou, 6),
                    "success_auc": round(e.success_auc, 6) if e.success_auc is not None else "",
                    "mean_fps": round(e.mean_fps, 2),
                    "peak_memory_mb": round(e.peak_memory_mb, 2),
                    "ees": round(e.ees, 6),
                    "wall_time_s": round(e.wall_time_s, 2),
                    "on_pareto_front": e.on_pareto_front,
                })
                writer.writerow(row)

        return p

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_config(
        self,
        tracker_cls: Type[BaseTracker],
        params: Dict[str, Any],
        idx: int = 0,
        total: int = 1,
    ) -> SweepEntry:
        """Run one configuration and return a :class:`SweepEntry`.

        If the tracker raises an exception for a given configuration (e.g.
        an out-of-bounds patch in a correlation filter), the entry is
        recorded with zero metrics so the sweep continues rather than
        aborting.
        """
        import math

        tracker = tracker_cls(**params)
        engine = BenchmarkEngine(verbose=False)

        t_start = time.perf_counter()
        try:
            result = engine.run(
                tracker, self.dataset,
                dataset_name=self.dataset_name,
                max_sequences=self.max_sequences,
            )
            wall_time = time.perf_counter() - t_start

            mem = result.peak_memory_mb
            fps = result.mean_fps
            iou = result.mean_iou
            ees = iou * math.log1p(fps) / (1.0 + mem / self.memory_budget_mb)
            success_auc = result.mean_success_auc
        except Exception as exc:
            wall_time = time.perf_counter() - t_start
            iou, fps, mem, ees = 0.0, 0.0, 0.0, 0.0
            success_auc = None
            if self.verbose:
                print(f"  [{idx + 1:>3}/{total}] {str(params):<45s} ERROR: {exc}")

        if self.verbose:
            print(
                f"  [{idx + 1:>3}/{total}] {str(params):<45s} "
                f"mIoU={iou:.4f}  FPS={fps:.1f}  EES={ees:.4f}  "
                f"({wall_time:.1f}s)"
            )

        return SweepEntry(
            params=params,
            mean_iou=iou,
            success_auc=success_auc,
            mean_fps=fps,
            peak_memory_mb=mem,
            ees=ees,
            wall_time_s=wall_time,
        )
