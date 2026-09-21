"""Systematic hyperparameter sweep engine for EOVOT.

:class:`HyperparamSweep` explores the full Cartesian product of a
``param_grid`` for a single tracker, evaluating each combination with the
benchmark engine and ranking results by a configurable primary metric.
The IoU–FPS Pareto frontier is computed automatically so researchers
can see which configurations are worth deploying on a given edge device.

Example::

    from eovot.experiment.sweep import HyperparamSweep
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, num_frames=50)
    result = HyperparamSweep(
        tracker_name="KCF",
        param_grid={
            "learning_rate": [0.05, 0.125, 0.25],
            "kernel_sigma": [0.5, 1.0, 2.0],
        },
        primary_metric="mean_iou",
        output_dir="results/sweeps/kcf",
    ).run(dataset)
    print(result.to_markdown())
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..benchmark.engine import BenchmarkEngine

_SUPPORTED_METRICS = frozenset({"mean_iou", "success_auc", "precision_auc", "mean_fps"})


@dataclass
class SweepEntry:
    """Result for one hyperparameter combination.

    Attributes:
        params:          The parameter dict used for this combination.
        mean_iou:        Mean IoU across all benchmark sequences.
        success_auc:     Success AUC (IoU threshold sweep).
        precision_auc:   Precision AUC (centre-distance threshold sweep).
        mean_fps:        Mean frames per second.
        peak_memory_mb:  Peak RSS memory in megabytes.
        wall_time_s:     Wall-clock time for the benchmark run in seconds.
        result_dict:     Full result dict from BenchmarkResult.to_dict().
    """

    params: Dict[str, Any]
    mean_iou: float
    success_auc: float
    precision_auc: float
    mean_fps: float
    peak_memory_mb: float
    wall_time_s: float
    result_dict: Dict = field(repr=False)

    @property
    def params_str(self) -> str:
        """Human-readable parameter string, keys sorted alphabetically."""
        return ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))


@dataclass
class SweepResult:
    """Aggregated results from a completed :class:`HyperparamSweep`.

    Attributes:
        tracker_name:   Tracker that was swept.
        param_grid:     Parameter grid used.
        primary_metric: Metric used for ranking.
        entries:        All combinations sorted descending by primary_metric.
        best:           The top-ranked entry.
        pareto_front:   Non-dominated entries on the IoU–FPS Pareto front.
    """

    tracker_name: str
    param_grid: Dict[str, List]
    primary_metric: str
    entries: List[SweepEntry]
    best: SweepEntry
    pareto_front: List[SweepEntry]

    def to_markdown(self) -> str:
        """Return a Markdown leaderboard table ranked by primary metric.

        Pareto-optimal entries are flagged with ✓ in the last column.
        """
        header = (
            "| Rank | Parameters "
            "| mIoU | Success AUC | FPS | Mem (MB) | Time (s) | Pareto |"
        )
        sep = "|------|------------|-----:|------------:|----:|---------:|---------:|:------:|"
        pareto_ids = {id(e) for e in self.pareto_front}
        rows = [header, sep]
        for rank, e in enumerate(self.entries, start=1):
            flag = "✓" if id(e) in pareto_ids else ""
            rows.append(
                f"| {rank} | {e.params_str} "
                f"| {e.mean_iou:.4f} | {e.success_auc:.4f} "
                f"| {e.mean_fps:.1f} | {e.peak_memory_mb:.1f} "
                f"| {e.wall_time_s:.2f} | {flag} |"
            )
        rows.append("")
        best_str = self.best.params_str or "(defaults)"
        rows.append(f"**Best** ({self.primary_metric}): `{best_str}`")
        rows.append(
            f"Pareto-optimal combinations: {len(self.pareto_front)} / {len(self.entries)}"
        )
        return "\n".join(rows)

    def to_dict(self) -> Dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "tracker_name": self.tracker_name,
            "param_grid": self.param_grid,
            "primary_metric": self.primary_metric,
            "best_params": self.best.params,
            "entries": [
                {
                    "params": e.params,
                    "mean_iou": e.mean_iou,
                    "success_auc": e.success_auc,
                    "precision_auc": e.precision_auc,
                    "mean_fps": e.mean_fps,
                    "peak_memory_mb": e.peak_memory_mb,
                    "wall_time_s": e.wall_time_s,
                }
                for e in self.entries
            ],
        }


class HyperparamSweep:
    """Grid-search hyperparameters for a single registered tracker.

    Evaluates every combination of the Cartesian product of ``param_grid``
    values using :class:`~eovot.benchmark.engine.BenchmarkEngine`, ranks
    the results by ``primary_metric``, and computes the IoU–FPS Pareto
    frontier.  Results can be saved to disk and interrupted runs can be
    resumed.

    Args:
        tracker_name:   Name registered in
                        :data:`~eovot.trackers.registry.TRACKER_REGISTRY`.
        param_grid:     Dict mapping parameter names to lists of candidate
                        values.  An empty dict runs the tracker once with
                        its default parameters.
        primary_metric: Metric for sorting results.  One of
                        ``"mean_iou"``, ``"success_auc"``,
                        ``"precision_auc"``, ``"mean_fps"``.
                        Higher is always better for all supported metrics.
        output_dir:     Optional directory for saving per-combination JSON
                        files, ``sweep_summary.json``, and
                        ``sweep_leaderboard.md``.
        verbose:        Print per-combination progress to stdout.
        tdp_watts:      Forwarded to
                        :class:`~eovot.benchmark.engine.BenchmarkEngine`
                        for energy profiling.  ``None`` disables it.
        resume:         When ``True`` and *output_dir* is set, skip
                        combinations whose ``combo_NNNN.json`` result file
                        already exists.

    Example::

        from eovot.experiment.sweep import HyperparamSweep
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=3, num_frames=30)
        result = HyperparamSweep(
            "MOSSE",
            {"learning_rate": [0.05, 0.125, 0.25], "sigma": [1.0, 2.0]},
        ).run(dataset)
        print(result.to_markdown())
    """

    def __init__(
        self,
        tracker_name: str,
        param_grid: Dict[str, List],
        primary_metric: str = "mean_iou",
        output_dir: Optional[str] = None,
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
        resume: bool = False,
    ) -> None:
        if primary_metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"primary_metric must be one of {sorted(_SUPPORTED_METRICS)}, "
                f"got '{primary_metric}'"
            )
        self.tracker_name = tracker_name
        self.param_grid = dict(param_grid)
        self.primary_metric = primary_metric
        self.output_dir = Path(output_dir) if output_dir else None
        self.verbose = verbose
        self.tdp_watts = tdp_watts
        self.resume = resume

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def combinations(self) -> List[Dict[str, Any]]:
        """Return the full Cartesian product of parameter grid values.

        Returns a list of one dict per combination.  An empty
        ``param_grid`` returns ``[{}]`` (a single run with default
        parameters).
        """
        if not self.param_grid:
            return [{}]
        keys = list(self.param_grid.keys())
        values_lists = [self.param_grid[k] for k in keys]
        return [
            dict(zip(keys, combo))
            for combo in itertools.product(*values_lists)
        ]

    def run(self, dataset, dataset_name: str = "synthetic") -> SweepResult:
        """Execute the full grid sweep and return ranked results.

        Args:
            dataset:       A dataset implementing the EOVOT dataset protocol
                           (``__len__``, ``__getitem__``).
            dataset_name:  Label used in per-combination result files and
                           the Markdown leaderboard.

        Returns:
            :class:`SweepResult` with entries sorted descending by
            ``primary_metric``.
        """
        from ..trackers.registry import build_tracker

        combos = self.combinations()
        engine = BenchmarkEngine(verbose=False, tdp_watts=self.tdp_watts)
        entries: List[SweepEntry] = []

        if self.verbose:
            print(
                f"[sweep] {self.tracker_name} — {len(combos)} combination(s) "
                f"(primary metric: {self.primary_metric})"
            )

        for idx, params in enumerate(combos, start=1):
            result_path = (
                self.output_dir / f"combo_{idx:04d}.json"
                if self.output_dir
                else None
            )

            if self.resume and result_path and result_path.exists():
                if self.verbose:
                    print(f"  [{idx}/{len(combos)}] resume — cached: {params}")
                entries.append(self._load_cached_entry(result_path, params))
                continue

            if self.verbose:
                pstr = ", ".join(f"{k}={v}" for k, v in params.items()) or "(defaults)"
                print(f"  [{idx}/{len(combos)}] {pstr}")

            tracker = build_tracker(self.tracker_name, **params)
            t0 = time.perf_counter()
            result = engine.run(
                tracker=tracker,
                dataset=dataset,
                dataset_name=dataset_name,
            )
            elapsed = round(time.perf_counter() - t0, 3)

            result_dict = result.to_dict()
            if result_path:
                result_path.write_text(
                    json.dumps(result_dict, indent=2), encoding="utf-8"
                )

            entries.append(self._entry_from_dict(result_dict, params, elapsed))

        entries.sort(
            key=lambda e: getattr(e, self.primary_metric), reverse=True
        )
        pareto = self._pareto_front(entries)
        best = entries[0]

        sweep_result = SweepResult(
            tracker_name=self.tracker_name,
            param_grid=self.param_grid,
            primary_metric=self.primary_metric,
            entries=entries,
            best=best,
            pareto_front=pareto,
        )

        if self.output_dir:
            (self.output_dir / "sweep_summary.json").write_text(
                json.dumps(sweep_result.to_dict(), indent=2), encoding="utf-8"
            )
            (self.output_dir / "sweep_leaderboard.md").write_text(
                sweep_result.to_markdown(), encoding="utf-8"
            )

        if self.verbose:
            best_str = best.params_str or "(defaults)"
            metric_val = getattr(best, self.primary_metric)
            print(
                f"\n[sweep] Best: {best_str} "
                f"— {self.primary_metric}={metric_val:.4f}"
            )
            print(f"        Pareto-optimal: {len(pareto)} / {len(entries)}")

        return sweep_result

    # ------------------------------------------------------------------
    @staticmethod
    def _entry_from_dict(
        result_dict: Dict, params: Dict, wall_time_s: float
    ) -> SweepEntry:
        """Build a :class:`SweepEntry` from a BenchmarkResult dict."""
        s = result_dict.get("summary", {})
        miou = float(s.get("mean_iou", 0.0))
        return SweepEntry(
            params=params,
            mean_iou=miou,
            success_auc=float(s.get("success_auc", miou)),
            precision_auc=float(s.get("precision_auc", 0.0)),
            mean_fps=float(s.get("mean_fps", 0.0)),
            peak_memory_mb=float(s.get("peak_memory_mb", 0.0)),
            wall_time_s=wall_time_s,
            result_dict=result_dict,
        )

    @staticmethod
    def _load_cached_entry(path: Path, params: Dict) -> SweepEntry:
        """Load a previously saved combo JSON as a :class:`SweepEntry`."""
        with open(path) as fh:
            cached = json.load(fh)
        return HyperparamSweep._entry_from_dict(cached, params, wall_time_s=0.0)

    @staticmethod
    def _pareto_front(entries: List[SweepEntry]) -> List[SweepEntry]:
        """Return the IoU–FPS Pareto front from *entries*.

        Entry *A* strictly dominates entry *B* when
        ``A.mean_iou > B.mean_iou`` **and** ``A.mean_fps > B.mean_fps``.
        The Pareto front is the set of non-dominated entries.
        """
        front: List[SweepEntry] = []
        for candidate in entries:
            dominated = any(
                other.mean_iou > candidate.mean_iou
                and other.mean_fps > candidate.mean_fps
                for other in entries
                if other is not candidate
            )
            if not dominated:
                front.append(candidate)
        return front
