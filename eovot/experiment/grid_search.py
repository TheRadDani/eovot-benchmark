"""Hyperparameter grid search for EOVOT trackers.

Evaluates every combination in a parameter grid against a benchmark dataset
to identify optimal tracker configurations per deployment scenario (accuracy,
throughput, memory, or a composite score).

Typical usage::

    from eovot.experiment.grid_search import GridSearchEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=10, num_frames=80, seed=42)

    engine = GridSearchEngine(
        tracker_cls=MOSSETracker,
        param_grid={
            "learning_rate": [0.075, 0.10, 0.125, 0.15],
            "sigma": [1.5, 2.0, 2.5],
        },
    )
    entries = engine.run(dataset, dataset_name="Synthetic")
    best = engine.best_config(entries, metric="mean_iou")
    print(engine.to_markdown(entries))

YAML-driven usage::

    from eovot.experiment.grid_search import GridSearchEngine
    import yaml

    with open("configs/grid_search/mosse_grid.yaml") as f:
        cfg = yaml.safe_load(f)

    engine = GridSearchEngine.from_config(cfg)
    entries = engine.run_from_config(cfg)
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class GridSearchEntry:
    """Result for a single parameter combination.

    Attributes:
        params: The hyperparameter dict used for this run.
        result: Full :class:`~eovot.benchmark.engine.BenchmarkResult`.
        elapsed_s: Wall-clock seconds spent on this combination.
    """

    params: Dict[str, Any]
    result: BenchmarkResult
    elapsed_s: float = 0.0

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                #
    # ------------------------------------------------------------------ #

    @property
    def mean_iou(self) -> float:
        return self.result.mean_iou

    @property
    def fps(self) -> float:
        return self.result.mean_fps

    @property
    def peak_memory_mb(self) -> float:
        return self.result.peak_memory_mb

    @property
    def success_auc(self) -> float:
        return self.result.mean_success_auc

    @property
    def precision_auc(self) -> float:
        return self.result.mean_precision_auc

    def to_dict(self) -> Dict[str, Any]:
        """Serialise entry to a JSON-compatible dict."""
        return {
            "params": self.params,
            "mean_iou": self.mean_iou,
            "success_auc": self.success_auc,
            "precision_auc": self.precision_auc,
            "fps": self.fps,
            "peak_memory_mb": self.peak_memory_mb,
            "elapsed_s": round(self.elapsed_s, 3),
        }


# ---------------------------------------------------------------------------
# Grid Search Engine
# ---------------------------------------------------------------------------

_METRIC_FN = {
    "mean_iou": lambda e: e.mean_iou,
    "success_auc": lambda e: e.success_auc,
    "precision_auc": lambda e: e.precision_auc,
    "fps": lambda e: e.fps,
}


class GridSearchEngine:
    """Systematic hyperparameter search for an EOVOT tracker.

    Evaluates all combinations in *param_grid* by instantiating the tracker
    with each combination and running a full benchmark, then reports which
    configuration achieves the best accuracy, throughput, or memory footprint.

    Args:
        tracker_cls: Tracker class that accepts every key in *param_grid* as a
            keyword argument to its constructor.
        param_grid: Mapping of parameter name → candidate values list.
            All combinations (Cartesian product) are evaluated.
        benchmark_kwargs: Extra keyword arguments forwarded verbatim to
            :class:`~eovot.benchmark.engine.BenchmarkEngine`.

    Example::

        engine = GridSearchEngine(
            KCFTracker,
            {
                "learning_rate": [0.05, 0.075, 0.1],
                "lambda_": [1e-4, 1e-3],
                "padding": [1.0, 1.5],
            },
        )
        entries = engine.run(synthetic_dataset, max_sequences=5)
        print(engine.best_config(entries))
    """

    def __init__(
        self,
        tracker_cls: Type[BaseTracker],
        param_grid: Dict[str, List[Any]],
        benchmark_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not param_grid:
            raise ValueError("param_grid must not be empty.")
        self.tracker_cls = tracker_cls
        self.param_grid = param_grid
        self._benchmark_kwargs: Dict[str, Any] = benchmark_kwargs or {}

    # ------------------------------------------------------------------ #
    # Core search                                                          #
    # ------------------------------------------------------------------ #

    def _iter_combinations(self) -> List[Dict[str, Any]]:
        """Return all parameter combinations as flat dicts."""
        keys = list(self.param_grid.keys())
        return [
            dict(zip(keys, combo))
            for combo in itertools.product(*[self.param_grid[k] for k in keys])
        ]

    def run(
        self,
        dataset: BaseDataset,
        dataset_name: str = "dataset",
        max_sequences: Optional[int] = None,
        verbose: bool = True,
    ) -> List[GridSearchEntry]:
        """Evaluate every parameter combination on *dataset*.

        Args:
            dataset: Dataset instance to benchmark on.
            dataset_name: Human-readable label used in reports.
            max_sequences: Cap on sequences per combination (speeds up search).
            verbose: Print a progress line after each combination.

        Returns:
            List of :class:`GridSearchEntry` objects, sorted by mean IoU
            descending (best first).
        """
        combos = self._iter_combinations()
        n = len(combos)
        if verbose:
            print(
                f"[GridSearch] {self.tracker_cls.__name__}: "
                f"{n} combinations × {max_sequences or 'all'} sequences"
            )

        engine = BenchmarkEngine(verbose=False, **self._benchmark_kwargs)
        entries: List[GridSearchEntry] = []

        for idx, params in enumerate(combos, 1):
            tracker = self.tracker_cls(**params)
            t0 = time.perf_counter()
            result = engine.run(tracker, dataset, dataset_name, max_sequences)
            elapsed = time.perf_counter() - t0

            entry = GridSearchEntry(params=params, result=result, elapsed_s=elapsed)
            entries.append(entry)

            if verbose:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                print(
                    f"  [{idx:>{len(str(n))}}/{n}] {param_str}"
                    f"  →  mIoU={entry.mean_iou:.4f}"
                    f"  FPS={entry.fps:6.1f}"
                    f"  ({elapsed:.1f}s)"
                )

        entries.sort(key=lambda e: e.mean_iou, reverse=True)
        return entries

    # ------------------------------------------------------------------ #
    # Analysis helpers                                                     #
    # ------------------------------------------------------------------ #

    def best_config(
        self,
        entries: List[GridSearchEntry],
        metric: str = "mean_iou",
    ) -> Dict[str, Any]:
        """Return the parameter dict of the highest-scoring combination.

        Args:
            entries: Output of :meth:`run`.
            metric: Ranking criterion — one of ``"mean_iou"``,
                ``"success_auc"``, ``"precision_auc"``, ``"fps"``.

        Returns:
            Dict of best hyperparameter values.

        Raises:
            ValueError: If *metric* is not recognised or *entries* is empty.
        """
        if not entries:
            raise ValueError("entries list is empty.")
        if metric not in _METRIC_FN:
            raise ValueError(
                f"Unknown metric '{metric}'. "
                f"Choose from: {list(_METRIC_FN)}"
            )
        return dict(max(entries, key=_METRIC_FN[metric]).params)

    def sensitivity_report(
        self,
        entries: List[GridSearchEntry],
        metric: str = "mean_iou",
    ) -> Dict[str, Dict[Any, float]]:
        """Compute per-parameter marginal sensitivity.

        For each parameter, averages *metric* over all combinations that share
        the same value for that parameter.  A large spread indicates the
        parameter has a strong influence on performance.

        Args:
            entries: Output of :meth:`run`.
            metric: Metric to average.

        Returns:
            ``{param_name: {value: mean_metric, ...}, ...}``
        """
        if metric not in _METRIC_FN:
            raise ValueError(f"Unknown metric '{metric}'.")
        fn = _METRIC_FN[metric]
        report: Dict[str, Dict[Any, float]] = {}

        for key in self.param_grid:
            buckets: Dict[Any, List[float]] = {}
            for entry in entries:
                v = entry.params[key]
                buckets.setdefault(v, []).append(fn(entry))
            report[key] = {v: float(sum(vals) / len(vals)) for v, vals in buckets.items()}

        return report

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def to_markdown(
        self,
        entries: List[GridSearchEntry],
        top_n: Optional[int] = None,
    ) -> str:
        """Format search results as a Markdown table.

        Args:
            entries: Output of :meth:`run` (already sorted by mIoU).
            top_n: Limit table to the top-N rows.  Shows all when ``None``.

        Returns:
            Markdown string ready for GitHub issues, wikis, or papers.
        """
        if not entries:
            return "_No results._\n"

        rows = entries[:top_n] if top_n else entries
        param_keys = list(rows[0].params.keys())

        header = (
            "| Rank | "
            + " | ".join(param_keys)
            + " | mIoU | Success AUC | FPS | Mem (MB) | Time (s) |"
        )
        sep = (
            "|------|"
            + "--------|" * len(param_keys)
            + "------|-------------|------|----------|----------|"
        )

        lines = [header, sep]
        for rank, entry in enumerate(rows, 1):
            vals = " | ".join(str(entry.params[k]) for k in param_keys)
            lines.append(
                f"| {rank} | {vals} | "
                f"{entry.mean_iou:.4f} | "
                f"{entry.success_auc:.4f} | "
                f"{entry.fps:.1f} | "
                f"{entry.peak_memory_mb:.1f} | "
                f"{entry.elapsed_s:.1f} |"
            )

        return "\n".join(lines) + "\n"

    def sensitivity_to_markdown(
        self,
        entries: List[GridSearchEntry],
        metric: str = "mean_iou",
    ) -> str:
        """Format parameter sensitivity as a Markdown section.

        Args:
            entries: Output of :meth:`run`.
            metric: Metric to report sensitivity for.

        Returns:
            Markdown string with one sub-table per hyperparameter.
        """
        report = self.sensitivity_report(entries, metric=metric)
        lines = [f"## Parameter Sensitivity ({metric})\n"]
        for param, mapping in report.items():
            lines.append(f"### `{param}`\n")
            lines.append(f"| Value | Mean {metric} |")
            lines.append("|-------|-------------|")
            for val, mean_m in sorted(mapping.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {val} | {mean_m:.4f} |")
            lines.append("")
        return "\n".join(lines)

    def save_json(
        self,
        entries: List[GridSearchEntry],
        path: str,
    ) -> Path:
        """Save all results to a JSON file.

        Args:
            entries: Output of :meth:`run`.
            path: Destination file path.

        Returns:
            Resolved :class:`~pathlib.Path` of the saved file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tracker": self.tracker_cls.__name__,
            "param_grid": {k: list(v) for k, v in self.param_grid.items()},
            "n_combinations": len(entries),
            "results": [e.to_dict() for e in entries],
        }
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out

    # ------------------------------------------------------------------ #
    # Factory — YAML-driven construction                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GridSearchEngine":
        """Construct a GridSearchEngine from a YAML config dict.

        Expected config shape::

            tracker:
              name: MOSSE          # must match TRACKER_REGISTRY
              param_grid:
                learning_rate: [0.075, 0.10, 0.125, 0.15]
                sigma: [1.5, 2.0, 2.5]

            benchmark:
              tdp_watts: null      # optional energy profiling

        Args:
            config: Parsed YAML dict.

        Returns:
            Configured :class:`GridSearchEngine` instance.
        """
        from ..trackers.kcf import KCFTracker
        from ..trackers.mosse import MOSSETracker
        from ..trackers.median_flow import MedianFlowTracker
        from ..trackers.csrt import CSRTTracker
        from ..trackers.mil import MILTracker

        _REGISTRY = {
            "MOSSE": MOSSETracker,
            "KCF": KCFTracker,
            "MedianFlow": MedianFlowTracker,
            "CSRT": CSRTTracker,
            "MIL": MILTracker,
        }

        tracker_cfg = config["tracker"]
        name = tracker_cfg["name"]
        if name not in _REGISTRY:
            raise ValueError(
                f"Unknown tracker '{name}'. "
                f"Available: {list(_REGISTRY)}"
            )

        param_grid = {k: list(v) for k, v in tracker_cfg.get("param_grid", {}).items()}
        if not param_grid:
            raise ValueError("tracker.param_grid must define at least one parameter.")

        bm_cfg = config.get("benchmark", {})
        benchmark_kwargs: Dict[str, Any] = {}
        if bm_cfg.get("tdp_watts") is not None:
            benchmark_kwargs["tdp_watts"] = float(bm_cfg["tdp_watts"])

        return cls(
            tracker_cls=_REGISTRY[name],
            param_grid=param_grid,
            benchmark_kwargs=benchmark_kwargs,
        )

    def run_from_config(
        self,
        config: Dict[str, Any],
    ) -> List[GridSearchEntry]:
        """Build dataset from config and run the search.

        Expected config shape (dataset section)::

            dataset:
              loader: SyntheticDataset
              params:
                num_sequences: 10
                num_frames: 100
                motion: linear
                seed: 42

        Args:
            config: Parsed YAML dict (must also contain a ``tracker`` section
                matching :meth:`from_config` expectations).

        Returns:
            Sorted :class:`GridSearchEntry` list.
        """
        from ..datasets.synthetic import SyntheticDataset
        from ..datasets.base import OTBDataset
        from ..datasets.got10k import GOT10kDataset
        from ..datasets.lasot import LaSOTDataset

        _DS_REGISTRY = {
            "SyntheticDataset": SyntheticDataset,
            "OTBDataset": OTBDataset,
            "GOT10kDataset": GOT10kDataset,
            "LaSOTDataset": LaSOTDataset,
        }

        ds_cfg = config.get("dataset", {})
        loader_name = ds_cfg.get("loader", "SyntheticDataset")
        if loader_name not in _DS_REGISTRY:
            raise ValueError(f"Unknown dataset loader '{loader_name}'.")

        ds_params = ds_cfg.get("params", {})
        dataset = _DS_REGISTRY[loader_name](**ds_params)
        dataset_name = ds_cfg.get("name", loader_name)

        bm_cfg = config.get("benchmark", {})
        max_seq = bm_cfg.get("max_sequences")
        verbose = bm_cfg.get("verbose", True)

        return self.run(
            dataset=dataset,
            dataset_name=dataset_name,
            max_sequences=max_seq,
            verbose=verbose,
        )
