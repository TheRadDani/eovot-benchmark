"""Hyperparameter ablation engine for systematic tracker evaluation.

:class:`AblationStudy` sweeps over a Cartesian product of hyperparameter values,
evaluates each configuration with :class:`~eovot.benchmark.engine.BenchmarkEngine`,
and returns a ranked :class:`AblationResult` with per-parameter sensitivity
analysis.

This module is the standard mechanism for justifying hyperparameter choices in
publications — you should be able to state: *"We use learning_rate=0.125 because
our ablation on GOT-10k/val shows it maximises success AUC; the sensitivity
analysis shows ΔAUClearning_rate = 0.031, compared to ΔAUCpadding = 0.007."*

Typical usage::

    from eovot.experiment.ablation import AblationStudy
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, num_frames=100, seed=42)
    study = AblationStudy(
        tracker_cls=KCFTracker,
        base_params={"learning_rate": 0.125, "padding": 1.5},
        param_grid={"learning_rate": [0.05, 0.075, 0.100, 0.125, 0.150]},
        dataset=dataset,
        dataset_name="Synthetic",
    )
    result = study.run()
    print(result.to_markdown_table())
    print(f"Best config: {result.best_config().params}")
    for entry in result.sensitivity_analysis():
        print(f"  {entry.param_name}: impact={entry.impact:.4f}  optimal={entry.optimal_value}")

Config-driven usage (see :func:`run_ablation_from_config`)::

    import yaml
    from eovot.experiment.ablation import run_ablation_from_config

    with open("configs/experiments/ablation_kcf.yaml") as f:
        config = yaml.safe_load(f)
    result_dict = run_ablation_from_config(config, output_dir="results/ablations")
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """One parameter configuration evaluated during an ablation study.

    Attributes:
        params: Full parameter dict passed to the tracker constructor.
        result: Benchmark result for this config; set by :meth:`AblationStudy.run`.
        wall_time_s: Wall-clock seconds taken by this tracker run.
    """

    params: Dict[str, Any]
    result: Optional[BenchmarkResult] = field(default=None, repr=False)
    wall_time_s: float = 0.0

    @property
    def success_auc(self) -> float:
        """Primary ranking metric: success AUC when available, otherwise mIoU."""
        if self.result is None:
            return 0.0
        sauc = getattr(self.result, "mean_success_auc", None)
        return sauc if sauc is not None else self.result.mean_iou

    @property
    def mean_iou(self) -> float:
        return self.result.mean_iou if self.result is not None else 0.0

    @property
    def mean_fps(self) -> float:
        return self.result.mean_fps if self.result is not None else 0.0

    @property
    def peak_memory_mb(self) -> float:
        return self.result.peak_memory_mb if self.result is not None else 0.0


@dataclass
class SensitivityEntry:
    """Per-parameter sensitivity report from a one-at-a-time sweep.

    Attributes:
        param_name: Tracker hyperparameter that was swept.
        values_tested: Ordered list of values evaluated (others held at base).
        success_aucs: Corresponding success AUC for each tested value.
        impact: ``max(success_aucs) − min(success_aucs)`` — proxy for how
            much this parameter affects tracking accuracy.
        optimal_value: Value that achieved the highest success AUC.
    """

    param_name: str
    values_tested: List[Any]
    success_aucs: List[float]
    impact: float
    optimal_value: Any

    def __str__(self) -> str:
        return (
            f"SensitivityEntry({self.param_name}  "
            f"impact={self.impact:.4f}  "
            f"optimal={self.optimal_value})"
        )


# ---------------------------------------------------------------------------
# AblationResult
# ---------------------------------------------------------------------------


class AblationResult:
    """Results from a completed hyperparameter ablation study.

    Attributes:
        tracker_name: Display name of the tracker under study.
        dataset_name: Dataset used for evaluation.
        configs: All evaluated :class:`AblationConfig` objects, sorted by
            success AUC descending (best first).
        base_params: The base parameter dict used when constructing the study.
    """

    def __init__(
        self,
        tracker_name: str,
        dataset_name: str,
        configs: List[AblationConfig],
        base_params: Dict[str, Any],
    ) -> None:
        self.tracker_name = tracker_name
        self.dataset_name = dataset_name
        self.configs = sorted(configs, key=lambda c: c.success_auc, reverse=True)
        self.base_params = base_params

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def best_config(self) -> AblationConfig:
        """Return the configuration with the highest success AUC."""
        if not self.configs:
            raise ValueError("No configs available.")
        return self.configs[0]

    def sensitivity_analysis(self) -> List[SensitivityEntry]:
        """One-at-a-time (OAT) sensitivity analysis across ablated parameters.

        For each swept parameter, this method filters the evaluated configs to
        those where **all other ablated parameters are held at their base values**
        (or the single value present if the base is not in the grid).  It then
        records the success AUC for each value of the swept parameter and
        computes the impact as ``max − min`` AUC.

        Returns:
            List of :class:`SensitivityEntry`, one per ablated parameter, sorted
            by impact descending (most influential parameter first).
        """
        # Identify which parameters were actually varied in the study
        ablated_params: List[str] = []
        if self.configs:
            first_params = self.configs[0].params
            ablated_params = [k for k in first_params if k in self.base_params or True]
            # Keep only parameters whose values differ across configs
            ablated_params = [
                k for k in first_params
                if len({cfg.params.get(k) for cfg in self.configs}) > 1
            ]

        entries: List[SensitivityEntry] = []
        for param in ablated_params:
            other_ablated = [p for p in ablated_params if p != param]

            # For each config, check whether all other ablated params match base
            sweep_configs = []
            for cfg in self.configs:
                other_match = all(
                    cfg.params.get(other_p) == self.base_params.get(other_p)
                    for other_p in other_ablated
                )
                if other_match:
                    sweep_configs.append(cfg)

            if not sweep_configs:
                continue

            values = [cfg.params.get(param) for cfg in sweep_configs]
            aucs = [cfg.success_auc for cfg in sweep_configs]
            impact = float(max(aucs) - min(aucs)) if len(aucs) > 1 else 0.0
            best_idx = int(np.argmax(aucs))
            entries.append(
                SensitivityEntry(
                    param_name=param,
                    values_tested=values,
                    success_aucs=aucs,
                    impact=impact,
                    optimal_value=values[best_idx],
                )
            )

        entries.sort(key=lambda e: e.impact, reverse=True)
        return entries

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_table(self) -> str:
        """Format the ranked ablation results as a Markdown table.

        Returns:
            Multi-line Markdown string ready to embed in papers or reports.
        """
        if not self.configs:
            return f"## Ablation: {self.tracker_name} — no results\n"

        param_names = sorted(self.configs[0].params.keys())
        header_params = " | ".join(param_names)
        sep_params = " | ".join(["---:"] * len(param_names))

        lines = [
            f"## Ablation Study: {self.tracker_name} on {self.dataset_name}\n",
            f"| Rank | {header_params} | Success AUC | mIoU | FPS | Mem (MB) |",
            f"|------|{sep_params}|------------:|-----:|----:|---------:|",
        ]
        for rank, cfg in enumerate(self.configs, start=1):
            param_vals = " | ".join(str(cfg.params.get(p, "—")) for p in param_names)
            lines.append(
                f"| {rank} | {param_vals} "
                f"| {cfg.success_auc:.4f} "
                f"| {cfg.mean_iou:.4f} "
                f"| {cfg.mean_fps:.1f} "
                f"| {cfg.peak_memory_mb:.1f} |"
            )
        return "\n".join(lines)

    def sensitivity_to_markdown(self) -> str:
        """Format the sensitivity analysis as a Markdown table.

        Returns:
            Multi-line Markdown string, or a note if no sweep data is available.
        """
        entries = self.sensitivity_analysis()
        if not entries:
            return "No single-parameter sweep data available for sensitivity analysis.\n"

        lines = [
            f"## Sensitivity Analysis: {self.tracker_name} on {self.dataset_name}\n",
            "| Rank | Parameter | Impact (ΔAUC) | Optimal Value | Values Tested |",
            "|------|-----------|-------------:|---------------|---------------|",
        ]
        for rank, e in enumerate(entries, start=1):
            values_str = ", ".join(str(v) for v in e.values_tested)
            lines.append(
                f"| {rank} | {e.param_name} "
                f"| {e.impact:.4f} "
                f"| {e.optimal_value} "
                f"| {values_str} |"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full ablation result to a JSON-compatible dict."""
        sensitivity = [
            {
                "param": e.param_name,
                "impact": round(e.impact, 6),
                "optimal_value": e.optimal_value,
                "values_tested": e.values_tested,
                "success_aucs": [round(a, 6) for a in e.success_aucs],
            }
            for e in self.sensitivity_analysis()
        ]
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "num_configs": len(self.configs),
            "best_config": {
                "params": self.best_config().params,
                "success_auc": round(self.best_config().success_auc, 6),
                "mean_iou": round(self.best_config().mean_iou, 6),
                "mean_fps": round(self.best_config().mean_fps, 2),
                "peak_memory_mb": round(self.best_config().peak_memory_mb, 2),
            },
            "sensitivity": sensitivity,
            "all_configs": [
                {
                    "params": cfg.params,
                    "success_auc": round(cfg.success_auc, 6),
                    "mean_iou": round(cfg.mean_iou, 6),
                    "mean_fps": round(cfg.mean_fps, 2),
                    "peak_memory_mb": round(cfg.peak_memory_mb, 2),
                    "wall_time_s": round(cfg.wall_time_s, 3),
                }
                for cfg in self.configs
            ],
        }


# ---------------------------------------------------------------------------
# AblationStudy
# ---------------------------------------------------------------------------


class AblationStudy:
    """Systematic hyperparameter grid search over a single tracker class.

    Each element of the Cartesian product of ``param_grid`` values is evaluated
    via :class:`~eovot.benchmark.engine.BenchmarkEngine` and ranked by success
    AUC.  The resulting :class:`AblationResult` also provides sensitivity
    analysis through one-at-a-time sweeps.

    Args:
        tracker_cls: Tracker class to instantiate for each config.  Must accept
            all keys in ``base_params`` and ``param_grid`` as keyword arguments.
        base_params: Default parameter values.  Grid overrides are merged on top,
            so parameters not in ``param_grid`` remain at their base values.
        param_grid: Mapping of parameter names to lists of candidate values.
            The full Cartesian product of all lists is evaluated.
        dataset: Dataset to evaluate against (iterated once per config).
        dataset_name: Human-readable label used in reports.
        verbose: Print per-configuration progress to stdout.  Default ``True``.
        tdp_watts: If set, enables CPU energy profiling.
        max_sequences: Limit evaluation to this many sequences per config.
            Useful for rapid sweeps before committing to full-dataset runs.

    Raises:
        ValueError: If ``param_grid`` is empty.

    Example::

        from eovot.experiment.ablation import AblationStudy
        from eovot.trackers.mosse import MOSSETracker
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=4, num_frames=60, seed=0)
        study = AblationStudy(
            tracker_cls=MOSSETracker,
            base_params={"learning_rate": 0.125, "sigma": 2.0},
            param_grid={"learning_rate": [0.05, 0.10, 0.125, 0.175, 0.25]},
            dataset=dataset,
            dataset_name="Synthetic",
        )
        result = study.run()
        print(result.to_markdown_table())
        print(result.sensitivity_to_markdown())
    """

    def __init__(
        self,
        tracker_cls: Type[BaseTracker],
        base_params: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
        max_sequences: Optional[int] = None,
    ) -> None:
        if not param_grid:
            raise ValueError("param_grid must contain at least one parameter to sweep.")
        self._tracker_cls = tracker_cls
        self._base_params = dict(base_params)
        self._param_grid = param_grid
        self._dataset = dataset
        self._dataset_name = dataset_name
        self._verbose = verbose
        self._engine = BenchmarkEngine(verbose=False, tdp_watts=tdp_watts)
        self._max_sequences = max_sequences

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> AblationResult:
        """Execute the full ablation sweep and return ranked results.

        Each config is evaluated by re-initialising the tracker from scratch
        on every sequence in the dataset.  The dataset is iterated once per
        configuration.

        Returns:
            :class:`AblationResult` with all configs sorted by success AUC
            (best first) and sensitivity analysis data attached.
        """
        tracker_display = self._tracker_cls.__name__.replace("Tracker", "")
        param_combos = self._generate_configs()
        n = len(param_combos)

        if self._verbose:
            print(f"\nAblation Study: {tracker_display} on {self._dataset_name}")
            print(
                f"Parameters: {list(self._param_grid.keys())}  "
                f"Configs: {n}  Sequences: "
                f"{self._max_sequences or len(self._dataset)}"
            )
            print("-" * 60)

        ablation_configs: List[AblationConfig] = []

        for i, params in enumerate(param_combos):
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            if self._verbose:
                print(f"[{i + 1:>3}/{n}] {param_str}")

            try:
                tracker = self._tracker_cls(**params)
            except TypeError as exc:
                raise ValueError(
                    f"Cannot instantiate {self._tracker_cls.__name__} "
                    f"with params {params}: {exc}"
                ) from exc

            t0 = time.perf_counter()
            bench_result = self._engine.run(
                tracker=tracker,
                dataset=self._dataset,
                dataset_name=self._dataset_name,
                max_sequences=self._max_sequences,
            )
            elapsed = time.perf_counter() - t0

            cfg = AblationConfig(params=params, result=bench_result, wall_time_s=elapsed)
            ablation_configs.append(cfg)

            if self._verbose:
                print(
                    f"         success_auc={cfg.success_auc:.4f}  "
                    f"mIoU={cfg.mean_iou:.4f}  "
                    f"FPS={cfg.mean_fps:.1f}  "
                    f"({elapsed:.1f}s)"
                )

        if self._verbose:
            best = max(ablation_configs, key=lambda c: c.success_auc)
            best_str = ", ".join(f"{k}={v}" for k, v in best.params.items())
            print("-" * 60)
            print(f"Best: {best_str}  (success_auc={best.success_auc:.4f})")

        return AblationResult(
            tracker_name=tracker_display,
            dataset_name=self._dataset_name,
            configs=ablation_configs,
            base_params=self._base_params,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_configs(self) -> List[Dict[str, Any]]:
        """Return the Cartesian product of param_grid values merged with base_params."""
        param_names = list(self._param_grid.keys())
        param_values = list(self._param_grid.values())
        configs: List[Dict[str, Any]] = []
        for combo in itertools.product(*param_values):
            cfg = dict(self._base_params)
            cfg.update(dict(zip(param_names, combo)))
            configs.append(cfg)
        return configs


# ---------------------------------------------------------------------------
# Config-driven entry point
# ---------------------------------------------------------------------------

#: Registry mapping config tracker names to tracker classes.
_TRACKER_REGISTRY: Dict[str, Type[BaseTracker]] = {}


def _populate_registry() -> None:
    """Lazily import and register all built-in tracker classes."""
    if _TRACKER_REGISTRY:
        return
    try:
        from ..trackers.mosse import MOSSETracker
        _TRACKER_REGISTRY["MOSSE"] = MOSSETracker  # type: ignore[assignment]
    except Exception:
        pass
    try:
        from ..trackers.kcf import KCFTracker
        _TRACKER_REGISTRY["KCF"] = KCFTracker  # type: ignore[assignment]
    except Exception:
        pass
    try:
        from ..trackers.csrt import CSRTTracker
        _TRACKER_REGISTRY["CSRT"] = CSRTTracker  # type: ignore[assignment]
    except Exception:
        pass
    try:
        from ..trackers.mil import MILTracker
        _TRACKER_REGISTRY["MIL"] = MILTracker  # type: ignore[assignment]
    except Exception:
        pass
    try:
        from ..trackers.median_flow import MedianFlowTracker
        _TRACKER_REGISTRY["MedianFlow"] = MedianFlowTracker  # type: ignore[assignment]
    except Exception:
        pass


def run_ablation_from_config(
    config: Dict[str, Any],
    output_dir: str = "results/ablations",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute an ablation study from a config dict and persist results.

    The config schema::

        experiment:
          name: "kcf-learning-rate-ablation"
          seed: 42
          tdp_watts: null

        dataset:
          loader: SyntheticDataset
          name: Synthetic
          num_sequences: 8
          num_frames: 100
          motion: linear
          seed: 42

        ablation:
          tracker: KCF
          base_params:
            learning_rate: 0.125
            padding: 1.5
          grid:
            learning_rate: [0.025, 0.05, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200]
          max_sequences: null

    Saved outputs (under ``output_dir/<experiment.name>/``):

    * ``ablation_results.json`` — full serialised :class:`AblationResult`
    * ``ablation_table.md`` — ranked configuration table
    * ``sensitivity_analysis.md`` — per-parameter OAT sensitivity report

    Args:
        config: Nested dict matching the schema above.
        output_dir: Root directory for output files.
        verbose: Print per-config progress.

    Returns:
        Serialised :class:`AblationResult` dict.
    """
    # Defer import to avoid circular dependencies
    from ..experiment.runner import ExperimentRunner  # noqa: PLC0415

    _populate_registry()

    exp_cfg = config.get("experiment", {})
    exp_name = exp_cfg.get("name", "ablation")
    tdp_watts = exp_cfg.get("tdp_watts", None)

    abl_cfg = config.get("ablation", {})
    tracker_name = abl_cfg.get("tracker")
    if tracker_name not in _TRACKER_REGISTRY:
        raise ValueError(
            f"Unknown tracker '{tracker_name}'. Available: {sorted(_TRACKER_REGISTRY)}"
        )
    tracker_cls = _TRACKER_REGISTRY[tracker_name]
    base_params: Dict[str, Any] = dict(abl_cfg.get("base_params", {}) or {})
    param_grid: Dict[str, List[Any]] = dict(abl_cfg.get("grid", {}) or {})
    max_sequences = abl_cfg.get("max_sequences", None)

    dataset_cfg = config.get("dataset", {})
    dataset = ExperimentRunner._build_dataset(dataset_cfg)
    dataset_name: str = dataset_cfg.get("name", dataset_cfg.get("loader", "dataset"))

    study = AblationStudy(
        tracker_cls=tracker_cls,
        base_params=base_params,
        param_grid=param_grid,
        dataset=dataset,
        dataset_name=dataset_name,
        verbose=verbose,
        tdp_watts=tdp_watts,
        max_sequences=max_sequences,
    )
    result = study.run()

    out_dir = Path(output_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result_dict = result.to_dict()
    (out_dir / "ablation_results.json").write_text(
        json.dumps(result_dict, indent=2), encoding="utf-8"
    )
    (out_dir / "ablation_table.md").write_text(
        result.to_markdown_table(), encoding="utf-8"
    )
    (out_dir / "sensitivity_analysis.md").write_text(
        result.sensitivity_to_markdown(), encoding="utf-8"
    )

    if verbose:
        print(f"\nResults saved to {out_dir}/")
        print(result.sensitivity_to_markdown())

    return result_dict
