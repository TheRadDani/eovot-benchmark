"""Experiment runner for EOVOT reproducible benchmarking.

:class:`ExperimentRunner` ties together the config system, tracker registry,
dataset registry, benchmark engine, and reporter into a single callable
that produces fully reproducible results.

Key responsibilities:

1. **Seed management** — sets ``random``, ``numpy``, and (if available)
   ``torch`` seeds before each tracker is initialised.
2. **Object construction** — builds tracker and dataset objects from the
   config registry, forwarding constructor kwargs from the config.
3. **Benchmarking** — drives :class:`~eovot.benchmark.engine.BenchmarkEngine`
   for each (tracker, dataset) pair.
4. **Reporting** — saves JSON / CSV results and a Markdown comparison table.

Example::

    from eovot.experiment import ExperimentConfig, ExperimentRunner

    cfg = ExperimentConfig.from_yaml("configs/comparison_experiment.yaml")
    runner = ExperimentRunner(cfg)
    results = runner.run()
    # results is a list of BenchmarkResult.to_dict() dicts
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset, OTBDataset
from ..datasets.got10k import GOT10kDataset
from ..datasets.lasot import LaSOTDataset
from ..reporting.reporter import BenchmarkReporter
from ..trackers.base import BaseTracker
from ..trackers.kcf import KCFTracker
from ..trackers.mosse import MOSSETracker
from .config import DatasetConfig, ExperimentConfig, TrackerConfig

# ---------------------------------------------------------------------------
# Registries — extend these dicts to add new trackers / datasets
# ---------------------------------------------------------------------------

TRACKER_REGISTRY: Dict[str, type] = {
    "MOSSE": MOSSETracker,
    "KCF": KCFTracker,
}

DATASET_REGISTRY: Dict[str, type] = {
    "OTBDataset": OTBDataset,
    "GOT10kDataset": GOT10kDataset,
    "LaSOTDataset": LaSOTDataset,
}


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy``, and optionally ``torch`` for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch not installed — that's fine for classical trackers


def build_tracker(cfg: TrackerConfig) -> BaseTracker:
    """Instantiate a tracker from its :class:`~eovot.experiment.config.TrackerConfig`.

    Args:
        cfg: Tracker configuration.

    Returns:
        A :class:`~eovot.trackers.base.BaseTracker` instance.

    Raises:
        ValueError: If ``cfg.name`` is not in :data:`TRACKER_REGISTRY`.
    """
    cls = TRACKER_REGISTRY.get(cfg.name)
    if cls is None:
        available = list(TRACKER_REGISTRY)
        raise ValueError(
            f"Unknown tracker '{cfg.name}'. "
            f"Available: {available}. "
            "Register new trackers in eovot.experiment.runner.TRACKER_REGISTRY."
        )
    return cls(**cfg.params)


def build_dataset(cfg: DatasetConfig) -> BaseDataset:
    """Instantiate a dataset from its :class:`~eovot.experiment.config.DatasetConfig`.

    Args:
        cfg: Dataset configuration.

    Returns:
        A :class:`~eovot.datasets.base.BaseDataset` instance.

    Raises:
        ValueError: If ``cfg.loader`` is not in :data:`DATASET_REGISTRY`.
    """
    cls = DATASET_REGISTRY.get(cfg.loader)
    if cls is None:
        available = list(DATASET_REGISTRY)
        raise ValueError(
            f"Unknown dataset loader '{cfg.loader}'. "
            f"Available: {available}. "
            "Register new datasets in eovot.experiment.runner.DATASET_REGISTRY."
        )
    kwargs: Dict[str, Any] = {"root": cfg.root}
    if cfg.loader in ("GOT10kDataset", "LaSOTDataset"):
        kwargs["split"] = cfg.split
    if cfg.max_sequences is not None:
        kwargs["max_sequences"] = cfg.max_sequences
    return cls(**kwargs)


class ExperimentRunner:
    """Drive a full EOVOT benchmark experiment from an :class:`ExperimentConfig`.

    Each (tracker × dataset) pair is evaluated independently.  The RNG is
    re-seeded before every tracker instantiation so that results are
    reproducible regardless of evaluation order.

    Args:
        config: Fully populated :class:`ExperimentConfig`.

    Example::

        cfg = ExperimentConfig.from_yaml("configs/comparison_experiment.yaml")
        runner = ExperimentRunner(cfg)
        all_results = runner.run()
        # Markdown comparison table printed to stdout
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._reporter = BenchmarkReporter(output_dir=config.output_dir)
        self._engine = BenchmarkEngine(verbose=config.verbose)

    def run(self) -> List[Dict[str, Any]]:
        """Run all (tracker × dataset) combinations and return result dicts.

        For each combination:

        1. Seeds the RNG with ``config.seed``.
        2. Constructs the tracker (with optional constructor kwargs).
        3. Constructs the dataset.
        4. Runs :class:`~eovot.benchmark.engine.BenchmarkEngine`.
        5. Saves JSON + CSV results.

        After all combinations, saves a Markdown comparison table and a
        consolidated ``summary.json`` under ``config.output_dir``.

        Returns:
            List of ``BenchmarkResult.to_dict()`` dicts, one per combination.
        """
        cfg = self.config
        all_results: List[Dict[str, Any]] = []

        print(f"\n{'=' * 70}")
        print(f"  EOVOT Experiment: {cfg.name}")
        print(f"  Seed: {cfg.seed}   Output: {cfg.output_dir}")
        print(f"  Trackers : {[t.name for t in cfg.trackers]}")
        print(f"  Datasets : {[d.label for d in cfg.datasets]}")
        print(f"{'=' * 70}\n")

        for dataset_cfg in cfg.datasets:
            dataset = build_dataset(dataset_cfg)
            max_seq = dataset_cfg.max_sequences or cfg.max_sequences

            for tracker_cfg in cfg.trackers:
                seed_everything(cfg.seed)
                tracker = build_tracker(tracker_cfg)

                bench_result: BenchmarkResult = self._engine.run(
                    tracker,
                    dataset,
                    dataset_name=dataset_cfg.label,
                    max_sequences=max_seq,
                )
                result_dict = bench_result.to_dict()
                result_dict["experiment"] = {
                    "name": cfg.name,
                    "seed": cfg.seed,
                    "tracker": tracker_cfg.name,
                    "dataset": dataset_cfg.label,
                }

                run_name = f"{cfg.name}-{tracker_cfg.name}-{dataset_cfg.label}"
                saved = self._reporter.save_all(result_dict, name=run_name)
                for fmt, path in saved.items():
                    print(f"  [{fmt.upper()}] {path}")

                all_results.append(result_dict)

        self._save_summary(all_results)

        if len(all_results) > 1:
            cmp_path = self._reporter.save_comparison(
                all_results, name=f"{cfg.name}-comparison"
            )
            print(f"\n[COMPARISON TABLE] → {cmp_path}")
            print("\n" + self._reporter.comparison_table(all_results))

        return all_results

    def _save_summary(self, results: List[Dict[str, Any]]) -> Path:
        """Write a consolidated ``summary.json`` with all results + metadata."""
        summary = {
            "experiment": self.config.name,
            "seed": self.config.seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "runs": [r.get("summary", {}) for r in results],
        }
        path = Path(self.config.output_dir) / f"{self.config.name}-summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"  [SUMMARY JSON] → {path}")
        return path
