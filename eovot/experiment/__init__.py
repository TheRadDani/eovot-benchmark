"""Experiment configuration and reproducibility system for EOVOT.

Provides:

- :class:`~eovot.experiment.config.ExperimentConfig` — declarative config
  loaded from YAML, with tracker / dataset / benchmark settings.
- :class:`~eovot.experiment.runner.ExperimentRunner` — seeds the RNG,
  builds tracker + dataset objects, drives :class:`BenchmarkEngine`, and
  saves reports.

Typical usage::

    from eovot.experiment import ExperimentConfig, ExperimentRunner

    cfg = ExperimentConfig.from_yaml("configs/comparison_experiment.yaml")
    runner = ExperimentRunner(cfg)
    results = runner.run()
"""

from .config import DatasetConfig, ExperimentConfig, TrackerConfig
from .runner import ExperimentRunner

__all__ = [
    "DatasetConfig",
    "ExperimentConfig",
    "TrackerConfig",
    "ExperimentRunner",
]
