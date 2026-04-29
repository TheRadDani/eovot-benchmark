"""Config sub-package — typed schema and validation for EOVOT experiments.

Provides a :func:`load_config` / :func:`validate_config` pair that converts
raw YAML dicts into fully typed, validated :class:`EOVOTConfig` objects,
surfacing all configuration errors up-front with clear messages.

Quick start::

    from eovot.config import load_config

    cfg = load_config("configs/default.yaml")
    print(cfg.experiment.name)       # "baseline-mosse-otb"
    print(cfg.tracker.name)          # "MOSSE"
    print(cfg.benchmark.tdp_watts)   # None

Programmatic construction::

    from eovot.config import validate_config

    cfg = validate_config({
        "experiment": {"name": "my-run"},
        "dataset":    {"name": "OTB100", "loader": "OTBDataset", "root": "/data"},
        "tracker":    {"name": "MOSSE"},
    })
"""

from .loader import ConfigValidationError, load_config, validate_config
from .schema import (
    BUILTIN_LOADERS,
    VALID_REPORT_FORMATS,
    BenchmarkConfig,
    DatasetConfig,
    EOVOTConfig,
    ExperimentConfig,
    ReportingConfig,
    TrackerConfig,
)

__all__ = [
    # Functions
    "load_config",
    "validate_config",
    # Exception
    "ConfigValidationError",
    # Schema dataclasses
    "EOVOTConfig",
    "ExperimentConfig",
    "DatasetConfig",
    "TrackerConfig",
    "BenchmarkConfig",
    "ReportingConfig",
    # Constants
    "BUILTIN_LOADERS",
    "VALID_REPORT_FORMATS",
]
