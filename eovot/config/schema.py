"""Typed configuration schema for EOVOT experiments.

Each section of an EOVOT YAML config maps to a frozen-style dataclass that
documents all accepted fields, their types, and their defaults.  Having a
single authoritative schema means scripts, tests, and documentation all agree
on what keys exist without ad-hoc ``config.get("key", default)`` calls.

Typical usage::

    from eovot.config.schema import EOVOTConfig, TrackerConfig

    # Build programmatically
    cfg = EOVOTConfig(
        experiment=ExperimentConfig(name="my-run"),
        dataset=DatasetConfig(name="OTB100", loader="OTBDataset", root="/data/OTB100"),
        tracker=TrackerConfig(name="MOSSE"),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Dataset loader class names that ship with EOVOT.
# Used by the validator to surface helpful errors on typos.
BUILTIN_LOADERS: frozenset = frozenset(
    ["OTBDataset", "GOT10kDataset", "LaSOTDataset", "SyntheticDataset"]
)

# Report format identifiers accepted by BenchmarkReporter.
VALID_REPORT_FORMATS: frozenset = frozenset(["json", "csv", "markdown"])


@dataclass
class ExperimentConfig:
    """Top-level experiment metadata."""

    name: str
    """Unique experiment identifier used in report file names."""

    output_dir: str = "results/"
    """Directory where JSON/CSV/Markdown reports are written."""

    seed: int = 42
    """Global RNG seed for reproducible runs."""


@dataclass
class DatasetConfig:
    """Dataset loading configuration."""

    name: str
    """Human-readable dataset label used in benchmark reports (e.g. ``"OTB100"``)."""

    loader: str
    """Dataset loader class name (e.g. ``"OTBDataset"``, ``"GOT10kDataset"``)."""

    root: str
    """Path to the dataset root directory on disk."""

    max_sequences: Optional[int] = None
    """Evaluate only the first *N* sequences.  ``None`` evaluates all."""

    split: Optional[str] = None
    """Dataset split to load (e.g. ``"train"``, ``"val"``, ``"test"``).
    Only applicable to GOT-10k and LaSOT."""


@dataclass
class TrackerConfig:
    """Tracker instantiation configuration."""

    name: str
    """Tracker class name or display label used in reports."""

    params: Dict[str, Any] = field(default_factory=dict)
    """Key-value hyper-parameters forwarded to the tracker constructor."""


@dataclass
class BenchmarkConfig:
    """Benchmark engine run settings."""

    verbose: bool = True
    """Print per-sequence progress to stdout during evaluation."""

    tdp_watts: Optional[float] = None
    """CPU Thermal Design Power in Watts for energy estimation.  Set to the
    device TDP (e.g. ``6.0`` for Raspberry Pi 4) to enable energy profiling.
    ``None`` disables energy profiling."""

    save_predictions: bool = False
    """Write per-frame predicted bounding boxes to disk alongside reports."""


@dataclass
class ReportingConfig:
    """Result export settings."""

    formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    """List of output formats.  Accepted values: ``"json"``, ``"csv"``, ``"markdown"``."""

    print_summary: bool = True
    """Print a formatted summary table to stdout after evaluation completes."""


@dataclass
class EOVOTConfig:
    """Root configuration object for a complete EOVOT experiment.

    All sub-configs have sensible defaults so callers only need to supply the
    three required string fields: ``experiment.name``, ``dataset.name /
    loader / root``, and ``tracker.name``.

    Example::

        cfg = EOVOTConfig(
            experiment=ExperimentConfig(name="mosse-otb"),
            dataset=DatasetConfig(name="OTB100", loader="OTBDataset", root="/data"),
            tracker=TrackerConfig(name="MOSSE"),
        )
        print(cfg.benchmark.verbose)   # True
        print(cfg.reporting.formats)   # ['json', 'csv']
    """

    experiment: ExperimentConfig
    dataset: DatasetConfig
    tracker: TrackerConfig
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
