"""Declarative experiment configuration for EOVOT.

Experiment configs are stored as YAML files and loaded into typed
dataclasses via :meth:`ExperimentConfig.from_yaml`.  This gives you:

* **Reproducibility** — seed is recorded alongside results.
* **Portability** — configs are plain text, version-controllable.
* **Composability** — one config may list several trackers and datasets
  for a full comparison run.

YAML schema::

    experiment:
      name: "mosse-vs-kcf-otb100"
      output_dir: "results/"
      seed: 42

    benchmark:
      verbose: true
      max_sequences: null      # null = all sequences

    trackers:
      - name: MOSSE
        params:
          learning_rate: 0.125
          sigma: 2.0
      - name: KCF

    datasets:
      - loader: OTBDataset
        root: /data/OTB100
        label: OTB100
      - loader: GOT10kDataset
        root: /data/GOT-10k
        split: val
        label: GOT-10k-val
        max_sequences: 50
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TrackerConfig:
    """Configuration for a single tracker.

    Attributes:
        name: Registry key used to look up the tracker class
            (e.g. ``"MOSSE"``, ``"KCF"``).
        params: Optional keyword arguments forwarded to the tracker
            constructor (e.g. ``{"learning_rate": 0.125}``).
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrackerConfig":
        return cls(name=d["name"], params=d.get("params") or {})


@dataclass
class DatasetConfig:
    """Configuration for a single dataset.

    Attributes:
        loader: Loader class name (``"OTBDataset"`` or ``"GOT10kDataset"``
            or ``"LaSOTDataset"``).
        root: Path to the dataset root directory.
        label: Human-readable label used in reports.  Defaults to *loader*.
        split: Dataset split (GOT-10k / LaSOT only).  Default: ``"val"``.
        max_sequences: Cap on sequence count.  ``None`` means all sequences.
    """

    loader: str
    root: str
    label: str = ""
    split: str = "val"
    max_sequences: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.loader

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetConfig":
        return cls(
            loader=d["loader"],
            root=d["root"],
            label=d.get("label") or d.get("name") or d["loader"],
            split=d.get("split", "val"),
            max_sequences=d.get("max_sequences"),
        )


@dataclass
class ExperimentConfig:
    """Full experiment configuration loaded from a YAML file.

    Attributes:
        name: Unique identifier for this experiment run.
        output_dir: Directory where reports (JSON, CSV, Markdown) are saved.
        seed: Integer RNG seed for reproducibility.
        verbose: Print per-sequence progress during evaluation.
        max_sequences: Global cap on sequence count (can be overridden per
            dataset via :attr:`DatasetConfig.max_sequences`).
        trackers: List of tracker configurations to evaluate.
        datasets: List of dataset configurations to evaluate on.

    Example::

        cfg = ExperimentConfig.from_yaml("configs/comparison_experiment.yaml")
        print(cfg.name, cfg.seed)
        for t in cfg.trackers:
            print(t.name, t.params)
    """

    name: str = "unnamed-experiment"
    output_dir: str = "results/"
    seed: int = 42
    verbose: bool = True
    max_sequences: Optional[int] = None
    trackers: List[TrackerConfig] = field(default_factory=list)
    datasets: List[DatasetConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load an :class:`ExperimentConfig` from a YAML file.

        Args:
            path: Path to the ``.yaml`` / ``.yml`` config file.

        Returns:
            A fully populated :class:`ExperimentConfig`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If required YAML keys are missing.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")

        with open(p) as fh:
            raw = yaml.safe_load(fh)

        exp = raw.get("experiment", {})
        bench = raw.get("benchmark", {})

        trackers_raw = raw.get("trackers", [])
        datasets_raw = raw.get("datasets", [])

        # Support legacy single-tracker / single-dataset format
        if not trackers_raw and "tracker" in raw:
            trackers_raw = [raw["tracker"]]
        if not datasets_raw and "dataset" in raw:
            datasets_raw = [raw["dataset"]]

        return cls(
            name=exp.get("name", "unnamed-experiment"),
            output_dir=exp.get("output_dir", "results/"),
            seed=exp.get("seed", 42),
            verbose=bench.get("verbose", True),
            max_sequences=bench.get("max_sequences"),
            trackers=[TrackerConfig.from_dict(t) for t in trackers_raw],
            datasets=[DatasetConfig.from_dict(d) for d in datasets_raw],
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Construct from a plain dictionary (useful for programmatic use).

        Args:
            d: Dictionary with the same structure as the YAML schema.

        Returns:
            A populated :class:`ExperimentConfig`.
        """
        exp = d.get("experiment", {})
        bench = d.get("benchmark", {})
        return cls(
            name=exp.get("name", "unnamed-experiment"),
            output_dir=exp.get("output_dir", "results/"),
            seed=exp.get("seed", 42),
            verbose=bench.get("verbose", True),
            max_sequences=bench.get("max_sequences"),
            trackers=[TrackerConfig.from_dict(t) for t in d.get("trackers", [])],
            datasets=[DatasetConfig.from_dict(ds) for ds in d.get("datasets", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise back to the YAML-compatible dict format."""
        return {
            "experiment": {
                "name": self.name,
                "output_dir": self.output_dir,
                "seed": self.seed,
            },
            "benchmark": {
                "verbose": self.verbose,
                "max_sequences": self.max_sequences,
            },
            "trackers": [{"name": t.name, "params": t.params} for t in self.trackers],
            "datasets": [
                {
                    "loader": d.loader,
                    "root": d.root,
                    "label": d.label,
                    "split": d.split,
                    "max_sequences": d.max_sequences,
                }
                for d in self.datasets
            ],
        }
