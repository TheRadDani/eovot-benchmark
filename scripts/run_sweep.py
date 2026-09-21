#!/usr/bin/env python3
"""CLI for running EOVOT hyperparameter sweeps from YAML configs.

Example
-------
Run a MOSSE learning-rate/sigma sweep on the synthetic dataset::

    python scripts/run_sweep.py configs/experiments/mosse_sweep.yaml \\
        --output-dir results/sweeps/mosse --verbose

Run with a real OTB dataset::

    python scripts/run_sweep.py configs/experiments/kcf_sweep.yaml \\
        --dataset-root /data/OTB100 --output-dir results/sweeps/kcf

Rank by FPS instead of accuracy::

    python scripts/run_sweep.py configs/experiments/mosse_sweep.yaml \\
        --primary-metric mean_fps --output-dir results/sweeps/mosse-fps

Resume an interrupted sweep::

    python scripts/run_sweep.py configs/experiments/mosse_sweep.yaml \\
        --output-dir results/sweeps/mosse --resume

Config schema::

    sweep:
      tracker: MOSSE              # tracker name (required)
      primary_metric: mean_iou    # metric to rank by (optional, default: mean_iou)
      tdp_watts: null             # float or null; enables energy profiling
      output_dir: null            # overridden by --output-dir

    param_grid:
      learning_rate: [0.05, 0.10, 0.125, 0.15, 0.20, 0.25]
      sigma: [1.0, 2.0, 3.0]

    dataset:
      loader: SyntheticDataset    # SyntheticDataset | OTBDataset | GOT10kDataset | LaSOTDataset
      name: synthetic-linear      # human-readable label in results
      # SyntheticDataset params:
      num_sequences: 5
      num_frames: 60
      frame_size: [320, 240]
      motion: linear
      seed: 42
      # Real dataset params (replace above with):
      # root: /data/OTB100
      # split: val                 # GOT10k / LaSOT only
      # max_sequences: 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def _load_yaml(path: str) -> dict:
    if yaml is None:
        raise ImportError(
            "pyyaml is required to load sweep configs: pip install pyyaml"
        )
    with open(path) as fh:
        return yaml.safe_load(fh)


def _build_dataset(cfg: dict, dataset_root_override: str | None = None):
    """Instantiate the dataset from config, optionally overriding the root."""
    from eovot.experiment.runner import ExperimentRunner

    dataset_cfg = dict(cfg.get("dataset", {}))
    if dataset_root_override:
        dataset_cfg["root"] = dataset_root_override
    if "loader" not in dataset_cfg:
        dataset_cfg["loader"] = "SyntheticDataset"
    return ExperimentRunner._build_dataset(dataset_cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="EOVOT hyperparameter sweep CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", help="Path to sweep YAML config file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for saving per-combination results and summary files",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Path to dataset root directory (overrides config; for OTB/GOT/LaSOT)",
    )
    parser.add_argument(
        "--primary-metric",
        default=None,
        choices=["mean_iou", "success_auc", "precision_auc", "mean_fps"],
        help="Metric used to rank combinations (overrides config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip combinations whose combo_NNNN.json result already exists",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print per-combination progress (default: on)",
    )
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Suppress per-combination progress output",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        default=None,
        metavar="PATH",
        help="Also write sweep_summary.json to this explicit path",
    )
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)

    sweep_cfg = cfg.get("sweep", {})
    tracker_name = sweep_cfg.get("tracker")
    if not tracker_name:
        parser.error("sweep.tracker must be specified in the config file")

    primary_metric = args.primary_metric or sweep_cfg.get("primary_metric", "mean_iou")
    tdp_watts = sweep_cfg.get("tdp_watts", None)
    output_dir = args.output_dir or sweep_cfg.get("output_dir", None)
    param_grid = cfg.get("param_grid", {})

    for k, v in param_grid.items():
        if not isinstance(v, list):
            parser.error(
                f"param_grid.{k} must be a list, got {type(v).__name__}. "
                "Wrap single values in brackets: [{v}]"
            )

    from eovot.experiment.sweep import HyperparamSweep

    sweep = HyperparamSweep(
        tracker_name=tracker_name,
        param_grid=param_grid,
        primary_metric=primary_metric,
        output_dir=output_dir,
        verbose=args.verbose,
        tdp_watts=tdp_watts,
        resume=args.resume,
    )

    dataset_cfg = cfg.get("dataset", {})
    dataset_name = dataset_cfg.get("name") or dataset_cfg.get("loader", "dataset")
    dataset = _build_dataset(cfg, dataset_root_override=args.dataset_root)

    result = sweep.run(dataset, dataset_name=dataset_name)

    print("\n" + result.to_markdown())

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\nSummary JSON written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
