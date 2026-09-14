"""CLI tool for EOVOT hyperparameter sweeps.

Evaluates a tracker over all combinations of the parameter grid defined in a
YAML config file and writes a ranked Markdown + CSV report to the output
directory.

Usage::

    python scripts/sweep_tracker.py --config configs/sweeps/kcf_learning_rate.yaml

    python scripts/sweep_tracker.py \\
        --config configs/sweeps/kcf_learning_rate.yaml \\
        --output-dir results/sweeps/kcf \\
        --max-sequences 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EOVOT hyperparameter sweep — grid search over tracker params."
    )
    p.add_argument(
        "--config", required=True, metavar="YAML",
        help="Path to sweep config YAML.",
    )
    p.add_argument(
        "--output-dir", default="results/sweeps", metavar="DIR",
        help="Directory to write Markdown, CSV, and JSON reports.",
    )
    p.add_argument(
        "--max-sequences", type=int, default=None, metavar="N",
        help="Limit evaluation to N sequences per config (fast preview).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-frame progress output.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    tracker_name: str = config["tracker"]["name"]
    base_params: dict = config["tracker"].get("base_params", {}) or {}
    param_grid: dict = config["sweep"]["param_grid"]
    dataset_cfg: dict = config["dataset"]

    dataset = _build_dataset(dataset_cfg)
    dataset_name: str = dataset_cfg.get("name", type(dataset).__name__)
    tdp_watts = config.get("profiling", {}).get("tdp_watts", None)

    from eovot.analysis.hyperparam_sweep import HyperparamSweeper

    sweeper = HyperparamSweeper(
        tracker_name=tracker_name,
        dataset=dataset,
        dataset_name=dataset_name,
        verbose=not args.quiet,
        tdp_watts=tdp_watts,
    )

    report = sweeper.sweep(
        param_grid=param_grid,
        base_params=base_params,
        max_sequences=args.max_sequences,
    )

    out_dir = Path(args.output_dir) / f"{tracker_name}-{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "sweep_report.md"
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    print(f"Markdown report → {md_path}")

    csv_path = out_dir / "sweep_results.csv"
    csv_path.write_text(report.to_csv(), encoding="utf-8")
    print(f"CSV results    → {csv_path}")

    json_path = report.save(out_dir / "sweep_results")
    print(f"JSON results   → {json_path}")

    best = report.best_config()
    if best:
        print(f"\nBest config:  {best.params_label()}")
        print(f"  success_auc = {best.success_auc:.4f}")
        print(f"  FPS         = {best.mean_fps:.1f}")

    return 0


def _build_dataset(cfg: dict):
    """Construct a dataset from a config dict section."""
    loader_name: str = cfg.get("loader", "SyntheticDataset")

    if loader_name == "SyntheticDataset":
        from eovot.datasets.synthetic import SyntheticDataset
        frame_size = tuple(cfg.get("frame_size", [320, 240]))
        return SyntheticDataset(
            num_sequences=cfg.get("num_sequences", 10),
            num_frames=cfg.get("num_frames", 100),
            frame_size=frame_size,
            motion=cfg.get("motion", "linear"),
            seed=cfg.get("seed", 42),
        )

    from eovot.datasets.base import OTBDataset
    from eovot.datasets.got10k import GOT10kDataset
    from eovot.datasets.lasot import LaSOTDataset

    loaders = {
        "OTBDataset": OTBDataset,
        "GOT10kDataset": GOT10kDataset,
        "LaSOTDataset": LaSOTDataset,
    }
    if loader_name not in loaders:
        raise ValueError(f"Unknown loader '{loader_name}'. Available: {list(loaders)}")

    cls = loaders[loader_name]
    root = cfg["root"]
    if loader_name == "OTBDataset":
        return cls(root=root)
    return cls(root=root, split=cfg.get("split", "val"),
               max_sequences=cfg.get("max_sequences", None))


if __name__ == "__main__":
    sys.exit(main())
