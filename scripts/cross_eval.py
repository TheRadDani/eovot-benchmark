#!/usr/bin/env python3
"""Cross-dataset evaluation CLI for EOVOT.

Evaluates one or more trackers across multiple datasets and exports a ranked
leaderboard with Aggregate Performance Scores (APS).

Usage::

    # Quick smoke test with built-in synthetic dataset:
    python scripts/cross_eval.py \\
        --trackers MOSSE KCF \\
        --synthetic --synthetic-sequences 5 --synthetic-frames 50

    # Evaluate on two OTB-style dataset roots:
    python scripts/cross_eval.py \\
        --trackers MOSSE KCF CSRT \\
        --datasets OTB50:/data/OTB50 OTB100:/data/OTB100 \\
        --output-dir results/cross_eval/

    # YAML config with custom weights and resume support:
    python scripts/cross_eval.py \\
        --config configs/experiments/cross_dataset.yaml

    # With energy profiling (laptop TDP):
    python scripts/cross_eval.py \\
        --trackers MOSSE KCF \\
        --synthetic --tdp-watts 15.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from eovot.datasets.base import OTBDataset
from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.lasot import LaSOTDataset
from eovot.datasets.synthetic import SyntheticDataset
from eovot.experiment.cross_eval import CrossDatasetEvaluator
from eovot.trackers.registry import build_tracker


DATASET_LOADERS = {
    "OTBDataset": OTBDataset,
    "GOT10kDataset": GOT10kDataset,
    "LaSOTDataset": LaSOTDataset,
    "SyntheticDataset": SyntheticDataset,
}


def _build_dataset(cfg: dict):
    """Instantiate a dataset from a YAML config block."""
    loader = cfg.get("loader", "OTBDataset")
    cls = DATASET_LOADERS[loader]
    kwargs = {}
    if "root" in cfg:
        kwargs["root"] = cfg["root"]
    if "split" in cfg and loader in ("GOT10kDataset", "LaSOTDataset"):
        kwargs["split"] = cfg["split"]
    if "max_sequences" in cfg and cfg["max_sequences"] is not None:
        kwargs["max_sequences"] = int(cfg["max_sequences"])
    if loader == "SyntheticDataset":
        kwargs.setdefault("num_sequences", cfg.get("num_sequences", 10))
        kwargs.setdefault("num_frames", cfg.get("num_frames", 100))
        kwargs.setdefault("motion", cfg.get("motion", "linear"))
        kwargs.pop("root", None)
        kwargs.pop("split", None)
    return cls(**kwargs)


def _run_from_config(config: dict, output_dir: Path) -> None:
    """Execute a cross-evaluation run defined by a YAML config dict."""
    exp_cfg = config.get("experiment", {})
    verbose: bool = bool(exp_cfg.get("verbose", True))
    max_seq = exp_cfg.get("max_sequences")
    tdp: float | None = exp_cfg.get("tdp_watts") or None
    resume: bool = bool(exp_cfg.get("resume", False))

    # Build datasets
    datasets = {}
    weights = {}
    for ds_cfg in config.get("datasets", []):
        name = ds_cfg["name"]
        datasets[name] = _build_dataset(ds_cfg)
        if "weight" in ds_cfg:
            weights[name] = float(ds_cfg["weight"])

    if not datasets:
        print("ERROR: No datasets specified in config.", file=sys.stderr)
        sys.exit(1)

    # Build trackers
    trackers = []
    for t_cfg in config.get("trackers", []):
        trackers.append(build_tracker(t_cfg["name"], **t_cfg.get("params", {})))

    if not trackers:
        print("ERROR: No trackers specified in config.", file=sys.stderr)
        sys.exit(1)

    # Load prior results for resume
    prior = None
    if resume and output_dir.is_dir():
        from eovot.experiment.cross_eval import CrossDatasetEvaluator
        prior = CrossDatasetEvaluator.load_prior_results(output_dir)
        if prior:
            print(f"Resuming: loaded cached results for {list(prior)}")

    evaluator = CrossDatasetEvaluator(
        datasets=datasets,
        dataset_weights=weights if weights else None,
        verbose=verbose,
        max_sequences=int(max_seq) if max_seq is not None else None,
        tdp_watts=tdp,
    )
    for t in trackers:
        evaluator.add_tracker(t)

    table = evaluator.run(prior_results=prior)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_name = exp_cfg.get("name", "cross_eval")

    json_path = table.save(output_dir / f"{exp_name}.json")
    csv_path = table.to_csv(output_dir / f"{exp_name}.csv")
    md_path = output_dir / f"{exp_name}.md"
    md_path.write_text(table.to_markdown(), encoding="utf-8")

    print("\n" + "=" * 60)
    print(table.to_markdown())
    print("=" * 60)
    print(f"\nOutputs written to {output_dir}:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  MD:   {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cross_eval",
        description="EOVOT cross-dataset tracker evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", help="Path to YAML experiment config.")
    parser.add_argument(
        "--trackers", nargs="+", metavar="NAME",
        help="Tracker names from the registry (e.g. MOSSE KCF).",
    )
    parser.add_argument(
        "--datasets", nargs="+", metavar="NAME:ROOT",
        help="Datasets as NAME:ROOT pairs for OTBDataset (e.g. OTB100:/data/OTB100).",
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="Add a synthetic dataset for quick testing.")
    parser.add_argument("--synthetic-sequences", type=int, default=5,
                        metavar="N")
    parser.add_argument("--synthetic-frames", type=int, default=80,
                        metavar="N")
    parser.add_argument("--max-sequences", type=int, default=None,
                        metavar="N", help="Limit sequences per dataset.")
    parser.add_argument("--tdp-watts", type=float, default=None,
                        metavar="W", help="CPU TDP for energy profiling.")
    parser.add_argument("--output-dir", default="results/cross_eval",
                        metavar="DIR")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-evaluated combinations.")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", dest="verbose", action="store_false")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Config-file path takes precedence
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        _run_from_config(config, output_dir)
        return

    # Build config from CLI flags
    if not args.trackers:
        parser.error("Provide --trackers or --config.")

    datasets_cfg = []
    if args.synthetic:
        datasets_cfg.append({
            "name": "Synthetic-Linear",
            "loader": "SyntheticDataset",
            "num_sequences": args.synthetic_sequences,
            "num_frames": args.synthetic_frames,
            "motion": "linear",
        })
        datasets_cfg.append({
            "name": "Synthetic-Circular",
            "loader": "SyntheticDataset",
            "num_sequences": args.synthetic_sequences,
            "num_frames": args.synthetic_frames,
            "motion": "circular",
        })

    if args.datasets:
        for entry in args.datasets:
            name, root = entry.split(":", 1)
            datasets_cfg.append({"name": name, "loader": "OTBDataset", "root": root})

    if not datasets_cfg:
        parser.error("Provide --synthetic or --datasets, or use --config.")

    config = {
        "experiment": {
            "name": "cross_eval",
            "verbose": args.verbose,
            "max_sequences": args.max_sequences,
            "tdp_watts": args.tdp_watts,
            "resume": args.resume,
        },
        "datasets": datasets_cfg,
        "trackers": [{"name": t, "params": {}} for t in args.trackers],
    }
    _run_from_config(config, output_dir)


if __name__ == "__main__":
    main()
