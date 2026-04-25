#!/usr/bin/env python3
"""CLI for running reproducible multi-tracker EOVOT experiments.

Loads a YAML experiment config, runs every listed tracker on the specified
dataset, and writes per-tracker JSON/CSV results, a ranked Markdown
leaderboard, and a ``metadata.json`` reproducibility snapshot to the output
directory.

Usage::

    # Validate config and print the experiment plan without running
    python scripts/run_experiment.py \\
        --config configs/experiments/multi_tracker.yaml \\
        --dry-run

    # Full run (MOSSE + KCF + MIL + MedianFlow on OTB100)
    python scripts/run_experiment.py \\
        --config configs/experiments/multi_tracker.yaml \\
        --output-dir results/experiments

    # Resume an interrupted run (skips trackers that already have results)
    python scripts/run_experiment.py \\
        --config configs/experiments/multi_tracker.yaml \\
        --resume

    # Suppress per-sequence verbose output
    python scripts/run_experiment.py \\
        --config configs/experiments/multi_tracker.yaml \\
        --quiet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml  # pyyaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.experiment.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_experiment",
        description="Run a reproducible EOVOT multi-tracker experiment from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a YAML experiment config file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/experiments",
        metavar="DIR",
        help="Root directory for all experiment outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip trackers whose result JSON already exists (resume interrupted runs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the experiment plan without running anything.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sequence verbose output.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    if args.dry_run:
        _print_plan(config, args.output_dir)
        return

    runner = ExperimentRunner(
        output_dir=args.output_dir,
        verbose=not args.quiet,
        resume=args.resume,
    )
    runner.run_from_config(config)


def _print_plan(config: dict, output_dir: str) -> None:
    """Print a human-readable experiment plan without running anything."""
    exp = config.get("experiment", {})
    ds = config.get("dataset", {})
    trackers = config.get("trackers", [])

    exp_name = exp.get("name", "unnamed-experiment")
    exp_dir = Path(output_dir) / exp_name

    print("\nEXPERIMENT PLAN  (dry-run — nothing will be executed)")
    print("=" * 58)
    print(f"  Name        : {exp_name}")
    print(f"  Output dir  : {exp_dir}")
    print(f"  Seed        : {exp.get('seed', 'none')}")
    print(f"  TDP         : {exp.get('tdp_watts', 'disabled')} W")
    print(f"  Dataset     : {ds.get('name', ds.get('loader', '?'))}")
    print(f"  Loader      : {ds.get('loader', 'OTBDataset')}")
    print(f"  Root        : {ds.get('root', '(not set)')}")
    print(f"  Max seqs    : {ds.get('max_sequences', 'all')}")
    print(f"\n  Trackers ({len(trackers)}):")
    for t in trackers:
        params = t.get("params") or {}
        param_str = f"  params={params}" if params else ""
        print(f"    - {t['name']}{param_str}")
    print("=" * 58)
    print()


if __name__ == "__main__":
    main()
