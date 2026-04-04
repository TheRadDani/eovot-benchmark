#!/usr/bin/env python3
"""Reproducible multi-tracker experiment runner for EOVOT.

Loads a YAML experiment config, seeds the RNG, evaluates all
(tracker × dataset) combinations, and saves JSON / CSV / Markdown reports.

Usage::

    # Run the default comparison experiment
    python scripts/run_experiment.py \\
        --config configs/comparison_experiment.yaml

    # Quick smoke-test with 5 sequences per dataset
    python scripts/run_experiment.py \\
        --config configs/comparison_experiment.yaml \\
        --max-sequences 5

    # Override seed and output directory
    python scripts/run_experiment.py \\
        --config configs/comparison_experiment.yaml \\
        --seed 7 \\
        --output-dir results/seed7/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repository root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.experiment.config import ExperimentConfig
from eovot.experiment.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_experiment",
        description="Run a reproducible EOVOT benchmark experiment from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML experiment config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the RNG seed specified in the config.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Override the max_sequences cap for all datasets (useful for quick tests).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory specified in the config.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sequence progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_sequences is not None:
        cfg.max_sequences = args.max_sequences
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.quiet:
        cfg.verbose = False

    runner = ExperimentRunner(cfg)
    results = runner.run()

    print(f"\nDone. {len(results)} benchmark run(s) completed.")
    print(f"Results saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
