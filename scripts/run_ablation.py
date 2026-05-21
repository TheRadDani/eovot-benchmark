"""CLI for running hyperparameter ablation studies from a YAML config file.

Usage::

    # Run a full ablation study
    python scripts/run_ablation.py --config configs/experiments/ablation_kcf.yaml

    # Save to a custom directory
    python scripts/run_ablation.py \\
        --config configs/experiments/ablation_kcf.yaml \\
        --output-dir results/ablations

    # Suppress per-config progress output
    python scripts/run_ablation.py \\
        --config configs/experiments/ablation_kcf.yaml \\
        --quiet

Output files are written to ``output_dir/<experiment.name>/``:

* ``ablation_results.json`` — full serialised results
* ``ablation_table.md``     — ranked Markdown table
* ``sensitivity_analysis.md`` — one-at-a-time sensitivity report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from eovot.experiment.ablation import run_ablation_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter ablation study for an EOVOT tracker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the ablation YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ablations",
        metavar="DIR",
        help="Root directory for output files (default: results/ablations).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-config benchmark progress output.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    run_ablation_from_config(
        config,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
