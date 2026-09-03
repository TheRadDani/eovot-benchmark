#!/usr/bin/env python3
"""CLI for tracker hyperparameter grid search and sensitivity analysis.

Usage examples::

    # Sweep KCF learning_rate and sigma on a synthetic dataset
    python scripts/sweep_hyperparams.py \\
        --tracker KCF \\
        --params "learning_rate=[0.075,0.125,0.200]" \\
        --params "sigma_spatial=[0.2,0.4,0.6]" \\
        --synthetic-sequences 10 \\
        --synthetic-frames 100 \\
        --output results/kcf_sweep.csv

    # MOSSE eta sweep on synthetic data
    python scripts/sweep_hyperparams.py \\
        --tracker MOSSE \\
        --params "eta=[0.05,0.10,0.20,0.30]" \\
        --synthetic-sequences 8 \\
        --synthetic-frames 80 \\
        --output results/mosse_eta.csv

The script prints a Markdown leaderboard, saves CSV, and highlights the
Pareto-optimal configurations for edge deployment decisions.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Make the project importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.analysis.hyperparam_sweep import HyperparamSweeper
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.registry import TRACKER_REGISTRY


def parse_param(s: str):
    """Parse 'key=value' or 'key=[v1,v2,...]' into (key, list_of_values)."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"Invalid param format: {s!r}. Expected key=value or key=[v1,v2]")
    key, raw = s.split("=", 1)
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw  # keep as string

    if not isinstance(value, list):
        value = [value]
    return key.strip(), value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EOVOT hyperparameter grid sweep for tracker configuration analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tracker", required=True,
        help=f"Tracker name. Available: {sorted(TRACKER_REGISTRY)}",
    )
    parser.add_argument(
        "--params", action="append", default=[], metavar="KEY=VALUES",
        help="Parameter to sweep. Repeat for multiple params. "
             "E.g. --params learning_rate=[0.075,0.125]",
    )
    parser.add_argument(
        "--synthetic-sequences", type=int, default=10, metavar="N",
        help="Number of synthetic sequences to evaluate on (default: 10).",
    )
    parser.add_argument(
        "--synthetic-frames", type=int, default=100, metavar="N",
        help="Frames per synthetic sequence (default: 100).",
    )
    parser.add_argument(
        "--max-sequences", type=int, default=None, metavar="N",
        help="Limit evaluation to first N sequences (default: all).",
    )
    parser.add_argument(
        "--memory-budget", type=float, default=512.0, metavar="MB",
        help="Memory budget in MiB for EES denominator (default: 512).",
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Path to write the results CSV (default: no file output).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-run progress output.",
    )
    args = parser.parse_args()

    # Validate tracker name.
    if args.tracker not in TRACKER_REGISTRY:
        print(f"Error: unknown tracker '{args.tracker}'. Available: {sorted(TRACKER_REGISTRY)}", file=sys.stderr)
        sys.exit(1)

    # Parse param grid.
    param_grid: dict = {}
    for raw_param in args.params:
        try:
            key, values = parse_param(raw_param)
            param_grid[key] = values
        except argparse.ArgumentTypeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    if not param_grid:
        print("Warning: no --params specified; running one evaluation with default configuration.")
        param_grid = {"__dummy__": [None]}

    tracker_cls = TRACKER_REGISTRY[args.tracker]

    # Build dataset.
    dataset = SyntheticDataset(
        num_sequences=args.synthetic_sequences,
        num_frames=args.synthetic_frames,
    )

    sweeper = HyperparamSweeper(
        dataset=dataset,
        dataset_name=f"Synthetic-{args.synthetic_sequences}seq",
        max_sequences=args.max_sequences,
        memory_budget_mb=args.memory_budget,
        verbose=not args.quiet,
    )

    # Handle dummy param (no real params given).
    if "__dummy__" in param_grid:
        param_grid.pop("__dummy__")
        results = sweeper.grid_search(tracker_cls, param_grid if param_grid else {})
    else:
        results = sweeper.grid_search(tracker_cls, param_grid)

    # Compute Pareto front.
    sweeper.pareto_front(results)

    # Sensitivity scores.
    sensitivity = sweeper.sensitivity_scores(results)

    # Print Markdown table.
    print("\n" + sweeper.to_markdown(results, title=f"{args.tracker} sweep"))

    # Print Pareto-optimal configs.
    pareto = [e for e in results if e.on_pareto_front]
    if pareto:
        print(f"\n### Pareto-Optimal Configurations ({len(pareto)} of {len(results)})")
        for e in pareto:
            print(f"  {e.params}  →  mIoU={e.mean_iou:.4f}  FPS={e.mean_fps:.1f}  EES={e.ees:.4f}")

    # Print sensitivity.
    if sensitivity:
        print("\n### Parameter Sensitivity (std of group-mean EES)")
        for k, v in sensitivity.items():
            print(f"  {k:30s}: {v:.5f}")

    # Save CSV.
    if args.output:
        out_path = sweeper.to_csv(results, args.output)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
