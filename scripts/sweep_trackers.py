#!/usr/bin/env python3
"""Hyperparameter sweep CLI for EOVOT trackers.

Runs a grid search over tracker parameters on a synthetic dataset (no
external data download required) and saves results as JSON + Markdown.

Usage examples::

    # Sweep KCF learning_rate and padding — 4×3 = 12 configs
    python scripts/sweep_trackers.py \\
        --tracker KCF \\
        --grid learning_rate=0.05,0.075,0.125,0.2 padding=1.0,1.5,2.0

    # Quick sweep: 5 sequences per config, random motion
    python scripts/sweep_trackers.py \\
        --tracker MOSSE \\
        --grid learning_rate=0.05,0.1,0.2 \\
        --num-sequences 5 --motion random

    # Use more frames for accurate FPS estimates
    python scripts/sweep_trackers.py \\
        --tracker KCF \\
        --grid learning_rate=0.075,0.125 kernel_sigma=0.3,0.5,0.7 \\
        --num-frames 200

Parameter grid format::

    --grid PARAM=VAL1,VAL2,VAL3 [PARAM2=VAL1,VAL2 ...]

    Values are parsed as Python literals (int/float/bool/string).
    Example: --grid learning_rate=0.05,0.125 padding=1.0,2.0

Output::

    results/sweeps/sweep-<TRACKER>-Synthetic-<MOTION>.json
    results/sweeps/sweep-<TRACKER>-Synthetic-<MOTION>.md
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Allow running as ``python scripts/sweep_trackers.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.datasets.synthetic import SyntheticDataset
from eovot.experiment.sweep import HyperparameterSweeper
from eovot.trackers.registry import available_trackers


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _parse_grid(grid_items: list) -> dict:
    """Parse ``["param=v1,v2", ...]`` into a ``{param: [v1, v2]}`` dict."""
    grid = {}
    for item in grid_items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Grid item must be 'PARAM=val1,val2,...', got: {item!r}"
            )
        key, vals_str = item.split("=", 1)
        values = []
        for raw in vals_str.split(","):
            raw = raw.strip()
            try:
                values.append(ast.literal_eval(raw))
            except (ValueError, SyntaxError):
                values.append(raw)
        grid[key.strip()] = values
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sweep_trackers",
        description=(
            "Hyperparameter grid-search sweep for EOVOT trackers on synthetic data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tracker",
        required=True,
        choices=available_trackers(),
        metavar="TRACKER",
        help=f"Tracker to sweep. Choices: {available_trackers()}",
    )
    parser.add_argument(
        "--grid",
        nargs="+",
        required=True,
        metavar="PARAM=VAL1,VAL2,...",
        help=(
            "Parameter grid specification. "
            "E.g.: --grid learning_rate=0.05,0.075,0.125 padding=1.0,1.5,2.0"
        ),
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10,
        metavar="N",
        help="Number of synthetic sequences evaluated per configuration.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        metavar="N",
        help="Frames per synthetic sequence.",
    )
    parser.add_argument(
        "--motion",
        default="linear",
        choices=["linear", "circular", "random"],
        help="Synthetic motion pattern.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for synthetic dataset reproducibility.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Cap on sequences evaluated per config (useful for quick tests).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/sweeps",
        help="Directory where JSON and Markdown sweep results are saved.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-configuration progress output.",
    )

    args = parser.parse_args()

    try:
        param_grid = _parse_grid(args.grid)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if not param_grid:
        parser.error("--grid must specify at least one parameter.")

    dataset_name = f"Synthetic-{args.motion.capitalize()}"
    dataset = SyntheticDataset(
        num_sequences=args.num_sequences,
        num_frames=args.num_frames,
        motion=args.motion,
        seed=args.seed,
    )

    n_configs = 1
    for vals in param_grid.values():
        n_configs *= len(vals)

    print(f"\nEOVOT Hyperparameter Sweep")
    print(f"  Tracker   : {args.tracker}")
    print(f"  Dataset   : {dataset_name}  ({args.num_sequences} seqs × {args.num_frames} frames)")
    print(f"  Grid      : {param_grid}")
    print(f"  Configs   : {n_configs}")
    print(f"  Output    : {args.output_dir}")

    sweeper = HyperparameterSweeper(
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    result = sweeper.sweep(
        tracker_name=args.tracker,
        param_grid=param_grid,
        dataset=dataset,
        dataset_name=dataset_name,
        max_sequences=args.max_sequences,
    )

    print("\n" + "=" * 70)
    print(f"  SWEEP COMPLETE: {args.tracker} on {dataset_name}")
    print("=" * 70)
    print(result.to_markdown_table())

    best = result.best_by_success_auc
    if best:
        print("\nBest configuration (by success AUC):")
        for k, v in sorted(best.params.items()):
            print(f"  {k}: {v}")
        if best.success_auc is not None:
            print(f"  → success_AUC = {best.success_auc:.4f}  mIoU = {best.mean_iou:.4f}")


if __name__ == "__main__":
    main()
