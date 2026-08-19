"""CLI for the EOVOT accuracy–efficiency Pareto sweep.

Exhaustively evaluates a base tracker across a grid of (skip_rate,
scale_factor) combinations, identifies the Pareto-optimal operating points,
and writes a CSV report plus a Markdown summary.

Usage examples
--------------
Synthetic dataset (no download required)::

    python scripts/pareto_sweep.py \\
        --tracker KCF \\
        --synthetic \\
        --synthetic-sequences 5 \\
        --skip-rates 1 2 3 4 \\
        --scale-factors 1.0 0.75 0.5 \\
        --output results/pareto_sweep_kcf.csv

Real dataset (OTB)::

    python scripts/pareto_sweep.py \\
        --tracker MOSSE \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100 \\
        --max-sequences 20 \\
        --skip-rates 1 2 4 \\
        --scale-factors 1.0 0.5 \\
        --memory-budget 512 \\
        --output results/pareto_sweep_mosse.csv

Installed as a package entry point (if configured in setup.py)::

    eovot-pareto --tracker KCF --synthetic --output pareto.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.datasets.base import OTBDataset
from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.lasot import LaSOTDataset
from eovot.datasets.synthetic import SyntheticDataset
from eovot.experiment.pareto_sweep import ParetoSweep
from eovot.trackers.registry import available_trackers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eovot-pareto",
        description=(
            "EOVOT Pareto Sweep — evaluate a tracker across (skip_rate × scale_factor) "
            "combinations and identify the Pareto-optimal operating points."
        ),
    )
    parser.add_argument(
        "--tracker", "-t",
        default="KCF",
        choices=available_trackers(),
        help="Base tracker to sweep (default: KCF).",
    )
    parser.add_argument(
        "--skip-rates",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        metavar="K",
        help="Frame-skip factors to evaluate (default: 1 2 3 4).",
    )
    parser.add_argument(
        "--scale-factors",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5],
        metavar="S",
        help="Resolution scale factors to evaluate (default: 1.0 0.75 0.5).",
    )
    parser.add_argument(
        "--memory-budget",
        type=float,
        default=512.0,
        metavar="MB",
        help="Memory budget in MB for EES computation (default: 512).",
    )
    parser.add_argument(
        "--tdp-watts",
        type=float,
        default=None,
        metavar="W",
        help="CPU TDP for energy estimation (default: disabled).",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/pareto_sweep.csv",
        metavar="PATH",
        help="CSV output path (default: results/pareto_sweep.csv).",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N sequences per combination.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-combination progress output.",
    )

    # Dataset options
    ds_group = parser.add_mutually_exclusive_group(required=True)
    ds_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the built-in synthetic dataset.",
    )
    ds_group.add_argument(
        "--dataset-root",
        metavar="DIR",
        help="Path to OTB/GOT-10k/LaSOT dataset root.",
    )

    # Synthetic dataset options
    syn = parser.add_argument_group("synthetic dataset options")
    syn.add_argument("--synthetic-sequences", type=int, default=5, metavar="N",
                     help="Number of synthetic sequences (default: 5).")
    syn.add_argument("--synthetic-frames", type=int, default=100, metavar="N",
                     help="Frames per synthetic sequence (default: 100).")
    syn.add_argument("--synthetic-motion", default="linear",
                     choices=["linear", "circular", "random"],
                     help="Synthetic target motion pattern (default: linear).")

    # Real dataset options
    real = parser.add_argument_group("real dataset options")
    real.add_argument("--dataset-name", default="OTB100",
                      help="Dataset label for reports (default: OTB100).")
    real.add_argument("--dataset-type", default="OTBDataset",
                      choices=["OTBDataset", "GOT10kDataset", "LaSOTDataset"],
                      help="Dataset loader class (default: OTBDataset).")
    real.add_argument("--dataset-split", default="val",
                      help="Dataset split for GOT-10k/LaSOT (default: val).")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Build dataset ---
    if args.synthetic:
        dataset = SyntheticDataset(
            num_sequences=args.synthetic_sequences,
            num_frames=args.synthetic_frames,
            motion=args.synthetic_motion,
        )
        dataset_name = f"Synthetic-{args.synthetic_motion.capitalize()}"
    else:
        dataset_root = args.dataset_root
        dataset_name = args.dataset_name
        if args.dataset_type == "GOT10kDataset":
            dataset = GOT10kDataset(dataset_root, split=args.dataset_split,
                                    max_sequences=args.max_sequences)
        elif args.dataset_type == "LaSOTDataset":
            dataset = LaSOTDataset(dataset_root, split=args.dataset_split,
                                   max_sequences=args.max_sequences)
        else:
            dataset = OTBDataset(dataset_root)

    # --- Run sweep ---
    sweep = ParetoSweep(
        skip_rates=args.skip_rates,
        scale_factors=args.scale_factors,
        memory_budget_mb=args.memory_budget,
        tdp_watts=args.tdp_watts,
        verbose=not args.quiet,
    )
    result = sweep.run(
        base_tracker_name=args.tracker,
        dataset=dataset,
        dataset_name=dataset_name,
        max_sequences=args.max_sequences,
    )

    # --- Print markdown summary ---
    print()
    print(result.to_markdown())

    # --- Save CSV ---
    out_path = result.save(args.output)
    print(f"\nCSV saved to {out_path}")


if __name__ == "__main__":
    main()
