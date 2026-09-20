#!/usr/bin/env python3
"""CLI for joint skip-rate × resolution-scale optimization.

Sweeps every combination of the requested skip rates and resolution scale
factors, runs the tracker on the chosen dataset, identifies the Pareto-
optimal operating points in (mIoU, FPS) space, and prints a ranked Markdown
table.  Optionally saves the table to a file.

Examples::

    # Quick synthetic sweep with KCF
    python scripts/joint_optimize.py \\
        --tracker KCF \\
        --synthetic --sequences 5 --frames 50 \\
        --skip-rates 1 2 3 4 \\
        --scale-factors 1.0 0.75 0.5

    # Save table to Markdown
    python scripts/joint_optimize.py \\
        --tracker MOSSE \\
        --synthetic \\
        --output results/joint_mosse.md

    # Set IoU floor and print best config
    python scripts/joint_optimize.py \\
        --tracker KCF \\
        --synthetic \\
        --min-iou 0.70
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Joint skip-rate × resolution-scale optimizer (EOVOT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Tracker
    p.add_argument(
        "--tracker",
        default="KCF",
        help="Tracker name from the registry (default: KCF)",
    )

    # Dataset — mutually exclusive: real vs synthetic
    ds_group = p.add_mutually_exclusive_group()
    ds_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the built-in SyntheticDataset (no download needed)",
    )
    ds_group.add_argument(
        "--dataset-root",
        metavar="PATH",
        help="Path to an OTB / GOT-10k / LaSOT dataset root directory",
    )
    p.add_argument(
        "--dataset-type",
        default="OTB",
        choices=["OTB", "GOT10k", "LaSOT"],
        help="Dataset type when --dataset-root is used (default: OTB)",
    )
    p.add_argument(
        "--sequences",
        type=int,
        default=5,
        metavar="N",
        help="Number of synthetic sequences (default: 5)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=50,
        metavar="N",
        help="Frames per synthetic sequence (default: 50)",
    )

    # Sweep parameters
    p.add_argument(
        "--skip-rates",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        metavar="K",
        help="Skip rates to sweep (default: 1 2 3 4)",
    )
    p.add_argument(
        "--scale-factors",
        nargs="+",
        type=float,
        default=[1.0, 0.75, 0.5, 0.25],
        metavar="S",
        help="Resolution scale factors to sweep (default: 1.0 0.75 0.5 0.25)",
    )

    # Constraints
    p.add_argument(
        "--min-iou",
        type=float,
        default=0.0,
        metavar="F",
        help="Minimum acceptable mIoU when printing optimal config (default: 0)",
    )
    p.add_argument(
        "--min-fps",
        type=float,
        default=0.0,
        metavar="F",
        help="Minimum acceptable FPS when printing optimal config (default: 0)",
    )

    # Output
    p.add_argument(
        "--output",
        metavar="PATH",
        help="Path to write the Markdown table (optional)",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress for each (skip, scale) pair",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Limit evaluation to first N sequences",
    )

    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # Lazy imports so --help is fast
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.analysis.joint_analysis import JointOptimizationAnalyzer
    from eovot.trackers.registry import build_tracker

    # Build tracker
    try:
        tracker = build_tracker(args.tracker)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Build dataset
    if args.synthetic or args.dataset_root is None:
        from eovot.datasets.synthetic import SyntheticDataset
        dataset = SyntheticDataset(
            num_sequences=args.sequences,
            frames_per_sequence=args.frames,
        )
        dataset_name = f"Synthetic({args.sequences}×{args.frames})"
    else:
        root = Path(args.dataset_root)
        if not root.exists():
            print(f"error: dataset root not found: {root}", file=sys.stderr)
            return 1
        if args.dataset_type == "GOT10k":
            from eovot.datasets.got10k import GOT10kDataset
            dataset = GOT10kDataset(root)
        elif args.dataset_type == "LaSOT":
            from eovot.datasets.lasot import LaSOTDataset
            dataset = LaSOTDataset(root)
        else:
            from eovot.datasets.otb import OTBDataset
            dataset = OTBDataset(root)
        dataset_name = f"{args.dataset_type}:{root.name}"

    engine   = BenchmarkEngine(verbose=False)
    analyzer = JointOptimizationAnalyzer(engine)

    print(
        f"Sweeping {args.tracker} over "
        f"{len(args.skip_rates)} skip rates × "
        f"{len(args.scale_factors)} scale factors "
        f"= {len(args.skip_rates) * len(args.scale_factors)} configs ..."
    )

    result = analyzer.analyze(
        tracker=tracker,
        dataset=dataset,
        dataset_name=dataset_name,
        skip_rates=args.skip_rates,
        scale_factors=args.scale_factors,
        max_sequences=args.max_sequences,
        verbose=args.verbose,
    )

    table = result.to_markdown_table()
    print()
    print(result.tracker_name, "on", result.dataset_name)
    print(table)
    print()
    print(f"Pareto-optimal configs: {len(result.pareto_front)}")

    # Optimal config recommendation
    try:
        skip, scale, iou, fps = result.optimal_config(
            min_iou=args.min_iou, min_fps=args.min_fps
        )
        print(
            f"Optimal (mIoU≥{args.min_iou}, FPS≥{args.min_fps}): "
            f"skip={skip}, scale={scale:.2f} → {fps:.1f} FPS, mIoU={iou:.4f}"
        )
    except ValueError as exc:
        print(f"No config meets constraints: {exc}", file=sys.stderr)

    # Save table
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
        print(f"\nTable saved to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
