#!/usr/bin/env python
"""Run the full EOVOT evaluation pipeline from the command line.

Evaluates one or more trackers on a synthetic dataset through all five
metric layers (accuracy, robustness, temporal consistency, attribute
analysis, efficiency scoring, statistical significance) and writes a
Markdown + JSON report.

Usage examples
--------------
# Default: MOSSE vs KCF vs DSST on a 10-sequence linear-motion dataset
python scripts/full_evaluation.py

# Compare KCF and DSST only, circular motion, more sequences
python scripts/full_evaluation.py --trackers KCF DSST --motion circular \\
    --num-sequences 20 --num-frames 150 --output-dir results/circular_eval

# All registered trackers, random motion
python scripts/full_evaluation.py --trackers MOSSE KCF DSST CSRT MIL \\
    --motion random --num-sequences 15 --output-dir results/full_random
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the package importable when the script is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.datasets.synthetic import SyntheticDataset
from eovot.evaluation import EvaluationPipeline
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker

_DEFAULT_TRACKERS = ["MOSSE", "KCF", "DSST"]
_DEFAULT_MOTION = "linear"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EOVOT full evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=_DEFAULT_TRACKERS,
        metavar="NAME",
        help=f"Tracker names from the registry. Available: {sorted(TRACKER_REGISTRY)}",
    )
    parser.add_argument(
        "--motion",
        choices=["linear", "circular", "random"],
        default=_DEFAULT_MOTION,
        help="Synthetic dataset motion type.",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10,
        metavar="N",
        help="Number of sequences in the synthetic dataset.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=120,
        metavar="N",
        help="Number of frames per sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the synthetic dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/evaluation",
        metavar="DIR",
        help="Directory to write the Markdown and JSON report.",
    )
    parser.add_argument(
        "--memory-budget-mb",
        type=float,
        default=512.0,
        metavar="MB",
        help="Memory budget (MB) for the Edge Efficiency Score denominator.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sequence progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Validate tracker names early.
    unknown = [t for t in args.trackers if t not in TRACKER_REGISTRY]
    if unknown:
        print(
            f"Unknown tracker(s): {unknown}\n"
            f"Available: {sorted(TRACKER_REGISTRY)}",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_name = f"Synthetic-{args.motion.capitalize()}"
    print(
        f"\nEOVOT Full Evaluation Pipeline\n"
        f"  Trackers  : {', '.join(args.trackers)}\n"
        f"  Dataset   : {dataset_name} "
        f"({args.num_sequences} seqs × {args.num_frames} frames)\n"
        f"  Output    : {args.output_dir}\n"
    )

    dataset = SyntheticDataset(
        num_sequences=args.num_sequences,
        num_frames=args.num_frames,
        motion=args.motion,
        seed=args.seed,
    )
    trackers = [build_tracker(name) for name in args.trackers]

    pipeline = EvaluationPipeline(
        output_dir=args.output_dir,
        memory_budget_mb=args.memory_budget_mb,
        verbose=not args.quiet,
    )

    t0 = time.perf_counter()
    report = pipeline.run(trackers, dataset, dataset_name=dataset_name)
    elapsed = time.perf_counter() - t0

    report_name = f"{'_'.join(args.trackers).lower()}_{args.motion}"
    paths = report.save(args.output_dir, name=report_name)

    print(f"\nEvaluation complete in {elapsed:.1f} s")
    print(f"  Markdown : {paths['markdown']}")
    print(f"  JSON     : {paths['json']}")

    # Print a quick accuracy summary to stdout.
    print("\nAccuracy Summary")
    print("-" * 52)
    for r in report.benchmark_results:
        sauc = r.mean_success_auc
        sauc_s = f"{sauc:.4f}" if sauc is not None else "  n/a "
        print(
            f"  {r.tracker_name:<14}  mIoU={r.mean_iou:.4f}  "
            f"AUC={sauc_s}  FPS={r.mean_fps:.1f}"
        )

    # Efficiency ranking.
    if report.efficiency:
        print("\nEfficiency Ranking (EES)")
        print("-" * 52)
        for rank, e in enumerate(report.efficiency, start=1):
            pareto = " *" if e.on_pareto_front else ""
            print(
                f"  {rank}. {e.tracker_name:<12}  EES={e.ees:.4f}  "
                f"mIoU={e.mean_iou:.4f}  FPS={e.fps:.1f}{pareto}"
            )
        print("  (* = Pareto-optimal)")


if __name__ == "__main__":
    main()
