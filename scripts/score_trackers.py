#!/usr/bin/env python3
"""Compute and rank trackers by Edge Efficiency Score.

Reads one or more EOVOT benchmark result JSON files (produced by
``run_benchmark.py`` or ``compare_trackers.py``) and outputs a ranked table
that combines accuracy, throughput, and memory efficiency into a single
deployability score.

Usage
-----
# Score two saved result files against a Raspberry Pi 4 device profile:
python scripts/score_trackers.py \\
    results/MOSSE-OTB100.json results/KCF-OTB100.json \\
    --fps-ref 30 --memory-ref 512

# Score all JSON files in a directory:
python scripts/score_trackers.py results/*.json --fps-ref 30 --memory-ref 512

# Save the Markdown ranking table:
python scripts/score_trackers.py results/*.json --output ranking.md

# Weight accuracy more heavily (e.g., for high-accuracy requirement):
python scripts/score_trackers.py results/*.json \\
    --weight-accuracy 0.7 --weight-speed 0.2 --weight-memory 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from eovot.metrics.efficiency import EdgeEfficiencyScorer, EfficiencyResult


def load_summary(path: Path) -> dict:
    """Load the summary sub-dict from an EOVOT result JSON file."""
    with open(path) as fh:
        data = json.load(fh)
    if "summary" not in data:
        print(f"  WARNING: '{path}' has no 'summary' key — skipping.", file=sys.stderr)
        return {}
    return data["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank EOVOT trackers by hardware-aware Edge Efficiency Score.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "result_files",
        metavar="FILE",
        nargs="+",
        type=Path,
        help="EOVOT benchmark result JSON files to score.",
    )
    parser.add_argument(
        "--fps-ref",
        type=float,
        default=30.0,
        metavar="FPS",
        help="Target FPS for the deployment device (default: 30).",
    )
    parser.add_argument(
        "--memory-ref",
        type=float,
        default=512.0,
        metavar="MB",
        help="Memory budget in MiB for the deployment device (default: 512).",
    )
    parser.add_argument(
        "--weight-accuracy",
        type=float,
        default=0.5,
        metavar="W",
        help="Relative weight for the accuracy component (default: 0.5).",
    )
    parser.add_argument(
        "--weight-speed",
        type=float,
        default=0.3,
        metavar="W",
        help="Relative weight for the throughput component (default: 0.3).",
    )
    parser.add_argument(
        "--weight-memory",
        type=float,
        default=0.2,
        metavar="W",
        help="Relative weight for the memory-efficiency component (default: 0.2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Optional path to save the Markdown ranking table.",
    )
    args = parser.parse_args()

    scorer = EdgeEfficiencyScorer(
        fps_ref=args.fps_ref,
        memory_ref_mb=args.memory_ref,
        weight_accuracy=args.weight_accuracy,
        weight_speed=args.weight_speed,
        weight_memory=args.weight_memory,
    )

    tracker_metrics: dict = {}
    for path in args.result_files:
        if not path.exists():
            print(f"  ERROR: '{path}' not found.", file=sys.stderr)
            continue
        summary = load_summary(path)
        if not summary:
            continue
        name = summary.get("tracker", path.stem)
        tracker_metrics[name] = {
            "mean_iou": summary.get("mean_iou", 0.0),
            "fps": summary.get("mean_fps", 0.0),
            "memory_mb": summary.get("peak_memory_mb", 0.0),
        }

    if not tracker_metrics:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    ranked = scorer.rank(tracker_metrics)
    table = scorer.summary_table(tracker_metrics)

    # Console output
    print("\n" + "=" * 70)
    print(" EDGE EFFICIENCY RANKING")
    print(f" Device profile: {args.fps_ref:.0f} FPS target, "
          f"{args.memory_ref:.0f} MB memory budget")
    print(f" Weights: accuracy={args.weight_accuracy:.2f}  "
          f"speed={args.weight_speed:.2f}  memory={args.weight_memory:.2f}")
    print("=" * 70)
    print(
        f"\n{'Rank':<5} {'Tracker':<20} {'Score':>8} "
        f"{'mIoU':>8} {'FPS':>8} {'Mem(MB)':>10}"
    )
    print("-" * 65)
    for i, r in enumerate(ranked, start=1):
        print(
            f"{i:<5} {r.tracker_name:<20} {r.edge_score:>8.4f} "
            f"{r.mean_iou:>8.4f} {r.fps:>8.1f} {r.memory_mb:>10.1f}"
        )
    print("=" * 70 + "\n")

    # Optional Markdown file output
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            fh.write("# EOVOT Edge Efficiency Report\n\n")
            fh.write(table)
            fh.write("\n")
        print(f"Ranking table saved to: {args.output}")


if __name__ == "__main__":
    main()
