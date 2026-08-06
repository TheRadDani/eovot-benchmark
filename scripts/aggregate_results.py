"""Aggregate multiple JSON benchmark result files into a ranked leaderboard.

Reads all ``.json`` result files from one or more directories produced by
:meth:`~eovot.benchmark.engine.BenchmarkResult.save` and outputs a ranked
Markdown table (and optional CSV) comparing all trackers on every dataset.

Usage::

    # Aggregate all results in results/ and print leaderboard
    python scripts/aggregate_results.py results/

    # Multiple directories
    python scripts/aggregate_results.py results/exp1/ results/exp2/

    # Custom sort metric, with MOSSE as baseline
    python scripts/aggregate_results.py results/ --sort success_auc --baseline MOSSE

    # Save to CSV
    python scripts/aggregate_results.py results/ --csv leaderboard.csv

    # Filter to a specific dataset
    python scripts/aggregate_results.py results/ --dataset OTB100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eovot.results.leaderboard import LeaderboardAggregator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate EOVOT benchmark JSON results into a leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "dirs",
        nargs="+",
        metavar="DIR",
        help="Directory (or directories) containing benchmark JSON files",
    )
    p.add_argument(
        "--sort",
        default="mean_iou",
        metavar="METRIC",
        help="Metric to sort by (default: mean_iou)",
    )
    p.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending (use for memory/latency metrics where lower is better)",
    )
    p.add_argument(
        "--baseline",
        default=None,
        metavar="TRACKER",
        help="Tracker name to use as baseline for Δ columns",
    )
    p.add_argument(
        "--dataset",
        default=None,
        metavar="NAME",
        help="Filter to a specific dataset name (default: all datasets)",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        metavar="METRIC",
        help="Metrics to include (default: mean_iou mean_fps peak_memory_mb)",
    )
    p.add_argument(
        "--csv",
        default=None,
        metavar="FILE",
        help="Save leaderboard to a CSV file",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Scan directories recursively for JSON files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    agg = LeaderboardAggregator(metrics=args.metrics)
    total = 0
    for d in args.dirs:
        try:
            n = agg.load_dir(d, recursive=args.recursive)
            total += n
            print(f"  Loaded {n} result(s) from {d}")
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc}", file=sys.stderr)

    if total == 0:
        print("No result files found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal results loaded: {total}")
    print(f"Datasets: {', '.join(agg.available_datasets())}")
    print(f"Trackers: {', '.join(agg.available_trackers())}")
    print()

    higher_is_better = not args.ascending
    datasets = [args.dataset] if args.dataset else agg.available_datasets()

    for ds in datasets:
        try:
            board = agg.build(
                dataset=ds,
                sort_by=args.sort,
                higher_is_better=higher_is_better,
                baseline=args.baseline,
            )
        except ValueError as exc:
            print(f"  WARNING: {exc}", file=sys.stderr)
            continue

        print(board.to_markdown(show_delta=True))
        print()

        if args.csv:
            # Append dataset name to CSV path when multiple datasets
            csv_path = args.csv
            if len(datasets) > 1:
                stem = Path(args.csv).stem
                suffix = Path(args.csv).suffix or ".csv"
                csv_path = str(Path(args.csv).parent / f"{stem}_{ds}{suffix}")
            saved = board.save_csv(csv_path)
            print(f"  CSV saved to: {saved}")


if __name__ == "__main__":
    main()
