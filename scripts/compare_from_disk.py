"""Load saved benchmark results from a ResultsBank and print a comparison table.

Usage
-----
    # Compare all saved results in the bank
    python scripts/compare_from_disk.py --bank results/bank

    # Filter to a specific dataset
    python scripts/compare_from_disk.py --bank results/bank --dataset OTB100

    # Sort by FPS instead of mIoU
    python scripts/compare_from_disk.py --bank results/bank --sort-by mean_fps

    # Export comparison table to a Markdown file
    python scripts/compare_from_disk.py --bank results/bank --output comparison.md

    # List available results without loading them
    python scripts/compare_from_disk.py --bank results/bank --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.results.bank import ResultsBank


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compare_from_disk",
        description="Load and compare EOVOT benchmark results from a ResultsBank directory.",
    )
    parser.add_argument(
        "--bank", "-b",
        required=True,
        metavar="DIR",
        help="Path to the ResultsBank directory.",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=None,
        metavar="NAME",
        help="Filter results to this dataset (case-insensitive substring match).",
    )
    parser.add_argument(
        "--sort-by",
        default="mean_iou",
        choices=["mean_iou", "mean_fps", "success_auc", "precision_auc"],
        help="Column to sort by (default: mean_iou).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help="Write the comparison table to this Markdown file (stdout if omitted).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available results and exit (no full table generation).",
    )
    args = parser.parse_args()

    bank = ResultsBank(args.bank)

    if args.list:
        entries = bank.list_results(dataset=args.dataset)
        if not entries:
            print("No results found.")
            return
        print(f"{'Tracker':<20s}  {'Dataset':<15s}  {'mIoU':>6s}  {'FPS':>7s}  {'Saved At'}")
        print("-" * 75)
        for e in entries:
            print(
                f"{e['tracker']:<20s}  {e['dataset']:<15s}  "
                f"{e['mean_iou']:>6.4f}  {e['mean_fps']:>7.1f}  {e['timestamp']}"
            )
        return

    table = bank.compare(dataset=args.dataset, sort_by=args.sort_by)

    if args.output:
        Path(args.output).write_text("# EOVOT Benchmark Comparison\n\n" + table + "\n")
        print(f"Comparison table written to {args.output}")
    else:
        print("\n# EOVOT Benchmark Comparison\n")
        print(table)


if __name__ == "__main__":
    main()
