#!/usr/bin/env python3
"""Per-attribute tracker comparison table generator.

Runs multiple trackers on a dataset (synthetic by default) and builds the
standard VOT-challenge attribute breakdown table: rows are challenge
attributes (fast_motion, scale_variation, …), columns are tracker names.

Usage examples
--------------
# Markdown to stdout, synthetic dataset
python scripts/attribute_comparison.py

# Save LaTeX + CSV to results/ directory
python scripts/attribute_comparison.py --out-dir results --formats latex csv

# KCF vs MOSSE vs CSRT on 10 sequences, 200 frames
python scripts/attribute_comparison.py --trackers KCF MOSSE CSRT \\
    --num-sequences 10 --num-frames 200

# Use success_auc as the comparison metric
python scripts/attribute_comparison.py --metric success_auc
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Ensure the project root is importable when run from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.reporting.attribute_report import MultiTrackerAttributeReport
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-attribute tracker comparison table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--trackers",
        nargs="+",
        default=["KCF", "LKOpticalFlow"],
        metavar="NAME",
        help="Tracker names from the registry.  Default: KCF LKOpticalFlow",
    )
    p.add_argument(
        "--metric",
        choices=["mean_iou", "success_auc", "precision_auc"],
        default="mean_iou",
        help="Metric used to fill table cells.  Default: mean_iou",
    )
    p.add_argument(
        "--num-sequences",
        type=int,
        default=8,
        metavar="N",
        help="Synthetic sequences to generate.  Default: 8",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=80,
        metavar="N",
        help="Frames per sequence.  Default: 80",
    )
    p.add_argument(
        "--motion",
        choices=["linear", "circular", "random"],
        default="circular",
        help="Synthetic motion pattern.  Default: circular",
    )
    p.add_argument(
        "--dataset-name",
        default="Synthetic",
        metavar="NAME",
        help="Label used in report headings.  Default: Synthetic",
    )
    p.add_argument(
        "--min-sequences",
        type=int,
        default=1,
        metavar="N",
        help="Min sequences that must exhibit an attribute for it to appear.  "
             "Default: 1",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to save report files.  If omitted, prints to stdout only.",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        choices=["markdown", "latex", "csv", "json"],
        default=["markdown"],
        metavar="FMT",
        help="Output formats when --out-dir is set.  Default: markdown",
    )
    p.add_argument(
        "--bold-best",
        action="store_true",
        default=True,
        help="Bold the best tracker per attribute row (default: on)",
    )
    p.add_argument(
        "--show-counts",
        action="store_true",
        default=False,
        help="Append (N) sequence counts to each Markdown cell",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-tracker progress to stderr",
    )
    return p


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Validate tracker names early.
    unknown = [t for t in args.trackers if t not in TRACKER_REGISTRY]
    if unknown:
        print(
            f"[error] Unknown tracker(s): {unknown}. "
            f"Available: {sorted(TRACKER_REGISTRY)}",
            file=sys.stderr,
        )
        return 1

    ds = SyntheticDataset(
        num_sequences=args.num_sequences,
        num_frames=args.num_frames,
        motion=args.motion,
    )
    engine = BenchmarkEngine(verbose=args.verbose)

    results = []
    for tracker_name in args.trackers:
        tracker = build_tracker(tracker_name)
        if args.verbose:
            print(f"Running {tracker_name}…", file=sys.stderr)
        t0 = time.perf_counter()
        result = engine.run(tracker, ds, dataset_name=args.dataset_name)
        elapsed = time.perf_counter() - t0
        if args.verbose:
            print(
                f"  {tracker_name}: mIoU={result.mean_iou:.4f}  "
                f"fps={result.mean_fps:.1f}  ({elapsed:.1f}s)",
                file=sys.stderr,
            )
        results.append(result)

    if not results:
        print("[error] No results produced.", file=sys.stderr)
        return 1

    report = MultiTrackerAttributeReport(
        results,
        metric=args.metric,
        min_sequences=args.min_sequences,
    )

    active = report.active_attributes()
    if args.verbose or not active:
        print(
            f"Active attributes ({len(active)}): {active or 'none'}",
            file=sys.stderr,
        )

    # Always print Markdown to stdout.
    print(report.to_markdown(show_counts=args.show_counts))

    # Rankings.
    print("\n### Overall ranking")
    for rank, (name, score) in enumerate(report.overall_ranking(), 1):
        print(f"  {rank}. {name}  ({score:.4f})")

    # Save files if requested.
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"attr_{args.metric}"

        if "markdown" in args.formats:
            p = report.save_markdown(args.out_dir / f"{stem}.md",
                                     show_counts=args.show_counts)
            print(f"Saved Markdown  → {p}", file=sys.stderr)

        if "latex" in args.formats:
            caption = (
                f"Per-attribute {args.metric} comparison on "
                f"{args.dataset_name}."
            )
            p = report.save_latex(args.out_dir / f"{stem}.tex",
                                  caption=caption)
            print(f"Saved LaTeX     → {p}", file=sys.stderr)

        if "csv" in args.formats:
            p = report.save_csv(args.out_dir / f"{stem}.csv")
            print(f"Saved CSV       → {p}", file=sys.stderr)

        if "json" in args.formats:
            p = report.save_json(args.out_dir / f"{stem}.json")
            print(f"Saved JSON      → {p}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
