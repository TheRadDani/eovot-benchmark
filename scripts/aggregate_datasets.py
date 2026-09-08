#!/usr/bin/env python
"""Cross-dataset rank aggregation for EOVOT benchmark results.

Loads saved BenchmarkResult JSON files (produced by scripts/run_benchmark.py
or BenchmarkResult.save()) from multiple datasets, applies multi-dataset rank
aggregation (Demšar 2006), and prints a comparison table in Markdown or LaTeX.

Usage::

    # Aggregate two JSON result files and print Markdown
    python scripts/aggregate_datasets.py \\
        results/mosse_got10k.json \\
        results/kcf_got10k.json \\
        results/mosse_lasot.json \\
        results/kcf_lasot.json

    # Use LaTeX output
    python scripts/aggregate_datasets.py --format latex results/*.json

    # Save to file
    python scripts/aggregate_datasets.py results/*.json --output leaderboard.md

    # Change primary ranking metric
    python scripts/aggregate_datasets.py results/*.json --metric mean_iou
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkResult
from eovot.metrics.aggregation import MultiDatasetAggregator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-dataset rank aggregation for EOVOT benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        metavar="RESULT_JSON",
        help="Paths to BenchmarkResult JSON files (at least 2 trackers × 2 datasets).",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "latex", "both"],
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--metric",
        default="success_auc",
        help="Primary metric key for final ranking (default: success_auc).",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Write output to FILE instead of stdout.",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Also print per-dataset breakdown tables.",
    )
    args = parser.parse_args()

    # --- Load results ---
    results = []
    for path in args.json_files:
        p = Path(path)
        if not p.exists():
            print(f"[warning] File not found: {p}", file=sys.stderr)
            continue
        try:
            results.append(BenchmarkResult.load(p))
        except Exception as exc:
            print(f"[warning] Could not load {p}: {exc}", file=sys.stderr)

    if len(results) < 2:
        print("[error] Need at least 2 result files to aggregate.", file=sys.stderr)
        sys.exit(1)

    n_trackers = len({r.tracker_name for r in results})
    n_datasets = len({r.dataset_name for r in results})
    print(
        f"Loaded {len(results)} result(s): "
        f"{n_trackers} tracker(s) × {n_datasets} dataset(s)",
        file=sys.stderr,
    )

    # --- Aggregate ---
    agg = MultiDatasetAggregator(primary_metric=args.metric)
    try:
        report = agg.aggregate(results)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Format output ---
    output_parts: list[str] = []

    if args.format in ("markdown", "both"):
        output_parts.append("## Cross-Dataset Tracker Ranking\n")
        output_parts.append(agg.to_markdown(report))
        if args.breakdown:
            output_parts.append("\n\n## Per-Dataset Breakdown\n")
            output_parts.append(agg.to_per_dataset_markdown(report))

    if args.format in ("latex", "both"):
        if output_parts:
            output_parts.append("\n\n---\n\n")
        output_parts.append(agg.to_latex(report))

    output = "\n".join(output_parts)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Saved to {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
