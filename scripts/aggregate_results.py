#!/usr/bin/env python3
"""CLI tool to aggregate and rank benchmark results across datasets and sessions.

Usage examples::

    # Rank all JSON results in a directory
    python scripts/aggregate_results.py results/

    # Filter to one dataset and output Markdown + CSV
    python scripts/aggregate_results.py results/ --dataset OTB100 --csv out.csv

    # Cross-dataset summary
    python scripts/aggregate_results.py results/*.json --cross-dataset

    # Custom edge-scoring weights
    python scripts/aggregate_results.py results/ \\
        --accuracy-weight 0.4 --speed-weight 0.4 --memory-weight 0.2
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.reporting.aggregator import ResultAggregator


def _collect_json_files(inputs: list[str]) -> list[Path]:
    """Expand directories to their *.json contents; keep plain file paths as-is."""
    files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif p.suffix == ".json":
            files.append(p)
        else:
            print(f"[warn] Skipping non-JSON path: {p}", file=sys.stderr)
    return files


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="aggregate_results",
        description="Aggregate EOVOT benchmark results across datasets and sessions.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="PATH",
        help="JSON result files or directories containing them.",
    )
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        default=None,
        help="Filter to a specific dataset (e.g. OTB100, GOT10k, LaSOT).",
    )
    parser.add_argument(
        "--sort-by",
        metavar="FIELD",
        default="composite",
        help=(
            "Sort field. One of: composite (default), mean_iou, success_auc, "
            "precision_auc, mean_fps, peak_memory_mb."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help="Show only the top N trackers.",
    )
    parser.add_argument(
        "--cross-dataset",
        action="store_true",
        help="Print cross-dataset aggregated summary instead of per-result leaderboard.",
    )
    parser.add_argument(
        "--csv",
        metavar="FILE",
        default=None,
        help="Write all ranked entries to a CSV file.",
    )
    parser.add_argument(
        "--json-out",
        metavar="FILE",
        default=None,
        help="Write all entries to a JSON file.",
    )
    parser.add_argument(
        "--accuracy-weight",
        type=float,
        default=0.5,
        help="Weight for accuracy in composite score (default: 0.5).",
    )
    parser.add_argument(
        "--speed-weight",
        type=float,
        default=0.3,
        help="Weight for FPS in composite score (default: 0.3).",
    )
    parser.add_argument(
        "--memory-weight",
        type=float,
        default=0.2,
        help="Weight for memory efficiency in composite score (default: 0.2).",
    )
    parser.add_argument(
        "--fps-scale",
        type=float,
        default=30.0,
        help="FPS value treated as 'full score' for speed component (default: 30).",
    )
    parser.add_argument(
        "--mem-scale",
        type=float,
        default=512.0,
        help="Memory (MB) treated as 'zero score' for memory component (default: 512).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    files = _collect_json_files(args.inputs)
    if not files:
        print("[error] No JSON files found in the provided inputs.", file=sys.stderr)
        sys.exit(1)

    agg = ResultAggregator(
        accuracy_weight=args.accuracy_weight,
        speed_weight=args.speed_weight,
        memory_weight=args.memory_weight,
        fps_scale=args.fps_scale,
        mem_scale=args.mem_scale,
    )

    loaded = 0
    for f in files:
        try:
            agg.load(f)
            loaded += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Skipping {f}: {exc}", file=sys.stderr)

    if loaded == 0:
        print("[error] No valid result files could be loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {loaded} file(s), {len(agg.entries())} entries.\n")

    if args.cross_dataset:
        print(agg.cross_dataset_summary())
    else:
        print(
            agg.leaderboard(
                dataset=args.dataset,
                sort_by=args.sort_by,
                top_n=args.top_n,
            )
        )

    if args.csv:
        agg.to_csv(args.csv)
        print(f"\nCSV written to {args.csv}")

    if args.json_out:
        agg.to_json(args.json_out)
        print(f"JSON written to {args.json_out}")


if __name__ == "__main__":
    main()
