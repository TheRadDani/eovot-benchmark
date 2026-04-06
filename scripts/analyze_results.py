#!/usr/bin/env python3
"""Statistical analysis CLI for EOVOT benchmark results.

Loads JSON result files produced by the benchmark engine and performs
pairwise statistical comparisons between trackers, outputting a ranked
summary table with significance annotations.

Usage
-----
    # Compare two trackers on per-sequence IoU
    python scripts/analyze_results.py \\
        results/MOSSE-OTB100.json \\
        results/KCF-OTB100.json

    # Compare multiple trackers and save the ranking table
    python scripts/analyze_results.py \\
        results/MOSSE-OTB100.json \\
        results/KCF-OTB100.json \\
        results/CSRT-OTB100.json \\
        --metric mean_iou \\
        --output-dir results/analysis/

    # Compare on FPS instead of IoU
    python scripts/analyze_results.py \\
        results/MOSSE-OTB100.json \\
        results/KCF-OTB100.json \\
        --metric fps

    # Adjust significance level and bootstrap samples
    python scripts/analyze_results.py \\
        results/MOSSE-OTB100.json \\
        results/KCF-OTB100.json \\
        --alpha 0.01 \\
        --n-bootstrap 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.analysis.stats import (
    compare_trackers,
    load_sequence_metric,
    rank_trackers,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_results",
        description="Statistical comparison of EOVOT tracker result JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        metavar="RESULT_JSON",
        help="One or more EOVOT result JSON files (one per tracker).",
    )
    parser.add_argument(
        "--metric",
        default="mean_iou",
        metavar="KEY",
        help=(
            "Per-sequence metric key to compare. "
            "Common choices: mean_iou, fps, peak_memory_mb, energy_per_frame_mj."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="Significance level for the Wilcoxon signed-rank test.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        metavar="N",
        help="Number of bootstrap resamples for confidence interval estimation.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "If provided, save ranking_<metric>.json and ranking_<metric>.md "
            "to this directory."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load per-sequence metric values from each result file
    # ------------------------------------------------------------------
    tracker_scores = {}
    for path in args.result_files:
        if not os.path.isfile(path):
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            sys.exit(1)
        try:
            name, values = load_sequence_metric(path, metric_key=args.metric)
        except KeyError as exc:
            print(
                f"[ERROR] Metric '{args.metric}' not found in {path}. "
                f"Missing key: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        if len(values) == 0:
            print(
                f"[WARNING] No sequences with metric '{args.metric}' in {path} "
                f"— skipping.",
                file=sys.stderr,
            )
            continue

        if name in tracker_scores:
            # Disambiguate duplicate tracker names using filename stem
            name = f"{name} ({Path(path).stem})"
        tracker_scores[name] = values
        print(
            f"  Loaded {len(values):>4d} sequences  "
            f"mean {args.metric}={values.mean():.4f}  "
            f"[{name}]"
        )

    if len(tracker_scores) < 2:
        print(
            "\n[ERROR] At least 2 result files with parsable sequences are required.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Rank and compare
    # ------------------------------------------------------------------
    print(f"\nRunning statistical analysis (metric={args.metric}, α={args.alpha}) …")
    table = rank_trackers(
        tracker_scores,
        metric=args.metric,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
    )

    # ------------------------------------------------------------------
    # Print results to stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(table.to_markdown())
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save to output directory if requested
    # ------------------------------------------------------------------
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        stem = f"ranking_{args.metric}"

        json_path = os.path.join(args.output_dir, f"{stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(table.to_dict(), f, indent=2)
        print(f"\n[JSON] Saved → {json_path}")

        md_path = os.path.join(args.output_dir, f"{stem}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(table.to_markdown())
            f.write("\n")
        print(f"[MD]   Saved → {md_path}")


if __name__ == "__main__":
    main()
