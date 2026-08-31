#!/usr/bin/env python3
"""CLI: compute pairwise tracker failure correlation from saved benchmark JSONs.

Usage::

    # Compare two pre-saved benchmark JSON results
    python scripts/tracker_correlation.py \\
        results/MOSSE-OTB100.json \\
        results/KCF-OTB100.json \\
        results/CSRT-OTB100.json

    # Save the Markdown report
    python scripts/tracker_correlation.py \\
        results/MOSSE-OTB100.json results/KCF-OTB100.json \\
        --output-dir results/

The script prints the Pearson-r matrix and oracle-union ensemble gain matrix,
identifies the most complementary and most redundant tracker pairs, and
optionally saves a Markdown report.

All JSON inputs must be results from the **same dataset** (same sequences in
the same order) so that per-frame IoU arrays are aligned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.analysis.tracker_correlation import TrackerCorrelationAnalyzer
from eovot.benchmark.engine import BenchmarkResult


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tracker_correlation",
        description="Compute pairwise failure correlation between EOVOT trackers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        metavar="RESULT_JSON",
        help="Two or more benchmark JSON files (same dataset, same sequences).",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.85,
        metavar="R",
        help="Pearson r above which two trackers are placed in the same cluster.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to save the Markdown report (skipped when omitted).",
    )
    args = parser.parse_args()

    if len(args.result_files) < 2:
        parser.error("At least two result JSON files are required.")

    results: dict[str, BenchmarkResult] = {}
    for path in args.result_files:
        br = BenchmarkResult.load(path)
        if br.tracker_name in results:
            print(
                f"Warning: duplicate tracker name '{br.tracker_name}' — "
                f"using '{path}' as override.",
                file=sys.stderr,
            )
        results[br.tracker_name] = br
        print(f"Loaded: {br.tracker_name}  ({len(br.sequence_results)} sequences)")

    analyzer = TrackerCorrelationAnalyzer(cluster_threshold=args.cluster_threshold)
    report = analyzer.analyze(results)

    print()
    print(report.to_markdown())

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / "tracker_correlation.md"
        md_path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"\n[Markdown] saved → {md_path}")


if __name__ == "__main__":
    main()
