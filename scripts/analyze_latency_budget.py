#!/usr/bin/env python3
"""Latency budget analysis CLI for EOVOT.

Loads one or more JSON benchmark result files and evaluates whether each
tracker can sustain a set of target frame rates on edge hardware.  For
every (tracker, target-FPS) pair the tool reports:

- The per-frame deadline in ms
- Mean and p99 latency vs. the deadline
- A Gaussian-estimated compliance rate (fraction of frames within budget)
- A plain-language verdict: SAFE / MARGINAL / UNSAFE

Usage::

    # Analyse a single result against standard frame-rate targets
    python scripts/analyze_latency_budget.py results/mosse-synthetic.json

    # Custom target FPS and multiple results
    python scripts/analyze_latency_budget.py \\
        results/mosse-synthetic.json results/kcf-synthetic.json \\
        --fps 15 30 60 120

    # Use worst-case (highest-latency) sequence instead of the mean
    python scripts/analyze_latency_budget.py results/csrt-otb100.json \\
        --worst-case

    # Save Markdown report
    python scripts/analyze_latency_budget.py results/mosse-synthetic.json \\
        --output results/latency_budget_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.analysis.latency_budget import LatencyBudgetAnalyzer
from eovot.benchmark.engine import BenchmarkResult


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="analyze_latency_budget",
        description=(
            "Evaluate whether EOVOT trackers can meet real-time frame-rate "
            "deadlines based on their measured latency distributions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        metavar="RESULT_JSON",
        help="Path(s) to JSON result files produced by BenchmarkEngine.run().",
    )
    parser.add_argument(
        "--fps",
        nargs="+",
        type=float,
        default=[15.0, 24.0, 30.0, 60.0],
        metavar="FPS",
        help="Target frame rates to evaluate (Hz).",
    )
    parser.add_argument(
        "--worst-case",
        action="store_true",
        default=False,
        help=(
            "Use worst-case (highest mean latency) sequence instead of the "
            "cross-sequence average. More conservative for deployment decisions."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help=(
            "Optional path to write the Markdown report. "
            "When omitted the report is printed to stdout only."
        ),
    )
    args = parser.parse_args()

    analyzer = LatencyBudgetAnalyzer(target_fps_list=args.fps)
    reports = []

    for path_str in args.result_files:
        path = Path(path_str)
        if not path.exists():
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            sys.exit(1)
        try:
            result = BenchmarkResult.load(path)
        except Exception as exc:
            print(f"[ERROR] Could not load {path}: {exc}", file=sys.stderr)
            sys.exit(1)

        report = analyzer.analyze_benchmark(result, aggregate=not args.worst_case)
        reports.append(report)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    separator = "-" * 70
    for report in reports:
        print(f"\n{separator}")
        print(f"  Latency Budget Analysis: {report.tracker_name} on {report.dataset_name}")
        print(separator)
        print(report.to_markdown_table())
        print()
        print(f"  Recommendation: {report.recommendations()}")
        print()

    if len(reports) > 1:
        print(separator)
        print("  Multi-Tracker Comparison")
        print(separator)
        print(LatencyBudgetAnalyzer.multi_tracker_table(reports))
        print()

    # ------------------------------------------------------------------
    # Optional Markdown file output
    # ------------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write("# EOVOT Latency Budget Analysis\n\n")
            for report in reports:
                fh.write(report.to_markdown_table())
                fh.write("\n\n")
                fh.write(f"**Recommendation:** {report.recommendations()}\n\n")
            if len(reports) > 1:
                fh.write("## Multi-Tracker Comparison\n\n")
                fh.write(LatencyBudgetAnalyzer.multi_tracker_table(reports))
                fh.write("\n")
        print(f"Markdown report saved → {out_path}")


if __name__ == "__main__":
    main()
