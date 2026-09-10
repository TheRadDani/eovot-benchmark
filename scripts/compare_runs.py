"""Compare two EOVOT benchmark result JSON files and report regressions/improvements.

Loads a *baseline* and a *candidate* :class:`~eovot.benchmark.engine.BenchmarkResult`
from JSON files (written by :meth:`~eovot.benchmark.engine.BenchmarkResult.save`)
and runs :class:`~eovot.analysis.comparator.BenchmarkComparator` to produce a
structured diff across accuracy, efficiency, and per-sequence metrics.

Usage
-----
    # Basic comparison (prints Markdown table to stdout)
    python scripts/compare_runs.py baseline.json candidate.json

    # Custom regression threshold (2 % relative change)
    python scripts/compare_runs.py baseline.json candidate.json --threshold 0.02

    # Save Markdown report
    python scripts/compare_runs.py baseline.json candidate.json --md report.md

    # Save CSV for CI artifact storage
    python scripts/compare_runs.py baseline.json candidate.json --csv comparison.csv

    # Save JSON for downstream pipelines
    python scripts/compare_runs.py baseline.json candidate.json --json comparison.json

    # Exit with code 1 if any metric regressed (useful in CI)
    python scripts/compare_runs.py baseline.json candidate.json --fail-on-regression

Exit codes
----------
    0 — comparison complete, no regressions (or --fail-on-regression not set)
    1 — regressions detected AND --fail-on-regression flag is set
    2 — argument or file error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkResult
from eovot.analysis.comparator import BenchmarkComparator


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compare_runs",
        description=(
            "Compare two EOVOT benchmark result JSON files and report "
            "per-metric regressions and improvements."
        ),
    )
    parser.add_argument(
        "baseline",
        metavar="BASELINE",
        help="Path to the baseline benchmark result JSON.",
    )
    parser.add_argument(
        "candidate",
        metavar="CANDIDATE",
        help="Path to the candidate benchmark result JSON (run under evaluation).",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.01,
        metavar="FRAC",
        help=(
            "Minimum relative change (as a fraction, not %%) required to flag "
            "a metric as improved or regressed. Default: 0.01 (1%%)."
        ),
    )
    parser.add_argument(
        "--md",
        metavar="PATH",
        default=None,
        help="Write the Markdown comparison table to this file.",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="Write a CSV summary to this file.",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Write the full comparison as JSON to this file.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if any metric regressed beyond the threshold.",
    )
    parser.add_argument(
        "--no-sequences",
        action="store_true",
        help="Skip per-sequence IoU delta computation.",
    )
    args = parser.parse_args()

    # --- Load results ---
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    for p in (baseline_path, candidate_path):
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(2)

    try:
        baseline = BenchmarkResult.load(baseline_path)
        candidate = BenchmarkResult.load(candidate_path)
    except Exception as exc:
        print(f"Error loading results: {exc}", file=sys.stderr)
        sys.exit(2)

    # --- Run comparison ---
    cmp = BenchmarkComparator(
        regression_threshold=args.threshold,
        sequence_match_by_name=not args.no_sequences,
    )
    result = cmp.compare(baseline, candidate)

    # --- Print Markdown table ---
    md = cmp.to_markdown_table(result)
    print(md)

    # --- Regression report ---
    if result.has_regressions:
        print()
        print(result.regression_report())

    # --- Optional outputs ---
    if args.md:
        Path(args.md).write_text(md + "\n", encoding="utf-8")
        print(f"\nMarkdown report written to {args.md}", file=sys.stderr)

    if args.csv:
        Path(args.csv).write_text(cmp.to_csv(result), encoding="utf-8")
        print(f"CSV written to {args.csv}", file=sys.stderr)

    if args.json:
        d = cmp.to_dict(result)
        Path(args.json).write_text(
            json.dumps(d, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"JSON written to {args.json}", file=sys.stderr)

    # --- CI exit code ---
    if args.fail_on_regression and result.has_regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
