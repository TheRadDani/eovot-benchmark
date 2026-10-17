#!/usr/bin/env python3
"""CI-grade performance regression checker for EOVOT benchmark results.

Compares two saved BenchmarkResult JSON files (baseline and candidate)
and exits with code 0 if no regressions are detected, or code 1 if a
regression is found.  Designed for use in GitHub Actions or pre-merge hooks.

Usage::

    # Basic check — exits 1 if any metric regressed
    python scripts/check_regression.py \\
        --baseline results/mosse_before.json \\
        --candidate results/mosse_after.json

    # Strict mode — any regression, even non-significant IoU drops, fails
    python scripts/check_regression.py \\
        --baseline results/kcf_v1.json \\
        --candidate results/kcf_v2.json \\
        --strict

    # Adjust sensitivity
    python scripts/check_regression.py \\
        --baseline results/mosse_v1.json \\
        --candidate results/mosse_v2.json \\
        --min-iou-delta 0.01 \\
        --significance-level 0.01

    # Write report to JSON for CI artifact upload
    python scripts/check_regression.py \\
        --baseline results/mosse_v1.json \\
        --candidate results/mosse_v2.json \\
        --output-json regression_report.json

GitHub Actions example::

    - name: Check for tracking regressions
      run: |
        python scripts/check_regression.py \\
          --baseline results/baseline.json \\
          --candidate results/candidate.json \\
          --output-json ${{ github.workspace }}/regression.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.benchmark.engine import BenchmarkResult
from eovot.analysis.regression_checker import RegressionChecker


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="check_regression",
        description="Detect performance regressions between two EOVOT benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline", "-b",
        required=True,
        metavar="JSON",
        help="Path to the baseline BenchmarkResult JSON (before the change).",
    )
    parser.add_argument(
        "--candidate", "-c",
        required=True,
        metavar="JSON",
        help="Path to the candidate BenchmarkResult JSON (after the change).",
    )
    parser.add_argument(
        "--min-iou-delta",
        type=float,
        default=0.005,
        metavar="FLOAT",
        help="Minimum absolute IoU drop to flag as a regression.",
    )
    parser.add_argument(
        "--min-fps-delta",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Minimum absolute FPS drop to flag as a regression.",
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help="p-value threshold for the IoU paired t-test.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Flag IoU regressions even when not statistically significant.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        metavar="PATH",
        help="Write the regression report dict to a JSON file.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress the text summary; only print the final status line.",
    )
    args = parser.parse_args()

    baseline_path  = Path(args.baseline)
    candidate_path = Path(args.candidate)

    for p in (baseline_path, candidate_path):
        if not p.exists():
            print(f"[error] File not found: {p}", file=sys.stderr)
            sys.exit(2)

    print(f"Loading baseline:  {baseline_path}")
    baseline = BenchmarkResult.load(baseline_path)
    print(f"Loading candidate: {candidate_path}")
    candidate = BenchmarkResult.load(candidate_path)

    checker = RegressionChecker(
        min_iou_delta=args.min_iou_delta,
        min_fps_delta=args.min_fps_delta,
        significance_level=args.significance_level,
        strict=args.strict,
    )
    report = checker.compare(baseline, candidate)

    if not args.quiet:
        print()
        print(report.summary())
    else:
        status = "REGRESSION" if report.has_regression else "OK"
        print(f"\n{status}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"\nReport written to {out_path}")

    sys.exit(1 if report.has_regression else 0)


if __name__ == "__main__":
    main()
