"""CLI to diff two BenchmarkResult JSON files and detect regressions.

Useful in tracker development workflows: when you tune a hyperparameter or
change an algorithm, run this tool against the saved baseline and candidate
result files to see which sequences improved and which regressed — before
committing the change.

Usage
-----
    # Compare two saved results:
    python scripts/diff_results.py baseline.json candidate.json

    # Set a custom regression threshold and rate limit:
    python scripts/diff_results.py baseline.json candidate.json \\
        --threshold 0.03 --rate-limit 0.15

    # Fail (exit code 1) when the candidate is rejected (useful in CI):
    python scripts/diff_results.py baseline.json candidate.json --fail-on-regression

    # Save the Markdown report to a file:
    python scripts/diff_results.py baseline.json candidate.json --output diff.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.analysis.regression import BenchmarkDiff, RegressionError
from eovot.benchmark.engine import BenchmarkResult


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two BenchmarkResult JSON files and detect regressions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("baseline", help="Path to baseline BenchmarkResult JSON.")
    p.add_argument("candidate", help="Path to candidate BenchmarkResult JSON.")
    p.add_argument(
        "--threshold", type=float, default=0.02,
        help="Absolute IoU change to count as regression/improvement (default: 0.02).",
    )
    p.add_argument(
        "--rate-limit", type=float, default=0.25,
        help="Max fraction of regressed sequences before candidate is rejected (default: 0.25).",
    )
    p.add_argument(
        "--require-improvement", action="store_true",
        help="Also reject candidate if aggregate mIoU does not improve.",
    )
    p.add_argument(
        "--fail-on-regression", action="store_true",
        help="Exit with code 1 when the candidate is rejected (CI mode).",
    )
    p.add_argument(
        "--output", default=None,
        help="Write Markdown report to this file (prints to stdout if omitted).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    baseline = BenchmarkResult.load(args.baseline)
    candidate = BenchmarkResult.load(args.candidate)

    diff = BenchmarkDiff(
        regression_threshold=args.threshold,
        regression_rate_limit=args.rate_limit,
        require_iou_improvement=args.require_improvement,
    )
    report = diff.compare(baseline, candidate)

    print(report)

    md = report.to_markdown()
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"\nMarkdown report saved to: {args.output}")
    else:
        print("\n" + md)

    if args.fail_on_regression:
        try:
            report.raise_if_regressed()
        except RegressionError as e:
            print(f"\n[CI] {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
