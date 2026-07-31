#!/usr/bin/env python3
"""CLI: compare two EOVOT benchmark result JSON files for regressions.

Loads two JSON result files produced by
:meth:`eovot.benchmark.engine.BenchmarkResult.save`, compares every tracked
scalar metric, and prints a colour-coded table with IMPROVED / PASS / WARN /
FAIL status per metric.

Usage
-----
::

    # Basic comparison
    python scripts/diff_results.py baseline.json current.json

    # Tighter thresholds (1 % warn, 3 % fail)
    python scripts/diff_results.py baseline.json current.json \\
        --warn-pct 1.0 --fail-pct 3.0

    # Plain output for CI log files
    python scripts/diff_results.py baseline.json current.json --no-color

    # Write a machine-readable JSON report alongside the terminal output
    python scripts/diff_results.py baseline.json current.json \\
        --output-json reports/regression_check.json

Exit codes (CI-compatible)
--------------------------
* ``0`` — all metrics PASS or IMPROVED
* ``1`` — at least one WARN, no FAILs
* ``2`` — at least one FAIL

Example workflow (GitHub Actions)
----------------------------------
::

    - name: Benchmark regression check
      run: |
        python scripts/run_benchmark.py --config configs/baseline.yaml \\
            --output-json /tmp/baseline.json
        python scripts/run_benchmark.py --config configs/current.yaml  \\
            --output-json /tmp/current.json
        python scripts/diff_results.py /tmp/baseline.json /tmp/current.json \\
            --warn-pct 2 --fail-pct 5 --no-color
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.analysis.regression import RegressionDetector, load_scalars_from_file


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="diff_results",
        description="Compare two EOVOT benchmark JSON result files for regressions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("baseline", type=Path, help="Baseline result JSON file.")
    p.add_argument("current",  type=Path, help="Current result JSON file to compare against baseline.")
    p.add_argument(
        "--warn-pct",
        type=float,
        default=2.0,
        metavar="PCT",
        help="Relative %% change that triggers a WARN (default: 2.0).",
    )
    p.add_argument(
        "--fail-pct",
        type=float,
        default=5.0,
        metavar="PCT",
        help="Relative %% change that triggers a FAIL (default: 5.0).",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour output (for plain CI log files).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional path to write the full regression report as JSON.",
    )
    p.add_argument(
        "--tracker",
        type=str,
        default=None,
        metavar="NAME",
        help="Tracker name override (inferred from JSON 'tracker' field if omitted).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="NAME",
        help="Dataset name override (inferred from JSON 'dataset' field if omitted).",
    )
    return p


def _infer_field(raw: dict, key: str, fallback: str) -> str:
    summary = raw.get("summary", raw)
    return str(summary.get(key, fallback))


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)

    for attr, label in [("baseline", "Baseline"), ("current", "Current")]:
        p = getattr(args, attr)
        if not p.exists():
            print(f"Error: {label} file not found: {p}", file=sys.stderr)
            return 2

    with open(args.baseline, "r", encoding="utf-8") as fh:
        baseline_raw = json.load(fh)
    with open(args.current, "r", encoding="utf-8") as fh:
        current_raw = json.load(fh)

    tracker = args.tracker or _infer_field(baseline_raw, "tracker", "unknown")
    dataset = args.dataset or _infer_field(baseline_raw, "dataset", "unknown")

    baseline_scalars = load_scalars_from_file(args.baseline)
    current_scalars  = load_scalars_from_file(args.current)

    if not baseline_scalars:
        print("Warning: no tracked metrics found in baseline file.", file=sys.stderr)
    if not current_scalars:
        print("Warning: no tracked metrics found in current file.", file=sys.stderr)

    detector = RegressionDetector(warn_pct=args.warn_pct, fail_pct=args.fail_pct)
    report = detector.compare(
        baseline_scalars,
        current_scalars,
        tracker_name=tracker,
        dataset_name=dataset,
    )

    print(report.summary(color=not args.no_color))

    if args.output_json is not None:
        out_path = report.to_json(args.output_json)
        print(f"\nReport saved: {out_path}")

    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
