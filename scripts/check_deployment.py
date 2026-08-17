#!/usr/bin/env python
"""CLI tool — check whether a benchmark result meets deployment requirements.

Reads a JSON result file produced by :class:`~eovot.benchmark.engine.BenchmarkEngine`
(via ``BenchmarkResult.save()`` or ``scripts/run_benchmark.py``), compares it
against a user-supplied deployment profile, and prints a GO / NO-GO report with
per-metric headroom values.

Usage examples
--------------
Check a KCF result against Raspberry Pi 4 requirements::

    python scripts/check_deployment.py \\
        results/KCF-Synthetic.json \\
        --device "Raspberry Pi 4" \\
        --fps 30 \\
        --memory 256 \\
        --latency-mean 33.3 \\
        --latency-p99 50 \\
        --min-iou 0.45

Save the report as JSON::

    python scripts/check_deployment.py \\
        results/KCF-Synthetic.json \\
        --fps 30 --memory 256 \\
        --output reports/KCF-pi4-deployment.json

Compare multiple trackers against the same profile::

    for tracker in MOSSE KCF CSRT; do
        python scripts/check_deployment.py results/${tracker}-Synthetic.json \\
            --fps 25 --memory 512 --min-iou 0.4
    done
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the package root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkResult
from eovot.analysis.deployment import DeploymentChecker, DeploymentProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a benchmark result meets edge deployment requirements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "result",
        metavar="RESULT_JSON",
        help="Path to a BenchmarkResult JSON file (from run_benchmark.py or BenchmarkResult.save()).",
    )
    parser.add_argument(
        "--device",
        default="unspecified",
        metavar="NAME",
        help="Human-readable name for the target deployment device (default: 'unspecified').",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        metavar="FPS",
        help="Minimum required mean frames per second.",
    )
    parser.add_argument(
        "--memory",
        type=float,
        default=None,
        metavar="MB",
        help="Maximum allowed peak memory usage in megabytes.",
    )
    parser.add_argument(
        "--latency-mean",
        type=float,
        default=None,
        metavar="MS",
        dest="latency_mean",
        help="Maximum allowed mean per-frame latency in milliseconds.",
    )
    parser.add_argument(
        "--latency-p99",
        type=float,
        default=None,
        metavar="MS",
        dest="latency_p99",
        help="Maximum allowed 99th-percentile latency in milliseconds.",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=None,
        metavar="IOU",
        dest="min_iou",
        help="Minimum required mean IoU across all sequences (0–1).",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=None,
        metavar="AUC",
        dest="min_auc",
        help="Minimum required mean success-curve AUC (0–1).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Save the deployment report to this JSON file.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print the report in Markdown format instead of plain text.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output; use the exit code only (0=GO, 1=NO-GO).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load result
    result_path = Path(args.result)
    if not result_path.exists():
        print(f"error: result file not found: {result_path}", file=sys.stderr)
        return 2
    try:
        result = BenchmarkResult.load(result_path)
    except Exception as exc:
        print(f"error: failed to load result: {exc}", file=sys.stderr)
        return 2

    # Build deployment profile from CLI args
    profile = DeploymentProfile(
        target_fps=args.fps,
        memory_budget_mb=args.memory,
        max_latency_mean_ms=args.latency_mean,
        max_latency_p99_ms=args.latency_p99,
        min_iou=args.min_iou,
        min_success_auc=args.min_auc,
        device_name=args.device,
    )

    # Check: at least one requirement must be specified
    requirements = [
        args.fps, args.memory, args.latency_mean,
        args.latency_p99, args.min_iou, args.min_auc,
    ]
    if all(r is None for r in requirements):
        print(
            "warning: no deployment requirements specified — report will be empty.\n"
            "Use --fps, --memory, --latency-mean, --latency-p99, --min-iou, or --min-auc.",
            file=sys.stderr,
        )

    # Run checker
    checker = DeploymentChecker()
    report = checker.check(result, profile)

    # Output
    if not args.quiet:
        if args.markdown:
            print(report.to_markdown())
        else:
            print(report)

    if args.output:
        saved = report.save(args.output)
        if not args.quiet:
            print(f"\nReport saved to: {saved}")

    # Exit code: 0 = deployable (GO), 1 = not deployable (NO-GO)
    return 0 if report.deployable else 1


if __name__ == "__main__":
    sys.exit(main())
