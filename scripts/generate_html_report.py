#!/usr/bin/env python3
"""Generate a self-contained HTML benchmark report from JSON result files.

Loads one or more BenchmarkResult JSON files (produced by run_benchmark.py or
run_experiment.py) and renders them into a single, portable HTML report with
embedded charts, a performance summary table, and an efficiency scatter plot.

Usage examples::

    # Report from two saved result files
    python scripts/generate_html_report.py results/MOSSE.json results/KCF.json

    # Custom output path and title
    python scripts/generate_html_report.py results/*.json \\
        --output-dir reports \\
        --name otb100_comparison \\
        --title "OTB-100 Tracker Comparison"

    # Use a larger memory budget for EES
    python scripts/generate_html_report.py results/*.json --memory-budget 2048
"""

import argparse
import sys
from pathlib import Path

# Allow running from the repository root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from eovot.benchmark.engine import BenchmarkResult
from eovot.reporting.html_reporter import HTMLReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML benchmark report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        help="Paths to BenchmarkResult JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to write the HTML report.",
    )
    parser.add_argument(
        "--name",
        default="report",
        help="Output filename stem (without .html extension).",
    )
    parser.add_argument(
        "--title",
        default="EOVOT Benchmark Report",
        help="Report title shown in the page header.",
    )
    parser.add_argument(
        "--memory-budget",
        type=float,
        default=512.0,
        metavar="MB",
        help="Memory budget (MB) for Edge Efficiency Score computation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = []
    for path_str in args.result_files:
        path = Path(path_str)
        if not path.exists():
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            sys.exit(1)
        try:
            result = BenchmarkResult.load(path)
            results.append(result)
            print(f"  Loaded: {result.tracker_name} ({len(result.sequence_results)} sequences)")
        except Exception as exc:
            print(f"[ERROR] Failed to load {path}: {exc}", file=sys.stderr)
            sys.exit(1)

    if not results:
        print("[ERROR] No results loaded.", file=sys.stderr)
        sys.exit(1)

    reporter = HTMLReporter(
        output_dir=args.output_dir,
        memory_budget_mb=args.memory_budget,
    )
    out_path = reporter.generate(results, title=args.title, name=args.name)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
