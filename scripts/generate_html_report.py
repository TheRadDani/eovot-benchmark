#!/usr/bin/env python3
"""Generate a self-contained HTML benchmark report from saved EOVOT result files.

Reads one or more JSON result files produced by
:meth:`~eovot.benchmark.engine.BenchmarkResult.save` and renders a single
HTML file containing a summary table, success curves, efficiency scatter
plot, and per-sequence details.  The output requires no internet access or
external assets.

Usage
-----
Single tracker::

    python scripts/generate_html_report.py \\
        --results results/MOSSE-OTB100.json \\
        --output  results/reports/mosse_report.html

Multiple trackers (comparison report)::

    python scripts/generate_html_report.py \\
        --results results/MOSSE-OTB100.json results/KCF-OTB100.json \\
        --title   "Classical Tracker Comparison" \\
        --output  results/reports/comparison.html

Run immediately after a benchmark::

    python scripts/run_benchmark.py --config configs/default.yaml
    python scripts/generate_html_report.py \\
        --results results/MOSSE-OTB100.json \\
        --output  results/reports/latest.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkResult
from eovot.reporting.html_report import HTMLReportGenerator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a self-contained HTML benchmark report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[0],
    )
    p.add_argument(
        "--results",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more JSON result files produced by BenchmarkResult.save().",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help=(
            "Output HTML file path.  Defaults to the directory of the first "
            "result file, named 'benchmark_report.html'."
        ),
    )
    p.add_argument(
        "--title",
        default="EOVOT Benchmark Report",
        help="Report title shown in the page header and browser tab.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    results = []
    for path_str in args.results:
        path = Path(path_str)
        if not path.exists():
            print(f"[error] Result file not found: {path}", file=sys.stderr)
            sys.exit(1)
        print(f"  Loading {path} …")
        results.append(BenchmarkResult.load(path))

    if not results:
        print("[error] No valid result files provided.", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
        output_dir = str(out_path.parent)
        filename = out_path.name
    else:
        output_dir = str(Path(args.results[0]).parent / "reports")
        filename = "benchmark_report.html"

    gen = HTMLReportGenerator(output_dir=output_dir)
    saved = gen.generate(results, filename=filename, title=args.title)
    print(f"\nReport written to: {saved}")
    print(f"Open with:  xdg-open '{saved}'  (Linux)  or  open '{saved}'  (macOS)")


if __name__ == "__main__":
    main()
