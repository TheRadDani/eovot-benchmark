#!/usr/bin/env python3
"""Generate a self-contained HTML benchmark report from saved EOVOT JSON results.

Reads one or more JSON files produced by the EOVOT benchmark engine and
renders a single HTML file with a comparison table, per-sequence breakdowns,
and embedded accuracy/FPS charts.

Usage::

    # Single tracker
    python scripts/generate_report.py \\
        --results results/MOSSE-OTB100.json

    # Multi-tracker comparison
    python scripts/generate_report.py \\
        --results results/MOSSE-OTB100.json results/KCF-OTB100.json \\
        --title "Classical Trackers on OTB-100" \\
        --name classical-comparison

    # Custom output directory, no embedded plots
    python scripts/generate_report.py \\
        --results results/*.json \\
        --output-dir reports/ \\
        --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.reporting.html_reporter import HTMLReporter


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate_report",
        description="Generate a self-contained HTML benchmark report from EOVOT JSON results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        metavar="JSON_FILE",
        help="One or more JSON result files from the EOVOT benchmark engine.",
    )
    parser.add_argument(
        "--name",
        default="benchmark-report",
        help="Base filename for the HTML output (without .html extension).",
    )
    parser.add_argument(
        "--title",
        default="EOVOT Benchmark Report",
        help="Page heading shown in the HTML report.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Directory where the HTML file is written.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable embedded matplotlib charts (useful when matplotlib is not installed).",
    )
    args = parser.parse_args()

    results = []
    for path_str in args.results:
        path = Path(path_str)
        if not path.exists():
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path, encoding="utf-8") as fh:
            results.append(json.load(fh))
        print(f"  [LOADED] {path}")

    reporter = HTMLReporter(output_dir=args.output_dir)
    out_path = reporter.save(
        results=results,
        name=args.name,
        title=args.title,
        embed_plots=not args.no_plots,
    )
    print(f"\n[HTML REPORT] saved → {out_path}")


if __name__ == "__main__":
    main()
