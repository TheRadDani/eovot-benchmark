#!/usr/bin/env python3
"""CLI for generating benchmark visualizations from saved EOVOT result files.

Reads one or more JSON result files produced by
:meth:`~eovot.benchmark.engine.BenchmarkEngine.run` (or saved via
:class:`~eovot.reporting.reporter.BenchmarkReporter`) and generates
publication-quality plots.

Requires ``matplotlib``::

    pip install matplotlib

Usage
-----
Plot success curves for two trackers::

    python scripts/plot_results.py \\
        --results results/MOSSE-OTB100.json results/KCF-OTB100.json \\
        --plot success \\
        --output plots/success_curves.png

Plot comparison bar chart::

    python scripts/plot_results.py \\
        --results results/MOSSE-OTB100.json results/KCF-OTB100.json \\
        --plot comparison \\
        --output plots/comparison.png

Generate all plot types at once::

    python scripts/plot_results.py \\
        --results results/MOSSE-OTB100.json results/KCF-OTB100.json \\
        --plot all \\
        --output-dir plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.visualization.plots import (
    plot_precision_curves,
    plot_success_curves,
    plot_tracker_comparison,
)

PLOT_TYPES = ("success", "precision", "comparison", "all")


def load_results(paths: List[str]) -> list:
    """Load and return benchmark result dicts from JSON files."""
    results = []
    for p in paths:
        with open(p) as fh:
            data = json.load(fh)
        # Support both raw engine dicts and legacy summary-only JSONs.
        if "summary" not in data:
            data = {"summary": data, "sequences": []}
        results.append(data)
        tracker = data.get("summary", {}).get("tracker_name", Path(p).stem)
        n_seq = len(data.get("sequences", []))
        print(f"  Loaded {tracker!r}: {n_seq} sequences from {p}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plot_results",
        description="Generate EOVOT benchmark visualizations from saved JSON result files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more JSON result files to visualize.",
    )
    parser.add_argument(
        "--plot",
        choices=PLOT_TYPES,
        default="all",
        help=(
            "Plot type: 'success' (overlap success curves), "
            "'precision' (centre-distance precision curves), "
            "'comparison' (bar chart across metrics), "
            "or 'all' to generate every type."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Output file path (PNG / PDF / SVG).  Used only when --plot is "
            "not 'all'.  If omitted, the plot is shown interactively."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="plots/",
        metavar="DIR",
        help="Output directory when --plot all is used.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom figure title.  Auto-generated from tracker names when omitted.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["mean_iou", "mean_fps", "peak_memory_mb"],
        metavar="METRIC",
        help="Metrics to include in the comparison bar chart.",
    )
    args = parser.parse_args()

    print("\nLoading result files …")
    results = load_results(args.results)
    if not results:
        print("[ERROR] No valid result files loaded.", file=sys.stderr)
        sys.exit(1)

    names = " vs ".join(
        r.get("summary", {}).get("tracker_name", f"tracker_{i}")
        for i, r in enumerate(results)
    )
    auto_title = names

    def _out(stem: str, ext: str = ".png") -> str:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        return str(outdir / f"{stem}{ext}")

    plot_type = args.plot

    if plot_type in ("success", "all"):
        title = args.title or f"Success Curves — {auto_title}"
        out = args.output if plot_type == "success" else _out("success_curves")
        plot_success_curves(results, output_path=out, title=title)

    if plot_type in ("precision", "all"):
        title = args.title or f"Precision Curves — {auto_title}"
        out = args.output if plot_type == "precision" else _out("precision_curves")
        plot_precision_curves(results, output_path=out, title=title)

    if plot_type in ("comparison", "all"):
        title = args.title or f"Tracker Comparison — {auto_title}"
        out = args.output if plot_type == "comparison" else _out("tracker_comparison")
        plot_tracker_comparison(results, metrics=args.metrics, output_path=out, title=title)


if __name__ == "__main__":
    main()
