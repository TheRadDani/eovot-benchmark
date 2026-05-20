"""CLI tool: estimate and compare algorithmic complexity of EOVOT trackers.

Reports parameter count, FLOPs per frame, and model size for all built-in
trackers.  Useful for understanding why certain trackers are faster or more
memory-efficient on edge hardware, and for reasoning about deployment cost
before running a full benchmark.

Usage
-----
    # Print table for all trackers (default patch_size=64 px):
    python scripts/analyze_complexity.py

    # Custom patch size:
    python scripts/analyze_complexity.py --patch-size 128

    # Single tracker with verbose notes:
    python scripts/analyze_complexity.py --tracker KCF --verbose

    # Save JSON report:
    python scripts/analyze_complexity.py --output complexity_report.json

    # Markdown table (for documentation or GitHub issues):
    python scripts/analyze_complexity.py --format markdown

Example output (patch_size=64)
-------------------------------
    EOVOT Tracker Complexity Analysis  (patch=64 px, scale=2.0×)

    ------------------------------------------------------
    Tracker          Params     MFLOPs/fr     Size (MB)
    ------------------------------------------------------
    MOSSE            16,384         0.060        0.0625
    KCF             507,904         0.887        1.9375
    CSRT            853,504         1.481        3.2578
    MIL                 256     2,949.120        0.0010
    MedianFlow          1,000         0.500        0.0038
    ------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Allow running directly from the scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.profiling.complexity import (
    SUPPORTED_TRACKERS,
    ComplexityReport,
    TrackerComplexityAnalyzer,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _table_plain(reports: List[ComplexityReport]) -> str:
    header = (
        f"{'Tracker':<14} {'Params':>12} {'MFLOPs/fr':>12} {'Size (MB)':>12}"
    )
    sep = "-" * len(header)
    lines = ["\n" + sep, header, sep]
    for r in reports:
        lines.append(
            f"{r.tracker_name:<14} "
            f"{r.param_count:>12,} "
            f"{r.mflops:>12.3f} "
            f"{r.model_size_mb:>12.4f}"
        )
    lines.append(sep + "\n")
    return "\n".join(lines)


def _table_markdown(reports: List[ComplexityReport]) -> str:
    lines = [
        "| Tracker | Params | MFLOPs/frame | Size (MB) |",
        "|---------|-------:|-------------:|----------:|",
    ]
    for r in reports:
        lines.append(
            f"| {r.tracker_name} "
            f"| {r.param_count:,} "
            f"| {r.mflops:.3f} "
            f"| {r.model_size_mb:.4f} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_complexity",
        description=(
            "Estimate algorithmic complexity (FLOPs, parameters, model size) "
            "for EOVOT trackers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/analyze_complexity.py\n"
            "  python scripts/analyze_complexity.py --patch-size 128 --tracker KCF\n"
            "  python scripts/analyze_complexity.py --format markdown --output report.md\n"
        ),
    )
    p.add_argument(
        "--tracker", "-t",
        choices=SUPPORTED_TRACKERS,
        default=None,
        metavar="NAME",
        help=(
            f"Analyse a single tracker. "
            f"Choices: {', '.join(SUPPORTED_TRACKERS)}. "
            "Defaults to all trackers."
        ),
    )
    p.add_argument(
        "--patch-size", "-p",
        type=int,
        default=64,
        metavar="N",
        help="Template patch side length in pixels (default: 64).",
    )
    p.add_argument(
        "--search-scale", "-s",
        type=float,
        default=2.0,
        metavar="SCALE",
        help="Search-region scale relative to patch side (default: 2.0).",
    )
    p.add_argument(
        "--format", "-f",
        choices=["plain", "markdown"],
        default="plain",
        help="Output table format (default: plain).",
    )
    p.add_argument(
        "--output", "-o",
        metavar="FILE",
        help=(
            "Save a JSON report to FILE. "
            "The JSON includes patch_size, search_scale, and a list of "
            "per-tracker dicts."
        ),
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-tracker algorithmic notes after the table.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    analyzer = TrackerComplexityAnalyzer(
        patch_size=args.patch_size,
        search_scale=args.search_scale,
    )

    names = [args.tracker] if args.tracker else SUPPORTED_TRACKERS
    reports = [analyzer.analyze(n) for n in names]

    print(
        f"\nEOVOT Tracker Complexity Analysis  "
        f"(patch={args.patch_size} px, scale={args.search_scale}×)"
    )

    if args.format == "markdown":
        print(_table_markdown(reports))
    else:
        print(_table_plain(reports))

    if args.verbose:
        print("Notes:")
        for r in reports:
            print(f"  [{r.tracker_name}] {r.notes}")
        print()

    if args.output:
        data = {
            "patch_size": args.patch_size,
            "search_scale": args.search_scale,
            "trackers": [r.to_dict() for r in reports],
        }
        out_path = Path(args.output)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"JSON report saved to {out_path}")


if __name__ == "__main__":
    main()
