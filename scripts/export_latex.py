"""CLI for exporting EOVOT benchmark results to a LaTeX table.

Reads one or more JSON result files produced by
:meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict` and writes a
publication-ready LaTeX ``table`` environment (booktabs style) to stdout
or a ``.tex`` file.

Usage
-----
    # Print to stdout from a single result file
    python scripts/export_latex.py results/mosse_otb100.json

    # Combine multiple files into one table and save
    python scripts/export_latex.py results/*.json -o paper/tables/comparison.tex

    # Choose specific columns and disable bold-facing
    python scripts/export_latex.py results/mosse.json \\
        --columns tracker miou fps ees \\
        --no-bold

    # Preview as Markdown (no LaTeX compiler needed)
    python scripts/export_latex.py results/mosse.json --markdown

    # Custom caption / label
    python scripts/export_latex.py results/*.json \\
        --caption "Tracker comparison on OTB-100" \\
        --label tab:otb100 \\
        -o paper/tables/otb100.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.reporting.latex_report import LaTeXReportGenerator, DEFAULT_COLUMNS, _BUILTIN_COLUMNS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_latex",
        description="Export EOVOT benchmark results to a LaTeX table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results",
        nargs="+",
        metavar="RESULT_JSON",
        help="One or more JSON files from BenchmarkResult.to_dict().",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="PATH",
        help="Write output to this file instead of stdout. "
             "A .tex extension is appended automatically when omitted.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=list(DEFAULT_COLUMNS),
        choices=sorted(_BUILTIN_COLUMNS),
        metavar="COL",
        help=(
            f"Columns to include. Available: {sorted(_BUILTIN_COLUMNS)}. "
            f"Default: {list(DEFAULT_COLUMNS)}."
        ),
    )
    parser.add_argument(
        "--no-bold",
        action="store_true",
        help="Disable automatic bold-facing of the best value per column.",
    )
    parser.add_argument(
        "--no-group",
        action="store_true",
        help="Disable \\midrule separators between datasets.",
    )
    parser.add_argument(
        "--memory-budget",
        type=float,
        default=512.0,
        metavar="MB",
        help="Memory ceiling (MB) used for EES computation. Default: 512.",
    )
    parser.add_argument(
        "--caption",
        default="EOVOT Tracker Comparison",
        help="Table caption string.",
    )
    parser.add_argument(
        "--label",
        default="tab:eovot-comparison",
        help="BibTeX cross-reference label.",
    )
    parser.add_argument(
        "--position",
        default="htbp",
        help="LaTeX float placement specifier. Default: htbp.",
    )
    parser.add_argument(
        "--no-preamble",
        action="store_true",
        help="Omit the package comment block from the output.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print a Markdown table preview instead of LaTeX.",
    )
    return parser


def _load_result(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Cannot parse JSON from {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    results = [_load_result(p) for p in args.results]

    try:
        gen = LaTeXReportGenerator(
            columns=args.columns,
            bold_best=not args.no_bold,
            group_by_dataset=not args.no_group,
            memory_budget_mb=args.memory_budget,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.markdown:
        output = gen.to_markdown_preview(results)
        if args.output:
            p = Path(args.output)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(output, encoding="utf-8")
            print(f"Markdown preview written to {p}", file=sys.stderr)
        else:
            print(output, end="")
        return

    if args.output:
        out_path = gen.save(
            results,
            args.output,
            caption=args.caption,
            label=args.label,
            position=args.position,
            include_preamble=not args.no_preamble,
        )
        print(f"LaTeX table written to {out_path}", file=sys.stderr)
    else:
        if args.no_preamble:
            output = gen.generate_table(
                results,
                caption=args.caption,
                label=args.label,
                position=args.position,
            )
        else:
            output = gen.generate_full(
                results,
                caption=args.caption,
                label=args.label,
                position=args.position,
            )
        print(output, end="")


if __name__ == "__main__":
    main()
