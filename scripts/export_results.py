"""Aggregate multiple EOVOT benchmark JSON results into comparison tables.

Loads two or more result JSON files produced by
:class:`~eovot.benchmark.engine.BenchmarkEngine` (or the experiment runner)
and writes a unified comparison CSV and Markdown table sorted by the
requested metric.

Usage
-----
    # Basic comparison of three saved results
    python scripts/export_results.py \\
        results/MOSSE-OTB100.json \\
        results/KCF-OTB100.json \\
        results/CSRT-OTB100.json \\
        --output-dir results/comparison/

    # Sort by success AUC (default), output Markdown too
    python scripts/export_results.py results/*.json \\
        --sort-by success_auc \\
        --output-dir results/comparison/ \\
        --name classical-comparison

    # Sort by FPS for edge-deployment ranking
    python scripts/export_results.py results/*.json \\
        --sort-by mean_fps \\
        --output-dir results/edge-report/

Output files
------------
    <output-dir>/<name>.csv   — Summary-level comparison CSV (one row/tracker)
    <output-dir>/<name>.md    — Markdown table for embedding in papers/READMEs

Exit codes
----------
    0   Success
    1   No valid result files found
    2   Output directory could not be created
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_result(path: str) -> Dict[str, Any]:
    """Load a single benchmark result JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_results",
        description="Aggregate EOVOT benchmark JSON results into comparison tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Exit codes")[0].strip(),
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        metavar="JSON",
        help="One or more benchmark result JSON files to aggregate.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="results/comparison/",
        metavar="DIR",
        help="Directory for output files (created if absent). Default: results/comparison/",
    )
    parser.add_argument(
        "--name", "-n",
        default="comparison",
        metavar="NAME",
        help="Base filename for output files (no extension). Default: comparison",
    )
    parser.add_argument(
        "--sort-by", "-s",
        default="success_auc",
        metavar="METRIC",
        help=(
            "Summary metric to sort by (descending). "
            "Common values: success_auc, mean_iou, mean_fps, peak_memory_mb. "
            "Default: success_auc"
        ),
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip writing the Markdown comparison table.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the CSV comparison table.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output messages.",
    )
    return parser


def _print_summary_table(results: List[Dict], sort_by: str) -> None:
    """Print a compact ASCII table to stdout."""
    summaries = [r.get("summary", r) for r in results]

    def _get(s: Dict, key: str, default: float = 0.0) -> float:
        val = s.get(key, default)
        return float(val) if val is not None else default

    summaries_sorted = sorted(
        summaries,
        key=lambda s: _get(s, sort_by, _get(s, "mean_iou")),
        reverse=True,
    )

    col_w = 14
    hdr = (
        f"{'Rank':<5} {'Tracker':<18} {'Dataset':<12} "
        f"{'mIoU':>{col_w}} {'Succ AUC':>{col_w}} {'FPS':>{col_w}} "
        f"{'Mem (MB)':>{col_w}} {'Sequences':>{col_w}}"
    )
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(f" EOVOT Comparison  [sorted by: {sort_by}]")
    print(sep)
    print(hdr)
    print(sep)

    for rank, s in enumerate(summaries_sorted, start=1):
        tracker = (s.get("tracker") or s.get("tracker_name", "?"))[:17]
        dataset = (s.get("dataset") or s.get("dataset_name", "?"))[:11]
        miou = _get(s, "mean_iou")
        sauc = _get(s, "success_auc", miou)
        fps = _get(s, "mean_fps")
        mem = _get(s, "peak_memory_mb")
        nseq = int(s.get("num_sequences", 0))
        print(
            f"{rank:<5} {tracker:<18} {dataset:<12} "
            f"{miou:>{col_w}.4f} {sauc:>{col_w}.4f} "
            f"{fps:>{col_w}.1f} {mem:>{col_w}.1f} "
            f"{nseq:>{col_w}}"
        )
    print(sep + "\n")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Load all result files
    results: List[Dict[str, Any]] = []
    for path in args.result_files:
        try:
            results.append(_load_result(path))
            if not args.quiet:
                print(f"[load] {path}")
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"[warn] Could not load {path}: {exc}", file=sys.stderr)

    if not results:
        print("[error] No valid result files loaded.", file=sys.stderr)
        return 1

    # Create output directory
    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[error] Cannot create output directory: {exc}", file=sys.stderr)
        return 2

    # Inline import to avoid circular dependency at module level
    from eovot.reporting.reporter import BenchmarkReporter

    reporter = BenchmarkReporter(output_dir=str(out_dir))

    paths_written: List[Path] = []

    if not args.no_csv:
        csv_path = reporter.save_comparison_csv(
            results, name=args.name, sort_by=args.sort_by
        )
        paths_written.append(csv_path)

    if not args.no_markdown:
        md_path = reporter.save_comparison(results, name=args.name)
        paths_written.append(md_path)

    if not args.quiet:
        _print_summary_table(results, sort_by=args.sort_by)
        for p in paths_written:
            print(f"[save] {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
