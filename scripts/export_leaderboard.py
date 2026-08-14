#!/usr/bin/env python3
"""Export a ranked tracker leaderboard from EOVOT benchmark JSON results.

Reads one or more ``BenchmarkResult`` JSON files (written by
:meth:`~eovot.benchmark.engine.BenchmarkResult.save`) and produces a
ranked comparison table in Markdown, CSV, and/or LaTeX format.

Usage examples
--------------
Print a Markdown table to stdout::

    python scripts/export_leaderboard.py results/MOSSE.json results/KCF.json

Export all formats to a directory::

    python scripts/export_leaderboard.py results/*.json \\
        --csv out/leaderboard.csv \\
        --latex out/leaderboard.tex \\
        --title "OTB-100 Leaderboard"

Rank by FPS instead of success AUC::

    python scripts/export_leaderboard.py results/*.json --sort-by mean_fps

Run against the built-in synthetic dataset for a quick demo::

    python scripts/export_leaderboard.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "result_files",
        nargs="*",
        metavar="RESULT.json",
        help="Path(s) to BenchmarkResult JSON files.",
    )
    p.add_argument(
        "--sort-by",
        default="success_auc",
        metavar="METRIC",
        help=(
            "Primary metric used for ranking. Choices: mean_iou, success_auc, "
            "precision_auc, mean_fps, peak_memory_mb, total_energy_j. "
            "Default: success_auc."
        ),
    )
    p.add_argument(
        "--csv",
        metavar="PATH",
        help="Write CSV leaderboard to this path.",
    )
    p.add_argument(
        "--latex",
        metavar="PATH",
        help="Write LaTeX table to this path.",
    )
    p.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip printing the Markdown table to stdout.",
    )
    p.add_argument(
        "--title",
        default=None,
        metavar="TEXT",
        help="Optional heading for the Markdown table.",
    )
    p.add_argument(
        "--latex-caption",
        default="Tracker performance comparison.",
        metavar="TEXT",
        help="Caption for the LaTeX table (default: 'Tracker performance comparison.').",
    )
    p.add_argument(
        "--latex-label",
        default="tab:leaderboard",
        metavar="LABEL",
        help="LaTeX \\label key for the table (default: tab:leaderboard).",
    )
    p.add_argument(
        "--no-dataset-col",
        action="store_true",
        help="Omit the Dataset column (useful when all results share a single dataset).",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Ignore RESULT_FILES and instead run a quick demo benchmark "
            "on the built-in SyntheticDataset with MOSSE and KCF."
        ),
    )
    return p.parse_args()


def _run_demo():
    """Run a minimal benchmark demo and return result dicts."""
    try:
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.registry import build_tracker, available_trackers
    except ImportError as exc:
        print(f"[error] Could not import EOVOT modules: {exc}", file=sys.stderr)
        print("Run from the repository root: python scripts/export_leaderboard.py --demo", file=sys.stderr)
        sys.exit(1)

    print("[demo] Running benchmark on SyntheticDataset (3 trackers × 5 sequences × 100 frames) ...",
          file=sys.stderr)
    ds = SyntheticDataset(num_sequences=5, num_frames=100, motion="linear", seed=99)
    engine = BenchmarkEngine(verbose=False)

    # Use three trackers that are always available without extra model files.
    demo_names = ["KCF", "CamShift"]
    # Add a third if something like LKOpticalFlow is registered.
    available = available_trackers()
    for extra in ["LKOpticalFlow", "MOSSE"]:
        if extra in available and extra not in demo_names:
            demo_names.append(extra)
            break

    results = []
    for name in demo_names:
        try:
            t = build_tracker(name)
            results.append(engine.run(t, ds, dataset_name="Synthetic").to_dict())
        except Exception as exc:
            print(f"[demo] Skipping {name}: {exc}", file=sys.stderr)

    return results


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Gather results
    # ------------------------------------------------------------------
    if args.demo:
        result_dicts = _run_demo()
    elif not args.result_files:
        print(
            "[error] No result files provided. Pass JSON file paths or use --demo.\n"
            "Usage:  python scripts/export_leaderboard.py results/*.json",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        result_dicts = []
        for path_str in args.result_files:
            p = Path(path_str)
            if not p.exists():
                print(f"[warning] File not found, skipping: {p}", file=sys.stderr)
                continue
            with open(p, encoding="utf-8") as fh:
                result_dicts.append(json.load(fh))
        if not result_dicts:
            print("[error] No valid result files loaded.", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Build leaderboard
    # ------------------------------------------------------------------
    try:
        from eovot.reporting.leaderboard import LeaderboardExporter
    except ImportError as exc:
        print(f"[error] Could not import LeaderboardExporter: {exc}", file=sys.stderr)
        sys.exit(1)

    exporter = LeaderboardExporter(primary_metric=args.sort_by)
    show_dataset = not args.no_dataset_col

    # ------------------------------------------------------------------
    # Markdown (stdout)
    # ------------------------------------------------------------------
    if not args.no_markdown:
        md = exporter.to_markdown(result_dicts, title=args.title, show_dataset=show_dataset)
        print(md)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    if args.csv:
        out_path = exporter.to_csv(result_dicts, args.csv, include_dataset=show_dataset)
        print(f"[leaderboard] CSV written → {out_path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------
    if args.latex:
        latex = exporter.to_latex(
            result_dicts,
            caption=args.latex_caption,
            label=args.latex_label,
            include_dataset=show_dataset,
        )
        out = Path(args.latex)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex, encoding="utf-8")
        print(f"[leaderboard] LaTeX written → {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
