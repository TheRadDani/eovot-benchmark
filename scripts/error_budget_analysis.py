"""CLI for tracking error budget decomposition.

Loads one or more benchmark result JSON files produced by BenchmarkEngine /
BenchmarkReporter and decomposes the IoU loss into centre, scale,
aspect-ratio, and residual error components.

Usage::

    # Single result file
    python scripts/error_budget_analysis.py results/MOSSE-OTB100.json

    # Multiple trackers — produces a comparison table
    python scripts/error_budget_analysis.py results/*.json --format markdown

    # JSON output for downstream processing
    python scripts/error_budget_analysis.py results/KCF-OTB100.json --format json

    # Save to file
    python scripts/error_budget_analysis.py results/*.json --output error_budget.md

Output example (markdown)::

    | Tracker | Dataset | IoU Loss | Center % | Scale % | AR % | Residual % | Dominant |
    |---------|---------|----------:|---------:|--------:|-----:|-----------:|:---------|
    | MOSSE   | OTB100  |   0.3412  |    71.2  |   18.4  |  5.1 |        5.3 | center   |
    | KCF     | OTB100  |   0.2891  |    40.6  |   43.1  |  9.8 |        6.5 | scale    |

Interpretation guide::

    center   — tracker is drifting (predicted centre far from GT)
    scale    — tracker is over/under-zooming
    AR       — tracker deforms the bounding box shape
    residual — other / interaction effects
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


def _load_result(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _build_arrays(result: dict):
    """Reconstruct prediction and GT arrays from a saved result dict.

    Returns a list of (preds, gts, seq_name) tuples — one per sequence.
    Sequences missing predictions or GT are silently skipped.
    """
    import numpy as np

    seqs = result.get("sequences", [])
    arrays = []
    for seq in seqs:
        preds_raw = seq.get("predictions")
        gts_raw = seq.get("ground_truths")
        if preds_raw is None or gts_raw is None:
            continue
        try:
            preds = np.array(preds_raw, dtype=np.float64)
            gts = np.array(gts_raw, dtype=np.float64)
            if preds.ndim == 2 and gts.ndim == 2 and preds.shape[1] == 4:
                arrays.append((preds, gts, seq.get("sequence_name", "unknown")))
        except (ValueError, TypeError):
            continue
    return arrays


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Decompose tracking IoU loss into geometric error components.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="Path(s) to benchmark result JSON files.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "text"],
        default="markdown",
        help="Output format. Default: markdown.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write output to this file instead of stdout.",
    )
    args = parser.parse_args(argv)

    try:
        from eovot.metrics.error_budget import ErrorBudgetAnalyzer, AggregateErrorBudget
    except ImportError as exc:
        print(f"ERROR: cannot import eovot — make sure the package is installed: {exc}")
        sys.exit(1)

    analyzer = ErrorBudgetAnalyzer()
    aggregates: List[AggregateErrorBudget] = []

    for path in args.results:
        try:
            result = _load_result(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: skipping {path}: {exc}", file=sys.stderr)
            continue

        summary = result.get("summary", {})
        tracker_name = summary.get("tracker") or summary.get("tracker_name", Path(path).stem)
        dataset_name = summary.get("dataset") or summary.get("dataset_name", "unknown")

        arrays = _build_arrays(result)
        if not arrays:
            print(
                f"WARNING: {path} contains no per-sequence predictions/GT arrays. "
                "Re-run benchmark with save_predictions=true.",
                file=sys.stderr,
            )
            continue

        budgets = [
            analyzer.analyze(preds, gts, tracker_name=tracker_name,
                             sequence_name=seq_name)
            for preds, gts, seq_name in arrays
        ]
        agg = analyzer.aggregate(budgets, tracker_name=tracker_name,
                                 dataset_name=dataset_name)
        aggregates.append(agg)

    if not aggregates:
        print("No valid results to analyse.", file=sys.stderr)
        sys.exit(1)

    if args.format == "markdown":
        output = ErrorBudgetAnalyzer.to_markdown_table(aggregates) + "\n"
    elif args.format == "json":
        import json as _json
        output = _json.dumps([a.to_dict() for a in aggregates], indent=2) + "\n"
    else:  # text
        output = "\n".join(str(a) for a in aggregates) + "\n"

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Error budget written to {args.output}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
