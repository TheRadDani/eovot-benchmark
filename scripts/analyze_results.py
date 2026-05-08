#!/usr/bin/env python3
"""analyze_results.py — post-hoc statistical analysis of EOVOT benchmark results.

Performs pairwise significance tests (Mann-Whitney U), bootstrap
confidence intervals, and Pareto frontier analysis across saved JSON
result files produced by the EOVOT benchmark engine.

Usage::

    # Pairwise significance test on IoU
    python scripts/analyze_results.py results/mosse.json results/kcf.json

    # Compare multiple trackers on FPS, output LaTeX table
    python scripts/analyze_results.py results/*.json --metric fps --latex

    # Pareto frontier: accuracy vs FPS
    python scripts/analyze_results.py results/*.json --pareto

    # Bootstrap 95% CI for each tracker's mean IoU
    python scripts/analyze_results.py results/*.json --ci

    # Write output to a file
    python scripts/analyze_results.py results/*.json --output analysis.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def _load(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _tracker_name(result: Dict, fallback: str) -> str:
    return result.get("summary", {}).get("tracker", fallback)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical analysis of EOVOT benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results",
        nargs="+",
        metavar="RESULT_JSON",
        help="One or more JSON result files produced by the benchmark engine.",
    )
    parser.add_argument(
        "--metric",
        default="iou",
        choices=["iou", "fps", "latency_ms", "memory_mb", "energy_j"],
        help="Per-sequence metric to compare in the significance test (default: iou).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the Mann-Whitney U test (default: 0.05).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for CI estimation (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible bootstrap sampling (default: 0).",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Emit a LaTeX booktabs table instead of Markdown.",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Show Pareto frontier analysis (accuracy vs FPS).",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Print 95%% bootstrap CI for each tracker's mean IoU.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Write analysis to FILE instead of stdout.",
    )
    args = parser.parse_args()

    if len(args.results) < 2 and not (args.pareto or args.ci):
        parser.error(
            "At least 2 result files are required for pairwise comparison. "
            "Use --pareto or --ci for single-file analyses."
        )

    result_dicts = [_load(p) for p in args.results]
    labels = [_tracker_name(rd, Path(p).stem) for rd, p in zip(result_dicts, args.results)]

    from eovot.analysis.statistics import TrackerStatistics
    from eovot.analysis.pareto import ParetoAnalyzer

    output_lines: List[str] = []

    # ------------------------------------------------------------------
    # Pareto frontier
    # ------------------------------------------------------------------
    if args.pareto:
        tracker_map = {label: rd for label, rd in zip(labels, result_dicts)}
        analyzer = ParetoAnalyzer()
        points = analyzer.analyze(tracker_map)
        output_lines.append("## Pareto Frontier Analysis\n")
        output_lines.append(analyzer.to_markdown(points))
        output_lines.append("")

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------
    if args.ci:
        stats = TrackerStatistics(
            alpha=args.alpha, n_bootstrap=args.n_bootstrap, seed=args.seed
        )
        output_lines.append("## Bootstrap 95% Confidence Interval — Mean IoU\n")
        output_lines.append("| Tracker | Mean IoU | CI Lower | CI Upper |")
        output_lines.append("|---------|----------|---------:|---------:|")
        for label, rd in zip(labels, result_dicts):
            mean_iou = rd.get("summary", {}).get("mean_iou", float("nan"))
            try:
                lo, hi = stats.auc_confidence_interval(rd)
            except ValueError as exc:
                output_lines.append(f"| {label} | error: {exc} | — | — |")
                continue
            output_lines.append(
                f"| {label} | {mean_iou:.4f} | {lo:.4f} | {hi:.4f} |"
            )
        output_lines.append("")

    # ------------------------------------------------------------------
    # Pairwise significance test
    # ------------------------------------------------------------------
    if len(args.results) >= 2:
        stats = TrackerStatistics(
            alpha=args.alpha, n_bootstrap=args.n_bootstrap, seed=args.seed
        )
        try:
            comparisons = stats.pairwise_comparison(result_dicts, metric=args.metric)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        output_lines.append(
            f"## Pairwise Significance Test  (metric={args.metric}, α={args.alpha})\n"
        )
        if args.latex:
            output_lines.append(stats.latex_table(comparisons))
        else:
            output_lines.append(stats.significance_table(comparisons))
        output_lines.append("")

        # Brief summary
        sig_count = sum(1 for c in comparisons if c.significant)
        output_lines.append(
            f"_{sig_count} of {len(comparisons)} pair(s) are statistically significant "
            f"at α={args.alpha}._"
        )

    text = "\n".join(output_lines)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Analysis written to {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
