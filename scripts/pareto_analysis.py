#!/usr/bin/env python3
"""CLI tool: Pareto front analysis for saved EOVOT benchmark results.

Loads one or more BenchmarkResult JSON files produced by
``BenchmarkResult.save()`` and prints a Pareto-front analysis in the
accuracy-efficiency space, including an Edge Efficiency Score (EES)
ranking for a user-specified device FPS budget.

Usage examples::

    # Basic analysis with default 30 FPS target
    python scripts/pareto_analysis.py results/kcf.json results/mosse.json

    # All results in a directory, Raspberry Pi 4 FPS budget
    python scripts/pareto_analysis.py results/*.json --target-fps 30

    # Use mean IoU instead of success AUC
    python scripts/pareto_analysis.py results/*.json --accuracy-key mean_iou

    # Save ranked table to CSV
    python scripts/pareto_analysis.py results/*.json --output ranking.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.analysis.pareto import ParetoAnalyzer, TrackerPoint


def load_point(json_path: Path, accuracy_key: str) -> TrackerPoint:
    with open(json_path) as f:
        data = json.load(f)
    summary = data.get("summary", data)
    tracker_name = summary.get("tracker", json_path.stem)
    accuracy = float(summary.get(accuracy_key, summary.get("mean_iou", 0.0)))
    fps = float(summary.get("mean_fps", 0.0))
    return TrackerPoint(
        name=tracker_name,
        accuracy=accuracy,
        fps=fps,
        extra={"source": str(json_path), "dataset": summary.get("dataset", "")},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pareto front analysis for EOVOT benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        metavar="JSON",
        help="BenchmarkResult JSON files (one per tracker)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        metavar="FPS",
        help="Device target FPS for the Edge Efficiency Score (default: 30)",
    )
    parser.add_argument(
        "--accuracy-key",
        default="success_auc",
        choices=["mean_iou", "success_auc", "precision_auc", "normalized_precision_auc"],
        help="Summary field to use as the accuracy metric (default: success_auc)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="CSV",
        help="Optional path to save the ranked table as a CSV file",
    )
    args = parser.parse_args()

    points = []
    for p in args.json_files:
        if not p.exists():
            print(f"Warning: {p} not found, skipping.", file=sys.stderr)
            continue
        try:
            points.append(load_point(p, args.accuracy_key))
        except Exception as exc:
            print(f"Warning: failed to load {p}: {exc}", file=sys.stderr)

    if not points:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    analyzer = ParetoAnalyzer(points)
    ranked = analyzer.rank_for_device(target_fps=args.target_fps)

    print(f"\nPareto Analysis  |  accuracy: {args.accuracy_key}  |  device target: {args.target_fps} FPS")
    print(analyzer.summary_table())

    pareto_names = [e.name for e in analyzer.pareto_optimal]
    print(f"\nPareto-optimal trackers: {pareto_names}")

    print(f"\nEdge ranking (target {args.target_fps} FPS — higher EES = better fit for this device):")
    for i, entry in enumerate(ranked, 1):
        ees_str = f"{entry.ees:.4f}" if entry.ees is not None else "N/A"
        star = " ★" if entry.is_pareto_optimal else ""
        print(
            f"  {i:>2}. {entry.name:<22}  EES={ees_str}  "
            f"acc={entry.accuracy:.4f}  fps={entry.fps:.1f}{star}"
        )

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "rank", "name", "accuracy", "fps", "latency_ms",
                    "pareto_rank", "is_pareto_optimal", "ees",
                ],
            )
            writer.writeheader()
            for i, entry in enumerate(ranked, 1):
                writer.writerow({
                    "rank": i,
                    "name": entry.name,
                    "accuracy": round(entry.accuracy, 4),
                    "fps": round(entry.fps, 2),
                    "latency_ms": round(entry.latency_ms, 3),
                    "pareto_rank": entry.pareto_rank,
                    "is_pareto_optimal": entry.is_pareto_optimal,
                    "ees": round(entry.ees, 4) if entry.ees is not None else "",
                })
        print(f"\nRanking table saved to {args.output}")


if __name__ == "__main__":
    main()
