"""CLI script for multi-objective Pareto frontier analysis of VOT benchmark results.

Loads one or more JSON files saved by BenchmarkResult.save(), builds a
ParetoFrontier, and reports the Pareto-optimal trackers in both accuracy-vs-latency
(2-D) and accuracy-vs-latency-vs-memory (3-D) objective spaces.

Usage examples::

    # Analyse all saved results in a directory
    python scripts/pareto_frontier.py results/

    # Apply a 20 ms latency budget and 256 MB memory budget
    python scripts/pareto_frontier.py results/ --max-latency 20 --max-memory 256

    # Use precision_auc as the accuracy axis
    python scripts/pareto_frontier.py results/ --accuracy precision_auc

    # Compute the 2-D hypervolume indicator (research metric)
    python scripts/pareto_frontier.py results/ --hypervolume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eovot.benchmark.engine import BenchmarkResult
from eovot.analysis.pareto import ParetoFrontier


def _load_results(paths: list[Path]) -> list[BenchmarkResult]:
    results = []
    for p in paths:
        if p.is_dir():
            for child in sorted(p.glob("*.json")):
                try:
                    results.append(BenchmarkResult.load(child))
                    print(f"  Loaded: {child.name}")
                except Exception as exc:
                    print(f"  Skipped {child.name}: {exc}", file=sys.stderr)
        elif p.suffix == ".json":
            try:
                results.append(BenchmarkResult.load(p))
                print(f"  Loaded: {p.name}")
            except Exception as exc:
                print(f"  Skipped {p.name}: {exc}", file=sys.stderr)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-objective Pareto frontier analysis for EOVOT benchmark results."
    )
    parser.add_argument(
        "paths", nargs="+", type=Path,
        help="JSON result files or directories containing them.",
    )
    parser.add_argument(
        "--accuracy", default="success_auc",
        choices=["success_auc", "mean_iou", "precision_auc"],
        help="Accuracy objective to use (default: success_auc).",
    )
    parser.add_argument(
        "--max-latency", type=float, default=None, metavar="MS",
        help="Latency budget constraint in milliseconds.",
    )
    parser.add_argument(
        "--max-memory", type=float, default=None, metavar="MB",
        help="Memory budget constraint in megabytes.",
    )
    parser.add_argument(
        "--hypervolume", action="store_true",
        help="Compute and print the 2-D hypervolume indicator.",
    )
    parser.add_argument(
        "--ref-latency", type=float, default=200.0, metavar="MS",
        help="Reference latency for hypervolume computation (default: 200 ms).",
    )
    args = parser.parse_args()

    print("\nLoading benchmark results...")
    results = _load_results(args.paths)
    if not results:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    frontier = ParetoFrontier.from_benchmark_results(results, accuracy_key=args.accuracy)
    print(f"\nLoaded {len(frontier)} tracker results.\n")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 2-D Pareto front
    # ---------------------------------------------------------------
    print(f"\n[2-D Pareto Front] accuracy={args.accuracy} vs latency")
    print("-" * 50)
    for p in frontier.frontier_2d:
        print(
            f"  {p.tracker_name:<20s} "
            f"{args.accuracy}={p.accuracy:.4f}  "
            f"lat={p.latency_ms:.2f}ms  "
            f"fps={p.fps:.1f}  "
            f"mem={p.memory_mb:.1f}MB"
        )

    # ---------------------------------------------------------------
    # 3-D Pareto front
    # ---------------------------------------------------------------
    print(f"\n[3-D Pareto Front] accuracy vs latency vs memory")
    print("-" * 50)
    for p in frontier.frontier_3d:
        print(
            f"  {p.tracker_name:<20s} "
            f"{args.accuracy}={p.accuracy:.4f}  "
            f"lat={p.latency_ms:.2f}ms  "
            f"mem={p.memory_mb:.1f}MB"
        )

    # ---------------------------------------------------------------
    # Efficiency ranking
    # ---------------------------------------------------------------
    print("\n[Efficiency Ranking] accuracy / ms (descending)")
    print("-" * 50)
    for i, p in enumerate(frontier.rank_by_efficiency(), start=1):
        print(f"  {i:2d}. {p.tracker_name:<20s} ratio={p.efficiency_ratio:.5f}")

    # ---------------------------------------------------------------
    # Constraint-aware best
    # ---------------------------------------------------------------
    if args.max_latency is not None or args.max_memory is not None:
        print("\n[Best Under Constraints]")
        best = frontier.best_under_constraints(
            max_latency_ms=args.max_latency, max_memory_mb=args.max_memory
        )
        if best:
            print(
                f"  Best: {best.tracker_name}  "
                f"{args.accuracy}={best.accuracy:.4f}  "
                f"lat={best.latency_ms:.2f}ms  "
                f"mem={best.memory_mb:.1f}MB"
            )
        else:
            print("  No tracker satisfies the given constraints.")

    # ---------------------------------------------------------------
    # Hypervolume
    # ---------------------------------------------------------------
    if args.hypervolume:
        hv = frontier.hypervolume_2d(ref_accuracy=0.0, ref_latency=args.ref_latency)
        print(f"\n[Hypervolume Indicator] ref_latency={args.ref_latency}ms")
        print(f"  HV = {hv:.6f}")

    # ---------------------------------------------------------------
    # Full Markdown table
    # ---------------------------------------------------------------
    print("\n[Tracker Comparison Table]")
    print(frontier.to_markdown())
    print("")


if __name__ == "__main__":
    main()
