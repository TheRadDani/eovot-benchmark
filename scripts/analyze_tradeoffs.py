#!/usr/bin/env python3
"""CLI for multi-objective tracker trade-off analysis.

Loads benchmark result JSON files produced by :class:`~eovot.benchmark.engine.BenchmarkEngine`
or :class:`~eovot.experiment.runner.ExperimentRunner` and prints a weighted composite
leaderboard together with an optional Pareto frontier report.

Usage examples::

    # Analyse a directory of per-tracker JSON results with balanced weights
    python scripts/analyze_tradeoffs.py results/my-experiment/

    # Use edge-deployment weights and show Pareto frontier (FPS vs AUC)
    python scripts/analyze_tradeoffs.py results/my-experiment/ --weights edge

    # Show Pareto frontier on a custom pair of objectives
    python scripts/analyze_tradeoffs.py results/ --pareto fps peak_memory_mb

    # Analyse individual JSON files with custom weights
    python scripts/analyze_tradeoffs.py MOSSE.json KCF.json --weights research
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def _load_results(paths: List[Path]) -> Dict[str, Dict]:
    """Parse JSON result files and extract per-tracker summary metrics."""
    tracker_metrics: Dict[str, Dict] = {}

    for p in paths:
        if not p.exists():
            print(f"[warn] File not found, skipping: {p}", file=sys.stderr)
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] Could not parse {p}: {exc}", file=sys.stderr)
            continue

        summary = data.get("summary", data)  # support both wrapped and flat formats
        tracker_name = summary.get("tracker", p.stem)

        tracker_metrics[tracker_name] = {
            "auc": float(summary.get("success_auc", summary.get("auc", 0.0))),
            "precision": float(summary.get("precision_auc", summary.get("precision", 0.0))),
            "fps": float(summary.get("mean_fps", summary.get("fps", 0.0))),
            "mean_latency_ms": float(
                summary.get("mean_latency_ms", 1000.0 / float(summary.get("mean_fps", 1.0))
            )),
            "peak_memory_mb": float(summary.get("peak_memory_mb", 0.0)),
            "mean_energy_j": float(summary.get("total_energy_j", 0.0)),
        }

    return tracker_metrics


def _collect_paths(inputs: List[str]) -> List[Path]:
    """Expand directories to JSON files and validate individual file paths."""
    paths: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            found = sorted(p.glob("*.json"))
            if not found:
                print(f"[warn] No JSON files found in directory: {p}", file=sys.stderr)
            paths.extend(found)
        else:
            paths.append(p)
    return paths


def _print_composite_table(df) -> None:
    """Pretty-print the composite leaderboard DataFrame."""
    cols = ["tracker", "auc", "fps", "peak_memory_mb", "accuracy_score",
            "speed_score", "memory_score", "energy_score", "composite_score"]
    present = [c for c in cols if c in df.columns]
    print(df[present].to_string(index=False))


def _print_pareto_table(pareto_names, pareto_df, obj_x, obj_y) -> None:
    """Pretty-print the Pareto frontier report."""
    print(f"\n{'=' * 60}")
    print(f"  PARETO FRONTIER: {obj_x}  vs  {obj_y}")
    print(f"{'=' * 60}")
    print(f"Non-dominated trackers: {pareto_names}\n")
    print(pareto_df.to_string(index=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-objective tracker trade-off analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSON result files or directories containing them.",
    )
    parser.add_argument(
        "--weights",
        choices=["balanced", "edge", "research", "energy"],
        default="balanced",
        help="Preset weight configuration (default: balanced).",
    )
    parser.add_argument(
        "--pareto",
        nargs=2,
        metavar=("OBJ_X", "OBJ_Y"),
        default=None,
        help="Show Pareto frontier for two objectives, e.g. --pareto fps auc.",
    )
    args = parser.parse_args(argv)

    # Import here so the script is importable without installing the package first.
    try:
        from eovot.metrics.scoring import PRESET_WEIGHTS, compute_composite_scores, pareto_frontier
    except ImportError as exc:
        print(f"[error] Could not import eovot: {exc}", file=sys.stderr)
        print("  Make sure you have installed the package: pip install -e .", file=sys.stderr)
        return 1

    paths = _collect_paths(args.inputs)
    if not paths:
        print("[error] No result files found.", file=sys.stderr)
        return 1

    tracker_metrics = _load_results(paths)
    if not tracker_metrics:
        print("[error] Could not extract metrics from any result file.", file=sys.stderr)
        return 1

    weights = PRESET_WEIGHTS[args.weights]

    print(f"\n{'=' * 60}")
    print(f"  EOVOT MULTI-OBJECTIVE LEADERBOARD  (weights: {args.weights})")
    print(f"  accuracy={weights.accuracy}  speed={weights.speed}  "
          f"memory={weights.memory}  energy={weights.energy}")
    print(f"{'=' * 60}")

    df = compute_composite_scores(tracker_metrics, weights=weights)
    _print_composite_table(df)

    if args.pareto is not None:
        obj_x, obj_y = args.pareto
        # Detect direction: latency, memory, energy are lower-is-better.
        lower_is_better = {"mean_latency_ms", "peak_memory_mb", "mean_energy_j"}
        x_hib = obj_x not in lower_is_better
        y_hib = obj_y not in lower_is_better
        pareto_names, pareto_df = pareto_frontier(
            tracker_metrics, obj_x, obj_y, x_hib, y_hib
        )
        _print_pareto_table(pareto_names, pareto_df, obj_x, obj_y)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
