#!/usr/bin/env python3
"""Pareto frontier analysis script for EOVOT.

Runs a set of registered trackers on a SyntheticDataset (or loads pre-saved
JSON result files) and prints the speed-accuracy Pareto frontier with Edge
Deployment Scores.

Usage — run trackers live::

    python scripts/pareto_analysis.py \\
        --trackers MOSSE KCF CSRT \\
        --num-sequences 20 \\
        --num-frames 150 \\
        --metric success_auc \\
        --min-fps 30 \\
        --out pareto_report.csv

Usage — load saved JSON results::

    python scripts/pareto_analysis.py \\
        --from-json results/MOSSE.json results/KCF.json results/CSRT.json \\
        --metric mean_iou
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.analysis.pareto import ParetoFrontierAnalyzer
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.registry import build_tracker, available_trackers


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pareto frontier analysis for EOVOT trackers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--trackers",
        nargs="+",
        metavar="NAME",
        help=f"Tracker names to benchmark live. Available: {available_trackers()}",
    )
    src.add_argument(
        "--from-json",
        nargs="+",
        metavar="PATH",
        help="Load pre-saved JSON result files instead of running trackers.",
    )
    p.add_argument("--num-sequences", type=int, default=10, metavar="N",
                   help="Number of synthetic sequences (live mode).")
    p.add_argument("--num-frames", type=int, default=100, metavar="N",
                   help="Frames per synthetic sequence (live mode).")
    p.add_argument("--seed", type=int, default=42,
                   help="SyntheticDataset random seed (live mode).")
    p.add_argument(
        "--metric",
        default="success_auc",
        choices=["mean_iou", "success_auc", "mean_eao", "precision_auc"],
        help="Accuracy metric for the y-axis of the Pareto plot.",
    )
    p.add_argument(
        "--min-fps",
        type=float,
        default=None,
        metavar="FPS",
        help="Print the best Pareto-optimal tracker that meets this FPS budget.",
    )
    p.add_argument("--out", default=None, metavar="PATH",
                   help="Optional path to write a CSV summary.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.from_json:
        results: list = []
        for path in args.from_json:
            with open(path) as fh:
                results.append(json.load(fh))
        print(f"Loaded {len(results)} result file(s).")
    else:
        trackers = args.trackers or available_trackers()
        dataset = SyntheticDataset(
            num_sequences=args.num_sequences,
            num_frames=args.num_frames,
            seed=args.seed,
        )
        engine = BenchmarkEngine(verbose=True)
        results = []
        for name in trackers:
            try:
                tracker = build_tracker(name)
            except ValueError as e:
                print(f"[skip] {e}")
                continue
            result = engine.run(tracker, dataset, dataset_name="Synthetic")
            results.append(result)

    if not results:
        print("No results to analyse.")
        sys.exit(1)

    analyzer = ParetoFrontierAnalyzer(accuracy_metric=args.metric)
    report = analyzer.analyze(results)

    print("\n" + report.to_markdown())

    best_eds = report.best_by_eds()
    if best_eds:
        print(f"\nBest EDS overall: {best_eds.tracker_name} "
              f"(FPS={best_eds.fps:.1f}, {args.metric}={best_eds.accuracy:.4f}, "
              f"EDS={best_eds.eds:.4f})")

    if args.min_fps is not None:
        rec = report.optimal_for_fps_budget(args.min_fps)
        if rec:
            print(f"Best tracker at ≥{args.min_fps} FPS: {rec.tracker_name} "
                  f"({args.metric}={rec.accuracy:.4f})")
        else:
            print(f"No tracker meets the ≥{args.min_fps} FPS budget.")

    if args.out:
        out_path = report.to_csv(args.out)
        print(f"\nCSV written to: {out_path}")


if __name__ == "__main__":
    main()
