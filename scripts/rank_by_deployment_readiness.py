#!/usr/bin/env python3
"""Rank trackers by Deployment-Readiness Score on a synthetic benchmark.

Runs all lightweight registered trackers on a configurable synthetic dataset
and ranks them using the four-component Deployment-Readiness Score (DRS):
Accuracy × Efficiency × Robustness × Temporal-Consistency.

This script is designed as a quick sanity-check and demonstration of the
DRS metric.  For production evaluation, replace SyntheticDataset with
OTBDataset, GOT10kDataset, or LaSOTDataset.

Usage::

    # Quick demo with defaults (5 sequences × 60 frames per tracker)
    python scripts/rank_by_deployment_readiness.py

    # Larger evaluation
    python scripts/rank_by_deployment_readiness.py \\
        --num-sequences 20 --num-frames 150 --motion circular

    # Edge-device budget (512 MB memory, 15 W TDP)
    python scripts/rank_by_deployment_readiness.py \\
        --memory-budget 512 --tdp-watts 15.0 --output results/drs_ranking.json

    # Custom DRS weights emphasising efficiency
    python scripts/rank_by_deployment_readiness.py \\
        --w-accuracy 0.25 --w-efficiency 0.45 --w-robustness 0.20 --w-smoothness 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.metrics.deployment import DeploymentReadinessAnalyzer
from eovot.trackers.registry import TRACKER_REGISTRY

# Trackers that run reliably on synthetic data without pre-trained files
_FAST_TRACKERS = ["MOSSE", "KCF", "LKOpticalFlow", "CamShift", "KalmanColor"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank trackers by Deployment-Readiness Score",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trackers", nargs="+", default=_FAST_TRACKERS,
                   help="Tracker names from the registry to evaluate.")
    p.add_argument("--num-sequences", type=int, default=5,
                   help="Number of synthetic sequences per tracker.")
    p.add_argument("--num-frames", type=int, default=60,
                   help="Frames per sequence.")
    p.add_argument("--motion", default="linear",
                   choices=["linear", "circular", "random"],
                   help="Synthetic motion pattern.")
    p.add_argument("--memory-budget", type=float, default=512.0,
                   help="Peak-memory budget in MB for the efficiency score.")
    p.add_argument("--tdp-watts", type=float, default=None,
                   help="CPU TDP in Watts for energy profiling (optional).")
    p.add_argument("--w-accuracy", type=float, default=0.35)
    p.add_argument("--w-efficiency", type=float, default=0.30)
    p.add_argument("--w-robustness", type=float, default=0.20)
    p.add_argument("--w-smoothness", type=float, default=0.15)
    p.add_argument("--output", type=str, default=None,
                   help="Optional path to save JSON results.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate weights
    total_w = args.w_accuracy + args.w_efficiency + args.w_robustness + args.w_smoothness
    if abs(total_w - 1.0) > 1e-6:
        print(
            f"ERROR: DRS weights must sum to 1.0, got {total_w:.4f}. "
            "Adjust --w-accuracy / --w-efficiency / --w-robustness / --w-smoothness.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate tracker names
    invalid = [t for t in args.trackers if t not in TRACKER_REGISTRY]
    if invalid:
        print(
            f"ERROR: Unknown trackers: {invalid}. "
            f"Available: {sorted(TRACKER_REGISTRY.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset = SyntheticDataset(
        num_sequences=args.num_sequences,
        num_frames=args.num_frames,
        motion=args.motion,
        seed=42,
    )
    dataset_name = f"Synthetic-{args.motion.capitalize()}"

    engine = BenchmarkEngine(verbose=True, tdp_watts=args.tdp_watts)
    analyzer = DeploymentReadinessAnalyzer(
        memory_budget_mb=args.memory_budget,
        weight_accuracy=args.w_accuracy,
        weight_efficiency=args.w_efficiency,
        weight_robustness=args.w_robustness,
        weight_smoothness=args.w_smoothness,
    )

    print(f"\n{'='*70}")
    print(f"EOVOT Deployment-Readiness Ranking")
    print(f"Dataset: {dataset_name}  ({args.num_sequences} seqs × {args.num_frames} fr)")
    print(f"Memory budget: {args.memory_budget:.0f} MB")
    print(
        f"DRS weights: A={args.w_accuracy} E={args.w_efficiency} "
        f"R={args.w_robustness} S={args.w_smoothness}"
    )
    print(f"{'='*70}\n")

    benchmark_results = []
    for name in args.trackers:
        tracker = TRACKER_REGISTRY[name]()
        result = engine.run(tracker, dataset, dataset_name=dataset_name)
        benchmark_results.append(result)

    scores = analyzer.rank(benchmark_results)

    print(f"\n{'='*70}")
    print("DEPLOYMENT-READINESS RANKING")
    print(f"{'='*70}")
    print(analyzer.to_markdown_table(scores))
    print()

    print("Detailed breakdown:")
    for rank, s in enumerate(scores, start=1):
        print(f"  {rank}. {s}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "dataset": dataset_name,
                "num_sequences": args.num_sequences,
                "num_frames": args.num_frames,
                "motion": args.motion,
                "memory_budget_mb": args.memory_budget,
                "weights": {
                    "accuracy": args.w_accuracy,
                    "efficiency": args.w_efficiency,
                    "robustness": args.w_robustness,
                    "smoothness": args.w_smoothness,
                },
            },
            "rankings": [s.to_dict() for s in scores],
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
