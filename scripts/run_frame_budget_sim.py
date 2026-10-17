#!/usr/bin/env python3
"""CLI: evaluate tracker accuracy under frame budget constraints.

Simulates what happens when an edge device can only process a fraction of
incoming camera frames.  For each configured budget rate the tracker runs
on a uniformly subsampled subset of frames; skipped frames propagate the
last known bounding box (zero-motion model).

Usage examples::

    # Evaluate MOSSE at 100%, 50%, 25%, and 10% frame budgets on OTB100
    python scripts/run_frame_budget_sim.py \\
        --tracker MOSSE \\
        --dataset-root /data/OTB100 \\
        --rates 1.0 0.5 0.25 0.1

    # Limit to 5 sequences, provide native FPS for effective-FPS column
    python scripts/run_frame_budget_sim.py \\
        --tracker KCF \\
        --dataset-root /data/OTB100 \\
        --rates 1.0 0.5 0.1 \\
        --max-sequences 5 \\
        --native-fps 300.0 \\
        --output results/kcf_frame_budget.json

Output is a JSON file containing per-sequence budget curves that can be
loaded for plotting or further analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the package importable when the script is run directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.datasets.base import OTBDataset
from eovot.simulation.frame_budget import FrameBudgetSimulator
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker

_TRACKER_REGISTRY = {
    "MOSSE": MOSSETracker,
    "KCF": KCFTracker,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate tracker accuracy under frame budget constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tracker",
        choices=list(_TRACKER_REGISTRY),
        default="MOSSE",
        help="Tracker to evaluate.",
    )
    p.add_argument(
        "--dataset-root",
        required=True,
        metavar="PATH",
        help="Root directory of an OTB-style dataset.",
    )
    p.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=[1.0, 0.75, 0.5, 0.25, 0.1],
        metavar="RATE",
        help="Frame budget rates to simulate (each must be in (0, 1]).",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Limit the number of sequences evaluated (default: all).",
    )
    p.add_argument(
        "--native-fps",
        type=float,
        default=None,
        metavar="FPS",
        help=(
            "Unthrottled tracker FPS used to compute effective FPS per budget. "
            "Run the standard benchmark first to obtain this value."
        ),
    )
    p.add_argument(
        "--output",
        default="results/frame_budget_sim.json",
        metavar="FILE",
        help="Output JSON file path.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sequence progress output.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    tracker_cls = _TRACKER_REGISTRY[args.tracker]
    dataset = OTBDataset(root=args.dataset_root)
    n_seq = min(len(dataset), args.max_sequences) if args.max_sequences else len(dataset)

    sim = FrameBudgetSimulator(budget_rates=args.rates, native_fps=args.native_fps or 0.0)

    if verbose:
        print(f"Frame Budget Simulation")
        print(f"  Tracker  : {args.tracker}")
        print(f"  Dataset  : {args.dataset_root}  ({n_seq} sequences)")
        print(f"  Rates    : {args.rates}")
        print(f"  Native FPS: {args.native_fps or 'not set'}")
        print()

    all_curves = []

    for i in range(n_seq):
        seq = dataset[i]
        tracker = tracker_cls()
        if verbose:
            print(f"[{i + 1}/{n_seq}] {seq.name}")
        curve = sim.simulate(tracker, seq, native_fps=args.native_fps)
        if verbose:
            FrameBudgetSimulator.print_curve(curve)
        all_curves.append(curve.to_dict())

    # Aggregate dataset-level mean per budget rate
    if len(all_curves) > 1:
        from eovot.simulation.frame_budget import BudgetCurve, BudgetPoint
        from eovot.metrics.accuracy import AccuracyMetrics

        rate_data: dict = {}
        for cd in all_curves:
            for pt in cd["points"]:
                r = pt["budget_rate"]
                rate_data.setdefault(r, {"iou": [], "sauc": [], "pauc": []})
                rate_data[r]["iou"].append(pt["mean_iou"])
                rate_data[r]["sauc"].append(pt["success_auc"])
                rate_data[r]["pauc"].append(pt["precision_auc"])

        import numpy as np
        dataset_mean = [
            {
                "budget_rate": r,
                "mean_iou": round(float(np.mean(v["iou"])), 4),
                "success_auc": round(float(np.mean(v["sauc"])), 4),
                "precision_auc": round(float(np.mean(v["pauc"])), 4),
            }
            for r, v in sorted(rate_data.items(), reverse=True)
        ]

        if verbose:
            print("\nDataset-level Mean:")
            print("-" * 55)
            print(f"{'Budget':>8}  {'mIoU':>8}  {'S-AUC':>8}  {'P-AUC':>8}")
            print("-" * 55)
            for row in dataset_mean:
                print(
                    f"{row['budget_rate']:>7.0%}  "
                    f"{row['mean_iou']:>8.4f}  "
                    f"{row['success_auc']:>8.4f}  "
                    f"{row['precision_auc']:>8.4f}"
                )
            print("-" * 55)
    else:
        dataset_mean = []

    output = {
        "tracker": args.tracker,
        "dataset_root": args.dataset_root,
        "budget_rates": args.rates,
        "native_fps": args.native_fps,
        "num_sequences": n_seq,
        "dataset_mean": dataset_mean,
        "curves": all_curves,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    if verbose:
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
