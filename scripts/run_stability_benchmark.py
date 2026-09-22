#!/usr/bin/env python3
"""CLI for multi-run benchmark stability analysis.

Runs a tracker N times on a dataset and reports cross-run variance in IoU,
FPS, and latency.  This is the recommended workflow before citing benchmark
numbers in a paper — high variance signals non-reproducible conditions.

Usage examples::

    # Quick synthetic demo (no dataset download needed)
    python scripts/run_stability_benchmark.py --tracker MOSSE --synthetic \\
        --runs 5 --warmup 3

    # Against a real OTB dataset
    python scripts/run_stability_benchmark.py \\
        --tracker KCF \\
        --dataset-root /data/OTB100 \\
        --runs 5 --warmup 5 \\
        --output results/stability/kcf_otb.json

    # Full comparison of all registered trackers
    python scripts/run_stability_benchmark.py \\
        --tracker MOSSE KCF CSRT \\
        --synthetic --sequences 5 --frames 50 \\
        --runs 3 --output results/stability/comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-run benchmark stability analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--tracker", nargs="+", default=["MOSSE"],
        help="Tracker name(s) from the registry (default: MOSSE)",
    )
    p.add_argument("--runs", type=int, default=3, help="Number of benchmark runs (default: 3)")
    p.add_argument(
        "--warmup", type=int, default=5,
        help="Frames to discard from latency stats per sequence (default: 5)",
    )
    p.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic dataset instead of a real one",
    )
    p.add_argument("--sequences", type=int, default=5, help="Synthetic sequences (default: 5)")
    p.add_argument("--frames", type=int, default=50, help="Frames per synthetic sequence (default: 50)")
    p.add_argument("--dataset-root", type=str, default=None, help="Path to OTB dataset root")
    p.add_argument("--max-sequences", type=int, default=None, help="Cap number of evaluated sequences")
    p.add_argument("--output", type=str, default=None, help="Write JSON results to this path")
    p.add_argument("--tdp-watts", type=float, default=None, help="CPU TDP for energy estimation")
    p.add_argument("--verbose", action="store_true", help="Print per-run progress")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    from eovot.benchmark.stability import MultiRunBenchmark
    from eovot.trackers.registry import build_tracker
    from eovot.datasets.synthetic import SyntheticDataset

    # --- Dataset ---
    if args.synthetic or args.dataset_root is None:
        dataset = SyntheticDataset(num_sequences=args.sequences, num_frames=args.frames)
        dataset_name = f"synthetic_{args.sequences}x{args.frames}"
    else:
        from eovot.datasets.base import OTBDataset
        dataset = OTBDataset(args.dataset_root)
        dataset_name = Path(args.dataset_root).name

    engine = MultiRunBenchmark(
        n_runs=args.runs,
        warmup_frames=args.warmup,
        tdp_watts=args.tdp_watts,
        verbose=args.verbose,
    )

    all_results = {}
    for tracker_name in args.tracker:
        try:
            tracker = build_tracker(tracker_name)
        except KeyError:
            print(f"[ERROR] Unknown tracker: {tracker_name}", file=sys.stderr)
            print("Available trackers: MOSSE, KCF, CSRT, MIL, CamShift", file=sys.stderr)
            return 1

        print(f"\n{'=' * 60}")
        print(f"Tracker: {tracker_name}")
        print(f"{'=' * 60}")

        result = engine.run(tracker, dataset, dataset_name, max_sequences=args.max_sequences)
        print(result.to_markdown())
        all_results[tracker_name] = result.summary()

    # --- Output ---
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nResults saved to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
