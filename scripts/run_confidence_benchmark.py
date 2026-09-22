#!/usr/bin/env python3
"""CLI for benchmarking trackers with appearance confidence estimation.

Runs a tracker wrapped with AppearanceConfidenceTracker on a dataset,
prints per-sequence confidence reports, and optionally saves results to JSON.

Usage examples::

    # Synthetic demo — no dataset download needed
    python scripts/run_confidence_benchmark.py --tracker MOSSE --synthetic

    # Multiple trackers compared
    python scripts/run_confidence_benchmark.py \\
        --tracker MOSSE KCF CSRT \\
        --synthetic --sequences 5 --frames 80

    # Against a real OTB dataset
    python scripts/run_confidence_benchmark.py \\
        --tracker KCF \\
        --dataset-root /data/OTB100 \\
        --output results/confidence/kcf_otb.json

    # Tune confidence thresholds
    python scripts/run_confidence_benchmark.py --tracker MOSSE --synthetic \\
        --update-thresh 0.65 --low-thresh 0.35 --drift-thresh 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark trackers with per-frame appearance confidence scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--tracker", nargs="+", default=["MOSSE"],
        help="Tracker name(s) from the registry (default: MOSSE)",
    )
    p.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    p.add_argument("--sequences", type=int, default=5, help="Synthetic sequences (default: 5)")
    p.add_argument("--frames", type=int, default=50, help="Frames per synthetic sequence (default: 50)")
    p.add_argument("--dataset-root", type=str, default=None, help="Path to OTB dataset root")
    p.add_argument("--max-sequences", type=int, default=None, help="Cap evaluated sequences")
    p.add_argument(
        "--update-thresh", type=float, default=0.70,
        help="NCC threshold for template refresh (default: 0.70)",
    )
    p.add_argument(
        "--low-thresh", type=float, default=0.40,
        help="NCC threshold below which frame is counted low-confidence (default: 0.40)",
    )
    p.add_argument(
        "--drift-thresh", type=float, default=0.30,
        help="Low-confidence frame ratio above which drift is flagged (default: 0.30)",
    )
    p.add_argument("--output", type=str, default=None, help="Write JSON results to this path")
    p.add_argument("--tdp-watts", type=float, default=None, help="CPU TDP for energy estimation")
    p.add_argument("--verbose", action="store_true", help="Print verbose benchmark progress")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    from eovot.trackers.confidence import AppearanceConfidenceTracker
    from eovot.trackers.registry import build_tracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    # --- Dataset ---
    if args.synthetic or args.dataset_root is None:
        dataset = SyntheticDataset(num_sequences=args.sequences, num_frames=args.frames)
        dataset_name = f"synthetic_{args.sequences}x{args.frames}"
    else:
        from eovot.datasets.base import OTBDataset
        dataset = OTBDataset(args.dataset_root)
        dataset_name = Path(args.dataset_root).name

    engine = BenchmarkEngine(verbose=args.verbose, tdp_watts=args.tdp_watts)
    all_results = {}

    for tracker_name in args.tracker:
        try:
            base = build_tracker(tracker_name)
        except KeyError:
            print(f"[ERROR] Unknown tracker: {tracker_name}", file=sys.stderr)
            return 1

        tracker = AppearanceConfidenceTracker(
            base,
            template_update_thresh=args.update_thresh,
            low_confidence_thresh=args.low_thresh,
            drift_ratio_thresh=args.drift_thresh,
        )

        print(f"\n{'=' * 60}")
        print(f"Tracker: {tracker_name}  (wrapped with AppearanceConfidenceTracker)")
        print(f"{'=' * 60}")

        bench_result = engine.run(
            tracker, dataset, dataset_name, max_sequences=args.max_sequences
        )
        conf_report = tracker.confidence_report()

        print(f"\n{conf_report}")
        print(f"  Mean IoU:  {bench_result.mean_iou:.4f}")
        print(f"  Mean FPS:  {bench_result.mean_fps:.1f}")

        entry = bench_result.summary()
        entry["confidence"] = conf_report.to_dict()
        all_results[tracker_name] = entry

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nResults saved to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
