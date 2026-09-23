#!/usr/bin/env python3
"""CLI: Tracker warm-up / cold-start analysis.

Measures how many frames a tracker needs to reach steady-state throughput
and quantifies the cold-start FPS penalty.  Useful for setting deployment
expectations on edge hardware where the first N frames may violate a
real-time SLA even though long-run FPS is adequate.

Synthetic example (no dataset download needed)::

    python scripts/analyze_warmup.py \\
        --tracker MOSSE --synthetic \\
        --sequences 5 --frames 200

Real OTB dataset::

    python scripts/analyze_warmup.py \\
        --tracker KCF --dataset-root /data/OTB100 \\
        --max-sequences 10 --output-json results/kcf_warmup.json
"""

from __future__ import annotations

import argparse
import json
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Warm-up / cold-start analysis for EOVOT trackers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tracker", default="MOSSE",
                   help="Tracker name from the EOVOT registry.")
    p.add_argument("--synthetic", action="store_true",
                   help="Use SyntheticDataset (no --dataset-root needed).")
    p.add_argument("--sequences", type=int, default=5,
                   help="Number of sequences (synthetic mode).")
    p.add_argument("--frames", type=int, default=200,
                   help="Frames per sequence (synthetic mode).")
    p.add_argument("--motion", default="linear",
                   choices=["linear", "circular", "random"],
                   help="Motion pattern (synthetic mode).")
    p.add_argument("--dataset-root", default=None,
                   help="Path to an OTB-layout dataset root.")
    p.add_argument("--max-sequences", type=int, default=None,
                   help="Limit evaluation to first N sequences.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit frames per sequence.")
    p.add_argument("--ema-alpha", type=float, default=0.1,
                   help="EMA smoothing factor for warm-up detection.")
    p.add_argument("--tolerance", type=float, default=0.10,
                   help="Steady-state band half-width (fraction of mean).")
    p.add_argument("--min-steady-frames", type=int, default=10,
                   help="Minimum frames in steady-state estimation window.")
    p.add_argument("--output-json", default=None,
                   help="Write JSON report to this file path.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-sequence output; print only summary.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    from eovot.trackers.registry import build_tracker
    from eovot.analysis.warmup_analysis import WarmupAnalyzer

    tracker = build_tracker(args.tracker)

    if args.synthetic:
        from eovot.datasets.synthetic import SyntheticDataset
        dataset = SyntheticDataset(
            num_sequences=args.sequences,
            num_frames=args.frames,
            motion=args.motion,
        )
        dataset_label = f"Synthetic({args.motion})"
    elif args.dataset_root:
        from eovot.datasets.base import OTBDataset
        dataset = OTBDataset(args.dataset_root)
        dataset_label = args.dataset_root
    else:
        print("ERROR: Provide --synthetic or --dataset-root.", file=sys.stderr)
        sys.exit(1)

    analyzer = WarmupAnalyzer(
        ema_alpha=args.ema_alpha,
        tolerance=args.tolerance,
        min_steady_frames=args.min_steady_frames,
    )

    print(f"Running warm-up analysis: {tracker.name} on {dataset_label}")
    report = analyzer.analyze_dataset(
        tracker,
        dataset,
        max_sequences=args.max_sequences,
        max_frames_per_seq=args.max_frames,
    )

    if not report.results:
        print("No results — all sequences were too short for analysis.")
        sys.exit(0)

    if not args.quiet:
        for r in report.results:
            print(r)
        print()

    print(report.to_markdown())
    print()
    print(f"Mean init time     : {report.mean_init_ms:.2f} ms")
    print(f"Mean warmup frames : {report.mean_warmup_frames:.1f}")
    print(f"Mean steady FPS    : {report.mean_steady_state_fps:.1f}")
    print(f"Mean speedup       : {report.mean_fps_improvement:.2f}x")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(report.summary_dict(), fh, indent=2)
        print(f"\nJSON report saved to {args.output_json}")


if __name__ == "__main__":
    main()
