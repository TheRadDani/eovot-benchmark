"""Compare AdaptiveBudgetTracker against a fixed-rate baseline.

Runs the same base tracker in three configurations on a synthetic dataset:

1. **Baseline** — full update every frame.
2. **Static skip (k=2)** — FrameSkipTracker with fixed skip_rate=2.
3. **Adaptive** — AdaptiveBudgetTracker with configurable thresholds.

Prints a Markdown comparison table showing mIoU, FPS, and skip_rate.

Usage
-----
    # Quick synthetic test:
    python scripts/compare_adaptive_budget.py

    # Tune thresholds:
    python scripts/compare_adaptive_budget.py \\
        --easy 0.03 --hard 0.12 --tracker KCF \\
        --sequences 10 --frames 200

    # Save report to JSON:
    python scripts/compare_adaptive_budget.py --output results/adaptive_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.adaptive_budget import AdaptiveBudgetTracker
from eovot.trackers.frame_skip import FrameSkipTracker
from eovot.trackers.registry import build_tracker, available_trackers


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Adaptive budget vs. fixed-skip vs. baseline comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tracker",
        default="MOSSE",
        metavar="NAME",
        help=f"Base tracker.  Choices: {available_trackers()}",
    )
    p.add_argument("--sequences", type=int, default=5, metavar="N")
    p.add_argument("--frames", type=int, default=120, metavar="N")
    p.add_argument(
        "--motion",
        default="linear",
        choices=["linear", "circular", "random"],
    )
    p.add_argument(
        "--easy",
        type=float,
        default=0.04,
        metavar="THRESH",
        help="Complexity below which AdaptiveBudgetTracker coasts.",
    )
    p.add_argument(
        "--hard",
        type=float,
        default=0.12,
        metavar="THRESH",
        help="Complexity at which AdaptiveBudgetTracker exits easy mode.",
    )
    p.add_argument(
        "--skip-rate",
        type=int,
        default=2,
        metavar="K",
        dest="skip_rate",
        help="Fixed skip interval for the FrameSkipTracker baseline.",
    )
    p.add_argument("--output", metavar="PATH", help="Save JSON report to this path.")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    engine = BenchmarkEngine(verbose=not args.quiet)
    dataset = SyntheticDataset(
        num_sequences=args.sequences,
        num_frames=args.frames,
        motion=args.motion,
    )
    dataset_name = f"Syn-{args.motion.capitalize()}"

    # Build the three variants
    baseline = build_tracker(args.tracker)
    fixed_skip = FrameSkipTracker(build_tracker(args.tracker), skip_rate=args.skip_rate)
    adaptive = AdaptiveBudgetTracker(
        build_tracker(args.tracker),
        easy_threshold=args.easy,
        hard_threshold=args.hard,
    )

    results = {}
    for label, tracker in [
        ("Baseline", baseline),
        (f"FixedSkip(k={args.skip_rate})", fixed_skip),
        (f"Adaptive(e={args.easy},h={args.hard})", adaptive),
    ]:
        r = engine.run(tracker, dataset, dataset_name=dataset_name)
        results[label] = r

    # Print comparison table
    print("\n" + "=" * 72)
    print("ADAPTIVE BUDGET vs STATIC SKIP vs BASELINE")
    print("=" * 72)
    print(f"{'Tracker':<36} {'mIoU':>7} {'FPS':>8} {'skip_rate':>10}")
    print("-" * 72)

    for label, r in results.items():
        skip_info = ""
        if label.startswith("Adaptive"):
            skip_info = f"{adaptive.skip_rate:.1%}"
        elif label.startswith("FixedSkip"):
            # Approximate: (k-1)/k frames skipped
            k = args.skip_rate
            skip_info = f"{(k - 1) / k:.1%}"
        else:
            skip_info = "0.0%"
        print(f"{label:<36} {r.mean_iou:>7.4f} {r.mean_fps:>8.1f} {skip_info:>10}")

    baseline_iou = results["Baseline"].mean_iou
    baseline_fps = results["Baseline"].mean_fps
    print("\nRelative to baseline:")
    for label, r in results.items():
        if label == "Baseline":
            continue
        iou_delta = r.mean_iou - baseline_iou
        fps_gain = r.mean_fps / baseline_fps if baseline_fps > 0 else float("nan")
        sign = "+" if iou_delta >= 0 else ""
        print(f"  {label:<34}  ΔmIoU={sign}{iou_delta:.4f}  FPS_gain={fps_gain:.2f}×")

    if args.output:
        report = {
            "config": {
                "tracker": args.tracker,
                "motion": args.motion,
                "sequences": args.sequences,
                "frames": args.frames,
                "adaptive_easy": args.easy,
                "adaptive_hard": args.hard,
                "fixed_skip_rate": args.skip_rate,
            },
            "results": {
                label: {
                    "mean_iou": round(r.mean_iou, 4),
                    "mean_fps": round(r.mean_fps, 2),
                    "peak_memory_mb": round(r.peak_memory_mb, 2),
                }
                for label, r in results.items()
            },
            "adaptive_skip_rate": adaptive.skip_rate,
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
