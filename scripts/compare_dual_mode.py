#!/usr/bin/env python3
"""CLI: Benchmark DualModeTracker and compare against its constituent trackers.

Runs a synthetic benchmark comparing a *fast* tracker, an *accurate* tracker,
and a DualModeTracker (fast + accurate) that switches dynamically based on
rolling-window latency.  Produces:

  1. Per-tracker performance table (mIoU, FPS, latency).
  2. DualModeTracker mode-usage summary (% frames in fast / accurate mode).
  3. Latency budget compliance table via LatencyBudgetAnalyzer.

Usage::

    # Quick demo (MOSSE vs KCF vs DualMode at 30 FPS default budget):
    python scripts/compare_dual_mode.py

    # Custom budget and trackers:
    python scripts/compare_dual_mode.py \\
        --fast CamShift --accurate KCF --budget-ms 20 \\
        --sequences 6 --frames 80

    # Save results to disk:
    python scripts/compare_dual_mode.py \\
        --output results/dual_mode_comparison/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.dual_mode import DualModeTracker
from eovot.trackers.registry import build_tracker, available_trackers
from eovot.analysis.latency_budget import LatencyBudgetAnalyzer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DualModeTracker against its constituent trackers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fast", default="MOSSE",
                        help="Fast tracker name (default: MOSSE)")
    parser.add_argument("--accurate", default="KCF",
                        help="Accurate tracker name (default: KCF)")
    parser.add_argument("--budget-ms", type=float, default=33.3,
                        help="Latency budget in ms (default: 33.3 ≈ 30 FPS)")
    parser.add_argument("--window-size", type=int, default=5,
                        help="DualMode rolling window size in frames (default: 5)")
    parser.add_argument("--sequences", type=int, default=5,
                        help="Number of synthetic sequences (default: 5)")
    parser.add_argument("--frames", type=int, default=60,
                        help="Frames per sequence (default: 60)")
    parser.add_argument("--motion", default="linear",
                        choices=["linear", "circular", "random"],
                        help="Synthetic dataset motion pattern (default: linear)")
    parser.add_argument("--target-fps", type=float, default=None,
                        help="FPS for compliance table (default: 1000/budget-ms)")
    parser.add_argument("--output", default=None,
                        help="Directory to save JSON results")
    parser.add_argument("--list-trackers", action="store_true",
                        help="Print available tracker names and exit")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_trackers:
        print("Available trackers:", ", ".join(available_trackers()))
        return

    target_fps = args.target_fps or (1000.0 / args.budget_ms)

    dataset = SyntheticDataset(
        num_sequences=args.sequences,
        frames_per_sequence=args.frames,
        motion=args.motion,
    )
    dataset_name = f"Synthetic({args.motion},{args.sequences}x{args.frames})"

    fast_t = build_tracker(args.fast)
    accurate_t = build_tracker(args.accurate)
    dual_t = DualModeTracker(
        fast=build_tracker(args.fast),
        accurate=build_tracker(args.accurate),
        budget_ms=args.budget_ms,
        window_size=args.window_size,
    )

    print(f"\nRunning benchmark: {args.fast} vs {args.accurate} vs DualMode")
    print(f"Budget: {args.budget_ms:.1f} ms/frame ({target_fps:.0f} FPS)")
    print(f"Dataset: {args.sequences} sequences × {args.frames} frames ({args.motion} motion)")
    print("-" * 60)

    engine = BenchmarkEngine(verbose=True)
    results = []
    for tracker in [fast_t, accurate_t, dual_t]:
        r = engine.run(tracker, dataset, dataset_name=dataset_name)
        results.append(r)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            saved = r.save(out_dir / f"{r.tracker_name}.json")
            print(f"Saved: {saved}")

    # ------ Performance table ------
    print("\n## Performance Summary\n")
    print("| Tracker | mIoU | FPS | Mean Lat (ms) | P95 Lat (ms) |")
    print("|---------|------|-----|--------------|-------------|")
    for r in results:
        lat_mean = float(
            sum(sr.profiling.latency_mean_ms for sr in r.sequence_results)
            / max(1, len(r.sequence_results))
        )
        p95 = max(
            (sr.profiling.latency_p95_ms for sr in r.sequence_results),
            default=0.0,
        )
        print(
            f"| {r.tracker_name} "
            f"| {r.mean_iou:.3f} "
            f"| {r.mean_fps:.1f} "
            f"| {lat_mean:.2f} "
            f"| {p95:.2f} |"
        )

    # ------ DualMode switch stats ------
    print(f"\n## DualModeTracker Statistics\n")
    print(f"  Budget:         {args.budget_ms:.1f} ms ({target_fps:.0f} FPS)")
    print(f"  Window size:    {args.window_size} frames")
    s = dual_t.stats
    print(f"  Total frames:   {s.total_frames}")
    print(f"  Accurate mode:  {s.accurate_frames} frames ({s.accurate_fraction:.1%})")
    print(f"  Fast mode:      {s.fast_frames} frames ({s.fast_fraction:.1%})")
    print(f"  Mode switches:  {s.num_switches}")
    if s.num_switches > 0:
        print("  Switch log:")
        for evt in s.switches[:5]:
            print(f"    {evt}")
        if s.num_switches > 5:
            print(f"    ... ({s.num_switches - 5} more)")

    # ------ Latency compliance ------
    analyzer = LatencyBudgetAnalyzer(target_fps_list=[target_fps])
    reports = [analyzer.analyze_benchmark(r) for r in results]
    print(f"\n## Latency Budget Compliance @ {target_fps:.0f} FPS\n")
    print(LatencyBudgetAnalyzer.multi_tracker_table(reports))


if __name__ == "__main__":
    main()
