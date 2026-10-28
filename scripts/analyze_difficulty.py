"""CLI: analyze sequence difficulty from a saved BenchmarkResult JSON.

Usage examples
--------------
# Profile all sequences in a saved benchmark result:
python scripts/analyze_difficulty.py results/KCF-OTB100.json

# Show only 'hard' and 'extreme' sequences:
python scripts/analyze_difficulty.py results/KCF-OTB100.json --min-tier hard

# Adjust objective weights:
python scripts/analyze_difficulty.py results/KCF-OTB100.json \\
    --motion-weight 0.5 --scale-weight 0.3 --deform-weight 0.1 --occlusion-weight 0.1

# Export to JSON:
python scripts/analyze_difficulty.py results/KCF-OTB100.json --output difficulty.json

# Use a synthetic dataset for a quick demo (no real data required):
python scripts/analyze_difficulty.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TIER_ORDER = {"easy": 0, "medium": 1, "hard": 2, "extreme": 3}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze per-sequence tracking difficulty from GT box trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "result_json",
        nargs="?",
        help="Path to a BenchmarkResult JSON produced by eovot.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run on a built-in synthetic dataset (no result JSON needed).",
    )
    parser.add_argument(
        "--min-tier",
        choices=["easy", "medium", "hard", "extreme"],
        default=None,
        help="Only show sequences at or above this difficulty tier.",
    )
    parser.add_argument(
        "--motion-weight", type=float, default=0.40, metavar="W",
        help="Weight for motion speed dimension (default: 0.40).",
    )
    parser.add_argument(
        "--scale-weight", type=float, default=0.30, metavar="W",
        help="Weight for scale volatility dimension (default: 0.30).",
    )
    parser.add_argument(
        "--deform-weight", type=float, default=0.20, metavar="W",
        help="Weight for aspect deformation dimension (default: 0.20).",
    )
    parser.add_argument(
        "--occlusion-weight", type=float, default=0.10, metavar="W",
        help="Weight for occlusion event rate dimension (default: 0.10).",
    )
    parser.add_argument(
        "--output", "-o", default=None, metavar="PATH",
        help="Write per-sequence difficulty JSON to this path.",
    )
    parser.add_argument(
        "--no-table", action="store_true",
        help="Skip Markdown table output (useful with --output).",
    )
    args = parser.parse_args()

    if not args.synthetic and args.result_json is None:
        parser.error("Provide a result JSON path or pass --synthetic.")

    from eovot.analysis.difficulty import SequenceDifficultyProfiler

    profiler = SequenceDifficultyProfiler(
        motion_weight=args.motion_weight,
        scale_weight=args.scale_weight,
        deform_weight=args.deform_weight,
        occlusion_weight=args.occlusion_weight,
    )

    if args.synthetic:
        results = _run_synthetic(profiler)
    else:
        results = _run_from_json(args.result_json, profiler)

    # Apply tier filter.
    if args.min_tier:
        min_order = TIER_ORDER[args.min_tier]
        results = {k: v for k, v in results.items() if TIER_ORDER[v.difficulty_tier] >= min_order}

    if not results:
        print("No sequences match the filter criteria.")
        sys.exit(0)

    # Aggregate stats.
    stats = profiler.aggregate_stats(results)
    print("\n=== Dataset Difficulty Summary ===")
    print(f"  Sequences profiled : {stats['num_sequences']}")
    print(f"  Mean difficulty    : {stats['mean_difficulty_score']:.4f} ± {stats['std_difficulty_score']:.4f}")
    print(f"  Mean motion speed  : {stats['mean_motion_speed']:.4f}")
    print(f"  Mean scale CV      : {stats['mean_scale_volatility']:.4f}")
    print(f"  Mean aspect deform : {stats['mean_aspect_deformation']:.4f}")
    print(f"  Mean occlusion/100 : {stats['mean_occlusion_events_per_100fr']:.3f}")
    tier_dist = stats["tier_distribution"]
    print(f"  Tier distribution  : easy={tier_dist['easy']}  medium={tier_dist['medium']}  "
          f"hard={tier_dist['hard']}  extreme={tier_dist['extreme']}")

    if not args.no_table:
        print("\n=== Per-Sequence Difficulty (sorted by score, hardest first) ===")
        print(profiler.to_markdown_table(results))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "aggregate": stats,
            "sequences": {name: r.to_dict() for name, r in results.items()},
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nDifficulty analysis written to: {out_path}")


def _run_from_json(json_path: str, profiler):
    """Load a BenchmarkResult from JSON and profile it."""
    from eovot.benchmark.engine import BenchmarkResult

    path = Path(json_path)
    if not path.exists():
        print(f"Error: file not found — {path}", file=sys.stderr)
        sys.exit(1)

    result = BenchmarkResult.load(path)
    results = profiler.profile_benchmark(result)

    if not results:
        print(
            "No sequences with stored GT boxes found.\n"
            "Make sure the benchmark was run with BenchmarkEngine (which stores "
            "ground_truths in every SequenceResult).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded: {result.tracker_name} on {result.dataset_name}  "
          f"({len(result.sequence_results)} sequences)")
    return results


def _run_synthetic(profiler):
    """Run profiler on a built-in synthetic dataset and return results dict."""
    import numpy as np
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    print("Running synthetic benchmark (MOSSE on 6 sequences)…")
    dataset = SyntheticDataset(
        num_sequences=6, num_frames=120,
        frame_size=(320, 240), bbox_size=(40, 40),
    )
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")
    results = profiler.profile_benchmark(result)
    print(f"Profiled {len(results)} sequences.")
    return results


if __name__ == "__main__":
    main()
