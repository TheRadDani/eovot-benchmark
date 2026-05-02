"""CLI: Rank trackers by Edge Deployment Score (EDS) for a target hardware profile.

Reads benchmark JSON result files (produced by run_benchmark.py or
run_experiment.py) and ranks each tracker using the Edge Deployment Score —
a weighted composite of accuracy, throughput, memory, and energy.

Usage examples::

    # Rank using Raspberry Pi 4 profile (reads all JSON files in results/)
    python scripts/rank_for_edge.py --results-dir results/ --profile raspberry_pi_4

    # Rank using custom weights and show Pareto-optimal trackers
    python scripts/rank_for_edge.py \\
        --results-dir results/ \\
        --accuracy-weight 0.5 --fps-weight 0.3 \\
        --memory-weight 0.1 --energy-weight 0.1 \\
        --target-fps 30 --max-memory 512 \\
        --pareto

    # Save report to file
    python scripts/rank_for_edge.py \\
        --results-dir results/ \\
        --profile jetson_nano \\
        --output results/edge_leaderboard.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.metrics.edge_score import (
    HARDWARE_PROFILES,
    EdgeDeploymentScorer,
    TrackerMetrics,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rank trackers by Edge Deployment Score.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--results-dir",
        required=True,
        metavar="DIR",
        help="Directory containing tracker JSON result files.",
    )
    p.add_argument(
        "--profile",
        choices=list(HARDWARE_PROFILES),
        default=None,
        metavar="PROFILE",
        help=(
            "Hardware profile to use for weights and constraints. "
            f"Choices: {list(HARDWARE_PROFILES)}."
        ),
    )
    p.add_argument(
        "--accuracy-weight",
        type=float,
        default=0.25,
        help="Weight for accuracy (IoU) axis [0, 1]. Default: 0.25.",
    )
    p.add_argument(
        "--fps-weight",
        type=float,
        default=0.25,
        help="Weight for throughput (FPS) axis [0, 1]. Default: 0.25.",
    )
    p.add_argument(
        "--memory-weight",
        type=float,
        default=0.25,
        help="Weight for memory (lower is better) axis [0, 1]. Default: 0.25.",
    )
    p.add_argument(
        "--energy-weight",
        type=float,
        default=0.25,
        help="Weight for energy (lower is better) axis [0, 1]. Default: 0.25.",
    )
    p.add_argument(
        "--target-fps",
        type=float,
        default=25.0,
        help="Minimum FPS required for deployment. Default: 25.",
    )
    p.add_argument(
        "--max-memory",
        type=float,
        default=512.0,
        metavar="MB",
        help="Maximum peak memory in MB. Default: 512.",
    )
    p.add_argument(
        "--pareto",
        action="store_true",
        help="Print Pareto-optimal trackers (accuracy vs. speed).",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save the Markdown leaderboard and suitability report to FILE.",
    )
    p.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print all available hardware profiles and exit.",
    )
    return p


def load_tracker_metrics(results_dir: str) -> list[TrackerMetrics]:
    """Load TrackerMetrics from all JSON result files in *results_dir*."""
    dir_path = Path(results_dir)
    if not dir_path.is_dir():
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        print(f"No JSON result files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    trackers: list[TrackerMetrics] = []
    for jf in json_files:
        try:
            with open(jf) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: skipping {jf.name}: {exc}", file=sys.stderr)
            continue

        summary = data.get("summary", data)
        try:
            tm = TrackerMetrics(
                name=summary.get("tracker", jf.stem),
                mean_iou=float(summary["mean_iou"]),
                fps=float(summary["mean_fps"]),
                peak_memory_mb=float(summary["peak_memory_mb"]),
                energy_per_frame_mj=float(summary.get("mean_energy_per_frame_mj", 0.0)),
            )
            trackers.append(tm)
        except (KeyError, TypeError) as exc:
            print(f"Warning: skipping {jf.name}, missing field: {exc}", file=sys.stderr)

    return trackers


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_profiles:
        print("Available hardware profiles:")
        for name, cfg in HARDWARE_PROFILES.items():
            print(f"  {name:<20s}  {cfg['description']}")
        return

    # Build scorer
    if args.profile:
        scorer = EdgeDeploymentScorer.from_hardware_profile(args.profile)
        print(f"\nUsing hardware profile: {args.profile}")
        print(f"  {HARDWARE_PROFILES[args.profile]['description']}")
    else:
        total_w = args.accuracy_weight + args.fps_weight + args.memory_weight + args.energy_weight
        if abs(total_w - 1.0) > 1e-6:
            print(
                f"Error: weights must sum to 1.0, got {total_w:.4f}",
                file=sys.stderr,
            )
            sys.exit(1)
        scorer = EdgeDeploymentScorer(
            weights={
                "accuracy": args.accuracy_weight,
                "fps": args.fps_weight,
                "memory": args.memory_weight,
                "energy": args.energy_weight,
            },
            target_fps=args.target_fps,
            max_memory_mb=args.max_memory,
        )
        print("\nUsing custom weights:")
        print(f"  accuracy={args.accuracy_weight}  fps={args.fps_weight}  "
              f"memory={args.memory_weight}  energy={args.energy_weight}")

    # Load tracker results
    trackers = load_tracker_metrics(args.results_dir)
    if not trackers:
        print("No valid tracker results found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoaded {len(trackers)} tracker(s) from {args.results_dir}")

    # Score
    scores = scorer.score(trackers)

    # Print leaderboard
    print("\n" + scorer.format_leaderboard(scores))

    # Print suitability report
    print("\n" + scorer.suitability_report(scores, profile_name=args.profile))

    # Pareto frontier
    if args.pareto:
        pareto_pts = scorer.pareto_frontier(trackers)
        optimal = [p.tracker_name for p in pareto_pts if p.is_pareto_optimal]
        print(f"\nPareto-optimal trackers (accuracy vs. speed): {optimal}")

    # Save output
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        content = scorer.format_leaderboard(scores)
        content += "\n\n"
        content += scorer.suitability_report(scores, profile_name=args.profile)
        out.write_text(content)
        print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
