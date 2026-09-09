"""CLI tool to generate an edge deployment scorecard from benchmark results.

Usage examples::

    # From a live benchmark run (synthetic dataset)
    python scripts/edge_scorecard.py --synthetic \\
        --trackers MOSSE KCF CSRT \\
        --devices rpi4 jetson_nano coral_board \\
        --target-fps 25 \\
        --output results/scorecard

    # From a saved BenchmarkResult JSON file
    python scripts/edge_scorecard.py \\
        --results results/benchmark_mosse.json \\
        --devices rpi4 jetson_nano \\
        --output results/scorecard

Outputs Markdown (.md), JSON (.json), and CSV (.csv) files at the
specified output prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.reporting.edge_scorecard import EdgeScorecard
from eovot.results.bank import BenchmarkResultsBank
from eovot.trackers.registry import build_tracker, available_trackers


def _run_benchmark_synthetic(
    tracker_names: list[str],
    num_sequences: int = 5,
    num_frames: int = 80,
) -> list:
    """Run a quick benchmark on synthetic data and return BenchmarkResults."""
    dataset = SyntheticDataset(
        num_sequences=num_sequences,
        num_frames=num_frames,
        motion="linear",
    )
    engine = BenchmarkEngine(verbose=False)
    results = []
    for name in tracker_names:
        try:
            tracker = build_tracker(name)
            result = engine.run(tracker, dataset, dataset_name="Synthetic")
            results.append(result)
        except Exception as exc:
            print(f"[warn] Skipping {name}: {exc}", file=sys.stderr)
    return results


def _load_results_from_json(paths: list[str]) -> list:
    """Load BenchmarkResult objects from persisted JSON files."""
    bank = BenchmarkResultsBank()
    results = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        result = bank._from_dict(data)
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an edge deployment scorecard for EOVOT trackers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="Run a quick benchmark on synthetic data before scoring.",
    )
    src.add_argument(
        "--results",
        nargs="+",
        metavar="JSON",
        help="Paths to saved BenchmarkResult JSON files.",
    )

    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["MOSSE", "KCF", "CSRT"],
        metavar="NAME",
        help=f"Tracker names (--synthetic only). Available: {available_trackers()}",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["rpi4", "rpi5", "jetson_nano", "jetson_xnx", "coral_board"],
        metavar="DEVICE",
        help="Device names to score against (default: all built-in devices).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=25.0,
        help="Minimum FPS for READY tier (default: 25).",
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=0.2,
        help="Weight of energy efficiency in Deployment Score (0-1, default: 0.2).",
    )
    parser.add_argument(
        "--sustained-seconds",
        type=float,
        default=60.0,
        help="Duration (s) for thermal throttle simulation (default: 60).",
    )
    parser.add_argument(
        "--output",
        default="results/edge_scorecard",
        help="Output path prefix (extensions .md/.json/.csv added automatically).",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=5,
        help="Synthetic sequences per tracker (--synthetic only, default: 5).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=80,
        help="Frames per synthetic sequence (--synthetic only, default: 80).",
    )

    args = parser.parse_args(argv)

    # ----------------------------------------------------------------
    # Acquire benchmark results
    # ----------------------------------------------------------------
    if args.synthetic:
        print(f"Running benchmark on synthetic data for: {args.trackers}")
        bench_results = _run_benchmark_synthetic(
            args.trackers,
            num_sequences=args.num_sequences,
            num_frames=args.num_frames,
        )
    else:
        print(f"Loading results from: {args.results}")
        bench_results = _load_results_from_json(args.results)

    if not bench_results:
        print("[error] No benchmark results to score.", file=sys.stderr)
        return 1

    # ----------------------------------------------------------------
    # Generate scorecard
    # ----------------------------------------------------------------
    scorecard = EdgeScorecard(
        target_fps=args.target_fps,
        devices=args.devices,
        energy_weight=args.energy_weight,
        sustained_seconds=args.sustained_seconds,
    )

    rows = scorecard.evaluate(bench_results)

    # ----------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------
    prefix = args.output

    md_path = prefix + ".md"
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_path).write_text(scorecard.to_markdown(rows))
    print(f"Markdown scorecard  -> {md_path}")

    json_path = prefix + ".json"
    scorecard.to_json(rows, json_path)
    print(f"JSON scorecard      -> {json_path}")

    csv_path = prefix + ".csv"
    scorecard.to_csv(rows, csv_path)
    print(f"CSV scorecard       -> {csv_path}")

    # ----------------------------------------------------------------
    # Print summary table
    # ----------------------------------------------------------------
    print("\n" + scorecard.to_markdown(rows))
    ready = sum(1 for r in rows if r.tier == "READY")
    marginal = sum(1 for r in rows if r.tier == "MARGINAL")
    not_feasible = sum(1 for r in rows if r.tier == "NOT FEASIBLE")
    print(
        f"Summary: {len(rows)} tracker×device pairs evaluated — "
        f"{ready} READY, {marginal} MARGINAL, {not_feasible} NOT FEASIBLE"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
