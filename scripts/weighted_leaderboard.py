"""Compute a sequence-discriminativeness-weighted leaderboard from saved results.

Loads one or more BenchmarkResult JSON files (one per tracker) and computes
a weighted ranking that emphasises sequences where trackers diverge in
performance (variance scheme) or sequences that are hard on average
(difficulty scheme).

Usage::

    # Compare three trackers using variance weighting
    python scripts/weighted_leaderboard.py \\
        results/mosse.json results/kcf.json results/csrt.json

    # Use difficulty weighting (emphasises hard sequences)
    python scripts/weighted_leaderboard.py \\
        results/mosse.json results/kcf.json \\
        --scheme difficulty

    # Show both schemes and per-sequence weight table
    python scripts/weighted_leaderboard.py \\
        results/*.json --compare-schemes --weights

    # Also run a quick synthetic benchmark and show the result
    python scripts/weighted_leaderboard.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _run_synthetic_demo() -> None:
    from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.camshift import CamShiftTracker
    from eovot.analysis.weighted_leaderboard import SequenceWeightedLeaderboard

    print("Running synthetic benchmark for demo …")
    dataset = SyntheticDataset(num_sequences=8, num_frames=40, seed=42)
    engine = BenchmarkEngine(verbose=False)

    results = []
    for tracker in [MOSSETracker(), KCFTracker(), CamShiftTracker()]:
        r = engine.run(tracker, dataset, dataset_name="Synthetic")
        results.append(r)
        print(f"  {r.tracker_name}: mIoU={r.mean_iou:.4f}  FPS={r.mean_fps:.1f}")

    lb = SequenceWeightedLeaderboard(results)
    print("\n" + lb.rank("variance").to_markdown_table())
    print("\n" + lb.rank("difficulty").to_markdown_table())
    print("\n" + lb.rank_delta_table())
    print("\n" + lb.sequence_weight_table("variance"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequence-discriminativeness-weighted benchmark leaderboard.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "result_files",
        nargs="*",
        help="Paths to BenchmarkResult JSON files (one per tracker).",
    )
    parser.add_argument(
        "--scheme", "-s",
        choices=["variance", "difficulty"],
        default="variance",
        help=(
            "Weighting scheme:\n"
            "  variance   — up-weight sequences where trackers diverge (default)\n"
            "  difficulty — up-weight sequences where the average IoU is mid-range"
        ),
    )
    parser.add_argument(
        "--compare-schemes",
        action="store_true",
        default=False,
        help="Show both variance and difficulty rankings side by side.",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        default=False,
        help="Also print a per-sequence weight table.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Write all output to this Markdown file.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Run a quick synthetic benchmark demo (ignores result_files).",
    )
    args = parser.parse_args()

    if args.synthetic:
        _run_synthetic_demo()
        return 0

    if len(args.result_files) < 2:
        print(
            "Error: provide at least 2 BenchmarkResult JSON files, or use --synthetic.",
            file=sys.stderr,
        )
        return 1

    try:
        from eovot.benchmark.engine import BenchmarkResult
        from eovot.analysis.weighted_leaderboard import SequenceWeightedLeaderboard
    except ImportError as exc:
        print(f"Error: could not import eovot ({exc})", file=sys.stderr)
        return 2

    results = []
    for path in args.result_files:
        try:
            results.append(BenchmarkResult.load(path))
        except Exception as exc:
            print(f"Error loading {path}: {exc}", file=sys.stderr)
            return 1

    try:
        lb = SequenceWeightedLeaderboard(results)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_parts = []
    if args.compare_schemes:
        output_parts.append(lb.rank("variance").to_markdown_table())
        output_parts.append(lb.rank("difficulty").to_markdown_table())
        output_parts.append(lb.rank_delta_table())
    else:
        output_parts.append(lb.rank(args.scheme).to_markdown_table())

    if args.weights:
        output_parts.append(lb.sequence_weight_table(args.scheme))

    output = "\n".join(output_parts)
    print(output)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output, encoding="utf-8")
        print(f"\nMarkdown written to: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
