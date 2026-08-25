#!/usr/bin/env python3
"""Tracker drift analysis CLI for EOVOT.

Loads one or more EOVOT benchmark JSON result files and produces a drift
analysis report — measuring how tracker accuracy degrades over sequence length.
This complements the discrete-failure analysis in ``eovot.metrics.robustness``
by capturing *gradual* accuracy decay, which is critical for setting
re-initialisation timeouts in long-running edge deployments.

Key metrics reported per sequence:

- **Stable duration** — consecutive frames where IoU stays above the stable
  threshold (how long you can trust the tracker).
- **Decay rate** — linear regression slope of IoU vs. frame (IoU/frame).
- **Stability ratio** — mean IoU (2nd half) / mean IoU (1st half); < 1 = drift.
- **Half-life** — estimated frame where IoU reaches half its initial value.
- **Collapse frame** — frame where IoU drops permanently below the collapse
  threshold.

Usage::

    # Single tracker — print table and aggregate summary
    python scripts/analyze_drift.py --results results/MOSSE-OTB100.json

    # Compare two trackers side by side
    python scripts/analyze_drift.py \\
        --results results/MOSSE-OTB100.json results/KCF-OTB100.json

    # Custom thresholds; save Markdown tables
    python scripts/analyze_drift.py \\
        --results results/*.json \\
        --stable-threshold 0.4 \\
        --collapse-threshold 0.15 \\
        --output-dir reports/drift/

    # Quick run on a synthetic benchmark (no dataset download needed)
    python -c "
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker
    import json, pathlib
    r = BenchmarkEngine(verbose=False).run(
        MOSSETracker(),
        SyntheticDataset(num_sequences=5, num_frames=200, motion='random'),
        dataset_name='Synthetic',
    )
    pathlib.Path('results').mkdir(exist_ok=True)
    r.save('results/MOSSE-Synthetic.json')
    "
    python scripts/analyze_drift.py --results results/MOSSE-Synthetic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.analysis.drift import DriftAnalyzer, DriftResult
from eovot.benchmark.engine import BenchmarkResult


def _load_result(path: Path) -> BenchmarkResult:
    """Deserialise a BenchmarkResult from a JSON file."""
    with open(path) as fh:
        data = json.load(fh)
    return BenchmarkResult.from_dict(data)


def _print_aggregate(agg: dict) -> None:
    """Print the aggregate summary block to stdout."""
    print(f"\n{'─' * 60}")
    print(f"  Tracker : {agg['tracker_name']}")
    print(f"  Seqs    : {agg['num_sequences']}")
    print(f"  Mean stable duration : {agg.get('mean_stable_duration_frames', 0):.1f} frames")
    print(f"  Mean decay rate      : {agg.get('mean_decay_rate_per_frame', 0):+.6f} IoU/frame")
    print(f"  Mean stability ratio : {agg.get('mean_stability_ratio', 1.0):.4f}")
    print(f"  Drifting sequences   : {agg.get('n_drifting_sequences', 0)}")
    print(f"  Collapsed sequences  : {agg.get('n_collapsed_sequences', 0)}")
    if "mean_half_life_frames" in agg:
        print(f"  Mean half-life       : {agg['mean_half_life_frames']:.1f} frames")
    print(f"\n  Verdict: {agg['verdict']}")
    print(f"{'─' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="analyze_drift",
        description=(
            "Measure tracker accuracy drift over sequence length from EOVOT JSON results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        metavar="JSON",
        help="One or more EOVOT JSON result files (from BenchmarkResult.save()).",
    )
    parser.add_argument(
        "--stable-threshold",
        type=float,
        default=0.3,
        metavar="T",
        help="IoU at or above which the tracker is considered stable.",
    )
    parser.add_argument(
        "--collapse-threshold",
        type=float,
        default=0.1,
        metavar="T",
        help="IoU below which the tracker is considered permanently collapsed.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=5,
        metavar="N",
        help="Frames at the start of each sequence to exclude from analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "If set, save per-tracker Markdown drift tables to this directory. "
            "Directory is created automatically if it does not exist."
        ),
    )
    args = parser.parse_args()

    analyzer = DriftAnalyzer(
        stable_threshold=args.stable_threshold,
        collapse_threshold=args.collapse_threshold,
        burn_in_frames=args.burn_in,
    )

    output_dir: Path | None = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_summaries = []

    for result_path in args.results:
        p = Path(result_path)
        if not p.exists():
            print(f"[WARN] File not found, skipping: {p}", file=sys.stderr)
            continue

        result = _load_result(p)
        report = analyzer.analyze_benchmark(result)
        agg = report["aggregate"]
        per_seq: dict[str, DriftResult] = report["per_sequence"]

        print(f"\n{'=' * 60}")
        print(f"  Drift Analysis: {result.tracker_name} on {result.dataset_name}")
        print(f"  Source: {p}")
        print(f"{'=' * 60}")

        table = analyzer.to_markdown_table(per_seq.values())
        print(table)
        _print_aggregate(agg)

        if output_dir is not None:
            fname = f"drift-{result.tracker_name}-{result.dataset_name}.md"
            out_path = output_dir / fname
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(f"# Drift Analysis: {result.tracker_name} on {result.dataset_name}\n\n")
                fh.write(table)
                fh.write(f"\n## Aggregate Summary\n\n")
                for k, v in agg.items():
                    fh.write(f"- **{k}**: {v}\n")
            print(f"  [saved] {out_path}")

        aggregate_summaries.append(agg)

    # Cross-tracker comparison when multiple results are loaded.
    if len(aggregate_summaries) > 1:
        print(f"\n{'=' * 60}")
        print("  CROSS-TRACKER DRIFT COMPARISON")
        print(f"{'=' * 60}")
        print(
            f"{'Tracker':<20} {'StableFr':>10} {'Decay/fr':>12} "
            f"{'Ratio':>8} {'Drifting':>10} {'HalfLife':>10}"
        )
        print("-" * 74)
        for agg in sorted(
            aggregate_summaries,
            key=lambda a: a.get("mean_stable_duration_frames", 0),
            reverse=True,
        ):
            hl = (
                f"{agg['mean_half_life_frames']:.0f}"
                if "mean_half_life_frames" in agg
                else "∞"
            )
            print(
                f"{agg['tracker_name']:<20} "
                f"{agg.get('mean_stable_duration_frames', 0):>10.1f} "
                f"{agg.get('mean_decay_rate_per_frame', 0):>+12.6f} "
                f"{agg.get('mean_stability_ratio', 1.0):>8.3f} "
                f"{agg.get('n_drifting_sequences', 0):>10} "
                f"{hl:>10}"
            )
        print()


if __name__ == "__main__":
    main()
