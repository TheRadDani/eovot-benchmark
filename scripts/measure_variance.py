"""CLI for benchmark measurement-variance analysis.

Runs a tracker N times on a dataset and reports metric stability statistics
(CV, confidence intervals, reliability flag).  Use this before reporting
single-run benchmark numbers to confirm that the measured metrics are
statistically stable.

Usage
-----
    # Quick check on synthetic data (no download required):
    python scripts/measure_variance.py \\
        --tracker MOSSE --synthetic --n-runs 5 --synthetic-frames 100

    # Check variance for KCF on an OTB-style dataset:
    python scripts/measure_variance.py \\
        --tracker KCF --dataset-root /data/OTB100 --n-runs 5 --max-sequences 10

    # Save Markdown report to file:
    python scripts/measure_variance.py \\
        --tracker CSRT --synthetic --n-runs 8 --output variance_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.analysis.variance_profiler import VarianceProfiler
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.base import OTBDataset
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.registry import build_tracker, available_trackers


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure benchmark metric variance over multiple runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--tracker", required=True, choices=available_trackers(),
                   help="Tracker to evaluate.")
    p.add_argument("--n-runs", type=int, default=5,
                   help="Number of independent benchmark runs (default: 5).")
    p.add_argument("--synthetic", action="store_true",
                   help="Use built-in synthetic dataset (no download required).")
    p.add_argument("--synthetic-sequences", type=int, default=5,
                   help="Number of synthetic sequences (default: 5).")
    p.add_argument("--synthetic-frames", type=int, default=100,
                   help="Frames per synthetic sequence (default: 100).")
    p.add_argument("--dataset-root", default=None,
                   help="Root directory of an OTB-style dataset.")
    p.add_argument("--dataset-name", default="OTBDataset",
                   help="Human-readable dataset name for reporting.")
    p.add_argument("--max-sequences", type=int, default=None,
                   help="Limit the number of sequences evaluated per run.")
    p.add_argument("--reliability-threshold", type=float, default=0.05,
                   help="CV threshold below which FPS is considered reliable (default: 0.05).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for bootstrap CI computation (default: 42).")
    p.add_argument("--output", default=None,
                   help="Write Markdown report to this file (prints to stdout if omitted).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.synthetic:
        dataset = SyntheticDataset(
            num_sequences=args.synthetic_sequences,
            num_frames=args.synthetic_frames,
            seed=args.seed,
        )
        dataset_name = "Synthetic"
    elif args.dataset_root:
        dataset = OTBDataset(args.dataset_root)
        dataset_name = args.dataset_name
    else:
        print("Error: provide --synthetic or --dataset-root.", file=sys.stderr)
        sys.exit(1)

    tracker = build_tracker(args.tracker)
    engine = BenchmarkEngine(verbose=False)

    print(f"Running {args.tracker} × {args.n_runs} times on {dataset_name} …")
    profiler = VarianceProfiler(
        n_runs=args.n_runs,
        seed=args.seed,
        reliability_threshold=args.reliability_threshold,
    )
    report = profiler.profile(
        tracker, dataset,
        dataset_name=dataset_name,
        engine=engine,
        max_sequences=args.max_sequences,
    )

    print(report)

    md = report.to_markdown()
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"\nMarkdown report saved to: {args.output}")
    else:
        print("\n" + md)


if __name__ == "__main__":
    main()
