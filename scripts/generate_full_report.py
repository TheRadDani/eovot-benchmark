#!/usr/bin/env python3
"""Generate a consolidated EOVOT research report from saved benchmark JSON files
or by running trackers live against a dataset.

The report covers four sections in one Markdown document:
  1. Accuracy Leaderboard (mIoU, success AUC, precision AUC)
  2. Efficiency Comparison (FPS, latency, memory, EES, Pareto front)
  3. Robustness Analysis (EAO, failures, survival rate)
  4. Per-Attribute Performance (scale variation, fast motion, occlusion, …)

Usage — from saved JSON results
--------------------------------
    python scripts/generate_full_report.py \\
        --results results/MOSSE-Synthetic.json results/KCF-Synthetic.json \\
        --output results/full_report.md

Usage — run trackers live (synthetic dataset, no download needed)
-----------------------------------------------------------------
    python scripts/generate_full_report.py \\
        --trackers MOSSE KCF \\
        --synthetic \\
        --synthetic-sequences 10 --synthetic-frames 100 \\
        --output results/full_report.md

Usage — run trackers live against a real dataset
-------------------------------------------------
    python scripts/generate_full_report.py \\
        --trackers MOSSE KCF CSRT \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100 \\
        --max-sequences 50 \\
        --output results/full_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.base import OTBDataset
from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.lasot import LaSOTDataset
from eovot.datasets.synthetic import SyntheticDataset
from eovot.reporting.full_report import FullReportGenerator
from eovot.trackers.registry import TRACKER_REGISTRY

DATASET_REGISTRY = {
    "OTBDataset": OTBDataset,
    "GOT10kDataset": GOT10kDataset,
    "LaSOTDataset": LaSOTDataset,
}


def _load_results(paths: list) -> list:
    results = []
    for p in paths:
        print(f"  Loading {p} …")
        results.append(BenchmarkResult.load(p))
    return results


def _run_live(args: argparse.Namespace) -> list:
    """Run benchmark live and return result objects."""
    if args.synthetic:
        frame_size = tuple(args.synthetic_size)
        dataset = SyntheticDataset(
            num_sequences=args.synthetic_sequences,
            num_frames=args.synthetic_frames,
            frame_size=frame_size,
            motion=args.synthetic_motion,
        )
        dataset_name = args.dataset_name or "Synthetic"
    else:
        loader_cls = DATASET_REGISTRY.get(args.dataset_loader)
        if loader_cls is None:
            print(f"[ERROR] Unknown dataset loader: {args.dataset_loader}", file=sys.stderr)
            sys.exit(1)
        if args.dataset_loader == "OTBDataset":
            dataset = loader_cls(args.dataset_root)
        else:
            dataset = loader_cls(
                args.dataset_root,
                split=args.split,
                max_sequences=args.max_sequences,
            )
        dataset_name = args.dataset_name or args.dataset_loader

    engine = BenchmarkEngine(
        verbose=not args.quiet,
        tdp_watts=args.tdp_watts,
    )

    results = []
    for tracker_name in args.trackers:
        cls = TRACKER_REGISTRY.get(tracker_name)
        if cls is None:
            print(f"[ERROR] Unknown tracker '{tracker_name}'.", file=sys.stderr)
            sys.exit(1)
        tracker = cls()
        print(f"\nEvaluating {tracker_name} …")
        result = engine.run(
            tracker=tracker,
            dataset=dataset,
            dataset_name=dataset_name,
            max_sequences=args.max_sequences,
        )
        results.append(result)
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_full_report",
        description="Generate a consolidated EOVOT Markdown research report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_argument_group(
        "input source (choose one: --results or --trackers)"
    )
    src.add_argument(
        "--results",
        nargs="+",
        metavar="PATH",
        help="Paths to saved BenchmarkResult JSON files.",
    )
    src.add_argument(
        "--trackers",
        nargs="+",
        choices=list(TRACKER_REGISTRY),
        metavar="TRACKER",
        help="Tracker names to evaluate live.",
    )

    ds = parser.add_argument_group("dataset (when using --trackers)")
    ds.add_argument("--dataset-root", metavar="DIR")
    ds.add_argument("--dataset-name", metavar="NAME", default=None)
    ds.add_argument(
        "--dataset-loader",
        default="OTBDataset",
        choices=list(DATASET_REGISTRY),
    )
    ds.add_argument("--split", default="val")
    ds.add_argument("--max-sequences", type=int, default=None, metavar="N")
    ds.add_argument("--tdp-watts", type=float, default=None, metavar="W")
    ds.add_argument("--quiet", action="store_true")

    syn = parser.add_argument_group("synthetic dataset")
    syn.add_argument("--synthetic", action="store_true")
    syn.add_argument("--synthetic-sequences", type=int, default=10, metavar="N")
    syn.add_argument("--synthetic-frames", type=int, default=100, metavar="N")
    syn.add_argument(
        "--synthetic-size", type=int, nargs=2, default=[320, 240], metavar=("W", "H")
    )
    syn.add_argument(
        "--synthetic-motion",
        default="linear",
        choices=["linear", "circular", "random"],
    )

    rep = parser.add_argument_group("report options")
    rep.add_argument(
        "--output", "-o",
        default="results/full_report.md",
        metavar="PATH",
        help="Destination Markdown file.",
    )
    rep.add_argument("--title", default=None, help="Custom report title.")
    rep.add_argument(
        "--memory-budget",
        type=float,
        default=512.0,
        metavar="MB",
        help="Memory budget for EES denominator.",
    )
    rep.add_argument(
        "--failure-threshold",
        type=float,
        default=0.1,
        metavar="T",
        help="IoU threshold below which a frame counts as a failure.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.results:
        results = _load_results(args.results)
    elif args.trackers:
        if not (args.synthetic or args.dataset_root):
            print(
                "[ERROR] Provide --synthetic or --dataset-root when using --trackers.",
                file=sys.stderr,
            )
            sys.exit(1)
        results = _run_live(args)
    else:
        parser.print_help()
        sys.exit(0)

    gen = FullReportGenerator(
        memory_budget_mb=args.memory_budget,
        failure_threshold=args.failure_threshold,
        title=args.title,
    )
    saved = gen.save(results, path=args.output)
    print(f"\nReport written to: {saved}")

    # Print a brief preview of the accuracy table.
    report = saved.read_text(encoding="utf-8")
    lines = report.split("\n")
    start = next((i for i, l in enumerate(lines) if "Accuracy Leaderboard" in l), 0)
    preview = "\n".join(lines[start : start + 12])
    print("\n" + preview)


if __name__ == "__main__":
    main()
