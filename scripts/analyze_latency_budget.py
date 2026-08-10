"""CLI: analyze tracker real-time latency budget compliance.

Usage
-----
    # Load results from JSON files and compare at 15, 30, 60 FPS
    python scripts/analyze_latency_budget.py \\
        --results results/MOSSE-Synthetic.json results/KCF-Synthetic.json \\
        --fps 15 30 60

    # Load from a ResultsBank directory
    python scripts/analyze_latency_budget.py \\
        --bank results/bank \\
        --fps 30

    # Run benchmarks inline (needs no saved results)
    python scripts/analyze_latency_budget.py \\
        --trackers MOSSE KCF CSRT \\
        --dataset synthetic --num-sequences 5 --num-frames 50 \\
        --fps 15 30 60 120

    # Use p95 instead of p99 for compliance check
    python scripts/analyze_latency_budget.py \\
        --bank results/bank --fps 30 --percentile 95

    # Output in machine-readable JSON
    python scripts/analyze_latency_budget.py \\
        --bank results/bank --fps 30 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.analysis.latency_budget import LatencyBudgetAnalyzer


def _load_results_from_files(paths):
    from eovot.benchmark.engine import BenchmarkResult
    results = []
    for p in paths:
        try:
            results.append(BenchmarkResult.load(p))
        except Exception as exc:
            print(f"Warning: could not load {p}: {exc}", file=sys.stderr)
    return results


def _load_results_from_bank(bank_dir, dataset_filter=None):
    from eovot.results.bank import ResultsBank
    bank = ResultsBank(bank_dir)
    return bank.load_all(dataset=dataset_filter)


def _run_inline(tracker_names, dataset, num_sequences, num_frames, seed):
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.registry import build_tracker
    from eovot.datasets.synthetic import SyntheticDataset

    if dataset != "synthetic":
        print(f"Inline mode only supports 'synthetic' dataset; got {dataset!r}",
              file=sys.stderr)
        sys.exit(1)

    ds = SyntheticDataset(num_sequences=num_sequences, num_frames=num_frames, seed=seed)
    engine = BenchmarkEngine(verbose=True)
    results = []
    for name in tracker_names:
        try:
            tracker = build_tracker(name)
        except ValueError as exc:
            print(f"Warning: {exc}", file=sys.stderr)
            continue
        results.append(engine.run(tracker, ds, dataset_name="Synthetic"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="analyze_latency_budget",
        description=(
            "Analyze tracker latency compliance against real-time FPS budgets. "
            "Loads results from JSON files, a ResultsBank, or runs inline."
        ),
    )

    # Input sources (mutually exclusive)
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--results", nargs="+", metavar="JSON_FILE",
        help="Path(s) to benchmark result JSON files.",
    )
    src.add_argument(
        "--bank", metavar="DIR",
        help="Path to a ResultsBank directory.",
    )
    src.add_argument(
        "--trackers", nargs="+", metavar="NAME",
        help="Tracker names to run inline on a synthetic dataset.",
    )

    # FPS targets
    parser.add_argument(
        "--fps", nargs="+", type=float, default=[15.0, 30.0, 60.0],
        metavar="FPS",
        help="Target FPS requirements to evaluate (default: 15 30 60).",
    )

    # Inline-mode options
    parser.add_argument(
        "--dataset", default="synthetic",
        help="Dataset for inline mode (currently only 'synthetic'). Default: synthetic.",
    )
    parser.add_argument(
        "--num-sequences", type=int, default=5, metavar="N",
        help="Number of synthetic sequences for inline mode (default: 5).",
    )
    parser.add_argument(
        "--num-frames", type=int, default=100, metavar="N",
        help="Frames per sequence for inline mode (default: 100).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for synthetic dataset (default: 42).",
    )

    # Bank filter
    parser.add_argument(
        "--dataset-filter", default=None, metavar="NAME",
        help="Filter bank results to this dataset name (substring match).",
    )

    # Analysis options
    parser.add_argument(
        "--percentile", type=int, choices=[95, 99], default=99,
        help="Latency percentile used for compliance check (default: 99).",
    )

    # Output options
    parser.add_argument(
        "--matrix", action="store_true",
        help="Print a compact compliance matrix in addition to the full table.",
    )
    parser.add_argument(
        "--recommend", action="store_true",
        help="Print the highest feasible FPS budget per tracker.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON to stdout (no Markdown table).",
    )

    args = parser.parse_args()

    # Load or run results
    if args.results:
        results = _load_results_from_files(args.results)
    elif args.bank:
        results = _load_results_from_bank(args.bank, dataset_filter=args.dataset_filter)
    elif args.trackers:
        results = _run_inline(
            args.trackers, args.dataset,
            args.num_sequences, args.num_frames, args.seed,
        )
    else:
        parser.error("Provide --results, --bank, or --trackers.")

    if not results:
        print("No results to analyze.", file=sys.stderr)
        sys.exit(1)

    analyzer = LatencyBudgetAnalyzer(compliance_percentile=args.percentile)
    report = analyzer.compare_trackers(results, args.fps)

    if args.json:
        print(json.dumps([e.to_dict() for e in report.entries], indent=2))
        return

    print("\n# Real-Time Latency Budget Compliance\n")
    print(f"Compliance threshold: p{args.percentile} latency ≤ budget\n")
    print(report.to_markdown())

    if args.matrix:
        print("\n## Compliance Matrix\n")
        print(report.to_compliance_matrix())

    if args.recommend:
        rec = analyzer.recommend_fps_budget(results)
        print("\n## Highest Feasible FPS Budget\n")
        print(f"{'Tracker':<20s}  {'Max Compliant FPS':>20s}")
        print("-" * 44)
        for name, fps in sorted(rec.items()):
            fps_str = f"{fps:.0f} FPS" if fps is not None else "not feasible"
            print(f"{name:<20s}  {fps_str:>20s}")


if __name__ == "__main__":
    main()
