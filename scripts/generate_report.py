"""CLI script to generate an HTML benchmark report from saved JSON results.

Usage
-----
Single-tracker report from a JSON file::

    python scripts/generate_report.py results/MOSSE-Synthetic.json

Multi-tracker report from multiple JSON files::

    python scripts/generate_report.py \\
        results/MOSSE-Synthetic.json \\
        results/KCF-Synthetic.json \\
        results/CSRT-Synthetic.json \\
        --output results/comparison_report.html \\
        --title "Classical Tracker Comparison"

Live run (no pre-existing JSON) with synthetic data::

    python scripts/generate_report.py \\
        --live \\
        --trackers MOSSE KCF CSRT \\
        --sequences 5 --frames 80 \\
        --output results/live_report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkResult
from eovot.reporting.html_report import HTMLReportGenerator


def _load_results(paths: list[str]) -> list[BenchmarkResult]:
    results = []
    for p in paths:
        try:
            r = BenchmarkResult.load(p)
            results.append(r)
            print(f"  Loaded: {p}  ({r.tracker_name} on {r.dataset_name})")
        except Exception as exc:
            print(f"  WARNING: could not load {p}: {exc}")
    return results


def _run_live(
    tracker_names: list[str],
    n_sequences: int,
    n_frames: int,
) -> list[BenchmarkResult]:
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.registry import build_tracker

    dataset = SyntheticDataset(
        num_sequences=n_sequences,
        num_frames=n_frames,
        motion="linear",
    )
    engine = BenchmarkEngine(verbose=True)
    results = []
    for name in tracker_names:
        try:
            tracker = build_tracker(name)
            result = engine.run(tracker, dataset, dataset_name="Synthetic")
            results.append(result)
        except Exception as exc:
            print(f"  WARNING: tracker '{name}' failed: {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML benchmark report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "json_files",
        nargs="*",
        metavar="FILE.json",
        help="Pre-computed benchmark result JSON files to include.",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/benchmark_report.html",
        help="Output HTML file path (default: results/benchmark_report.html).",
    )
    parser.add_argument(
        "--title", "-t",
        default="EOVOT Benchmark Report",
        help="Report title shown in the HTML page heading.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run trackers live on a synthetic dataset instead of loading JSON files.",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["MOSSE", "KCF"],
        metavar="NAME",
        help="Tracker names for --live mode (default: MOSSE KCF).",
    )
    parser.add_argument(
        "--sequences",
        type=int,
        default=5,
        metavar="N",
        help="Sequences per dataset for --live mode (default: 5).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=80,
        metavar="N",
        help="Frames per sequence for --live mode (default: 80).",
    )

    args = parser.parse_args()

    print(f"\nEOVOT HTML Report Generator")
    print("=" * 40)

    results: list[BenchmarkResult] = []

    if args.live:
        print(f"Running live benchmark: {args.trackers}")
        results = _run_live(args.trackers, args.sequences, args.frames)
    elif args.json_files:
        print("Loading result files:")
        results = _load_results(args.json_files)
    else:
        parser.error("Provide JSON files or use --live to run a fresh benchmark.")

    if not results:
        print("ERROR: no results to report.")
        sys.exit(1)

    out_path = Path(args.output)
    gen = HTMLReportGenerator(output_dir=str(out_path.parent))
    report_path = gen.generate(results, out_path.name, title=args.title)

    print(f"\nReport written: {report_path}")
    print(f"Open in browser: file://{report_path.resolve()}")


if __name__ == "__main__":
    main()
