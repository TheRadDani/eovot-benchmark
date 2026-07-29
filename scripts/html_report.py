"""CLI: Generate an interactive HTML benchmark report.

Loads one or more saved JSON result files (or runs a synthetic demo) and
produces a self-contained HTML file that can be opened in any browser with
no internet connection required.

Usage
-----
Demo (no external data)::

    python scripts/html_report.py --demo

From saved JSON results::

    python scripts/html_report.py \\
        --results results/MOSSE-Synthetic.json results/KCF-Synthetic.json \\
        --output  results/report.html \\
        --title   "MOSSE vs KCF on Synthetic"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def _demo() -> None:
    """Run benchmark on SyntheticDataset and open the generated report."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.reporting.html_reporter import HTMLReporter
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker

    print("Running synthetic demo (5 sequences × 80 frames) …")
    dataset = SyntheticDataset(num_sequences=5, num_frames=80, seed=42)
    engine = BenchmarkEngine(verbose=True, tdp_watts=10.0)

    results = [
        engine.run(MOSSETracker(), dataset, dataset_name="Synthetic"),
        engine.run(KCFTracker(),   dataset, dataset_name="Synthetic"),
    ]

    reporter = HTMLReporter(output_dir="results/")
    path = reporter.generate(results, name="demo_report", title="EOVOT Demo — MOSSE vs KCF")
    print(f"\nReport written to: {path}")
    print("Open in your browser to explore the interactive charts.")


def _from_json(result_paths: List[str], output: str, title: str) -> None:
    """Reconstruct lightweight BenchmarkResult objects from JSON and generate HTML."""
    from eovot.benchmark.engine import SequenceResult, BenchmarkResult
    from eovot.profiling.profiler import ProfilingResult
    from eovot.reporting.html_reporter import HTMLReporter

    benchmark_results = []

    for path in result_paths:
        with open(path) as fh:
            data = json.load(fh)

        summary = data.get("summary", {})
        tracker_name = summary.get("tracker", Path(path).stem)
        dataset_name = summary.get("dataset", "unknown")

        seq_results = []
        for seq in data.get("sequences", []):
            mean_iou = float(seq.get("mean_iou", 0.0))
            fps = float(seq.get("fps", 1.0))
            lat = float(seq.get("mean_latency_ms", 0.0))
            mem = float(seq.get("peak_memory_mb", 0.0))
            profiling = ProfilingResult(
                tracker_name=tracker_name,
                frame_count=1,
                latency_mean_ms=lat,
                latency_std_ms=0.0,
                latency_p95_ms=lat,
                fps=fps,
                peak_memory_mb=mem,
            )
            seq_results.append(
                SequenceResult(
                    sequence_name=seq.get("sequence_name", "?"),
                    ious=np.array([mean_iou]),
                    profiling=profiling,
                )
            )

        benchmark_results.append(
            BenchmarkResult(
                tracker_name=tracker_name,
                dataset_name=dataset_name,
                sequence_results=seq_results,
            )
        )
        print(f"Loaded '{tracker_name}' — {len(seq_results)} sequences")

    out_path = Path(output)
    reporter = HTMLReporter(output_dir=str(out_path.parent))
    path = reporter.generate(benchmark_results, name=out_path.stem, title=title)
    print(f"\nReport written to: {path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained interactive HTML benchmark report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run a synthetic demo and generate a report.")
    parser.add_argument("--results", nargs="+", metavar="JSON",
                        help="Path(s) to benchmark result JSON files.")
    parser.add_argument("--output", default="results/report.html",
                        help="Output HTML path. Default: results/report.html")
    parser.add_argument("--title", default="EOVOT Benchmark Report",
                        help="Report page title.")
    args = parser.parse_args(argv)

    if args.demo:
        _demo()
        return

    if not args.results:
        parser.print_help()
        sys.exit(1)

    _from_json(args.results, args.output, args.title)


if __name__ == "__main__":
    main()
