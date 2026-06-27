#!/usr/bin/env python3
"""CLI tool: generate an EOVOT HTML report from benchmark result JSON files.

Reads one or more ``*.json`` result files produced by
:class:`~eovot.reporting.reporter.BenchmarkReporter` (or
:meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`) and renders a
self-contained HTML report with interactive charts.

Usage examples::

    # Single tracker result
    python scripts/generate_html_report.py results/MOSSE-OTB100.json

    # Compare multiple trackers
    python scripts/generate_html_report.py results/*.json --name comparison

    # Custom output directory and title
    python scripts/generate_html_report.py results/KCF.json \\
        --output-dir reports/ \\
        --title "KCF on OTB100" \\
        --name kcf_otb100

    # Live benchmark then report
    python scripts/generate_html_report.py \\
        --run-synthetic --trackers MOSSE KCF --name synthetic_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running the script from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_result_from_json(path: Path):
    """Reconstruct a lightweight BenchmarkResult-like object from a JSON file."""
    from eovot.benchmark.engine import BenchmarkResult, SequenceResult
    from eovot.profiling.profiler import ProfilingResult

    with open(path) as fh:
        data = json.load(fh)

    summary = data.get("summary", {})
    tracker_name = summary.get("tracker") or summary.get("tracker_name", path.stem)
    dataset_name = summary.get("dataset") or summary.get("dataset_name", "unknown")

    result = BenchmarkResult(tracker_name=tracker_name, dataset_name=dataset_name)

    import numpy as np

    for seq in data.get("sequences", []):
        fps = float(seq.get("fps", 0.0) or 0.0)
        lat = float(seq.get("mean_latency_ms", 0.0) or 0.0)
        mem = float(seq.get("peak_memory_mb", 0.0) or 0.0)
        mean_iou = float(seq.get("mean_iou", 0.0) or 0.0)
        nframes = max(int(fps * lat / 1000.0) if lat > 0 else 1, 1)

        # Reconstruct synthetic IoU array from stored mean
        ious = np.full(nframes, mean_iou, dtype=np.float64)

        profiling = ProfilingResult(
            tracker_name=tracker_name,
            frame_count=nframes,
            fps=fps,
            latency_mean_ms=lat,
            latency_std_ms=0.0,
            latency_p95_ms=lat,
            peak_memory_mb=mem,
        )
        seq_result = SequenceResult(
            sequence_name=seq.get("sequence_name", "?"),
            ious=ious,
            profiling=profiling,
        )
        result.sequence_results.append(seq_result)

    return result


def _run_synthetic_benchmark(tracker_names: list[str]):
    """Run a quick synthetic benchmark and return BenchmarkResult objects."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5, num_frames=50, motion="linear")
    engine = BenchmarkEngine(verbose=True)
    results = []

    tracker_map = _build_tracker_map()
    for name in tracker_names:
        cls = tracker_map.get(name.upper())
        if cls is None:
            print(f"[warn] Unknown tracker '{name}', skipping.", file=sys.stderr)
            continue
        results.append(engine.run(cls(), dataset, dataset_name="Synthetic"))

    return results


def _build_tracker_map() -> dict:
    mapping = {}
    try:
        from eovot.trackers.mosse import MOSSETracker
        mapping["MOSSE"] = MOSSETracker
    except ImportError:
        pass
    try:
        from eovot.trackers.kcf import KCFTracker
        mapping["KCF"] = KCFTracker
    except ImportError:
        pass
    try:
        from eovot.trackers.csrt import CSRTTracker
        mapping["CSRT"] = CSRTTracker
    except ImportError:
        pass
    try:
        from eovot.trackers.median_flow import MedianFlowTracker
        mapping["MEDIANFLOW"] = MedianFlowTracker
    except ImportError:
        pass
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an EOVOT HTML benchmark report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "json_files",
        nargs="*",
        metavar="RESULT_JSON",
        help="Benchmark result JSON files to include in the report.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="results/",
        metavar="DIR",
        help="Directory to write the HTML file (default: results/).",
    )
    parser.add_argument(
        "--name", "-n",
        default="report",
        metavar="NAME",
        help="Base filename for the HTML output (default: report).",
    )
    parser.add_argument(
        "--title", "-t",
        default="EOVOT Benchmark Report",
        metavar="TITLE",
        help="Page title shown in the report header.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=50,
        metavar="N",
        help="Maximum rows in the per-sequence table (default: 50).",
    )
    parser.add_argument(
        "--run-synthetic",
        action="store_true",
        help="Run a quick synthetic benchmark instead of loading JSON files.",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["MOSSE", "KCF"],
        metavar="TRACKER",
        help="Trackers to use with --run-synthetic (default: MOSSE KCF).",
    )

    args = parser.parse_args()

    from eovot.reporting.html_reporter import HTMLReporter

    reporter = HTMLReporter(
        output_dir=args.output_dir,
        title=args.title,
        max_sequences_in_table=args.max_sequences,
    )

    if args.run_synthetic:
        results = _run_synthetic_benchmark(args.trackers)
    elif args.json_files:
        results = [_load_result_from_json(Path(p)) for p in args.json_files]
    else:
        parser.print_help()
        sys.exit(0)

    if not results:
        print("[error] No results to report.", file=sys.stderr)
        sys.exit(1)

    path = reporter.save(results, name=args.name)
    print(f"\nReport written to: {path}")
    print("Open in any web browser to view interactive charts.\n")


if __name__ == "__main__":
    main()
