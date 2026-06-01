#!/usr/bin/env python
"""Generate a self-contained HTML benchmark dashboard from saved JSON results.

Reads one or more JSON files produced by BenchmarkReporter.save_json() and
emits a single self-contained .html file with leaderboard, success curves,
and efficiency scatter plot.

Usage
-----
Single tracker result::

    python scripts/generate_html_report.py results/MOSSE.json

Multiple tracker results (comparison dashboard)::

    python scripts/generate_html_report.py results/MOSSE.json results/KCF.json \\
        --output results/dashboard.html

Live experiment (run + report in one step)::

    python scripts/generate_html_report.py \\
        --live --tracker MOSSE KCF --dataset-root /data/OTB100 \\
        --output results/live_dashboard.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.metrics.accuracy import AccuracyMetrics
from eovot.profiling.profiler import ProfilingResult
from eovot.reporting.html_reporter import HTMLReporter


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate an HTML benchmark dashboard from JSON result files."
    )
    p.add_argument(
        "json_files",
        nargs="*",
        metavar="RESULT.json",
        help="JSON files saved by BenchmarkReporter.save_json().",
    )
    p.add_argument(
        "--output", "-o",
        default="results/dashboard.html",
        help="Output HTML file path. Default: results/dashboard.html",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Run a quick live benchmark instead of loading JSON files.",
    )
    p.add_argument(
        "--tracker",
        nargs="+",
        default=["MOSSE", "KCF"],
        help="Tracker names for --live mode. Default: MOSSE KCF",
    )
    p.add_argument(
        "--dataset-root",
        default=None,
        help="Dataset root directory for --live mode. Uses SyntheticDataset if omitted.",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=5,
        help="Maximum sequences for --live mode. Default: 5",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for SyntheticDataset in --live mode. Default: 42",
    )
    return p.parse_args()


def _load_result_from_json(path: Path) -> BenchmarkResult:
    """Reconstruct a BenchmarkResult from a saved JSON file."""
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    tracker_name = summary.get("tracker") or summary.get("tracker_name", path.stem)
    dataset_name = summary.get("dataset") or summary.get("dataset_name", "unknown")

    sequence_results = []
    for seq in data.get("sequences", []):
        import numpy as np
        # Reconstruct a minimal ProfilingResult-like object
        profiling = _DictProxy({
            "fps": float(seq.get("fps", 0.0)),
            "latency_mean_ms": float(seq.get("mean_latency_ms", 0.0)),
            "peak_memory_mb": float(seq.get("peak_memory_mb", 0.0)),
        })
        miou = float(seq.get("mean_iou", 0.0))
        ious = np.array([miou])  # scalar summary; per-frame data not in JSON
        sauc = seq.get("success_auc")
        pauc = seq.get("precision_auc")
        acc = None
        if sauc is not None and pauc is not None:
            acc = _DictProxy({"success_auc": float(sauc), "precision_auc": float(pauc)})
        sr = SequenceResult(
            sequence_name=seq.get("sequence_name", "?"),
            ious=ious,
            profiling=profiling,
            accuracy_metrics=acc,
        )
        sequence_results.append(sr)

    result = BenchmarkResult(
        tracker_name=tracker_name,
        dataset_name=dataset_name,
        sequence_results=sequence_results,
    )
    return result


class _DictProxy:
    """Lightweight attribute-access wrapper around a plain dict."""

    def __init__(self, d: dict) -> None:
        self.__dict__.update(d)


def _run_live(args: argparse.Namespace) -> list:
    """Run a quick benchmark and return a list of BenchmarkResult objects."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    _TRACKER_MAP = {
        "MOSSE": lambda: __import__(
            "eovot.trackers.mosse", fromlist=["MOSSETracker"]
        ).MOSSETracker(),
        "KCF": lambda: __import__(
            "eovot.trackers.kcf", fromlist=["KCFTracker"]
        ).KCFTracker(),
        "CSRT": lambda: __import__(
            "eovot.trackers.csrt", fromlist=["CSRTTracker"]
        ).CSRTTracker(),
        "MedianFlow": lambda: __import__(
            "eovot.trackers.median_flow", fromlist=["MedianFlowTracker"]
        ).MedianFlowTracker(),
    }

    dataset = SyntheticDataset(
        num_sequences=args.max_sequences,
        seed=args.seed,
    )
    engine = BenchmarkEngine(verbose=True)
    results = []
    for name in args.tracker:
        if name not in _TRACKER_MAP:
            print(f"[warn] Unknown tracker '{name}', skipping.", file=sys.stderr)
            continue
        tracker = _TRACKER_MAP[name]()
        results.append(engine.run(tracker, dataset, dataset_name="Synthetic"))
    return results


def main() -> None:
    args = _parse_args()
    output = Path(args.output)

    if args.live:
        results = _run_live(args)
    elif args.json_files:
        results = [_load_result_from_json(Path(p)) for p in args.json_files]
    else:
        print(
            "Error: provide JSON file paths or use --live to run a quick benchmark.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not results:
        print("Error: no results to report.", file=sys.stderr)
        sys.exit(1)

    reporter = HTMLReporter(output_dir=str(output.parent))
    path = reporter.save_html(results, name=output.stem)
    print(f"Dashboard written to: {path}")


if __name__ == "__main__":
    main()
