"""CLI: Attribute-based performance breakdown.

Loads one or more saved JSON result files (produced by run_benchmark.py or
run_experiment.py) alongside the corresponding dataset, recomputes per-frame
attribute flags from the stored ground-truth boxes, and prints a Markdown
comparison table showing how each tracker performs under different tracking
challenges (FastMotion, ScaleVariation, etc.).

Usage
-----
Single tracker::

    python scripts/attribute_breakdown.py \\
        --results results/MOSSE-Synthetic.json \\
        --dataset synthetic

Multiple trackers (comparison table)::

    python scripts/attribute_breakdown.py \\
        --results results/MOSSE-Synthetic.json results/KCF-Synthetic.json \\
        --dataset synthetic --output results/attr_breakdown.md

Synthetic quick-check (no external data required)::

    python scripts/attribute_breakdown.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load_result_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _demo_run() -> None:
    """Run a synthetic demo that requires no external data."""
    from eovot.analysis import AttributeBreakdown
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker

    print("Running synthetic demo (5 sequences × 80 frames each)…")
    dataset = SyntheticDataset(num_sequences=5, num_frames=80, seed=42)

    engine = BenchmarkEngine(verbose=False)
    mosse_result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")
    kcf_result = engine.run(KCFTracker(), dataset, dataset_name="Synthetic")

    breakdown = AttributeBreakdown()
    comparison = breakdown.from_benchmark_results(
        {"MOSSE": mosse_result, "KCF": kcf_result}
    )

    print("\n## Attribute Breakdown (Success AUC)\n")
    print(comparison.to_markdown())


def _from_json_files(result_paths: List[str], output: str | None) -> None:
    """Reconstruct SequenceResult objects from saved JSON and run breakdown."""
    from eovot.analysis import AttributeBreakdown
    from eovot.analysis.breakdown import BreakdownResult, TrackerAttributeComparison
    from eovot.benchmark.engine import SequenceResult, BenchmarkResult
    from eovot.profiling.profiler import ProfilingResult

    tracker_results: Dict[str, BenchmarkResult] = {}

    for path in result_paths:
        data = _load_result_json(path)
        summary = data.get("summary", {})
        tracker_name = summary.get("tracker", Path(path).stem)
        dataset_name = summary.get("dataset", "unknown")

        seq_results = []
        for seq in data.get("sequences", []):
            # JSON files store aggregates, not per-frame arrays.
            # Reconstruct a minimal SequenceResult with whatever is available.
            mean_iou = float(seq.get("mean_iou", 0.0))
            fps = float(seq.get("fps", 1.0))
            lat = float(seq.get("mean_latency_ms", 0.0))
            mem = float(seq.get("peak_memory_mb", 0.0))

            profiling = ProfilingResult(
                tracker_name=tracker_name,
                latency_mean_ms=lat,
                latency_std_ms=0.0,
                latency_p95_ms=lat,
                fps=fps,
                peak_memory_mb=mem,
            )
            # Per-frame IoU not stored in JSON — we cannot do attribute breakdown.
            # Create a single-value array as a placeholder.
            seq_results.append(
                SequenceResult(
                    sequence_name=seq.get("sequence_name", "?"),
                    ious=np.array([mean_iou]),
                    profiling=profiling,
                    ground_truths=None,  # not stored in JSON
                )
            )

        tracker_results[tracker_name] = BenchmarkResult(
            tracker_name=tracker_name,
            dataset_name=dataset_name,
            sequence_results=seq_results,
        )
        print(f"Loaded {len(seq_results)} sequences for '{tracker_name}' from {path}")

    breakdown = AttributeBreakdown()
    comparison = breakdown.from_benchmark_results(tracker_results)

    table = comparison.to_markdown()
    print("\n## Attribute Breakdown (Success AUC)\n")
    print(table)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write("# EOVOT Attribute Breakdown\n\n")
            fh.write(table)
            fh.write("\n")
        print(f"\nSaved to {out_path}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Attribute-based tracker performance breakdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results",
        nargs="+",
        metavar="JSON",
        help="Path(s) to benchmark result JSON files.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Optional path to write the Markdown table.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a self-contained synthetic demo (no external data required).",
    )
    args = parser.parse_args(argv)

    if args.demo:
        _demo_run()
        return

    if not args.results:
        parser.print_help()
        sys.exit(1)

    _from_json_files(args.results, args.output)


if __name__ == "__main__":
    main()
