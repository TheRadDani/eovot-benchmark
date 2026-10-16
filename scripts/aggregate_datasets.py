#!/usr/bin/env python3
"""Cross-dataset tracker ranking using MultiDatasetAggregator.

Loads benchmark result JSON files (one or more per tracker/dataset
combination) and produces a composite ranking that normalises performance
across datasets using z-score normalization.

Usage
-----
Evaluate trackers and save results first::

    python scripts/run_benchmark.py --tracker MOSSE --dataset got10k \\
        --output results/mosse_got10k.json

    python scripts/run_benchmark.py --tracker KCF --dataset got10k \\
        --output results/kcf_got10k.json

    # Repeat for LaSOT, OTB, etc.

Then aggregate::

    python scripts/aggregate_datasets.py results/*.json

Or point at a directory::

    python scripts/aggregate_datasets.py --dir results/ \\
        --dataset-weights GOT-10k:2.0 LaSOT:2.0 Synthetic:0.5 \\
        --metric-weights mean_iou:3.0 mean_fps:1.0 peak_memory_mb:0.5 \\
        --output ranking.csv

Demo mode (no real data required)::

    python scripts/aggregate_datasets.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.analysis.multi_dataset import AGGREGATION_METRICS, MultiDatasetAggregator
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.csrt import CSRTTracker


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """Run the aggregator on synthetic data to demonstrate the workflow."""
    print("=== MultiDatasetAggregator demo (synthetic data) ===\n")

    engine = BenchmarkEngine(verbose=False)

    # Simulate two 'datasets' with different random seeds
    ds_a = SyntheticDataset(num_sequences=3, num_frames=30, frame_size=(128, 128), seed=1)
    ds_b = SyntheticDataset(num_sequences=3, num_frames=30, frame_size=(96, 96),  seed=2)

    trackers = [
        ("MOSSE", MOSSETracker()),
        ("KCF",   KCFTracker()),
        ("CSRT",  CSRTTracker()),
    ]

    results_a = [engine.run(t, ds_a, dataset_name="Synthetic-A") for _, t in trackers]
    results_b = [engine.run(t, ds_b, dataset_name="Synthetic-B") for _, t in trackers]

    agg = MultiDatasetAggregator(
        dataset_weights={"Synthetic-A": 1.0, "Synthetic-B": 1.0},
        metric_weights={"mean_iou": 2.0, "mean_fps": 1.0, "peak_memory_mb": 0.5},
    )
    agg.add_results(results_a)
    agg.add_results(results_b)

    print(agg.to_markdown())
    print()
    d = agg.summary_dict()
    print(f"Trackers evaluated: {d['num_trackers']}  across {d['num_datasets']} dataset(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_weights(pairs: list[str]) -> dict[str, float]:
    """Parse 'KEY:VALUE' strings into a dict."""
    out: dict[str, float] = {}
    for p in (pairs or []):
        if ":" not in p:
            print(f"Warning: skipping malformed weight spec '{p}' (expected KEY:VALUE)")
            continue
        k, v = p.split(":", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            print(f"Warning: could not parse weight value in '{p}'")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate EOVOT benchmark results across multiple datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files", nargs="*", type=Path, metavar="JSON",
        help="Benchmark result JSON files (output of run_benchmark.py).",
    )
    parser.add_argument(
        "--dir", type=Path, metavar="DIR",
        help="Load all *.json files from this directory.",
    )
    parser.add_argument(
        "--dataset-weights", nargs="*", metavar="DS:W",
        help="Dataset weights as DATASET_NAME:WEIGHT (e.g. GOT-10k:2.0).",
    )
    parser.add_argument(
        "--metric-weights", nargs="*", metavar="METRIC:W",
        help="Metric weights as METRIC_KEY:WEIGHT (e.g. mean_iou:3.0).",
    )
    parser.add_argument(
        "--min-datasets", type=int, default=1, metavar="N",
        help="Minimum datasets a tracker must appear on to be ranked (default: 1).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, metavar="PATH",
        help="Write ranking to CSV (if .csv) or JSON (if .json).",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run a built-in demo on synthetic data (requires no files).",
    )

    args = parser.parse_args()

    if args.demo:
        _run_demo()
        return 0

    # Collect JSON files
    files: list[Path] = list(args.files or [])
    if args.dir:
        files.extend(sorted(args.dir.glob("*.json")))

    if not files:
        print("Error: no JSON files specified. Pass result JSON paths or use --demo.")
        return 1

    # Build aggregator
    agg = MultiDatasetAggregator(
        dataset_weights=_parse_weights(args.dataset_weights),
        metric_weights=_parse_weights(args.metric_weights),
        min_datasets=args.min_datasets,
    )

    loaded = 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            agg.add_result(data)
            loaded += 1
        except Exception as exc:
            print(f"Warning: could not load {path}: {exc}")

    if loaded == 0:
        print("Error: no valid result files loaded.")
        return 1

    print(f"Loaded {loaded} result file(s).")
    print()
    print(agg.to_markdown())

    if args.output:
        if args.output.suffix == ".json":
            out = agg.to_json(args.output)
        else:
            out = agg.to_csv(args.output)
        print(f"\nRanking written to: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
