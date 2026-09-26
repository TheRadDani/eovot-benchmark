#!/usr/bin/env python3
"""Comprehensive multi-metric tracker comparison CLI for EOVOT.

Runs every registered classical tracker on a synthetic (or real) dataset,
computes ALL EOVOT metric dimensions — accuracy, robustness, temporal
consistency, and edge efficiency — and writes a unified Markdown + CSV + JSON
report without requiring any external dataset download.

This script is designed to be:

* **Self-contained**: works out-of-the-box with the built-in synthetic dataset.
* **Reproducible**: fixed RNG seed and snapshot-style JSON output.
* **Publishable**: outputs a ready-to-paste Markdown leaderboard table.

Usage
-----
Quick comparison (synthetic data, all classical trackers, 8 sequences × 150 frames)::

    python scripts/compare_all_metrics.py

Custom synthetic dataset::

    python scripts/compare_all_metrics.py \\
        --sequences 20 --frames 300 --motion circular

Select specific trackers::

    python scripts/compare_all_metrics.py \\
        --trackers MOSSE KCF CSRT

Real dataset (OTB-style)::

    python scripts/compare_all_metrics.py \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100 \\
        --trackers MOSSE KCF CSRT

With CPU energy profiling (Raspberry Pi 4 TDP)::

    python scripts/compare_all_metrics.py \\
        --sequences 10 --frames 200 \\
        --tdp-watts 6.0

Save output to a custom directory::

    python scripts/compare_all_metrics.py \\
        --output-dir results/full_comparison
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.base import OTBDataset
from eovot.datasets.synthetic import SyntheticDataset
from eovot.reporting.full_report import FullBenchmarkReport
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker

# Classical trackers that work without any external model files
CLASSICAL_TRACKERS = [
    "MOSSE",
    "KCF",
    "CSRT",
    "MIL",
    "CamShift",
    "LKOpticalFlow",
    "MedianFlow",
]


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_all_metrics",
        description=(
            "Run all EOVOT classical trackers and produce a full multi-metric "
            "comparison report (accuracy + robustness + temporal + efficiency)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset selection
    ds_group = p.add_argument_group("Dataset")
    ds_group.add_argument(
        "--dataset-root", default=None, metavar="PATH",
        help="Path to an OTB-style dataset root. When omitted, a synthetic "
             "dataset is generated automatically.",
    )
    ds_group.add_argument(
        "--dataset-name", default="Synthetic", metavar="NAME",
        help="Human-readable dataset label used in reports (default: Synthetic).",
    )
    ds_group.add_argument(
        "--sequences", type=int, default=8, metavar="N",
        help="Number of synthetic sequences (default: 8).",
    )
    ds_group.add_argument(
        "--frames", type=int, default=150, metavar="N",
        help="Frames per synthetic sequence (default: 150).",
    )
    ds_group.add_argument(
        "--motion", default="linear",
        choices=["linear", "circular", "random"],
        help="Motion pattern for synthetic sequences (default: linear).",
    )
    ds_group.add_argument(
        "--max-sequences", type=int, default=None, metavar="N",
        help="Limit evaluation to the first N sequences of a real dataset.",
    )

    # Tracker selection
    tr_group = p.add_argument_group("Trackers")
    tr_group.add_argument(
        "--trackers", nargs="+", default=None, metavar="NAME",
        help=(
            "Tracker names to evaluate (default: all classical trackers). "
            f"Available: {', '.join(CLASSICAL_TRACKERS)}."
        ),
    )

    # Profiling
    prof_group = p.add_argument_group("Profiling")
    prof_group.add_argument(
        "--tdp-watts", type=float, default=None, metavar="W",
        help=(
            "Enable CPU energy profiling using this TDP (Watts). "
            "Example: 6.0 for Raspberry Pi 4, 10.0 for Jetson Nano."
        ),
    )
    prof_group.add_argument(
        "--memory-budget", type=float, default=512.0, metavar="MB",
        help="Memory budget (MB) for Edge Efficiency Score (default: 512).",
    )

    # Output
    out_group = p.add_argument_group("Output")
    out_group.add_argument(
        "--output-dir", default="results/full_comparison", metavar="DIR",
        help="Directory to write JSON, CSV, and Markdown files (default: results/full_comparison).",
    )
    out_group.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-sequence progress output.",
    )
    out_group.add_argument(
        "--no-save", action="store_true",
        help="Print report to stdout only; do not write any files.",
    )

    return p


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _resolve_trackers(names: Optional[List[str]]) -> List[str]:
    """Resolve tracker names, skipping any not in the registry."""
    if names is None:
        names = CLASSICAL_TRACKERS
    available = set(TRACKER_REGISTRY.keys())
    resolved: List[str] = []
    for name in names:
        if name in available:
            resolved.append(name)
        else:
            print(f"  [warn] Tracker '{name}' not found in registry — skipping.")
    return resolved


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ---- Dataset ----
    if args.dataset_root is not None:
        print(f"Loading dataset from: {args.dataset_root}")
        dataset = OTBDataset(args.dataset_root)
        if args.max_sequences is not None:
            # Wrap with a slice view
            class _Sliced:
                def __init__(self, ds, n):
                    self._ds = ds
                    self._n = min(n, len(ds))
                def __len__(self): return self._n
                def __getitem__(self, idx): return self._ds[idx]
            dataset = _Sliced(dataset, args.max_sequences)  # type: ignore[assignment]
    else:
        dataset = SyntheticDataset(
            num_sequences=args.sequences,
            num_frames=args.frames,
            motion=args.motion,
        )

    dataset_name = args.dataset_name

    # ---- Trackers ----
    tracker_names = _resolve_trackers(args.trackers)
    if not tracker_names:
        print("No valid trackers selected. Exiting.")
        sys.exit(1)

    print(
        f"\nEOVOT Full Multi-Metric Comparison\n"
        f"{'─' * 50}\n"
        f"Dataset  : {dataset_name} ({len(dataset)} sequences)\n"
        f"Trackers : {', '.join(tracker_names)}\n"
        f"Energy   : {'enabled (TDP=' + str(args.tdp_watts) + ' W)' if args.tdp_watts else 'disabled'}\n"
    )

    engine = BenchmarkEngine(
        verbose=not args.quiet,
        tdp_watts=args.tdp_watts,
    )

    # ---- Run benchmarks ----
    results = []
    t_start = time.perf_counter()
    for name in tracker_names:
        tracker = build_tracker(name)
        result = engine.run(tracker, dataset, dataset_name=dataset_name)
        results.append(result)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'─' * 50}")
    print(f"Total evaluation time: {elapsed:.1f}s\n")

    # ---- Full multi-metric report ----
    report = FullBenchmarkReport(results, memory_budget_mb=args.memory_budget)
    md = report.to_markdown(
        title=f"EOVOT Full Benchmark — {dataset_name}"
    )

    print(md)

    if not args.no_save:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        md_path = out / "full_comparison.md"
        csv_path = report.to_csv(out / "full_comparison.csv")
        json_path = report.to_json(out / "full_comparison.json")
        md_path.write_text(md, encoding="utf-8")

        # Also save individual JSON results for reproducibility
        ind = out / "individual"
        ind.mkdir(exist_ok=True)
        for r in results:
            r.save(ind / f"{r.tracker_name}.json")

        print(f"\nFiles written to: {out.resolve()}")
        print(f"  {md_path.name}")
        print(f"  {csv_path.name}")
        print(f"  {json_path.name}")
        print(f"  individual/{', '.join(r.tracker_name + '.json' for r in results)}")


if __name__ == "__main__":
    main()
