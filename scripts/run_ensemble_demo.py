#!/usr/bin/env python3
"""Ensemble tracker demo — compare individual trackers vs. consensus fusion.

Runs MOSSE, KCF, CamShift individually and as an ensemble on synthetic
sequences.  Prints a comparison table showing how the ensemble performs
relative to each member on accuracy and FPS.

Usage::

    # Quick demo (20 sequences, 80 frames each)
    python scripts/run_ensemble_demo.py

    # Custom sequence count and length
    python scripts/run_ensemble_demo.py --num-sequences 50 --num-frames 150

    # With a specific motion pattern
    python scripts/run_ensemble_demo.py --motion circular --num-frames 200

    # Verbose per-sequence output
    python scripts/run_ensemble_demo.py --verbose

This script requires only the synthetic dataset — no external data download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.camshift import CamShiftTracker
from eovot.trackers.ensemble import TrackerEnsemble, build_default_ensemble
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-sequences", type=int, default=20,
                   help="Number of synthetic sequences to evaluate on (default: 20)")
    p.add_argument("--num-frames", type=int, default=80,
                   help="Frames per sequence (default: 80)")
    p.add_argument("--motion", choices=["linear", "circular", "random"], default="linear",
                   help="Synthetic target motion pattern (default: linear)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-sequence progress")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(results: list[BenchmarkResult]) -> str:
    """Format results as a Markdown comparison table."""
    header = (
        "| Tracker           | mIoU  | Success AUC | FPS    | Mem (MB) |\n"
        "|-------------------|------:|------------:|-------:|---------:|"
    )
    rows = [header]
    for r in results:
        sauc = r.mean_success_auc
        sauc_str = f"{sauc:.4f}" if sauc is not None else "  n/a  "
        rows.append(
            f"| {r.tracker_name:<17s} "
            f"| {r.mean_iou:.4f} "
            f"| {sauc_str:>11s} "
            f"| {r.mean_fps:>6.1f} "
            f"| {r.peak_memory_mb:>8.1f} |"
        )
    return "\n".join(rows)


def _improvement_note(individual: list[BenchmarkResult], ensemble: BenchmarkResult) -> str:
    """Print a quick relative-improvement summary."""
    best_iou = max(r.mean_iou for r in individual)
    ens_iou = ensemble.mean_iou
    delta = ens_iou - best_iou
    sign = "+" if delta >= 0 else ""
    return (
        f"  Best individual mIoU : {best_iou:.4f}\n"
        f"  Ensemble mIoU        : {ens_iou:.4f}  ({sign}{delta:.4f} relative to best member)\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print("EOVOT — TrackerEnsemble Demo")
    print("=" * 60)
    print(f"  Sequences : {args.num_sequences}")
    print(f"  Frames    : {args.num_frames}")
    print(f"  Motion    : {args.motion}")
    print(f"  Seed      : {args.seed}")
    print()

    dataset = SyntheticDataset(
        num_sequences=args.num_sequences,
        num_frames=args.num_frames,
        motion=args.motion,  # type: ignore[arg-type]
        seed=args.seed,
    )
    engine = BenchmarkEngine(verbose=args.verbose)
    dataset_name = f"Synthetic-{args.motion.capitalize()}"

    # ---- Individual trackers ----------------------------------------
    individuals = [MOSSETracker(), KCFTracker(), CamShiftTracker()]
    individual_results: list[BenchmarkResult] = []

    print("Running individual trackers …")
    for tracker in individuals:
        r = engine.run(tracker, dataset, dataset_name=dataset_name)
        individual_results.append(r)

    # ---- Ensemble -------------------------------------------------------
    print("\nRunning TrackerEnsemble (MOSSE + KCF + CamShift) …")
    ensemble = build_default_ensemble()
    ensemble_result = engine.run(ensemble, dataset, dataset_name=dataset_name)

    # ---- Two-tracker variants -------------------------------------------
    print("\nRunning 2-member ensembles for comparison …")
    ens_mk = TrackerEnsemble([MOSSETracker(), KCFTracker()], name="Ensemble-MK")
    ens_mc = TrackerEnsemble([MOSSETracker(), CamShiftTracker()], name="Ensemble-MC")
    ens_mk_result = engine.run(ens_mk, dataset, dataset_name=dataset_name)
    ens_mc_result = engine.run(ens_mc, dataset, dataset_name=dataset_name)

    # ---- Report ---------------------------------------------------------
    all_results = individual_results + [ens_mk_result, ens_mc_result, ensemble_result]
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(_fmt(all_results))
    print()
    print("ENSEMBLE SUMMARY")
    print(_improvement_note(individual_results, ensemble_result))

    print("=" * 60)
    print("Member trackers in the default ensemble:")
    for name in ensemble.member_names():
        print(f"  • {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
