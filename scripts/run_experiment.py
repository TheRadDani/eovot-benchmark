#!/usr/bin/env python3
"""YAML-driven experiment runner for EOVOT.

Loads a YAML experiment config, runs every listed tracker on the specified
dataset, writes per-tracker JSON/CSV results, a ranked Markdown leaderboard,
a metadata.json reproducibility snapshot, and an Edge Efficiency Score (EES)
ranking to the output directory.

Usage::

    # Validate config and print the experiment plan without running
    python scripts/run_experiment.py \\
        --config configs/experiments/multi_tracker.yaml \\
        --dry-run

    # Full run (MOSSE + KCF + MIL + MedianFlow on OTB100)
    python scripts/run_experiment.py \\
        --config configs/experiments/multi_tracker.yaml \\
        --output-dir results/experiments

    # Resume an interrupted run (skips trackers that already have results)
    python scripts/run_experiment.py \\
        --config configs/experiments/multi_tracker.yaml \\
        --resume

    # Project results onto six built-in edge device profiles
    python scripts/run_experiment.py \\
        --config configs/experiments/synthetic_demo.yaml \\
        --simulate-devices

    # Override the energy-profiling TDP value from the command line
    python scripts/run_experiment.py \\
        --config configs/experiments/multi_tracker.yaml \\
        --tdp-watts 6.0

    # List available trackers and exit
    python scripts/run_experiment.py --list-trackers

    # List built-in edge device profiles and exit
    python scripts/run_experiment.py --list-devices
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # pyyaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.experiment.runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Informational helpers (no experiment needed)
# ---------------------------------------------------------------------------

def _list_trackers() -> None:
    from eovot.trackers.registry import available_trackers
    names = available_trackers()
    print(f"\nAvailable trackers ({len(names)}):")
    for name in names:
        print(f"  • {name}")
    print()


def _list_devices() -> None:
    from eovot.profiling.device_sim import KNOWN_DEVICES
    print(f"\nBuilt-in edge device profiles ({len(KNOWN_DEVICES)}):")
    col = max(len(k) for k in KNOWN_DEVICES) + 2
    print(f"  {'Key':<{col}} {'Display name':<40} TDP")
    print(f"  {'─'*col} {'─'*40} ───")
    for key in sorted(KNOWN_DEVICES):
        p = KNOWN_DEVICES[key]
        print(f"  {key:<{col}} {p.display_name:<40} {p.tdp_watts} W")
    print()


# ---------------------------------------------------------------------------
# Dry-run plan printer
# ---------------------------------------------------------------------------

def _print_plan(config: dict, output_dir: str) -> None:
    """Print a human-readable experiment plan without running anything."""
    exp = config.get("experiment", {})
    ds = config.get("dataset", {})
    trackers = config.get("trackers", [])

    exp_name = exp.get("name", "unnamed-experiment")
    exp_dir = Path(output_dir) / exp_name

    print("\nEXPERIMENT PLAN  (dry-run — nothing will be executed)")
    print("=" * 60)
    print(f"  Name        : {exp_name}")
    print(f"  Output dir  : {exp_dir}")
    print(f"  Seed        : {exp.get('seed', 'none')}")
    tdp = exp.get("tdp_watts")
    print(f"  Energy      : {'TDP = ' + str(tdp) + ' W' if tdp else 'disabled'}")
    print(f"\n  Dataset")
    print(f"    Loader    : {ds.get('loader', 'OTBDataset')}")
    root = ds.get("root")
    if root:
        print(f"    Root      : {root}")
    if ds.get("split"):
        print(f"    Split     : {ds.get('split')}")
    ms = ds.get("max_sequences")
    print(f"    Max seqs  : {ms if ms is not None else 'all'}")
    print(f"\n  Trackers ({len(trackers)})")
    for t in trackers:
        params = t.get("params") or {}
        param_str = f"  {params}" if params else ""
        print(f"    • {t['name']}{param_str}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Post-run analysis
# ---------------------------------------------------------------------------

def _show_ees_ranking(
    results: List[Dict[str, Any]],
    memory_budget_mb: float = 512.0,
) -> None:
    """Print an Edge Efficiency Score ranking table for all completed trackers."""
    rows = []
    for r in results:
        s = r.get("summary", {})
        mean_iou = float(s.get("mean_iou", 0.0))
        fps = float(s.get("mean_fps", 0.0))
        mem = float(s.get("peak_memory_mb", 0.0))
        ees = mean_iou * math.log1p(fps) / (1.0 + mem / memory_budget_mb)
        rows.append(
            {
                "tracker": s.get("tracker", "?"),
                "mean_iou": mean_iou,
                "fps": fps,
                "mem_mb": mem,
                "ees": ees,
            }
        )
    rows.sort(key=lambda x: x["ees"], reverse=True)

    print(f"\n{'─' * 60}")
    print(f"  Edge Efficiency Score Ranking  (memory budget: {memory_budget_mb:.0f} MB)")
    print(f"  EES = mIoU × log1p(FPS) / (1 + mem / budget)")
    print(f"{'─' * 60}")
    print(f"  {'Rank':<5} {'Tracker':<18} {'mIoU':>7} {'FPS':>8} {'Mem(MB)':>9} {'EES':>8}")
    print(f"  {'─'*5} {'─'*18} {'─'*7} {'─'*8} {'─'*9} {'─'*8}")
    for rank, row in enumerate(rows, start=1):
        print(
            f"  {rank:<5} {row['tracker']:<18} "
            f"{row['mean_iou']:>7.4f} {row['fps']:>8.1f} "
            f"{row['mem_mb']:>9.1f} {row['ees']:>8.4f}"
        )
    print()


def _simulate_devices(results: List[Dict[str, Any]]) -> None:
    """Project each tracker's profiling result onto all built-in edge devices."""
    from eovot.profiling.device_sim import DeviceSimulator
    from eovot.profiling.profiler import ProfilingResult

    sim = DeviceSimulator()

    for result_dict in results:
        s = result_dict.get("summary", {})
        tracker_name = s.get("tracker", "?")
        mean_fps = float(s.get("mean_fps", 0.0))
        peak_mem = float(s.get("peak_memory_mb", 0.0))
        lat_ms = 1_000.0 / mean_fps if mean_fps > 0.0 else 9_999.0

        # Reconstruct a minimal ProfilingResult for the simulator from summary dict.
        prof = ProfilingResult(
            tracker_name=tracker_name,
            frame_count=0,
            fps=mean_fps,
            latency_mean_ms=lat_ms,
            latency_std_ms=0.0,
            latency_p95_ms=lat_ms,
            peak_memory_mb=peak_mem,
        )

        device_results = sim.simulate_all(prof)

        print(f"\n{'─' * 60}")
        print(f"  Edge Device Projection: {tracker_name}")
        print(f"  (host: {mean_fps:.1f} FPS,  {peak_mem:.0f} MB)")
        print(f"{'─' * 60}")
        print(sim.to_markdown_table(device_results))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_experiment",
        description="Run a reproducible EOVOT multi-tracker experiment from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See configs/experiments/ for example YAML config files.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to a YAML experiment config file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/experiments",
        metavar="DIR",
        help="Root directory for all experiment outputs. Default: results/experiments",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip trackers whose result JSON already exists (resume interrupted runs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the experiment plan without running anything.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sequence verbose output.",
    )
    parser.add_argument(
        "--tdp-watts",
        type=float,
        default=None,
        metavar="W",
        help=(
            "Override experiment.tdp_watts from the command line. "
            "Enables per-frame CPU energy estimation. "
            "Examples: 6.0 (Raspberry Pi 4), 10.0 (Jetson Nano), 15.0 (laptop)."
        ),
    )
    parser.add_argument(
        "--simulate-devices",
        action="store_true",
        help=(
            "After benchmarking, project each tracker's profiling result onto "
            "all six built-in edge device profiles and print the fleet table."
        ),
    )
    parser.add_argument(
        "--ees-budget",
        type=float,
        default=512.0,
        metavar="MB",
        help=(
            "Memory budget in MB used for the Edge Efficiency Score ranking. "
            "Default: 512 MB."
        ),
    )
    parser.add_argument(
        "--list-trackers",
        action="store_true",
        help="Print available tracker names and exit.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print built-in edge device profiles and exit.",
    )

    args = parser.parse_args()

    # Informational modes — no config required.
    if args.list_trackers:
        _list_trackers()
        return

    if args.list_devices:
        _list_devices()
        return

    if not args.config:
        parser.error(
            "--config is required (or use --list-trackers / --list-devices to "
            "explore what is available)."
        )

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    # CLI overrides
    if args.tdp_watts is not None:
        config.setdefault("experiment", {})["tdp_watts"] = args.tdp_watts

    if args.dry_run:
        _print_plan(config, args.output_dir)
        return

    runner = ExperimentRunner(
        output_dir=args.output_dir,
        verbose=not args.quiet,
        resume=args.resume,
    )
    output = runner.run_from_config(config)
    results = output.get("results", [])

    # Always show EES ranking when multiple trackers ran.
    if len(results) >= 1:
        _show_ees_ranking(results, memory_budget_mb=args.ees_budget)

    # Optional edge device fleet table.
    if args.simulate_devices and results:
        _simulate_devices(results)


if __name__ == "__main__":
    main()
