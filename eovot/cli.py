"""Unified command-line interface for EOVOT.

Replaces the collection of standalone scripts in ``scripts/`` with a single
``eovot`` entry point offering discoverable subcommands:

Subcommands
-----------
``eovot run``
    Benchmark one tracker on a dataset and print a summary.  Works out of the
    box with synthetic data — no dataset download required.

``eovot compare``
    Run several trackers on a dataset, produce a ranked leaderboard, and
    optionally save JSON results.

``eovot simulate``
    Project a saved benchmark JSON result onto a set of edge hardware profiles
    (Raspberry Pi 4, Jetson Nano, …) and print an estimated performance table.

``eovot list``
    Discover available trackers and/or registered edge device profiles.

Examples
--------
Quick smoke-test with synthetic data (no dataset needed)::

    eovot run --tracker KCF --dataset synthetic --sequences 5 --frames 80

Compare three classical trackers::

    eovot compare --trackers KCF MOSSE CSRT --dataset synthetic --sequences 8

Simulate a saved result on two edge targets::

    eovot simulate --input results/KCF-Synthetic.json --devices rpi4 jetson_nano

Discover everything::

    eovot list --trackers --devices
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Dataset builder helpers
# ---------------------------------------------------------------------------

def _build_synthetic_dataset(sequences: int, frames: int, motion: str, seed: int):
    from eovot.datasets.synthetic import SyntheticDataset
    return SyntheticDataset(
        num_sequences=sequences,
        num_frames=frames,
        motion=motion,  # type: ignore[arg-type]
        seed=seed,
    )


def _build_otb_dataset(root: str):
    from eovot.datasets.base import OTBDataset
    return OTBDataset(root=root)


def _build_got10k_dataset(root: str, split: str):
    from eovot.datasets.got10k import GOT10kDataset
    return GOT10kDataset(root=root, split=split)


def _build_lasot_dataset(root: str, split: str):
    from eovot.datasets.lasot import LaSOTDataset
    return LaSOTDataset(root=root, split=split)


def _resolve_dataset(args: argparse.Namespace):
    """Return (dataset, dataset_name) from CLI args."""
    dtype = (args.dataset or "synthetic").lower()
    if dtype == "synthetic":
        dataset = _build_synthetic_dataset(
            sequences=getattr(args, "sequences", 5),
            frames=getattr(args, "frames", 100),
            motion=getattr(args, "motion", "linear"),
            seed=getattr(args, "seed", 42),
        )
        return dataset, "Synthetic"
    if dtype == "otb":
        root = getattr(args, "dataset_root", None) or ""
        if not root:
            print("ERROR: --dataset-root is required for --dataset otb", file=sys.stderr)
            sys.exit(1)
        return _build_otb_dataset(root), "OTB"
    if dtype == "got10k":
        root = getattr(args, "dataset_root", None) or ""
        split = getattr(args, "split", "val")
        if not root:
            print("ERROR: --dataset-root is required for --dataset got10k", file=sys.stderr)
            sys.exit(1)
        return _build_got10k_dataset(root, split), "GOT10k"
    if dtype == "lasot":
        root = getattr(args, "dataset_root", None) or ""
        split = getattr(args, "split", "test")
        if not root:
            print("ERROR: --dataset-root is required for --dataset lasot", file=sys.stderr)
            sys.exit(1)
        return _build_lasot_dataset(root, split), "LaSOT"
    print(f"ERROR: Unknown dataset type '{dtype}'. Choose: synthetic, otb, got10k, lasot", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Benchmark a single tracker and print a summary."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.registry import build_tracker, available_trackers

    tracker_name = args.tracker
    if tracker_name not in available_trackers():
        print(
            f"ERROR: Unknown tracker '{tracker_name}'.\n"
            f"Available: {', '.join(available_trackers())}",
            file=sys.stderr,
        )
        sys.exit(1)

    tracker = build_tracker(tracker_name)
    dataset, dataset_name = _resolve_dataset(args)

    max_seq = getattr(args, "max_sequences", None)
    if max_seq is not None:
        max_seq = int(max_seq)

    tdp = getattr(args, "tdp", None)
    engine = BenchmarkEngine(verbose=True, tdp_watts=tdp)
    result = engine.run(tracker, dataset, dataset_name=dataset_name, max_sequences=max_seq)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for k, v in result.summary().items():
        print(f"  {k:<30s}: {v}")

    output = getattr(args, "output", None)
    if output:
        saved = result.save(output)
        print(f"\nResult saved to: {saved}")


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> None:
    """Run multiple trackers, print a ranked leaderboard."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.metrics.efficiency import EfficiencyMetricsEngine
    from eovot.trackers.registry import build_tracker, available_trackers
    from eovot.reporting.reporter import BenchmarkReporter

    tracker_names: List[str] = args.trackers
    unknown = [t for t in tracker_names if t not in available_trackers()]
    if unknown:
        print(
            f"ERROR: Unknown tracker(s): {', '.join(unknown)}.\n"
            f"Available: {', '.join(available_trackers())}",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset, dataset_name = _resolve_dataset(args)
    max_seq = getattr(args, "max_sequences", None)
    if max_seq is not None:
        max_seq = int(max_seq)

    tdp = getattr(args, "tdp", None)
    engine = BenchmarkEngine(verbose=not args.quiet, tdp_watts=tdp)

    results = []
    for name in tracker_names:
        tracker = build_tracker(name)
        r = engine.run(tracker, dataset, dataset_name=dataset_name, max_sequences=max_seq)
        results.append(r)

    # Ranked leaderboard by EES
    memory_budget = getattr(args, "memory_budget", 512.0)
    eff_engine = EfficiencyMetricsEngine(memory_budget_mb=memory_budget)
    ranking = eff_engine.rank_trackers(results)

    print("\n" + "=" * 60)
    print(f"LEADERBOARD — {dataset_name} ({len(results)} trackers)")
    print("=" * 60)
    print(eff_engine.to_markdown_table(ranking))

    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        reporter = BenchmarkReporter(output_dir=output_dir)
        for r in results:
            reporter.save_json(r.to_dict(), name=f"{r.tracker_name}-{dataset_name}")
        print(f"\nIndividual results saved to: {output_dir}/")


# ---------------------------------------------------------------------------
# Subcommand: simulate
# ---------------------------------------------------------------------------

def cmd_simulate(args: argparse.Namespace) -> None:
    """Project a saved benchmark JSON onto edge hardware profiles."""
    from eovot.benchmark.engine import BenchmarkResult
    from eovot.profiling.device_sim import DeviceSimulator
    from eovot.profiling.profiler import ProfilingResult

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    br = BenchmarkResult.load(input_path)
    tracker_name = br.tracker_name
    mean_fps = br.mean_fps
    mean_latency_ms = 1000.0 / mean_fps if mean_fps > 0 else 0.0

    host_prof = ProfilingResult(
        tracker_name=tracker_name,
        frame_count=sum(len(s.ious) for s in br.sequence_results),
        fps=mean_fps,
        latency_mean_ms=mean_latency_ms,
        latency_std_ms=0.0,
        latency_p95_ms=mean_latency_ms,
        peak_memory_mb=br.peak_memory_mb,
    )

    device_names: Optional[List[str]] = getattr(args, "devices", None) or None
    sustained: float = getattr(args, "sustained", 0.0)
    cal: float = getattr(args, "calibration", 1.0)

    sim = DeviceSimulator(host_calibration_factor=cal)
    device_results = sim.simulate_all(
        host_prof, sustained_seconds=sustained, device_names=device_names
    )

    print(f"\nDevice simulation for: {tracker_name} (host FPS={mean_fps:.1f})")
    print(f"Sustained load: {sustained:.0f}s  |  Host calibration: {cal:.2f}x")
    print()
    print(sim.to_markdown_table(device_results))

    output = getattr(args, "output", None)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"tracker": tracker_name, "devices": sim.to_summary_dict(device_results)}
        with open(out_path, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"\nSimulation saved to: {out_path}")


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    """List available trackers and/or edge device profiles."""
    show_trackers = getattr(args, "trackers", False)
    show_devices = getattr(args, "devices", False)

    if not show_trackers and not show_devices:
        show_trackers = True
        show_devices = True

    if show_trackers:
        from eovot.trackers.registry import available_trackers
        names = available_trackers()
        print("Available trackers:")
        for n in names:
            print(f"  {n}")

    if show_devices:
        from eovot.profiling.device_sim import KNOWN_DEVICES
        print("\nEdge device profiles:")
        for key, profile in sorted(KNOWN_DEVICES.items()):
            print(f"  {key:<18s}  {profile.display_name}")
            if getattr(args, "verbose", False):
                print(f"    speed_factor={profile.cpu_speed_factor:.2f}  "
                      f"TDP={profile.tdp_watts}W  "
                      f"RAM={profile.memory_limit_mb/1024:.1f}GB")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eovot",
        description="EOVOT — Edge-Optimized Visual Object Tracking Benchmark Suite",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ---- Common dataset flags shared across subcommands ----
    def _add_dataset_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--dataset",
            default="synthetic",
            metavar="TYPE",
            help="Dataset type: synthetic | otb | got10k | lasot  (default: synthetic)",
        )
        p.add_argument(
            "--dataset-root",
            dest="dataset_root",
            metavar="PATH",
            help="Root directory for real datasets (required for non-synthetic types).",
        )
        p.add_argument(
            "--sequences",
            type=int,
            default=5,
            metavar="N",
            help="Number of sequences for synthetic dataset.  (default: 5)",
        )
        p.add_argument(
            "--frames",
            type=int,
            default=100,
            metavar="N",
            help="Frames per sequence for synthetic dataset.  (default: 100)",
        )
        p.add_argument(
            "--motion",
            default="linear",
            choices=["linear", "circular", "random"],
            help="Motion pattern for synthetic dataset.  (default: linear)",
        )
        p.add_argument(
            "--seed",
            type=int,
            default=42,
            metavar="N",
            help="Random seed for synthetic dataset.  (default: 42)",
        )
        p.add_argument(
            "--split",
            default="val",
            metavar="SPLIT",
            help="Dataset split for GOT10k/LaSOT.  (default: val)",
        )
        p.add_argument(
            "--max-sequences",
            dest="max_sequences",
            type=int,
            default=None,
            metavar="N",
            help="Limit evaluation to the first N sequences.",
        )
        p.add_argument(
            "--tdp",
            type=float,
            default=None,
            metavar="WATTS",
            help="CPU TDP in watts to enable energy profiling (e.g. 7.5 for RPi 4).",
        )

    # ---- run ----
    run_p = sub.add_parser("run", help="Benchmark a single tracker.")
    run_p.add_argument(
        "--tracker",
        required=True,
        metavar="NAME",
        help="Tracker name (e.g. KCF, MOSSE, CSRT).  Use 'eovot list --trackers' to see all.",
    )
    run_p.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save full benchmark result to this JSON file.",
    )
    _add_dataset_args(run_p)

    # ---- compare ----
    cmp_p = sub.add_parser("compare", help="Compare multiple trackers and print a leaderboard.")
    cmp_p.add_argument(
        "--trackers",
        nargs="+",
        required=True,
        metavar="NAME",
        help="One or more tracker names to compare.",
    )
    cmp_p.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        metavar="DIR",
        help="Directory to write per-tracker JSON results.",
    )
    cmp_p.add_argument(
        "--memory-budget",
        dest="memory_budget",
        type=float,
        default=512.0,
        metavar="MB",
        help="Memory budget for Edge Efficiency Score (MB).  (default: 512)",
    )
    cmp_p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sequence progress output.",
    )
    _add_dataset_args(cmp_p)

    # ---- simulate ----
    sim_p = sub.add_parser(
        "simulate",
        help="Project a saved benchmark JSON onto edge device profiles.",
    )
    sim_p.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help="Path to a benchmark JSON file produced by 'eovot run --output'.",
    )
    sim_p.add_argument(
        "--devices",
        nargs="*",
        default=None,
        metavar="DEVICE",
        help=(
            "Device keys to simulate (e.g. rpi4 jetson_nano).  "
            "Omit to simulate all built-in devices.  "
            "Use 'eovot list --devices' to see all keys."
        ),
    )
    sim_p.add_argument(
        "--sustained",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Duration of continuous tracking to model thermal throttling.  "
            "0 (default) = cold-start / burst scenario."
        ),
    )
    sim_p.add_argument(
        "--calibration",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Host-to-reference calibration factor.  >1 if benchmark machine is "
            "faster than reference i7; <1 if slower.  (default: 1.0)"
        ),
    )
    sim_p.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save simulation results to this JSON file.",
    )

    # ---- list ----
    lst_p = sub.add_parser("list", help="List available trackers and device profiles.")
    lst_p.add_argument(
        "--trackers",
        action="store_true",
        help="Print the list of registered tracker names.",
    )
    lst_p.add_argument(
        "--devices",
        action="store_true",
        help="Print the list of registered edge device profile keys.",
    )
    lst_p.add_argument(
        "--verbose",
        action="store_true",
        help="Include hardware specs alongside device names.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Parse ``argv`` and dispatch to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "run": cmd_run,
        "compare": cmd_compare,
        "simulate": cmd_simulate,
        "list": cmd_list,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
