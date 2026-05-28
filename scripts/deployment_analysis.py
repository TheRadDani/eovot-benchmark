"""CLI for edge deployment analysis.

Runs a set of trackers on a synthetic dataset (no external data required)
and generates a full deployment analysis report projecting performance
onto all built-in edge devices.

Usage::

    python scripts/deployment_analysis.py \\
        --trackers MOSSE KCF \\
        --output results/deployment \\
        --sustained 60

    # With explicit device subset:
    python scripts/deployment_analysis.py \\
        --trackers MOSSE KCF CSRT \\
        --devices rpi4 rpi5 jetson_nano \\
        --output results/deployment

    # Adjust memory budget and thermal scenario:
    python scripts/deployment_analysis.py \\
        --trackers MOSSE KCF \\
        --memory-budget 256 \\
        --sustained 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.experiment.runner import ExperimentRunner
from eovot.reporting.deployment_report import DeploymentReportEngine


_DEFAULT_TRACKERS = ["MOSSE", "KCF", "CSRT", "MedianFlow"]

_TRACKER_PARAMS = {
    "MOSSE": {"learning_rate": 0.125, "sigma": 2.0},
    "KCF": {"learning_rate": 0.075, "padding": 1.5},
    "CSRT": {},
    "MedianFlow": {},
    "MIL": {},
    "ScaleAdaptiveMOSSE": {"learning_rate": 0.125, "n_scales": 7, "scale_step": 1.03},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run benchmark and generate edge deployment analysis report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--trackers",
        nargs="+",
        default=_DEFAULT_TRACKERS,
        metavar="NAME",
        help="Tracker names to benchmark. Default: %(default)s",
    )
    p.add_argument(
        "--devices",
        nargs="+",
        default=None,
        metavar="DEVICE",
        help=(
            "Edge device keys to include. "
            "Default: all built-in devices "
            "(rpi4, rpi5, jetson_nano, jetson_xnx, coral_board, snapdragon888)."
        ),
    )
    p.add_argument(
        "--output",
        default="results/deployment",
        metavar="DIR",
        help="Output directory for the report. Default: %(default)s",
    )
    p.add_argument(
        "--num-sequences",
        type=int,
        default=10,
        metavar="N",
        help="Number of synthetic sequences to benchmark. Default: %(default)s",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=100,
        metavar="N",
        help="Frames per synthetic sequence. Default: %(default)s",
    )
    p.add_argument(
        "--memory-budget",
        type=float,
        default=512.0,
        metavar="MB",
        help="Memory budget for EES computation (MB). Default: %(default)s",
    )
    p.add_argument(
        "--sustained",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Sustained-load duration for thermal throttling model. "
            "0 = cold-start burst. Default: %(default)s"
        ),
    )
    p.add_argument(
        "--tdp-watts",
        type=float,
        default=None,
        metavar="W",
        help=(
            "Host CPU TDP for energy profiling during benchmarking. "
            "Omit to disable energy measurement."
        ),
    )
    p.add_argument("--quiet", action="store_true", help="Suppress benchmark progress output.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dataset = SyntheticDataset(
        num_sequences=args.num_sequences,
        num_frames=args.num_frames,
        frame_size=(320, 240),
        bbox_size=(40, 40),
        motion="linear",
        seed=42,
    )

    engine = BenchmarkEngine(verbose=not args.quiet, tdp_watts=args.tdp_watts)
    bench_results = []

    for name in args.trackers:
        params = _TRACKER_PARAMS.get(name, {})
        cfg = {"name": name, "params": params}
        try:
            tracker = ExperimentRunner._build_tracker(cfg)
        except ValueError as exc:
            print(f"[skip] {exc}", file=sys.stderr)
            continue
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        bench_results.append(result)

    if not bench_results:
        print("No valid trackers — exiting.", file=sys.stderr)
        return 1

    report_engine = DeploymentReportEngine(
        memory_budget_mb=args.memory_budget,
        sustained_seconds=args.sustained,
        device_names=args.devices,
    )
    report = report_engine.analyze(bench_results)
    paths = report_engine.save(report, output_dir=args.output)

    print(f"\nDeployment report written:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")

    print("\n" + report_engine.to_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
