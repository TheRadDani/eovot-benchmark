"""CLI for generating EOVOT edge deployment reports.

Runs a tracker on a synthetic or real dataset, then generates a
self-contained Markdown report covering accuracy, host profiling,
edge device projections, and go/no-go deployment verdicts.

Usage examples
--------------
# Quick demo using the built-in synthetic dataset (no data download required):
    python scripts/edge_deploy.py \\
        --tracker MOSSE \\
        --dataset synthetic \\
        --num-sequences 5 --num-frames 50 \\
        --tdp-watts 15.0 \\
        --min-fps 15.0 \\
        --output results/mosse_edge_report.md

# Real OTB dataset:
    python scripts/edge_deploy.py \\
        --tracker KCF \\
        --dataset otb \\
        --dataset-root /data/OTB100 \\
        --max-sequences 10 \\
        --output results/kcf_edge_report.md

# Save JSON in addition to Markdown:
    python scripts/edge_deploy.py \\
        --tracker CSRT \\
        --dataset synthetic \\
        --json results/csrt_edge.json \\
        --output results/csrt_edge_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.reporting.edge_report import EdgeDeploymentReporter
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


def _build_dataset(args: argparse.Namespace):
    if args.dataset == "synthetic":
        return SyntheticDataset(
            num_sequences=args.num_sequences,
            num_frames=args.num_frames,
            motion=args.motion,
            seed=args.seed,
        ), "Synthetic"

    if args.dataset == "otb":
        from eovot.datasets.base import OTBDataset
        if not args.dataset_root:
            print("[ERROR] --dataset-root is required for OTB.", file=sys.stderr)
            sys.exit(1)
        return OTBDataset(args.dataset_root), "OTB"

    if args.dataset == "got10k":
        from eovot.datasets.got10k import GOT10kDataset
        if not args.dataset_root:
            print("[ERROR] --dataset-root is required for GOT-10k.", file=sys.stderr)
            sys.exit(1)
        return GOT10kDataset(
            args.dataset_root,
            split=args.split,
            max_sequences=args.max_sequences,
        ), "GOT10k"

    if args.dataset == "lasot":
        from eovot.datasets.lasot import LaSOTDataset
        if not args.dataset_root:
            print("[ERROR] --dataset-root is required for LaSOT.", file=sys.stderr)
            sys.exit(1)
        return LaSOTDataset(
            args.dataset_root,
            split=args.split,
            max_sequences=args.max_sequences,
        ), "LaSOT"

    print(f"[ERROR] Unknown dataset: {args.dataset}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="edge_deploy",
        description="EOVOT — Edge Deployment Report Generator",
    )

    # Tracker
    parser.add_argument(
        "--tracker", "-t",
        default="MOSSE",
        choices=sorted(TRACKER_REGISTRY),
        help="Tracker to evaluate (default: MOSSE).",
    )

    # Dataset
    parser.add_argument(
        "--dataset", "-d",
        default="synthetic",
        choices=["synthetic", "otb", "got10k", "lasot"],
        help="Dataset to use (default: synthetic).",
    )
    parser.add_argument("--dataset-root", metavar="DIR",
                        help="Root directory for OTB/GOT-10k/LaSOT datasets.")
    parser.add_argument("--max-sequences", type=int, default=None, metavar="N",
                        help="Limit evaluation to the first N sequences.")
    parser.add_argument("--split", default="val",
                        help="Dataset split for GOT-10k/LaSOT (default: val).")

    # Synthetic-dataset parameters
    parser.add_argument("--num-sequences", type=int, default=5, metavar="N",
                        help="Sequences for synthetic dataset (default: 5).")
    parser.add_argument("--num-frames", type=int, default=100, metavar="N",
                        help="Frames per synthetic sequence (default: 100).")
    parser.add_argument("--motion", default="linear",
                        choices=["linear", "circular", "random"],
                        help="Motion pattern for synthetic dataset (default: linear).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for synthetic dataset (default: 42).")

    # Profiling / energy
    parser.add_argument("--tdp-watts", type=float, default=None, metavar="W",
                        help=(
                            "Enable CPU energy estimation with this host TDP in watts "
                            "(e.g. 15.0 for a laptop, 6.0 for Raspberry Pi 4)."
                        ))

    # Report parameters
    parser.add_argument("--min-fps", type=float, default=10.0, metavar="FPS",
                        help="Minimum FPS for a 'deployable' verdict (default: 10).")
    parser.add_argument("--memory-budget", type=float, default=512.0, metavar="MB",
                        help="Memory budget in MB for EES computation (default: 512).")
    parser.add_argument("--sustained-seconds", type=float, default=60.0, metavar="S",
                        help="Sustained run duration for thermal model (default: 60 s).")
    parser.add_argument("--devices", nargs="+", metavar="DEVICE",
                        help="Subset of device names to project (e.g. rpi4 jetson_nano).")

    # Output
    parser.add_argument("--output", "-o", default=None, metavar="PATH",
                        help="Path for the Markdown report (default: stdout).")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="Also save a JSON summary to this path.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-sequence benchmark progress.")

    args = parser.parse_args()

    dataset, dataset_name = _build_dataset(args)

    tracker = build_tracker(args.tracker)
    engine = BenchmarkEngine(verbose=not args.quiet, tdp_watts=args.tdp_watts)
    result = engine.run(
        tracker=tracker,
        dataset=dataset,
        dataset_name=dataset_name,
        max_sequences=args.max_sequences,
    )

    reporter = EdgeDeploymentReporter(
        min_fps=args.min_fps,
        memory_budget_mb=args.memory_budget,
        sustained_seconds=args.sustained_seconds,
        device_names=args.devices,
    )

    md = reporter.generate(result)

    if args.output:
        out_path = reporter.save(result, path=args.output)
        print(f"Report saved to {out_path}")
    else:
        print(md)

    if args.json:
        data = reporter.to_dict(result)
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"JSON summary saved to {json_path}")


if __name__ == "__main__":
    main()
