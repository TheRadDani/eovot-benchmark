"""Run a multi-dataset cross-evaluation experiment.

Evaluates one or more trackers across multiple datasets and prints a unified
cross-dataset leaderboard sorted by frame-count-weighted mIoU.

Usage
-----
    # Run from a YAML config file:
    python scripts/run_multi_dataset.py --config configs/experiments/multi_dataset_demo.yaml

    # Quick smoke test (synthetic datasets, no downloads needed):
    python scripts/run_multi_dataset.py --synthetic --trackers MOSSE KCF

    # Save results to JSON:
    python scripts/run_multi_dataset.py --config ... --output results/multi_dataset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from eovot.benchmark.multi_dataset import MultiDatasetEvaluator
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.registry import build_tracker, available_trackers


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EOVOT multi-dataset cross-evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        metavar="PATH",
        help="YAML experiment config (overrides all other flags when given).",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use built-in synthetic datasets for a zero-download smoke test.",
    )
    p.add_argument(
        "--trackers",
        nargs="+",
        metavar="NAME",
        default=["MOSSE", "KCF"],
        help=f"Trackers to evaluate.  Choices: {available_trackers()}",
    )
    p.add_argument(
        "--sequences",
        type=int,
        default=3,
        metavar="N",
        help="Sequences per synthetic dataset (ignored with --config).",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=80,
        metavar="N",
        help="Frames per synthetic sequence (ignored with --config).",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Cap sequences evaluated per dataset (useful for quick tests).",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        help="Save results to this JSON file path.",
    )
    p.add_argument(
        "--tdp-watts",
        type=float,
        default=None,
        metavar="W",
        help="CPU TDP in Watts for energy estimation (e.g. 6.0 for Raspberry Pi 4).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-sequence output.")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    if args.config:
        with open(args.config, "r") as fh:
            config = yaml.safe_load(fh)
        result = MultiDatasetEvaluator.from_config(config, verbose=not args.quiet)
    elif args.synthetic:
        trackers = [build_tracker(name) for name in args.trackers]
        datasets = [
            (
                SyntheticDataset(
                    num_sequences=args.sequences,
                    num_frames=args.frames,
                    motion="linear",
                ),
                "Syn-Linear",
            ),
            (
                SyntheticDataset(
                    num_sequences=args.sequences,
                    num_frames=args.frames,
                    motion="random",
                ),
                "Syn-Random",
            ),
            (
                SyntheticDataset(
                    num_sequences=args.sequences,
                    num_frames=args.frames,
                    motion="circular",
                ),
                "Syn-Circular",
            ),
        ]
        ev = MultiDatasetEvaluator(verbose=not args.quiet, tdp_watts=args.tdp_watts)
        result = ev.run(trackers, datasets, max_sequences=args.max_sequences)
    else:
        print(
            "ERROR: Provide --config or --synthetic.  "
            "Run with --help for usage.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n" + "=" * 70)
    print("CROSS-DATASET LEADERBOARD")
    print("=" * 70)
    print(result.cross_dataset_table())

    if args.output:
        saved = result.save(args.output)
        print(f"\nResults saved to: {saved}")
    else:
        entries = result.aggregate()
        print("\nAggregate summary (JSON):")
        print(
            json.dumps(
                [
                    {
                        "tracker": e.tracker_name,
                        "mean_iou": round(e.mean_iou, 4),
                        "mean_fps": round(e.mean_fps, 2),
                        "total_frames": e.total_frames,
                    }
                    for e in entries
                ],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
