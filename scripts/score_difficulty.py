#!/usr/bin/env python3
"""CLI: Rank tracking sequences by ground-truth difficulty score.

Computes the EOVOT difficulty score for every sequence in a dataset using
six factors derived from ground-truth bounding boxes alone (no tracker
evaluation required).  Prints a ranked Markdown table and optionally writes
a JSON report.

Usage examples::

    # Synthetic dataset (no downloads required)
    python scripts/score_difficulty.py \\
        --dataset synthetic \\
        --motion random \\
        --num-sequences 20 \\
        --frame-size 240 320

    # GOT-10k validation split
    python scripts/score_difficulty.py \\
        --dataset got10k \\
        --dataset-root /data/GOT-10k \\
        --split val \\
        --frame-size 720 1280 \\
        --top 20 \\
        --out results/got10k_difficulty.json

    # OTB-100
    python scripts/score_difficulty.py \\
        --dataset otb \\
        --dataset-root /data/OTB100 \\
        --frame-size 480 640

    # LaSOT testing split
    python scripts/score_difficulty.py \\
        --dataset lasot \\
        --dataset-root /data/LaSOT \\
        --split test \\
        --num-sequences 50 \\
        --frame-size 360 640
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="score_difficulty",
        description="Rank tracking sequences by ground-truth difficulty score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        choices=["synthetic", "got10k", "otb", "lasot"],
        default="synthetic",
        help="Dataset to score.",
    )
    p.add_argument(
        "--dataset-root",
        metavar="DIR",
        help="Root directory for got10k / otb / lasot datasets.",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split (GOT-10k / LaSOT only).",
    )
    p.add_argument(
        "--motion",
        choices=["linear", "circular", "random"],
        default="random",
        help="Motion pattern (synthetic dataset only).",
    )
    p.add_argument(
        "--num-sequences",
        type=int,
        default=20,
        metavar="N",
        help="Number of sequences to score.",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=100,
        metavar="N",
        help="Frames per sequence (synthetic dataset only).",
    )
    p.add_argument(
        "--frame-size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        help="Frame height and width for small-target and OOV factors.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="Print the N hardest sequences.",
    )
    p.add_argument(
        "--out",
        metavar="FILE",
        help="Write the full difficulty report to a JSON file.",
    )
    return p


def _build_dataset(args):
    if args.dataset == "synthetic":
        from eovot.datasets.synthetic import SyntheticDataset
        return (
            SyntheticDataset(
                num_sequences=args.num_sequences,
                num_frames=args.num_frames,
                motion=args.motion,
            ),
            f"Synthetic-{args.motion.capitalize()}",
        )

    if args.dataset_root is None:
        print(
            f"Error: --dataset-root is required for {args.dataset}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dataset == "got10k":
        from eovot.datasets.got10k import GOT10kDataset
        return (
            GOT10kDataset(
                args.dataset_root,
                split=args.split,
                max_sequences=args.num_sequences,
            ),
            f"GOT-10k-{args.split}",
        )

    if args.dataset == "otb":
        from eovot.datasets.base import OTBDataset
        return OTBDataset(args.dataset_root), "OTB"

    if args.dataset == "lasot":
        from eovot.datasets.lasot import LaSOTDataset
        return (
            LaSOTDataset(
                args.dataset_root,
                split=args.split,
                max_sequences=args.num_sequences,
            ),
            f"LaSOT-{args.split}",
        )

    print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    args = _build_parser().parse_args()
    dataset, dataset_name = _build_dataset(args)
    frame_size = tuple(args.frame_size) if args.frame_size else None

    from eovot.metrics.difficulty import score_dataset

    print(f"\nScoring {len(dataset)} sequences from {dataset_name} ...", flush=True)
    report = score_dataset(dataset, frame_size=frame_size)

    n_print = min(args.top, len(report))
    print(f"\nTop-{n_print} hardest sequences:\n")
    # Build a sub-report for display
    from eovot.metrics.difficulty import DifficultyReport
    sub_report = DifficultyReport(report.hardest(n_print))
    print(sub_report.to_markdown())

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"\nFull report written to: {out_path}")


if __name__ == "__main__":
    main()
