#!/usr/bin/env python3
"""Analyze and report sequence difficulty for an EOVOT dataset.

Computes per-sequence difficulty scores from ground-truth annotations (no
frame decoding required) and prints a ranked table to stdout.

Usage::

    # Analyze a synthetic dataset (no data download required)
    python scripts/analyze_difficulty.py --dataset synthetic --num-sequences 10

    # Show only the 5 hardest sequences
    python scripts/analyze_difficulty.py --dataset synthetic --top-k 5

    # Filter to hard sequences (score >= 0.4) and save results
    python scripts/analyze_difficulty.py --dataset synthetic \\
        --min-score 0.4 --output results/difficulty.json

    # Analyze an OTB-style dataset
    python scripts/analyze_difficulty.py --dataset otb --root /data/OTB100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eovot.datasets.difficulty import DifficultyAnalyzer
from eovot.datasets.synthetic import SyntheticDataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rank and filter tracking sequences by difficulty.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        choices=["synthetic", "otb"],
        default="synthetic",
        help="Dataset type to analyze.",
    )
    p.add_argument("--root", default=None, help="Path to OTB dataset root.")
    p.add_argument(
        "--num-sequences", type=int, default=10,
        help="Number of synthetic sequences to generate.",
    )
    p.add_argument(
        "--num-frames", type=int, default=100,
        help="Frames per synthetic sequence.",
    )
    p.add_argument(
        "--motion",
        choices=["linear", "circular", "random"],
        default="linear",
        help="Motion pattern for synthetic sequences.",
    )
    p.add_argument("--frame-width", type=int, default=320)
    p.add_argument("--frame-height", type=int, default=240)
    p.add_argument(
        "--min-score", type=float, default=0.0,
        help="Minimum overall_score to include in output.",
    )
    p.add_argument(
        "--max-score", type=float, default=1.0,
        help="Maximum overall_score to include in output.",
    )
    p.add_argument(
        "--top-k", type=int, default=None,
        help="Show only the top-K hardest sequences.",
    )
    p.add_argument(
        "--output", default=None,
        help="Save results to this JSON path.",
    )
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.dataset == "synthetic":
        dataset = SyntheticDataset(
            num_sequences=args.num_sequences,
            num_frames=args.num_frames,
            motion=args.motion,
        )
    else:
        from eovot.datasets.base import OTBDataset
        if args.root is None:
            print("Error: --root is required for OTB dataset.", file=sys.stderr)
            sys.exit(1)
        dataset = OTBDataset(args.root)

    analyzer = DifficultyAnalyzer(
        frame_size=(args.frame_width, args.frame_height)
    )

    print(f"Analyzing {len(dataset)} sequences ...\n")
    difficulties = analyzer.analyze_dataset(dataset)

    filtered = analyzer.filter(difficulties, min_score=args.min_score, max_score=args.max_score)
    ranked = analyzer.rank(filtered)
    if args.top_k is not None:
        ranked = ranked[: args.top_k]

    col_w = 34
    header = (
        f"{'Sequence':<{col_w}} {'Speed':>8} {'Scale':>8} "
        f"{'AR-chg':>8} {'OOV':>8} {'Score':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for d in ranked:
        print(
            f"{d.name:<{col_w}} "
            f"{d.motion_speed:>8.2f} "
            f"{d.scale_variation:>8.4f} "
            f"{d.aspect_ratio_change:>8.4f} "
            f"{d.out_of_view_ratio:>8.4f} "
            f"{d.overall_score:>8.4f}"
        )

    if difficulties:
        stats = analyzer.summary(difficulties)
        print()
        print(
            f"Dataset summary ({len(difficulties)} sequences) — "
            f"score: mean={stats['overall_score_mean']:.4f}  "
            f"std={stats['overall_score_std']:.4f}  "
            f"min={stats['overall_score_min']:.4f}  "
            f"max={stats['overall_score_max']:.4f}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": args.dataset,
            "num_sequences_analyzed": len(difficulties),
            "num_sequences_shown": len(ranked),
            "sequences": [d.to_dict() for d in ranked],
            "summary": analyzer.summary(difficulties),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
