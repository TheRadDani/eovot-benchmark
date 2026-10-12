"""CLI script: profile the difficulty of every sequence in a dataset.

Usage examples::

    # Synthetic dataset (no data download needed)
    python scripts/profile_sequence_difficulty.py --dataset synthetic \\
        --num-sequences 20 --num-frames 150

    # OTB dataset
    python scripts/profile_sequence_difficulty.py --dataset otb \\
        --root /data/OTB100 --output results/difficulty/otb.md

    # GOT-10k validation split
    python scripts/profile_sequence_difficulty.py --dataset got10k \\
        --root /data/GOT-10k --split val --output results/difficulty/got10k.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_dataset(args):
    """Instantiate the requested dataset."""
    if args.dataset == "synthetic":
        from eovot.datasets.synthetic import SyntheticDataset
        return SyntheticDataset(
            num_sequences=args.num_sequences,
            num_frames=args.num_frames,
            motion=args.motion,
            seed=args.seed,
        )
    if args.dataset == "otb":
        from eovot.datasets.base import OTBDataset
        return OTBDataset(root=args.root)
    if args.dataset == "got10k":
        from eovot.datasets.got10k import GOT10kDataset
        return GOT10kDataset(root=args.root, split=args.split,
                             max_sequences=args.max_sequences)
    if args.dataset == "lasot":
        from eovot.datasets.lasot import LaSOTDataset
        return LaSOTDataset(root=args.root, split=args.split,
                            max_sequences=args.max_sequences)
    raise ValueError(f"Unknown dataset: {args.dataset}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile sequence difficulty from ground-truth boxes."
    )
    parser.add_argument(
        "--dataset", choices=["synthetic", "otb", "got10k", "lasot"],
        default="synthetic", help="Dataset to profile."
    )
    parser.add_argument("--root", default="", help="Dataset root directory.")
    parser.add_argument("--split", default="val", help="Dataset split (got10k/lasot).")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Limit number of sequences.")
    # Synthetic-only args
    parser.add_argument("--num-sequences", type=int, default=10,
                        help="Number of synthetic sequences.")
    parser.add_argument("--num-frames", type=int, default=100,
                        help="Frames per synthetic sequence.")
    parser.add_argument("--motion", default="linear",
                        choices=["linear", "circular", "random"],
                        help="Motion model for synthetic dataset.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic dataset.")
    # Output args
    parser.add_argument("--output", default="",
                        help="Optional path to write Markdown report.")
    parser.add_argument("--sort", choices=["hard", "easy", "name"],
                        default="hard",
                        help="Sort order: hardest first, easiest first, or by name.")
    # Profiler weight overrides
    parser.add_argument("--w-motion", type=float, default=0.40)
    parser.add_argument("--w-scale", type=float, default=0.25)
    parser.add_argument("--w-aspect", type=float, default=0.15)
    parser.add_argument("--w-oov", type=float, default=0.20)

    args = parser.parse_args()

    from eovot.metrics.difficulty import SequenceDifficultyProfiler

    dataset = build_dataset(args)
    profiler = SequenceDifficultyProfiler(
        weight_motion=args.w_motion,
        weight_scale=args.w_scale,
        weight_aspect=args.w_aspect,
        weight_oov=args.w_oov,
    )

    sequences = {}
    for seq in dataset:
        sequences[seq.name] = seq.ground_truth

    descending = args.sort != "easy"
    ranked = profiler.rank_by_difficulty(sequences, descending=descending)

    if args.sort == "name":
        ranked.sort(key=lambda r: r.sequence_name)

    # Print summary to stdout
    report = profiler.profile_dataset(sequences)
    s = report["summary"]
    print(f"\nDataset difficulty summary ({s['num_sequences']} sequences)")
    print("-" * 60)
    print(f"  Mean composite:  {s['mean_composite']:.4f}  ±{s['std_composite']:.4f}")
    print(f"  Range:           [{s['min_composite']:.4f}, {s['max_composite']:.4f}]")
    print(f"  Easy  (<0.25):   {s['easy_count']}")
    print(f"  Medium (0.25-0.60): {s['medium_count']}")
    print(f"  Hard  (>0.60):   {s['hard_count']}")
    print("-" * 60)

    table = profiler.to_markdown_table(ranked)
    print(f"\n{table}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Sequence Difficulty Report\n",
            f"Dataset: `{args.dataset}`  |  "
            f"Sequences: {s['num_sequences']}  |  "
            f"Mean score: {s['mean_composite']:.4f}\n",
            f"| Tier | Count |",
            f"|:----:|------:|",
            f"| easy | {s['easy_count']} |",
            f"| medium | {s['medium_count']} |",
            f"| hard | {s['hard_count']} |",
            f"\n## Per-Sequence Breakdown\n",
            table,
        ]
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nMarkdown report written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
