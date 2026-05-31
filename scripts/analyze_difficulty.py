#!/usr/bin/env python3
"""Sequence difficulty analysis CLI for EOVOT datasets.

Analyses a dataset's ground-truth bounding boxes and produces a per-sequence
difficulty report (score, tier, challenge tags) plus a dataset-level summary.
No frame images are loaded — only GT boxes are read, so the analysis is fast
even for large datasets.

Outputs (written to --output-dir):
  <name>-difficulty.json   Full per-sequence difficulty profiles (JSON)
  <name>-difficulty.md     Formatted Markdown table sorted by difficulty

Usage
-----
    # Analyse the SyntheticDataset (no download required):
    python scripts/analyze_difficulty.py \\
        --dataset-loader SyntheticDataset \\
        --dataset-name Synthetic \\
        --num-sequences 20 \\
        --motion random

    # Analyse a real OTB dataset:
    python scripts/analyze_difficulty.py \\
        --dataset-loader OTBDataset \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100

    # Show only hard sequences:
    python scripts/analyze_difficulty.py \\
        --dataset-loader SyntheticDataset \\
        --dataset-name Synthetic \\
        --filter-tier hard \\
        --top 10

    # Use custom challenge thresholds:
    python scripts/analyze_difficulty.py \\
        --dataset-loader SyntheticDataset \\
        --dataset-name Synthetic \\
        --scr-threshold 0.20 \\
        --mv-threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.datasets.difficulty import (
    DifficultyFilteredDataset,
    SequenceDifficultyAnalyzer,
)


def _build_dataset(args: argparse.Namespace):
    """Instantiate the requested dataset from CLI arguments."""
    from eovot.datasets.base import OTBDataset
    from eovot.datasets.got10k import GOT10kDataset
    from eovot.datasets.lasot import LaSOTDataset
    from eovot.datasets.synthetic import SyntheticDataset

    loader = args.dataset_loader
    if loader == "SyntheticDataset":
        return SyntheticDataset(
            num_sequences=args.num_sequences,
            num_frames=args.num_frames,
            motion=args.motion,
            seed=args.seed,
        )
    if loader == "OTBDataset":
        if not args.dataset_root:
            print("[ERROR] --dataset-root is required for OTBDataset.", file=sys.stderr)
            sys.exit(1)
        return OTBDataset(root=args.dataset_root)
    if loader == "GOT10kDataset":
        if not args.dataset_root:
            print("[ERROR] --dataset-root is required for GOT10kDataset.", file=sys.stderr)
            sys.exit(1)
        return GOT10kDataset(
            root=args.dataset_root,
            split=args.split,
            max_sequences=args.max_sequences,
        )
    if loader == "LaSOTDataset":
        if not args.dataset_root:
            print("[ERROR] --dataset-root is required for LaSOTDataset.", file=sys.stderr)
            sys.exit(1)
        return LaSOTDataset(
            root=args.dataset_root,
            split=args.split,
            max_sequences=args.max_sequences,
        )
    print(f"[ERROR] Unknown dataset loader: {loader}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="analyze_difficulty",
        description="Analyse EOVOT dataset difficulty from ground-truth bounding boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset selection
    parser.add_argument(
        "--dataset-loader",
        default="SyntheticDataset",
        choices=["SyntheticDataset", "OTBDataset", "GOT10kDataset", "LaSOTDataset"],
        help="Dataset loader to use.",
    )
    parser.add_argument("--dataset-root", default=None,
                        help="Root path for OTB / GOT-10k / LaSOT datasets.")
    parser.add_argument("--dataset-name", default="dataset",
                        help="Human-readable label used in output filenames.")
    parser.add_argument("--split", default="val",
                        help="Dataset split (GOT-10k / LaSOT only).")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Cap on sequences to analyse.")

    # SyntheticDataset options
    parser.add_argument("--num-sequences", type=int, default=10,
                        help="Number of sequences (SyntheticDataset only).")
    parser.add_argument("--num-frames", type=int, default=100,
                        help="Frames per sequence (SyntheticDataset only).")
    parser.add_argument("--motion", default="random",
                        choices=["linear", "circular", "random"],
                        help="Motion pattern (SyntheticDataset only).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (SyntheticDataset only).")

    # Difficulty thresholds
    parser.add_argument("--scr-threshold", type=float, default=0.15,
                        help="SCR threshold for SCALE_CHANGE tag.")
    parser.add_argument("--mv-threshold", type=float, default=0.10,
                        help="MV threshold for FAST_MOTION tag.")
    parser.add_argument("--arj-threshold", type=float, default=0.12,
                        help="ARJ threshold for DEFORMATION tag.")
    parser.add_argument("--dfr-threshold", type=float, default=0.05,
                        help="DFR threshold for OCCLUSION tag.")

    # Output / filtering options
    parser.add_argument("--output-dir", default="results/difficulty",
                        help="Directory for output files.")
    parser.add_argument("--filter-tier", default=None,
                        choices=["easy", "medium", "hard"],
                        help="If set, only report sequences with this difficulty tier.")
    parser.add_argument("--top", type=int, default=None,
                        help="Show only the top N hardest sequences.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress console output (write files only).")

    args = parser.parse_args()

    dataset = _build_dataset(args)

    analyzer = SequenceDifficultyAnalyzer(
        scr_threshold=args.scr_threshold,
        mv_threshold=args.mv_threshold,
        arj_threshold=args.arj_threshold,
        dfr_threshold=args.dfr_threshold,
    )

    if not args.quiet:
        print(f"\nAnalysing {len(dataset)} sequences from {args.dataset_name} ...")

    difficulties = analyzer.analyze_dataset(dataset)
    summary = analyzer.dataset_summary(difficulties)

    if args.filter_tier:
        indices = analyzer.filter_by_tier(difficulties, [args.filter_tier])
        difficulties = [difficulties[i] for i in indices]
        if not args.quiet:
            print(
                f"  → Filtered to {len(difficulties)} "
                f"'{args.filter_tier}' sequences."
            )

    if not args.quiet:
        print("\n" + "=" * 60)
        print(f"  DIFFICULTY SUMMARY — {args.dataset_name}")
        print("=" * 60)
        print(f"  Total sequences : {summary['num_sequences']}")
        print(f"  Mean difficulty : {summary['mean_difficulty']:.3f} "
              f"± {summary['std_difficulty']:.3f}")
        tc = summary["tier_counts"]
        print(f"  Tier breakdown  : easy={tc['easy']}  "
              f"medium={tc['medium']}  hard={tc['hard']}")
        cc = summary["challenge_counts"]
        print(f"  Challenges      : "
              f"SCALE_CHANGE={cc.get('SCALE_CHANGE', 0)}  "
              f"FAST_MOTION={cc.get('FAST_MOTION', 0)}  "
              f"DEFORMATION={cc.get('DEFORMATION', 0)}  "
              f"OCCLUSION={cc.get('OCCLUSION', 0)}")
        print("=" * 60)

        table = analyzer.to_markdown_table(difficulties, top_n=args.top)
        print("\n" + table)

    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{args.dataset_name}-difficulty"

    json_path = out_dir / f"{base}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset": args.dataset_name,
                "summary": summary,
                "sequences": [d.to_dict() for d in difficulties],
            },
            fh, indent=2,
        )

    md_path = out_dir / f"{base}.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Difficulty Analysis — {args.dataset_name}\n\n")
        fh.write("## Summary\n\n")
        fh.write(f"- Total sequences: {summary['num_sequences']}\n")
        fh.write(f"- Mean difficulty: {summary['mean_difficulty']:.3f} "
                 f"± {summary['std_difficulty']:.3f}\n")
        tc = summary["tier_counts"]
        fh.write(f"- Tiers: easy={tc['easy']}, medium={tc['medium']}, hard={tc['hard']}\n\n")
        fh.write("## Per-Sequence Difficulty\n\n")
        fh.write(analyzer.to_markdown_table(difficulties, top_n=args.top))
        fh.write("\n")

    if not args.quiet:
        print(f"\nResults saved to:")
        print(f"  JSON → {json_path}")
        print(f"  Markdown → {md_path}")


if __name__ == "__main__":
    main()
