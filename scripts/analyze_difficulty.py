#!/usr/bin/env python3
"""Sequence difficulty scoring CLI for EOVOT.

Scores every sequence in a dataset by tracking difficulty and produces a
Markdown report with per-tier accuracy breakdown when a benchmark JSON is
also supplied.

Usage examples::

    # Score all sequences in an OTB dataset
    python scripts/analyze_difficulty.py \\
        --dataset-root /data/OTB100

    # Score with custom output directory
    python scripts/analyze_difficulty.py \\
        --dataset-root /data/OTB100 \\
        --output-dir results/difficulty/

    # Full stratified analysis from a benchmark JSON
    python scripts/analyze_difficulty.py \\
        --dataset-root /data/OTB100 \\
        --benchmark-json results/MOSSE-OTB100.json

    # GOT-10k validation split
    python scripts/analyze_difficulty.py \\
        --dataset-loader GOT10kDataset \\
        --dataset-root /data/GOT-10k \\
        --split val \\
        --output-dir results/difficulty/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.datasets.base import OTBDataset
from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.lasot import LaSOTDataset
from eovot.metrics.difficulty import SequenceDifficultyAnalyzer

DATASET_REGISTRY = {
    "OTBDataset": OTBDataset,
    "GOT10kDataset": GOT10kDataset,
    "LaSOTDataset": LaSOTDataset,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="analyze_difficulty",
        description="Score tracking sequences by difficulty and (optionally) produce stratified reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", required=True, help="Path to dataset root.")
    parser.add_argument(
        "--dataset-loader",
        default="OTBDataset",
        choices=list(DATASET_REGISTRY),
        help="Dataset loader class.",
    )
    parser.add_argument("--split", default="val", help="Split for GOT-10k / LaSOT.")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="results/difficulty/",
        help="Directory for CSV and JSON output.",
    )
    parser.add_argument(
        "--benchmark-json",
        default=None,
        help="Path to a BenchmarkResult JSON file (enables stratified accuracy report).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Load dataset and extract GT boxes
    # -------------------------------------------------------------------
    cls = DATASET_REGISTRY[args.dataset_loader]
    if args.dataset_loader == "GOT10kDataset":
        dataset = cls(root=args.dataset_root, split=args.split,
                      max_sequences=args.max_sequences)
    elif args.dataset_loader == "LaSOTDataset":
        dataset = cls(root=args.dataset_root, split=args.split,
                      max_sequences=args.max_sequences)
    else:
        dataset = cls(root=args.dataset_root)

    n = min(len(dataset), args.max_sequences) if args.max_sequences else len(dataset)
    print(f"Loaded {n} sequences from {args.dataset_root}")

    sequences_gt = {}
    for i in range(n):
        seq = dataset[i]
        sequences_gt[seq.name] = seq.ground_truth

    # -------------------------------------------------------------------
    # Score sequences
    # -------------------------------------------------------------------
    analyzer = SequenceDifficultyAnalyzer()
    difficulties = analyzer.score_dataset(sequences_gt)
    difficulties.sort(key=lambda d: d.overall_score, reverse=True)

    print(f"\nScored {len(difficulties)} sequences")
    tier_counts = {t: sum(1 for d in difficulties if d.tier == t)
                   for t in ("easy", "medium", "hard")}
    print(f"  easy={tier_counts['easy']}  medium={tier_counts['medium']}  hard={tier_counts['hard']}")

    # -------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------
    csv_path = out_dir / "difficulty_scores.csv"
    with open(csv_path, "w", newline="") as fh:
        fieldnames = ["sequence_name", "tier", "overall_score", "motion_score",
                      "scale_score", "aspect_score", "length_score", "num_frames"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for d in difficulties:
            writer.writerow(d.to_dict())
    print(f"\nDifficulty CSV → {csv_path}")

    # -------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------
    json_path = out_dir / "difficulty_scores.json"
    with open(json_path, "w") as fh:
        json.dump([d.to_dict() for d in difficulties], fh, indent=2)
    print(f"Difficulty JSON → {json_path}")

    # -------------------------------------------------------------------
    # Print top-10 hardest sequences
    # -------------------------------------------------------------------
    print("\nTop-10 hardest sequences:")
    print(f"{'Name':<35} {'Tier':<8} {'Overall':>7} {'Motion':>7} {'Scale':>6} {'Frames':>7}")
    print("-" * 75)
    for d in difficulties[:10]:
        print(f"{d.sequence_name:<35} {d.tier:<8} {d.overall_score:>7.4f} "
              f"{d.motion_score:>7.4f} {d.scale_score:>6.4f} {d.num_frames:>7}")

    # -------------------------------------------------------------------
    # Optional: stratified accuracy report from benchmark JSON
    # -------------------------------------------------------------------
    if args.benchmark_json:
        from eovot.benchmark.engine import BenchmarkResult
        bench = BenchmarkResult.load(args.benchmark_json)
        report = analyzer.stratified_report(bench)
        md = report.to_markdown()
        md_path = out_dir / "stratified_report.md"
        md_path.write_text(md)
        print(f"\nStratified report → {md_path}")
        print("\n" + md)


if __name__ == "__main__":
    main()
