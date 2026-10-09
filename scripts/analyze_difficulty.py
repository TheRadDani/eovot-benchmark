#!/usr/bin/env python3
"""Sequence difficulty scoring CLI for EOVOT.

Scores every sequence in a dataset by tracking difficulty and produces a
Markdown report with per-tier accuracy breakdown when a benchmark JSON is
also supplied.

Usage examples::

    # Score all sequences in an OTB dataset (240×320 frames)
    python scripts/analyze_difficulty.py \\
        --dataset-root /data/OTB100 \\
        --frame-height 240 --frame-width 320

    # Save CSV and JSON output
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
        --frame-height 480 --frame-width 640
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
from eovot.metrics.difficulty import (
    SequenceDifficultyScorer,
    score_dataset,
    stratify_benchmark_result,
)

DATASET_REGISTRY = {
    "OTBDataset": OTBDataset,
    "GOT10kDataset": GOT10kDataset,
    "LaSOTDataset": LaSOTDataset,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="analyze_difficulty",
        description=(
            "Score tracking sequences by difficulty and "
            "produce difficulty-stratified reports."
        ),
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
        "--frame-height", type=int, default=None,
        help="Frame height in pixels (enables small_target and out-of-view scoring).",
    )
    parser.add_argument(
        "--frame-width", type=int, default=None,
        help="Frame width in pixels.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/difficulty/",
        help="Directory for CSV, JSON, and Markdown output.",
    )
    parser.add_argument(
        "--benchmark-json",
        default=None,
        help="Path to a BenchmarkResult JSON (enables difficulty-stratified accuracy report).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_size = None
    if args.frame_height and args.frame_width:
        frame_size = (args.frame_height, args.frame_width)

    # -------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------
    cls = DATASET_REGISTRY[args.dataset_loader]
    if args.dataset_loader in ("GOT10kDataset", "LaSOTDataset"):
        dataset = cls(root=args.dataset_root, split=args.split,
                      max_sequences=args.max_sequences)
    else:
        dataset = cls(root=args.dataset_root)

    n = min(len(dataset), args.max_sequences) if args.max_sequences else len(dataset)
    print(f"Loaded {n} sequences from {args.dataset_root}")

    # -------------------------------------------------------------------
    # Score sequences
    # -------------------------------------------------------------------
    scorer = SequenceDifficultyScorer(frame_size=frame_size)
    report = score_dataset(dataset, scorer=scorer)

    tier_counts = {"easy": 0, "medium": 0, "hard": 0}
    from eovot.metrics.difficulty import _assign_tier
    for entry in report:
        tier_counts[_assign_tier(entry.difficulty_score)] += 1

    print(f"\nScored {len(report)} sequences")
    print(f"  easy={tier_counts['easy']}  medium={tier_counts['medium']}  hard={tier_counts['hard']}")

    # -------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------
    csv_path = out_dir / "difficulty_scores.csv"
    with open(csv_path, "w", newline="") as fh:
        fieldnames = ["sequence", "difficulty_score", "motion_speed", "scale_change",
                      "aspect_ratio_change", "small_target", "out_of_view_risk", "deformation"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in report:
            row = {"sequence": entry.sequence_name}
            row.update(entry.factors.to_dict())
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nDifficulty CSV → {csv_path}")

    # -------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------
    json_path = out_dir / "difficulty_scores.json"
    with open(json_path, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print(f"Difficulty JSON → {json_path}")

    # -------------------------------------------------------------------
    # Print top-10 hardest
    # -------------------------------------------------------------------
    print("\nTop-10 hardest sequences:")
    print(f"{'Name':<35} {'Score':>6} {'Motion':>7} {'Scale':>6}")
    print("-" * 60)
    for entry in report.hardest(10):
        f = entry.factors
        print(f"{entry.sequence_name:<35} {f.difficulty_score:>6.4f} "
              f"{f.motion_speed:>7.4f} {f.scale_change:>6.4f}")

    # -------------------------------------------------------------------
    # Optional: stratified accuracy report
    # -------------------------------------------------------------------
    if args.benchmark_json:
        from eovot.benchmark.engine import BenchmarkResult
        bench = BenchmarkResult.load(args.benchmark_json)
        strat_report = stratify_benchmark_result(bench, frame_size=frame_size, scorer=scorer)
        md = strat_report.to_markdown()
        md_path = out_dir / "stratified_report.md"
        md_path.write_text(md)
        print(f"\nStratified report → {md_path}")
        print("\n" + md)


if __name__ == "__main__":
    main()
