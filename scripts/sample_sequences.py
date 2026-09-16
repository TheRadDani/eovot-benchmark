#!/usr/bin/env python3
"""CLI: score and sample dataset sequences by tracking difficulty.

Usage examples::

    # Score all sequences in an OTB-100 dataset
    python scripts/sample_sequences.py \\
        --dataset-root /data/OTB100 --dataset-type otb

    # Sample 20 stratified sequences for quick evaluation
    python scripts/sample_sequences.py \\
        --dataset-root /data/GOT10k/test --dataset-type got10k \\
        --sample 20 --bins 5 --seed 42 \\
        --output sampled_sequences.txt

    # Bias toward motion-heavy sequences
    python scripts/sample_sequences.py \\
        --dataset-root /data/LaSOT --dataset-type lasot \\
        --weight motion_speed 2.0 --weight target_smallness 1.5 \\
        --sample 30

    # Show top-10 hardest sequences
    python scripts/sample_sequences.py \\
        --dataset-root /data/OTB100 --dataset-type otb --top 10
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from eovot.datasets.sampler import SequenceDifficultyScorer, StratifiedSampler


def _load_dataset(dataset_type: str, root: str):
    if dataset_type == "otb":
        from eovot.datasets.base import OTBDataset
        return OTBDataset(root)
    if dataset_type == "got10k":
        from eovot.datasets.got10k import GOT10kDataset
        return GOT10kDataset(root)
    if dataset_type == "lasot":
        from eovot.datasets.lasot import LaSOTDataset
        return LaSOTDataset(root)
    raise ValueError(f"Unknown dataset type: {dataset_type!r}. Choose otb/got10k/lasot.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score and sample dataset sequences by tracking difficulty."
    )
    parser.add_argument(
        "--dataset-root", required=True,
        help="Path to dataset root directory.",
    )
    parser.add_argument(
        "--dataset-type", default="otb", choices=["otb", "got10k", "lasot"],
        help="Dataset type (default: otb).",
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Number of sequences to sample stratified by difficulty."
             " If omitted, all sequences are scored and listed.",
    )
    parser.add_argument(
        "--bins", type=int, default=5, metavar="K",
        help="Number of difficulty strata for stratified sampling (default: 5).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--weight", nargs=2, action="append", metavar=("COMPONENT", "VALUE"),
        help=(
            "Set a difficulty component weight (repeatable). "
            "Components: motion_speed, scale_variability, aspect_change, target_smallness."
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help="Write sampled sequence names to this file, one per line.",
    )
    parser.add_argument(
        "--top", type=int, default=None, metavar="N",
        help="Show only the top-N hardest sequences (overrides --sample).",
    )
    args = parser.parse_args()

    weights = None
    if args.weight:
        weights = {}
        for comp, val in args.weight:
            try:
                weights[comp] = float(val)
            except ValueError:
                parser.error(f"Weight value must be a number, got: {val!r}")

    print(f"Loading {args.dataset_type} dataset from: {args.dataset_root}")
    try:
        dataset = _load_dataset(args.dataset_type, args.dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(dataset)} sequences…")
    scorer = SequenceDifficultyScorer(weights=weights)
    scored = scorer.score_dataset(dataset)

    if args.top is not None:
        display = list(reversed(scored[-args.top:]))
        print(f"\nTop-{args.top} hardest sequences:")
        print(f"{'Rank':<5} {'Name':<40} {'Difficulty':>10}")
        print("-" * 58)
        for rank, s in enumerate(display, 1):
            print(f"{rank:<5} {s.name:<40} {s.difficulty_score:>10.4f}")
        return

    if args.sample is not None:
        sampler = StratifiedSampler(n_bins=args.bins, seed=args.seed)
        selected = sampler.sample(scored, n=args.sample)
        print(f"\nStratified sample: {len(selected)} / {len(scored)} sequences")
        print(f"{'Rank':<5} {'Name':<40} {'Difficulty':>10}  Components")
        print("-" * 85)
        for rank, s in enumerate(selected, 1):
            comps = "  ".join(f"{k[:4]}={v:.2f}" for k, v in s.components.items())
            print(f"{rank:<5} {s.name:<40} {s.difficulty_score:>10.4f}  {comps}")
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(s.name for s in selected) + "\n")
            print(f"\nSaved sequence names to: {out}")
        return

    print(f"\nAll {len(scored)} sequences (easiest → hardest):")
    print(f"{'Rank':<5} {'Name':<40} {'Difficulty':>10}")
    print("-" * 58)
    for rank, s in enumerate(scored, 1):
        print(f"{rank:<5} {s.name:<40} {s.difficulty_score:>10.4f}")


if __name__ == "__main__":
    main()
