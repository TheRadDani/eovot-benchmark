"""CLI: compute and display sequence difficulty attributes for a dataset.

Usage examples::

    # Show attribute breakdown for all OTB-100 sequences
    python scripts/analyze_attributes.py --dataset OTBDataset --root /data/OTB100

    # GOT-10k validation split
    python scripts/analyze_attributes.py --dataset GOT10kDataset --root /data/GOT-10k --split val

    # Save full JSON report
    python scripts/analyze_attributes.py --dataset OTBDataset --root /data/OTB100 \\
        --output results/attributes.json

    # Only show sequences with fast motion
    python scripts/analyze_attributes.py --dataset OTBDataset --root /data/OTB100 --filter FM
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure eovot package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.analysis.sequence_attributes import tag_sequences, _ALL_FLAGS as ALL_FLAGS
from eovot.datasets.base import OTBDataset
from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.lasot import LaSOTDataset

_DATASET_REGISTRY = {
    "OTBDataset": OTBDataset,
    "GOT10kDataset": GOT10kDataset,
    "LaSOTDataset": LaSOTDataset,
}

_FLAG_FULL = {
    "SV": "Scale Variation",
    "ARC": "Aspect Ratio Change",
    "FM": "Fast Motion",
    "LR": "Low Resolution",
    "PO": "Partial Occlusion",
    "DEF": "Deformation",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute sequence difficulty attributes for a VOT dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, choices=list(_DATASET_REGISTRY),
                   help="Dataset loader class name.")
    p.add_argument("--root", required=True,
                   help="Path to dataset root directory.")
    p.add_argument("--split", default="val",
                   help="Dataset split (for GOT-10k / LaSOT). Default: val.")
    p.add_argument("--max-sequences", type=int, default=None,
                   help="Limit number of sequences analysed.")
    p.add_argument("--filter", choices=ALL_FLAGS, default=None,
                   help="Only print sequences that have this attribute flag.")
    p.add_argument("--output", default=None,
                   help="Save full results as JSON to this path.")
    p.add_argument("--summary-only", action="store_true",
                   help="Print only the aggregate summary table, not per-sequence list.")
    return p


def load_dataset(args: argparse.Namespace):
    cls = _DATASET_REGISTRY[args.dataset]
    kwargs = {"root": args.root}
    if args.dataset in ("GOT10kDataset", "LaSOTDataset"):
        kwargs["split"] = args.split
    if args.max_sequences is not None:
        kwargs["max_sequences"] = args.max_sequences
    return cls(**kwargs)


def print_summary_table(tagged: dict) -> None:
    counts = {f: 0 for f in ALL_FLAGS}
    total = len(tagged)
    for attrs in tagged.values():
        for flag in ALL_FLAGS:
            if attrs.flags.to_dict().get(flag, False):
                counts[flag] += 1

    print(f"\n{'='*52}")
    print(f"  Attribute Summary  ({total} sequences total)")
    print(f"{'='*52}")
    print(f"  {'Attribute':<25} {'Count':>6}  {'%':>6}")
    print(f"  {'-'*42}")
    for flag in ALL_FLAGS:
        pct = 100.0 * counts[flag] / total if total else 0.0
        print(f"  {_FLAG_FULL[flag] + ' (' + flag + ')':<25} {counts[flag]:>6}  {pct:>5.1f}%")
    print(f"{'='*52}\n")


def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading {args.dataset} from {args.root} ...")
    dataset = load_dataset(args)
    print(f"  → {len(dataset)} sequences found.")

    print("Computing sequence attributes ...")
    tagged = tag_sequences(dataset)

    if args.filter:
        filtered = {
            n: a for n, a in tagged.items()
            if a.flags.to_dict().get(args.filter, False)
        }
        print(f"\nSequences with {_FLAG_FULL[args.filter]} ({args.filter}): "
              f"{len(filtered)} / {len(tagged)}")
        if not args.summary_only:
            for name, attrs in sorted(filtered.items()):
                print(f"  {name:<35} {attrs.flags}")
    else:
        print_summary_table(tagged)
        if not args.summary_only:
            print(f"{'Sequence':<35} {'Frames':>6}  {'Area':>8}  {'Flags'}")
            print("-" * 75)
            for name, attrs in sorted(tagged.items()):
                flags_str = str(attrs.flags)
                print(
                    f"  {name:<33} {attrs.num_frames:>6}  "
                    f"{attrs.mean_area_px:>8.0f}  {flags_str}"
                )

    if args.output:
        out = {name: attrs.to_dict() for name, attrs in tagged.items()}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()
