#!/usr/bin/env python3
"""Per-attribute success and precision curve analysis for EOVOT.

Runs a tracker on a synthetic dataset, groups sequences by auto-detected
VOT challenge attributes, and computes pooled success/precision curves for
each attribute group.  Results are written as JSON and CSV; an optional
Markdown summary is printed to stdout.

Usage::

    # Quick run on synthetic data (no real dataset needed)
    python scripts/per_attribute_analysis.py \\
        --tracker KCF \\
        --output-dir results/per_attribute

    # Compare two trackers, save curves for each
    python scripts/per_attribute_analysis.py \\
        --trackers MOSSE KCF DSST \\
        --num-sequences 30 \\
        --num-frames 200 \\
        --output-dir results/per_attribute

    # Use a real dataset (OTB-style directory)
    python scripts/per_attribute_analysis.py \\
        --trackers KCF DSST \\
        --dataset-root /data/OTB100 \\
        --dataset-loader OTBDataset \\
        --output-dir results/per_attribute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.metrics.attribute_curves import AttributeCurveAnalyzer
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker

# Dataset loaders available for real datasets.
_DATASET_LOADERS = {}
try:
    from eovot.datasets.base import OTBDataset
    _DATASET_LOADERS["OTBDataset"] = OTBDataset
except ImportError:
    pass
try:
    from eovot.datasets.got10k import GOT10kDataset
    _DATASET_LOADERS["GOT10kDataset"] = GOT10kDataset
except ImportError:
    pass
try:
    from eovot.datasets.lasot import LaSOTDataset
    _DATASET_LOADERS["LaSOTDataset"] = LaSOTDataset
except ImportError:
    pass


def _build_synthetic(num_sequences: int, num_frames: int, motion: str, seed: int):
    return SyntheticDataset(
        num_sequences=num_sequences,
        num_frames=num_frames,
        frame_size=(320, 240),
        bbox_size=(40, 40),
        motion=motion,
        seed=seed,
    )


def _build_real(loader_name: str, root: str, split: str, max_sequences):
    if loader_name not in _DATASET_LOADERS:
        raise ValueError(
            f"Dataset loader '{loader_name}' is not available. "
            f"Available: {list(_DATASET_LOADERS)}"
        )
    cls = _DATASET_LOADERS[loader_name]
    if loader_name in ("GOT10kDataset", "LaSOTDataset"):
        return cls(root=root, split=split, max_sequences=max_sequences)
    return cls(root=root)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="per_attribute_analysis",
        description=(
            "Compute per-attribute success/precision curves for one or more "
            "EOVOT trackers and save results as JSON and CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["MOSSE", "KCF"],
        metavar="TRACKER",
        help=f"Tracker names. Choices: {sorted(TRACKER_REGISTRY)}",
    )
    parser.add_argument(
        "--output-dir",
        default="results/per_attribute",
        help="Directory for JSON and CSV output.",
    )

    # Synthetic dataset options (used when --dataset-root is not given).
    grp_syn = parser.add_argument_group("Synthetic dataset (default)")
    grp_syn.add_argument("--num-sequences", type=int, default=20, metavar="N")
    grp_syn.add_argument("--num-frames", type=int, default=150, metavar="N")
    grp_syn.add_argument(
        "--motion",
        default="linear",
        choices=["linear", "circular", "random"],
        help="Motion pattern for the synthetic dataset.",
    )
    grp_syn.add_argument("--seed", type=int, default=42)

    # Real dataset options.
    grp_real = parser.add_argument_group("Real dataset (optional)")
    grp_real.add_argument(
        "--dataset-root",
        default=None,
        metavar="DIR",
        help="Path to dataset root.  When given, synthetic options are ignored.",
    )
    grp_real.add_argument(
        "--dataset-loader",
        default="OTBDataset",
        choices=list(_DATASET_LOADERS) or ["OTBDataset"],
        help="Dataset loader class.",
    )
    grp_real.add_argument("--split", default="val")
    grp_real.add_argument("--max-sequences", type=int, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = BenchmarkEngine(verbose=True)
    curve_analyzer = AttributeCurveAnalyzer()

    for tracker_name in args.trackers:
        if tracker_name not in TRACKER_REGISTRY:
            print(
                f"[WARN] '{tracker_name}' is not in the registry. "
                f"Available: {sorted(TRACKER_REGISTRY)}",
                file=sys.stderr,
            )
            continue

        print(f"\n{'=' * 60}")
        print(f"  Tracker: {tracker_name}")
        print(f"{'=' * 60}")

        tracker = build_tracker(tracker_name)

        if args.dataset_root:
            dataset = _build_real(
                args.dataset_loader,
                args.dataset_root,
                args.split,
                args.max_sequences,
            )
            dataset_name = args.dataset_loader
        else:
            dataset = _build_synthetic(
                args.num_sequences, args.num_frames, args.motion, args.seed
            )
            dataset_name = f"Synthetic-{args.motion}-{args.num_sequences}seq"

        result = engine.run(
            tracker=tracker,
            dataset=dataset,
            dataset_name=dataset_name,
        )

        curves = curve_analyzer.compute(result)

        # Save outputs.
        stem = f"{tracker_name}-{dataset_name}"
        json_path = curves.save_json(output_dir / f"{stem}_attribute_curves.json")
        csv_path = curves.save_csv(output_dir / f"{stem}_attribute_curves.csv")

        print(f"\n[JSON] {json_path}")
        print(f"[CSV]  {csv_path}")
        print()
        print(curves.summary_table())

    print(f"\nAll results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
