#!/usr/bin/env python3
"""Cross-dataset generalization evaluation CLI for EOVOT.

Runs one or more trackers on multiple datasets and produces a Markdown
report showing per-dataset accuracy, a Spearman rank-correlation matrix
between datasets, and a per-tracker generalization score.

Usage examples::

    # Quick sanity check with the synthetic dataset (no real data needed)
    python scripts/cross_dataset_eval.py \\
        --trackers MOSSE KCF \\
        --synthetic \\
        --synthetic-sequences 10 \\
        --output results/cross_dataset.md

    # Real datasets (OTB + GOT-10k)
    python scripts/cross_dataset_eval.py \\
        --trackers MOSSE KCF CSRT \\
        --dataset-roots OTB100:/data/OTB100 GOT10k:/data/GOT-10k \\
        --max-sequences 20 \\
        --output results/cross_dataset.md

    # With CPU energy estimation
    python scripts/cross_dataset_eval.py \\
        --trackers MOSSE KCF \\
        --synthetic \\
        --tdp-watts 15.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.analysis.cross_dataset import CrossDatasetEvaluator
from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.lasot import LaSOTDataset
from eovot.datasets.synthetic import SyntheticDataset
from eovot.datasets.base import OTBDataset


_REAL_DATASET_LOADERS = {
    "OTBDataset": lambda root, split, max_seq: OTBDataset(root=root),
    "GOT10kDataset": lambda root, split, max_seq: GOT10kDataset(root=root, split=split, max_sequences=max_seq),
    "LaSOTDataset": lambda root, split, max_seq: LaSOTDataset(root=root, split=split, max_sequences=max_seq),
}


def _parse_dataset_roots(specs):
    """Parse ``name:path`` strings into a ``{name: path}`` dict."""
    out = {}
    for spec in specs:
        parts = spec.split(":", 1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"--dataset-roots entry must be 'Name:path', got: {spec!r}"
            )
        out[parts[0].strip()] = parts[1].strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cross_dataset_eval",
        description="Cross-dataset generalization evaluation for EOVOT trackers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Tracker selection
    parser.add_argument(
        "--trackers",
        nargs="+",
        required=True,
        metavar="NAME",
        help="One or more tracker registry keys, e.g. MOSSE KCF CSRT.",
    )

    # Dataset configuration
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the built-in SyntheticDataset (no real data required). "
             "Creates two synthetic datasets: SyntheticA and SyntheticB.",
    )
    parser.add_argument(
        "--synthetic-sequences",
        type=int,
        default=10,
        metavar="N",
        help="Number of sequences per synthetic dataset. Default: 10.",
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=[],
        metavar="NAME:PATH",
        help=(
            "Real datasets as 'Name:path' pairs. "
            "Name must be one of: OTBDataset, GOT10kDataset, LaSOTDataset."
        ),
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split for GOT10k / LaSOT. Default: val.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Cap on sequences per dataset (for fast evaluation). Default: no cap.",
    )

    # Profiling
    parser.add_argument(
        "--tdp-watts",
        type=float,
        default=None,
        metavar="W",
        help="CPU TDP in Watts for energy estimation. Omit to skip energy profiling.",
    )

    # Reporting
    parser.add_argument(
        "--rank-metric",
        default="success_auc",
        choices=["success_auc", "mean_iou", "mean_fps"],
        help="Metric used to rank trackers within each dataset. Default: success_auc.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write the Markdown report to this file (also printed to stdout).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-tracker progress output.",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Build dataset dict
    # -----------------------------------------------------------------------
    datasets = {}

    if args.synthetic:
        datasets["SyntheticA"] = SyntheticDataset(
            num_sequences=args.synthetic_sequences, seed=0
        )
        datasets["SyntheticB"] = SyntheticDataset(
            num_sequences=args.synthetic_sequences, seed=42
        )

    if args.dataset_roots:
        for name, root in _parse_dataset_roots(args.dataset_roots).items():
            loader_fn = _REAL_DATASET_LOADERS.get(name)
            if loader_fn is None:
                parser.error(
                    f"Unknown dataset loader '{name}'. "
                    f"Available: {list(_REAL_DATASET_LOADERS)}"
                )
            datasets[name] = loader_fn(root, args.split, args.max_sequences)

    if not datasets:
        parser.error("No datasets specified. Use --synthetic or --dataset-roots.")

    # -----------------------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------------------
    evaluator = CrossDatasetEvaluator(
        verbose=not args.quiet,
        tdp_watts=args.tdp_watts,
        rank_metric=args.rank_metric,
        max_sequences=args.max_sequences,
    )

    print(f"\nCross-dataset evaluation")
    print(f"  Trackers : {args.trackers}")
    print(f"  Datasets : {list(datasets.keys())}")
    print(f"  Metric   : {args.rank_metric}")
    print()

    result = evaluator.evaluate(args.trackers, datasets)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    report = evaluator.to_markdown_report(result)
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nReport written to: {out_path}")

    # Print generalization scores summary to stdout for CI integration
    print("\n## Generalization Scores Summary")
    gen_scores = evaluator.generalization_scores(result)
    for tracker, score in gen_scores.items():
        score_str = f"{score:.3f}" if score == score else "n/a"  # NaN check
        print(f"  {tracker:<20s} ρ_mean = {score_str}")


if __name__ == "__main__":
    main()
