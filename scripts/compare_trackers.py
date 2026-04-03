#!/usr/bin/env python3
"""Multi-tracker comparison CLI for EOVOT.

Runs two or more trackers on the same dataset and writes a side-by-side
comparison table in Markdown and CSV formats.

Usage::

    # Quick sanity check (10 sequences)
    python scripts/compare_trackers.py \\
        --trackers MOSSE KCF \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100 \\
        --max-sequences 10

    # Full OTB100 evaluation
    python scripts/compare_trackers.py \\
        --trackers MOSSE KCF \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100 \\
        --output-dir results/

    # GOT-10k validation split
    python scripts/compare_trackers.py \\
        --trackers MOSSE KCF \\
        --dataset-loader GOT10kDataset \\
        --dataset-root /data/GOT-10k \\
        --split val

    # With CPU energy estimation (laptop TDP)
    python scripts/compare_trackers.py \\
        --trackers MOSSE KCF \\
        --dataset-root /data/OTB100 \\
        --tdp-watts 15.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as ``python scripts/compare_trackers.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.base import OTBDataset
from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.lasot import LaSOTDataset
from eovot.reporting.reporter import BenchmarkReporter
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker

# ---------------------------------------------------------------------------
# Registries — add new trackers / datasets here without touching the CLI code
# ---------------------------------------------------------------------------

TRACKER_REGISTRY = {
    "MOSSE": MOSSETracker,
    "KCF": KCFTracker,
}

DATASET_REGISTRY = {
    "OTBDataset": OTBDataset,
    "GOT10kDataset": GOT10kDataset,
    "LaSOTDataset": LaSOTDataset,
}


def _build_dataset(loader_name: str, root: str, split: str, max_sequences):
    """Instantiate the dataset, handling different constructor signatures."""
    cls = DATASET_REGISTRY[loader_name]
    if loader_name == "GOT10kDataset":
        return cls(root=root, split=split, max_sequences=max_sequences)
    if loader_name == "LaSOTDataset":
        return cls(root=root, split=split, max_sequences=max_sequences)
    return cls(root=root)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compare_trackers",
        description="Run multiple EOVOT trackers on a dataset and generate a comparison table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["MOSSE", "KCF"],
        choices=list(TRACKER_REGISTRY),
        metavar="TRACKER",
        help=f"One or more tracker names. Choices: {list(TRACKER_REGISTRY)}",
    )
    parser.add_argument(
        "--dataset-loader",
        default="OTBDataset",
        choices=list(DATASET_REGISTRY),
        help="Dataset loader class to use.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Human-readable label used in reports (defaults to --dataset-loader).",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split (for GOT-10k: train | val | test; for LaSOT: train | test | all).",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Cap on the number of sequences evaluated (useful for quick tests).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Directory where JSON, CSV, and Markdown reports are saved.",
    )
    parser.add_argument(
        "--tdp-watts",
        type=float,
        default=None,
        metavar="W",
        help=(
            "Enable CPU energy estimation using this TDP value in Watts. "
            "Example: 6.0 for Raspberry Pi 4, 15.0 for a laptop CPU."
        ),
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name or args.dataset_loader
    reporter = BenchmarkReporter(output_dir=args.output_dir)
    engine = BenchmarkEngine(verbose=True, tdp_watts=args.tdp_watts)

    all_results = []

    for tracker_name in args.trackers:
        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {tracker_name} on {dataset_name}")
        print(f"{'=' * 60}")

        tracker = TRACKER_REGISTRY[tracker_name]()
        dataset = _build_dataset(
            args.dataset_loader, args.dataset_root, args.split, args.max_sequences
        )

        result = engine.run(
            tracker,
            dataset,
            dataset_name=dataset_name,
            max_sequences=args.max_sequences,
        )
        result_dict = result.to_dict()
        result_dict["summary"].setdefault("dataset_name", dataset_name)

        reporter.print_summary(result_dict)

        run_name = f"{tracker_name}-{dataset_name}"
        saved = reporter.save_all(result_dict, name=run_name)
        for fmt, path in saved.items():
            print(f"  [{fmt.upper()}] saved → {path}")

        all_results.append(result_dict)

    # -----------------------------------------------------------------------
    # Comparison table (only meaningful with 2+ trackers)
    # -----------------------------------------------------------------------
    if len(all_results) > 1:
        cmp_path = reporter.save_comparison(all_results, name=f"comparison-{dataset_name}")
        print(f"\n[COMPARISON TABLE] saved → {cmp_path}")
        print("\n" + reporter.comparison_table(all_results))


if __name__ == "__main__":
    main()
