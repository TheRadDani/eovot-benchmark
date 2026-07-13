"""Sequence difficulty analysis script for EOVOT.

Runs one or more trackers on a dataset, then uses :class:`SequenceAnalyzer`
to break down performance by difficulty tier (EASY / MEDIUM / HARD).

Usage
-----
    # Quick test on the built-in SyntheticDataset (no data download required)
    python scripts/analyze_sequence_difficulty.py

    # Run on a real dataset with multiple trackers:
    python scripts/analyze_sequence_difficulty.py \\
        --dataset-loader OTBDataset \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100 \\
        --trackers MOSSE KCF CSRT \\
        --max-sequences 20 \\
        --output-dir results/difficulty_analysis

Output
------
For each tracker the script writes:
  - ``<output_dir>/<tracker>_difficulty_report.md`` — full Markdown report
  - ``<output_dir>/<tracker>_difficulty_summary.txt`` — console summary

It also prints a comparison table of per-tier mIoU across all trackers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.analysis.sequence_analyzer import SequenceAnalyzer
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


def _build_dataset(args: argparse.Namespace):
    if args.dataset_loader == "SyntheticDataset":
        return SyntheticDataset(
            num_sequences=args.max_sequences or 10,
            num_frames=80,
            frame_size=(320, 240),
            bbox_size=(40, 30),  # keep target well within frame boundaries
            motion="linear",
            seed=42,
        ), "Synthetic-Linear"

    from eovot.datasets.base import OTBDataset
    from eovot.datasets.got10k import GOT10kDataset
    from eovot.datasets.lasot import LaSOTDataset

    loaders = {
        "OTBDataset": OTBDataset,
        "GOT10kDataset": GOT10kDataset,
        "LaSOTDataset": LaSOTDataset,
    }
    cls = loaders.get(args.dataset_loader)
    if cls is None:
        print(f"[ERROR] Unknown loader: {args.dataset_loader}", file=sys.stderr)
        sys.exit(1)
    if args.dataset_loader == "OTBDataset":
        return cls(root=args.dataset_root), args.dataset_name
    return cls(root=args.dataset_root, split="val"), args.dataset_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EOVOT Sequence Difficulty Analyzer"
    )
    parser.add_argument(
        "--dataset-loader", default="SyntheticDataset",
        choices=["SyntheticDataset", "OTBDataset", "GOT10kDataset", "LaSOTDataset"],
        help="Dataset loader class (default: SyntheticDataset).",
    )
    parser.add_argument("--dataset-root", default=None,
                        help="Path to dataset root (required for real datasets).")
    parser.add_argument("--dataset-name", default="Synthetic",
                        help="Label used in output files.")
    parser.add_argument(
        "--trackers", nargs="+",
        default=["KCF", "CamShift"],
        choices=list(TRACKER_REGISTRY),
        help="Trackers to evaluate (default: KCF CamShift).",
    )
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Evaluate only the first N sequences.")
    parser.add_argument("--output-dir", default="results/difficulty_analysis",
                        help="Directory for output files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, dataset_name = _build_dataset(args)
    engine = BenchmarkEngine(verbose=True)
    analyzer = SequenceAnalyzer()

    all_reports = {}
    for tracker_name in args.trackers:
        tracker = build_tracker(tracker_name)
        result = engine.run(
            tracker=tracker,
            dataset=dataset,
            dataset_name=dataset_name,
            max_sequences=args.max_sequences,
        )
        report = SequenceAnalyzer.from_benchmark_result(result)
        all_reports[tracker_name] = report

        # Write Markdown report
        md_path = output_dir / f"{tracker_name}_difficulty_report.md"
        md_path.write_text(report["markdown_table"], encoding="utf-8")

        # Write summary text
        txt_path = output_dir / f"{tracker_name}_difficulty_summary.txt"
        txt_path.write_text(report["tier_summary"], encoding="utf-8")

        print(report["tier_summary"])

    # Cross-tracker comparison
    if len(all_reports) > 1:
        from eovot.analysis.sequence_analyzer import DifficultyTier
        print("\n" + "=" * 70)
        print("  CROSS-TRACKER DIFFICULTY COMPARISON")
        print("=" * 70)
        header = f"{'Tier':<8}" + "".join(f"{n:>14}" for n in args.trackers)
        print(header)
        print("-" * len(header))
        for tier in DifficultyTier:
            row = f"{tier.value:<8}"
            for tracker_name in args.trackers:
                tp = all_reports[tracker_name]["tier_performance"].get(tier)
                if tp and tp.num_sequences > 0:
                    row += f"{tp.mean_iou:>13.4f} "
                else:
                    row += f"{'—':>13} "
            print(row)
        print("=" * 70)


if __name__ == "__main__":
    main()
