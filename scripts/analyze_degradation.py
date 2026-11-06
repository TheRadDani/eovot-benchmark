#!/usr/bin/env python3
"""Image degradation robustness analysis CLI for EOVOT.

Evaluates one or more trackers across four degradation types
(Gaussian noise, Gaussian blur, JPEG compression, brightness reduction)
at configurable severity levels and writes robustness reports.

Usage::

    # Quick sanity check using the synthetic dataset
    python scripts/analyze_degradation.py \\
        --trackers MOSSE KCF \\
        --synthetic \\
        --synthetic-sequences 3 --synthetic-frames 40

    # On OTB-100
    python scripts/analyze_degradation.py \\
        --trackers MOSSE KCF CSRT \\
        --dataset-root /data/OTB100 \\
        --max-sequences 10 \\
        --output-dir results/degradation/

    # Custom severity sweep
    python scripts/analyze_degradation.py \\
        --trackers MOSSE \\
        --synthetic \\
        --severities 0.0 0.2 0.4 0.6 0.8 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.analysis.degradation_analysis import (
    DegradationRobustnessAnalyzer,
    DegradationReport,
    DegradationType,
)
from eovot.datasets.base import OTBDataset
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.registry import TRACKER_REGISTRY


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_degradation",
        description="EOVOT — Image Degradation Robustness Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["MOSSE", "KCF"],
        choices=list(TRACKER_REGISTRY),
        metavar="TRACKER",
        help="Trackers to evaluate.",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        metavar="S",
        help="Severity levels (0=clean, 1=maximum degradation).",
    )
    parser.add_argument(
        "--degradation-types",
        nargs="+",
        default=None,
        choices=[dt.value for dt in DegradationType],
        metavar="TYPE",
        help="Subset of degradation types to test (default: all four).",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        metavar="DIR",
        help="Path to an OTB-style dataset root directory.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Limit evaluation to the first N sequences.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/degradation/",
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-severity progress.",
    )

    syn = parser.add_argument_group("synthetic dataset")
    syn.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the built-in synthetic dataset (no download required).",
    )
    syn.add_argument("--synthetic-sequences", type=int, default=5, metavar="N")
    syn.add_argument("--synthetic-frames", type=int, default=60, metavar="N")
    syn.add_argument(
        "--synthetic-motion",
        default="linear",
        choices=["linear", "circular", "random"],
    )
    return parser


def _report_to_dict(report: DegradationReport) -> dict:
    return {
        "tracker": report.tracker_name,
        "mean_dts": round(report.mean_dts, 4),
        "sweeps": [
            {
                "degradation_type": s.degradation_type.value,
                "dts": round(s.dts, 4),
                "clean_iou": round(s.clean_iou, 4),
                "half_life_severity": round(s.half_life, 4) if s.half_life is not None else None,
                "severities": [round(v, 4) for v in s.severities],
                "mean_ious": [round(v, 4) for v in s.mean_ious],
            }
            for s in report.sweeps
        ],
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.synthetic and not args.dataset_root:
        parser.print_help()
        sys.exit(0)

    # Build dataset
    if args.synthetic:
        dataset = SyntheticDataset(
            num_sequences=args.synthetic_sequences,
            num_frames=args.synthetic_frames,
            motion=args.synthetic_motion,
        )
        dataset_label = "Synthetic"
    else:
        dataset = OTBDataset(args.dataset_root)
        dataset_label = Path(args.dataset_root).name

    # Optionally cap the number of sequences
    if args.max_sequences is not None:
        from eovot.datasets.base import BaseDataset, Sequence

        class _CappedDataset(BaseDataset):
            def __init__(self, inner, cap):
                self._inner = inner
                self._cap = min(cap, len(inner))

            def __len__(self):
                return self._cap

            def __getitem__(self, idx):
                return self._inner[idx]

        dataset = _CappedDataset(dataset, args.max_sequences)

    # Build degradation type filter
    if args.degradation_types:
        deg_types = [DegradationType(v) for v in args.degradation_types]
    else:
        deg_types = None  # all

    analyzer = DegradationRobustnessAnalyzer(
        severities=sorted(set(args.severities)),
        degradation_types=deg_types,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_reports = []

    for tracker_name in args.trackers:
        print(f"\n{'=' * 60}")
        print(f"  Degradation analysis: {tracker_name}  |  {dataset_label}")
        print(f"{'=' * 60}")

        tracker = TRACKER_REGISTRY[tracker_name]()
        report = analyzer.analyze(tracker, dataset, verbose=args.verbose)
        all_reports.append(report)
        print(report)

        # Save per-tracker JSON
        json_path = output_dir / f"degradation_{tracker_name}_{dataset_label}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(_report_to_dict(report), fh, indent=2)
        print(f"\n  [JSON] saved → {json_path}")

    # Comparison table (always written)
    if all_reports:
        md_table = analyzer.to_markdown_table(all_reports)
        print("\n" + md_table)

        md_path = output_dir / f"degradation_comparison_{dataset_label}.md"
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(f"# EOVOT Degradation Robustness Comparison — {dataset_label}\n\n")
            fh.write(md_table)
            fh.write("\n\n")
            fh.write("## Column key\n\n")
            fh.write(
                "Each column shows the **Degradation Tolerance Score (DTS)** for that\n"
                "degradation type.  DTS is the normalised AUC of the severity → mean-IoU\n"
                "curve: `1.0` means zero IoU loss across the full severity range;\n"
                "`0.0` means immediate collapse at any degradation.\n"
            )
        print(f"\n  [MARKDOWN] saved → {md_path}")


if __name__ == "__main__":
    main()
