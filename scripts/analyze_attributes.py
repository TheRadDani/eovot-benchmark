#!/usr/bin/env python3
"""Per-attribute benchmark analysis CLI for EOVOT.

Runs one or more trackers on a dataset, auto-annotates sequences with
challenge attributes inferred from ground-truth box trajectories, and
produces a per-attribute breakdown table showing where each tracker
excels or struggles.

Usage examples
--------------
Single tracker, synthetic dataset::

    python scripts/analyze_attributes.py \\
        --trackers MOSSE \\
        --dataset synthetic \\
        --num-sequences 20 --num-frames 150

Multiple trackers, OTB100::

    python scripts/analyze_attributes.py \\
        --trackers MOSSE KCF CSRT \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100 \\
        --output-dir results/attributes

The script prints a Markdown table (and optionally saves it) showing
mIoU and Success AUC for each challenge attribute, enabling direct
comparison of tracker strengths and weaknesses across challenge categories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.base import OTBDataset
from eovot.datasets.synthetic import SyntheticDataset
from eovot.metrics.attributes import AttributeAnalyzer, auto_annotate_from_gt
from eovot.trackers.csrt import CSRTTracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.median_flow import MedianFlowTracker
from eovot.trackers.mil import MILTracker
from eovot.trackers.mosse import MOSSETracker

TRACKER_REGISTRY = {
    "MOSSE": MOSSETracker,
    "KCF": KCFTracker,
    "CSRT": CSRTTracker,
    "MIL": MILTracker,
    "MedianFlow": MedianFlowTracker,
}


def build_annotations(result: BenchmarkResult, frame_size=(640, 480)) -> Dict:
    """Infer attribute annotations from GT boxes stored in BenchmarkResult."""
    annotations = {}
    for sr in result.sequence_results:
        if sr.ground_truths is not None and len(sr.ground_truths) > 1:
            annotations[sr.sequence_name] = auto_annotate_from_gt(
                sr.ground_truths, frame_size=frame_size
            )
    return annotations


def run_analysis(args: argparse.Namespace) -> None:
    # --- Build dataset ---
    if args.dataset == "synthetic":
        dataset = SyntheticDataset(
            num_sequences=args.num_sequences,
            num_frames=args.num_frames,
            frame_size=(args.frame_width, args.frame_height),
            motion=args.motion,
            seed=args.seed,
        )
        dataset_name = f"Synthetic-{args.motion}"
        frame_size = (args.frame_width, args.frame_height)
    else:
        if not args.dataset_root:
            raise ValueError("--dataset-root is required for OTBDataset.")
        dataset = OTBDataset(root=args.dataset_root)
        dataset_name = args.dataset_name or "OTB"
        frame_size = (args.frame_width, args.frame_height)

    # --- Build trackers ---
    trackers = []
    for name in args.trackers:
        if name not in TRACKER_REGISTRY:
            raise ValueError(
                f"Unknown tracker '{name}'. Available: {list(TRACKER_REGISTRY)}"
            )
        trackers.append(TRACKER_REGISTRY[name]())

    # --- Run benchmark ---
    engine = BenchmarkEngine(verbose=not args.quiet)
    results: List[BenchmarkResult] = []
    for tracker in trackers:
        result = engine.run(
            tracker=tracker,
            dataset=dataset,
            dataset_name=dataset_name,
            max_sequences=args.max_sequences,
        )
        results.append(result)

    # --- Auto-annotate from the first result's GT boxes ---
    annotations = build_annotations(results[0], frame_size=frame_size)

    attr_counts = {k: len(v) for k, v in {}.items()}
    n_annotated = sum(1 for s in annotations if annotations[s])
    print(f"\nAnnotated {n_annotated}/{len(annotations)} sequences with attributes.")

    # --- Attribute analysis ---
    analyzer = AttributeAnalyzer()

    if len(results) == 1:
        breakdown = analyzer.analyze(results[0], annotations)
        table = analyzer.to_markdown_table(breakdown)
        print("\n" + table)
    else:
        multi = analyzer.compare_trackers(results, annotations)
        table = analyzer.to_comparison_table(multi, metric="mean_iou")
        print("\n## mIoU by Attribute\n")
        print(table)

        table_auc = analyzer.to_comparison_table(multi, metric="mean_success_auc")
        print("\n## Success AUC by Attribute\n")
        print(table_auc)

    # --- Save outputs ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if len(results) == 1:
            breakdown = analyzer.analyze(results[0], annotations)
            records = [r.to_dict() for r in breakdown]
        else:
            records = []
            for tracker_name, breakdown in multi.items():
                records.extend(r.to_dict() for r in breakdown)

        json_path = out_dir / "attribute_analysis.json"
        with open(json_path, "w") as fh:
            json.dump(records, fh, indent=2)
        print(f"\nSaved JSON results → {json_path}")

        md_path = out_dir / "attribute_analysis.md"
        md_path.write_text(table, encoding="utf-8")
        print(f"Saved Markdown table → {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-attribute benchmark analysis for EOVOT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["MOSSE"],
        metavar="TRACKER",
        help=f"Tracker(s) to evaluate. Available: {list(TRACKER_REGISTRY)}",
    )
    parser.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "otb"],
        help="Dataset type (default: synthetic).",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        metavar="PATH",
        help="Root directory for OTB dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        metavar="NAME",
        help="Human-readable dataset label (default: derived from --dataset).",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10,
        help="Number of synthetic sequences (default: 10).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=150,
        help="Frames per synthetic sequence (default: 150).",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=320,
        help="Frame width in pixels (default: 320).",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=240,
        help="Frame height in pixels (default: 240).",
    )
    parser.add_argument(
        "--motion",
        default="linear",
        choices=["linear", "circular", "random"],
        help="Synthetic motion pattern (default: linear).",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of sequences evaluated per tracker.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic dataset (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to write JSON and Markdown outputs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sequence benchmark progress.",
    )

    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
