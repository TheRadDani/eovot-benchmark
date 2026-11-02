#!/usr/bin/env python3
"""CLI tool: analyse the IoU distribution of benchmark results.

Loads one or more JSON result files produced by the EOVOT benchmark engine
and produces a distribution-level analysis for each tracker.  Unlike the
standard summary (which only reports mean IoU), this tool reveals *how* IoU
values are distributed — whether a tracker is consistently mediocre or
alternates between perfect tracking and total failure.

Key output metrics
------------------
- **Bimodality Coefficient (BC)**: > 0.556 signals an oscillating tracker.
- **Failure ratio**: fraction of frames with IoU < 0.1.
- **Excellent ratio**: fraction of frames with IoU > 0.5.
- **IQR**: interquartile range showing distribution spread.

Usage
-----
Single tracker::

    python scripts/analyze_iou_distribution.py \\
        --results results/KCF-OTB100.json

Multiple trackers (comparison table)::

    python scripts/analyze_iou_distribution.py \\
        --results results/MOSSE-OTB100.json results/KCF-OTB100.json \\
        --output-dir results/analysis/

Using the synthetic dataset for a quick run::

    python scripts/run_experiment.py \\
        --config configs/experiments/full_tracker_comparison.yaml
    python scripts/analyze_iou_distribution.py \\
        --results results/experiments/full-tracker-comparison-synthetic/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from eovot.analysis.iou_distribution import IoUDistributionAnalyzer
import numpy as np


def _load_result(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_seq_ious(result: Dict) -> Dict[str, np.ndarray]:
    """Extract per-sequence IoU arrays from a benchmark result dict."""
    seq_ious: Dict[str, np.ndarray] = {}
    for seq in result.get("sequences", []):
        name = seq.get("sequence_name", "?")
        raw = seq.get("ious")
        if raw is not None:
            seq_ious[name] = np.array(raw, dtype=np.float64)
        elif "mean_iou" in seq:
            seq_ious[name] = np.array([float(seq["mean_iou"])], dtype=np.float64)
    return seq_ious


def _tracker_name(result: Dict) -> str:
    s = result.get("summary", {})
    return s.get("tracker") or s.get("tracker_name", "unknown")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyse per-frame IoU distributions of EOVOT benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        metavar="RESULT_JSON",
        help="One or more JSON result files from the EOVOT benchmark engine.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Directory to write per-tracker distribution JSON and comparison table.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.1,
        metavar="F",
        help="IoU below which a frame counts as a failure.  Default: 0.1.",
    )
    parser.add_argument(
        "--excellent-threshold",
        type=float,
        default=0.5,
        metavar="E",
        help="IoU above which a frame counts as excellent.  Default: 0.5.",
    )
    parser.add_argument(
        "--per-sequence",
        action="store_true",
        help="Print per-sequence results in addition to the aggregate summary.",
    )
    args = parser.parse_args(argv)

    analyzer = IoUDistributionAnalyzer(
        failure_threshold=args.failure_threshold,
        excellent_threshold=args.excellent_threshold,
    )

    all_aggregates: List[Dict] = []

    for result_path in args.results:
        p = Path(result_path)
        if not p.exists():
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
            return 1

        result = _load_result(p)
        tracker = _tracker_name(result)
        seq_ious = _extract_seq_ious(result)

        if not seq_ious:
            print(
                f"[WARN] No per-frame IoU data in {p.name}. "
                "Re-run the benchmark with a current engine version.",
                file=sys.stderr,
            )
            continue

        agg_result = analyzer.analyze_benchmark(seq_ious, tracker_name=tracker)
        agg = agg_result["aggregate"]
        all_aggregates.append(agg)

        print(f"\n{'=' * 60}")
        print(f"  Tracker : {tracker}")
        print(f"  File    : {p.name}")
        print(f"{'=' * 60}")
        print(f"  Sequences         : {agg['num_sequences']}")
        print(f"  Mean IoU          : {agg['mean_iou']:.4f}")
        print(f"  Mean Median IoU   : {agg['mean_median_iou']:.4f}")
        print(f"  Mean Std IoU      : {agg['mean_std_iou']:.4f}")
        print(f"  Mean Failure %    : {agg['mean_failure_ratio'] * 100:.1f} %  (IoU < {args.failure_threshold})")
        print(f"  Mean Excellent %  : {agg['mean_excellent_ratio'] * 100:.1f} %  (IoU > {args.excellent_threshold})")
        bc = agg['mean_bimodality_coefficient']
        bimodal_flag = "  <- OSCILLATING TRACKER" if bc > 5.0 / 9.0 else ""
        print(f"  Mean BC           : {bc:.4f}{bimodal_flag}")
        print(
            f"  Bimodal sequences : {agg['bimodal_sequence_count']} / {agg['num_sequences']} "
            f"({agg['bimodal_sequence_fraction'] * 100:.0f} %)"
        )

        if args.per_sequence:
            print(f"\n  {'Sequence':<30s} {'mIoU':>6} {'Median':>6} {'BC':>6} {'Fail%':>7} {'Excel%':>7}")
            print(f"  {'-' * 65}")
            for seq_name, seq_result in agg_result["per_sequence"].items():
                bimodal = " !" if seq_result.is_bimodal else "  "
                print(
                    f"  {seq_name:<30s} "
                    f"{seq_result.mean_iou:>6.3f} "
                    f"{seq_result.median_iou:>6.3f} "
                    f"{seq_result.bimodality_coefficient:>6.3f}{bimodal} "
                    f"{seq_result.failure_ratio * 100:>6.1f} "
                    f"{seq_result.excellent_ratio * 100:>6.1f}"
                )

        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{tracker}-iou-distribution.json"
            per_seq_dicts = {
                k: v.to_dict() for k, v in agg_result["per_sequence"].items()
            }
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump({"aggregate": agg, "per_sequence": per_seq_dicts}, fh, indent=2)
            print(f"\n  Distribution data saved -> {out_path}")

    if len(all_aggregates) > 1:
        rows = sorted(
            all_aggregates,
            key=lambda x: x.get("mean_bimodality_coefficient", 1.0),
        )
        print(f"\n{'=' * 60}")
        print("  TRACKER COMPARISON (sorted by BC, most consistent first)")
        print(f"{'=' * 60}")
        print(analyzer.to_markdown_table(rows))

        if args.output_dir:
            out_dir = Path(args.output_dir)
            table_path = out_dir / "iou-distribution-comparison.md"
            with open(table_path, "w", encoding="utf-8") as fh:
                fh.write("# EOVOT IoU Distribution Comparison\n\n")
                fh.write(analyzer.to_markdown_table(rows))
                fh.write("\n\n**BC** = Bimodality Coefficient (> 0.556 = oscillating tracker)\n")
            print(f"\n  Comparison table saved -> {table_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
