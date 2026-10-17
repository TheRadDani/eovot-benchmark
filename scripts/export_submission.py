#!/usr/bin/env python3
"""Export EOVOT benchmark results in standard leaderboard submission formats.

Reads a saved :class:`~eovot.benchmark.engine.BenchmarkResult` JSON file and
writes per-sequence prediction files in the exact format expected by GOT-10k,
LaSOT, and OTB evaluation toolkits.

Usage::

    # Export as GOT-10k OPE submission (default)
    python scripts/export_submission.py \\
        --result results/mosse_got10k.json \\
        --output submissions/mosse \\
        --protocol got10k

    # Export in all three protocols at once
    python scripts/export_submission.py \\
        --result results/kcf_otb100.json \\
        --output submissions/kcf \\
        --all-protocols

    # Also write a per-sequence CSV for offline analysis
    python scripts/export_submission.py \\
        --result results/csrt_lasot.json \\
        --output submissions/csrt \\
        --protocol lasot \\
        --csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.benchmark.engine import BenchmarkResult
from eovot.reporting.result_exporter import StandardResultExporter


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="export_submission",
        description="Export EOVOT benchmark results in standard evaluation formats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result", "-r",
        required=True,
        metavar="JSON",
        help="Path to a saved BenchmarkResult JSON file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="submissions",
        metavar="DIR",
        help="Root output directory for exported files.",
    )
    parser.add_argument(
        "--protocol", "-p",
        default="got10k",
        choices=["got10k", "lasot", "otb"],
        help="Export format: got10k | lasot | otb.",
    )
    parser.add_argument(
        "--all-protocols",
        action="store_true",
        help="Export in all three protocols at once (ignores --protocol).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write a per-sequence CSV summary for offline analysis.",
    )
    args = parser.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        print(f"[error] Result file not found: {result_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading result from {result_path} …")
    result = BenchmarkResult.load(result_path)
    print(
        f"  Tracker:   {result.tracker_name}\n"
        f"  Dataset:   {result.dataset_name}\n"
        f"  Sequences: {len(result.sequence_results)}\n"
        f"  mIoU:      {result.mean_iou:.4f}\n"
        f"  FPS:       {result.mean_fps:.1f}"
    )

    exporter = StandardResultExporter(output_dir=args.output)

    if args.all_protocols:
        all_paths = exporter.export_all_protocols(result)
        for proto, paths in all_paths.items():
            print(f"\n[{proto}] Exported {len(paths)} sequence file(s) → {exporter.output_dir / proto}/")
    else:
        paths = exporter.export(result, protocol=args.protocol)
        print(f"\n[{args.protocol}] Exported {len(paths)} sequence file(s) → {exporter.output_dir / args.protocol}/")

    if args.csv:
        csv_path = exporter.export_summary_csv(result)
        print(f"[csv]    Per-sequence CSV written → {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
