"""CLI: cold-start analysis for a set of trackers on a synthetic dataset.

Measures tracker initialization latency vs steady-state update latency and
produces a Markdown report + JSON summary.  Designed to be run without any
external dataset download — uses the built-in SyntheticDataset.

Usage::

    python scripts/analyze_cold_start.py \\
        --trackers MOSSE KCF CSRT \\
        --num-sequences 10 \\
        --num-frames 100 \\
        --output results/cold_start

The script:
1. Runs each tracker on a SyntheticDataset (linear motion by default).
2. Collects ``init_latency_ms`` and ``mean_latency_ms`` per sequence.
3. Produces ``cold_start_report.md`` and ``cold_start_report.json``.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the package is importable when run from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.registry import build_tracker, available_trackers
from eovot.analysis.cold_start import ColdStartAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cold-start latency analysis for EOVOT trackers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["MOSSE", "KCF", "CSRT"],
        help=f"Tracker names to evaluate. Available: {available_trackers()}",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10,
        help="Number of synthetic sequences per tracker.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Frames per synthetic sequence.",
    )
    parser.add_argument(
        "--motion",
        choices=["linear", "circular", "random"],
        default="linear",
        help="SyntheticDataset motion pattern.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/cold_start",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--warn-ratio",
        type=float,
        default=10.0,
        help="Init-to-update ratio threshold that triggers a warning.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticDataset(
        num_sequences=args.num_sequences,
        num_frames=args.num_frames,
        motion=args.motion,
    )
    engine = BenchmarkEngine(verbose=True)
    analyzer = ColdStartAnalyzer(warn_threshold_ratio=args.warn_ratio)

    results = []
    for name in args.trackers:
        try:
            tracker = build_tracker(name)
        except ValueError as e:
            print(f"[skip] {e}")
            continue
        result = engine.run(tracker, dataset, dataset_name=f"Synthetic-{args.motion}")
        results.append(result)

    reports = analyzer.compare(results)

    # Write per-tracker Markdown reports.
    for report in reports:
        md_path = out_dir / f"cold_start_{report.tracker_name}.md"
        md_path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"  → {md_path}")

    # Write aggregate comparison.
    comparison_md = analyzer.to_comparison_markdown(reports)
    comp_path = out_dir / "cold_start_comparison.md"
    comp_path.write_text(comparison_md, encoding="utf-8")
    print(f"  → {comp_path}")

    # Write JSON summary.
    summary = [r.to_dict() for r in reports]
    json_path = out_dir / "cold_start_report.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  → {json_path}")

    print("\n" + comparison_md)


if __name__ == "__main__":
    main()
